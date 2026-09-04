"""Three-class next-session return forecaster.

Classes are encoded as 0=down, 1=neutral and 2=up.  The default thresholds
are -1% and +1%, but callers may choose economically meaningful alternatives.
"""

from __future__ import annotations

from dataclasses import replace
from fnmatch import fnmatch
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             log_loss, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .config import ForecastConfig
from .features import technical_warmup_bars
from .labels import TARGET_METADATA_COLUMNS
from .pipeline import _feature_columns, _select_model_rows, build_dataset


CLASS_NAMES = {0: "down", 1: "neutral", 2: "up"}


def _date_mask(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.Series:
    timestamp = pd.to_datetime(frame["timestamp"])
    mask = pd.Series(True, index=frame.index)
    if start:
        mask &= timestamp >= pd.Timestamp(start)
    if end:
        end_timestamp = pd.Timestamp(end)
        if end_timestamp == end_timestamp.normalize():
            end_timestamp += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        mask &= timestamp <= end_timestamp
    return mask


def _chronological_splits(frame: pd.DataFrame, config: ForecastConfig) -> Dict[str, pd.DataFrame]:
    if not config.test_start_date:
        test_index = int(len(frame) * (1.0 - config.test_fraction))
        train_pool, test = frame.iloc[:test_index].copy(), frame.iloc[test_index:].copy()
    else:
        test = frame.loc[_date_mask(frame, config.test_start_date, config.test_end_date)].copy()
        explicit_validation = bool(config.validation_start_date or config.validation_end_date)
        if explicit_validation:
            if not (config.validation_start_date and config.validation_end_date):
                raise ValueError("validation_start_date and validation_end_date must be provided together")
            train = frame.loc[_date_mask(frame, config.train_start_date, config.train_end_date)].copy()
            validation = frame.loc[_date_mask(
                frame, config.validation_start_date, config.validation_end_date
            )].copy()
            if not config.train_end_date:
                train = train[pd.to_datetime(train["timestamp"]) < pd.Timestamp(config.validation_start_date)]
            splits = {"train": train, "validation": validation, "test": test}
            if min(map(len, splits.values())) == 0:
                raise ValueError("An explicit train, validation, or test range has no labeled rows")
            if not (
                pd.Timestamp(train["timestamp"].iloc[-1])
                < pd.Timestamp(validation["timestamp"].iloc[0])
                <= pd.Timestamp(validation["timestamp"].iloc[-1])
                < pd.Timestamp(test["timestamp"].iloc[0])
            ):
                raise ValueError("Date ranges must be chronological: train < validation < test")
            train = train[pd.to_datetime(train[TARGET_METADATA_COLUMNS[0]]) < pd.Timestamp(validation["timestamp"].iloc[0])]
            validation = validation[pd.to_datetime(validation[TARGET_METADATA_COLUMNS[0]]) < pd.Timestamp(test["timestamp"].iloc[0])]
            return {"train": train, "validation": validation, "test": test}

        train_pool = frame.loc[_date_mask(frame, config.train_start_date, config.train_end_date)].copy()
        if not config.train_end_date:
            train_pool = train_pool[pd.to_datetime(train_pool["timestamp"]) < pd.Timestamp(config.test_start_date)]

    validation_start = int(len(train_pool) * (1.0 - config.validation_fraction))
    train = train_pool.iloc[:validation_start].copy()
    validation = train_pool.iloc[validation_start:].copy()
    if len(validation):
        train = train[pd.to_datetime(train[TARGET_METADATA_COLUMNS[0]]) < pd.Timestamp(validation["timestamp"].iloc[0])]
    if len(test):
        validation = validation[pd.to_datetime(validation[TARGET_METADATA_COLUMNS[0]]) < pd.Timestamp(test["timestamp"].iloc[0])]
    return {"train": train, "validation": validation, "test": test}


def _classify(values: pd.Series, down_threshold: float, up_threshold: float) -> np.ndarray:
    values = pd.to_numeric(values, errors="coerce").to_numpy(float)
    return np.where(values <= down_threshold, 0, np.where(values >= up_threshold, 2, 1)).astype(int)


def _metrics(actual: np.ndarray, probability: np.ndarray) -> Dict[str, Any]:
    predicted = probability.argmax(axis=1)
    report = classification_report(
        actual, predicted, labels=[0, 1, 2],
        target_names=[CLASS_NAMES[i] for i in range(3)],
        output_dict=True, zero_division=0,
    )
    result: Dict[str, Any] = {
        "rows": int(len(actual)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "log_loss": float(log_loss(actual, probability, labels=[0, 1, 2])),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1, 2]).tolist(),
        "actual_class_counts": {CLASS_NAMES[i]: int((actual == i).sum()) for i in range(3)},
        "predicted_class_counts": {CLASS_NAMES[i]: int((predicted == i).sum()) for i in range(3)},
        "per_class": {name: report[name] for name in ("down", "neutral", "up")},
    }
    try:
        result["roc_auc_ovr_macro"] = float(
            roc_auc_score(actual, probability, labels=[0, 1, 2], multi_class="ovr", average="macro")
        )
    except ValueError:
        result["roc_auc_ovr_macro"] = float("nan")
    return result


def train_three_class_forecaster(
    config: ForecastConfig,
    *,
    down_threshold: float = -0.01,
    up_threshold: float = 0.01,
    class_weighting: str | None = None,
    exclude_features: tuple[str, ...] = (),
    xgboost_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Train a down/neutral/up XGBoost model and save complete artifacts.

    ``class_weighting`` accepts ``None`` or ``"balanced"``. ``exclude_features``
    accepts exact feature names or shell-style patterns such as ``"tech_atr_*"``.
    Thresholds and
    class weights are derived from training configuration only; validation and
    test observations remain unweighted.
    """
    if not down_threshold < up_threshold:
        raise ValueError("down_threshold must be less than up_threshold")
    if class_weighting not in {None, "balanced"}:
        raise ValueError("class_weighting must be None or 'balanced'")
    if not 0.0 < config.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    # Reuse the exact-return label builder; class labels are derived below.
    run_config = replace(config, target_mode="exact_return", model_types=("xgboost",),
                         prediction_model="xgboost")
    out = Path(run_config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    data = build_dataset(run_config)
    warmup = max(run_config.min_history_bars, technical_warmup_bars(run_config))
    labeled = data.dropna(subset=["target_exact_return"]).iloc[warmup:].copy()
    labeled = _select_model_rows(labeled, run_config)
    features = _feature_columns(
        labeled,
        use_standard_technical_features=run_config.use_standard_technical_features,
        use_chan_bsp_features=run_config.use_chan_bsp_features,
    )
    excluded = [
        feature for feature in features
        if any(fnmatch(feature, pattern) for pattern in exclude_features)
    ]
    features = [feature for feature in features if feature not in set(excluded)]
    splits = _chronological_splits(labeled, run_config)
    if not features or min(map(len, splits.values())) == 0:
        raise ValueError("Not enough features or rows after chronological splitting")
    classes = {
        name: _classify(part["target_exact_return"], down_threshold, up_threshold)
        for name, part in splits.items()
    }
    if set(np.unique(classes["train"])) != {0, 1, 2}:
        raise ValueError("Training data must contain down, neutral, and up classes")

    allowed = {
        "n_estimators", "max_depth", "learning_rate", "subsample",
        "colsample_bytree", "min_child_weight", "reg_alpha", "reg_lambda", "n_jobs",
    }
    supplied = xgboost_params or {}
    unknown = set(supplied).difference(allowed)
    if unknown:
        raise ValueError(f"Unknown XGBoost parameters: {sorted(unknown)}")
    params = {
        "n_estimators": run_config.n_estimators,
        "max_depth": run_config.xgb_max_depth,
        "learning_rate": run_config.xgb_learning_rate,
        "subsample": run_config.xgb_subsample,
        "colsample_bytree": run_config.xgb_colsample_bytree,
        "min_child_weight": run_config.xgb_min_child_weight,
        "reg_alpha": run_config.xgb_reg_alpha,
        "reg_lambda": run_config.xgb_reg_lambda,
        "n_jobs": run_config.n_jobs,
    }
    params.update(supplied)
    estimator = XGBClassifier(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        tree_method="hist", random_state=run_config.random_seed, **params,
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", estimator),
    ])
    sample_weight = (
        compute_sample_weight(class_weight="balanced", y=classes["train"])
        if class_weighting == "balanced" else None
    )
    fit_kwargs = {"model__sample_weight": sample_weight} if sample_weight is not None else {}
    model.fit(splits["train"][features], classes["train"], **fit_kwargs)
    probabilities = {
        name: model.predict_proba(part[features]) for name, part in splits.items()
    }
    report = {
        "thresholds": {"down": float(down_threshold), "up": float(up_threshold)},
        "class_weighting": class_weighting,
        "excluded_feature_patterns": list(exclude_features),
        "excluded_features": excluded,
        "rows": {name: int(len(part)) for name, part in splits.items()},
        "splits": {
            name: _metrics(classes[name], probabilities[name]) for name in splits
        },
    }
    saved_config = run_config.to_dict()
    saved_config["target_mode"] = "three_class_return"
    artifact = {
        "artifact_version": 1,
        "model": model,
        "features": features,
        "config": saved_config,
        "thresholds": report["thresholds"],
        "class_names": CLASS_NAMES,
    }
    joblib.dump(artifact, out / "three_class_model.joblib")
    test = splits["test"][["timestamp", "target_exact_return"]].copy()
    test["target_return_class"] = classes["test"]
    test["target_class_name"] = [CLASS_NAMES[value] for value in classes["test"]]
    for index, name in CLASS_NAMES.items():
        test[f"probability_{name}"] = probabilities["test"][:, index]
    test["predicted_return_class"] = probabilities["test"].argmax(axis=1)
    test["predicted_class_name"] = test["predicted_return_class"].map(CLASS_NAMES)
    test.to_csv(out / "three_class_test_predictions.csv", index=False)
    pd.DataFrame({"feature": features}).to_csv(out / "three_class_feature_manifest.csv", index=False)
    (out / "three_class_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "three_class_config.json").write_text(json.dumps(saved_config, indent=2), encoding="utf-8")
    if run_config.verbose:
        test_metrics = report["splits"]["test"]
        print(
            f"[Three-class] test accuracy={test_metrics['accuracy']:.4f}, "
            f"balanced accuracy={test_metrics['balanced_accuracy']:.4f}, "
            f"macro F1={test_metrics['macro_f1']:.4f}, "
            f"AUC={test_metrics['roc_auc_ovr_macro']:.4f} | "
            f"elapsed={(perf_counter() - started):.1f}s",
            flush=True,
        )
    return {
        "model_path": str(out / "three_class_model.joblib"),
        "predictions_path": str(out / "three_class_test_predictions.csv"),
        "metrics": report,
        "feature_count": len(features),
        "excluded_features": excluded,
    }
