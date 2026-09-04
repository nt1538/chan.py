"""End-to-end, leakage-resistant dataset and model pipeline."""

from __future__ import annotations

import json
from dataclasses import replace
from time import perf_counter
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, log_loss, mean_absolute_error,
                             mean_squared_error, roc_auc_score, r2_score)
from sklearn.pipeline import Pipeline

from .chan_state import add_chan_features
from .config import ForecastConfig
from .features import (add_technical_features, normalize_ohlcv,
                       technical_feature_kwargs, technical_warmup_bars)
from .labels import (TARGET_COLUMNS, TARGET_METADATA_COLUMNS,
                     add_same_time_direction_label, add_same_time_return_label,
                     add_trading_day_extreme_labels,
                     target_columns)
from .models import fit_lstm, make_xgboost, predict_lstm


NON_FEATURE_COLUMNS = {"timestamp", *TARGET_COLUMNS, "target_exact_return", "target_up",
                       *TARGET_METADATA_COLUMNS}
NONSTATIONARY_COLUMNS = {
    "open", "high", "low", "close", "volume",
    "tech_obv", "tech_vwap_session", "tech_ad_line", "chan_tech_ad_line",
}


def _select_model_rows(df: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    """Filter/sparsify observations without changing feature or label history."""
    timestamps = pd.to_datetime(df["timestamp"])
    selected = pd.Series(True, index=df.index)
    try:
        start_minutes = int(pd.Timedelta(f"{config.regular_session_start}:00").total_seconds() // 60)
        end_minutes = int(pd.Timedelta(f"{config.regular_session_end}:00").total_seconds() // 60)
    except (TypeError, ValueError) as exc:
        raise ValueError("regular_session_start/end must use HH:MM format") from exc
    if not (0 <= start_minutes < 1440 and 0 <= end_minutes < 1440 and start_minutes <= end_minutes):
        raise ValueError("regular session times must satisfy 00:00 <= start <= end <= 23:59")
    minute_of_day = timestamps.dt.hour * 60 + timestamps.dt.minute
    if config.regular_session_only:
        selected &= minute_of_day.between(start_minutes, end_minutes, inclusive="both")
    if config.sample_every_minutes is not None:
        interval = int(config.sample_every_minutes)
        if interval < 1:
            raise ValueError("sample_every_minutes must be positive or None")
        anchor = start_minutes if config.regular_session_only else 0
        selected &= ((minute_of_day - anchor) % interval).eq(0)
    return df.loc[selected].copy()


def build_dataset(config: ForecastConfig) -> pd.DataFrame:
    """Load bars, add causal features, then build the configured future label."""
    started = perf_counter()
    if config.verbose:
        print(f"[Data] Loading {config.input_csv}", flush=True)
    raw = normalize_ohlcv(pd.read_csv(config.input_csv))
    if config.verbose:
        print(f"[Data] Loaded {len(raw):,} normalized rows: {raw.timestamp.min()} to {raw.timestamp.max()}", flush=True)
    # Chan rebuilding is intentionally expensive. Restrict it to the requested
    # experiment period while retaining enough earlier rows to warm a complete
    # Chan window, technical rollups, and the LSTM sequence.
    if config.train_start_date:
        first = int(raw["timestamp"].searchsorted(pd.Timestamp(config.train_start_date), side="left"))
        technical_warmup = technical_warmup_bars(config)
        warmup = max(config.chan_window_bars, config.min_history_bars,
                     config.lstm_sequence_length, technical_warmup)
        raw = raw.iloc[max(0, first - warmup):].copy()
    if config.test_end_date:
        end_day = pd.Timestamp(config.test_end_date).normalize()
        days = pd.Index(raw["timestamp"].dt.normalize().drop_duplicates())
        positions = np.flatnonzero(days <= end_day)
        if len(positions):
            final_day_index = min(len(days) - 1, int(positions[-1]) + config.horizon_days)
            raw = raw[raw["timestamp"].dt.normalize() <= days[final_day_index]].copy()
    if config.verbose:
        print(f"[Data] Working period including warm-up: {raw.timestamp.min()} to {raw.timestamp.max()} ({len(raw):,} rows)", flush=True)
        print("[Features] Calculating causal price/volume technical features", flush=True)
    data = add_technical_features(
        raw.reset_index(drop=True),
        windows=config.technical_windows,
        rsi_periods=config.technical_rsi_periods,
        atr_periods=config.technical_atr_periods,
        macd_periods=config.technical_macd_periods,
        **technical_feature_kwargs(config),
        regular_session_start=config.regular_session_start,
        regular_session_end=config.regular_session_end,
    )
    if config.enable_chan:
        data = add_chan_features(
            data, config.symbol, config.chan_window_bars,
            verbose=config.verbose, progress_every_rows=config.progress_every_rows,
        )
    elif config.verbose:
        print("[Chan] Disabled", flush=True)
    if config.verbose:
        print(
            f"[Labels] Building targets through the same time "
            f"{config.horizon_days} trading sessions later",
            flush=True,
        )
    if config.target_mode == "exact_return":
        result = add_same_time_return_label(data, config.horizon_days)
    elif config.target_mode == "up_direction":
        result = add_same_time_direction_label(data, config.horizon_days)
    else:
        result = add_trading_day_extreme_labels(data, config.horizon_days)
    targets = target_columns(config.target_mode)
    if config.verbose:
        labeled = result[targets].notna().all(axis=1).sum()
        print(f"[Data] Dataset ready: {len(result):,} rows, {len(result.columns):,} columns, {labeled:,} labeled | {perf_counter()-started:.1f}s", flush=True)
    return result


def _feature_columns(
    df: pd.DataFrame,
    *,
    use_standard_technical_features: bool = True,
    use_chan_bsp_features: bool = True,
) -> List[str]:
    """Select stationary numeric fields, excluding disabled feature families."""
    selected = []
    for column in df.select_dtypes(include=[np.number, "bool"]).columns:
        normalized = str(column).lower().replace(" ", "_")
        if column in NON_FEATURE_COLUMNS or normalized in NONSTATIONARY_COLUMNS:
            continue
        # ``tech_*`` is reserved for the standard technical feature family.
        # Do not match ``chan_tech_*``: those belong to the Chan feature set.
        if not use_standard_technical_features and normalized.startswith("tech_"):
            continue
        if not use_chan_bsp_features and normalized.startswith("chan_") and "bsp" in normalized:
            continue
        selected.append(column)
    return selected


def _metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return {"mae": float(mean_absolute_error(y, pred)), "rmse": float(mean_squared_error(y, pred) ** 0.5), "r2": float(r2_score(y, pred))}


def _classification_metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predicted = (probability >= threshold).astype(int)
    result = {
        "accuracy": float(accuracy_score(y, predicted)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "positive_rate": float(y.mean()),
        "predicted_positive_rate": float(predicted.mean()),
    }
    result["roc_auc"] = float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else float("nan")
    return result


def _model_prediction(model, values, target_mode: str):
    if target_mode == "up_direction":
        return model.predict_proba(values)[:, 1]
    return model.predict(values)


def _encode_target(target: str, values):
    """Use a positive magnitude internally for the signed maximum-loss target."""
    array = np.asarray(values)
    return -array if target == "target_max_loss_2d" else array


def _restore_target(target: str, values):
    array = np.asarray(values)
    return -array if target == "target_max_loss_2d" else array


def train_forecaster(
    config: ForecastConfig,
    *,
    model_types: tuple[str, ...] | list[str] | None = None,
    prediction_model: str | None = None,
    xgboost_params: Dict[str, Any] | None = None,
    lstm_params: Dict[str, Any] | None = None,
    ensemble_xgboost_weight: float | None = None,
    train_start_date: str | None = None,
    train_end_date: str | None = None,
    validation_start_date: str | None = None,
    validation_end_date: str | None = None,
    validation_fraction: float | None = None,
    test_start_date: str | None = None,
    test_end_date: str | None = None,
    chan_window_bars: int | None = None,
    use_standard_technical_features: bool | None = None,
    use_chan_bsp_features: bool | None = None,
    regular_session_only: bool | None = None,
    regular_session_start: str | None = None,
    regular_session_end: str | None = None,
    sample_every_minutes: int | None = None,
    verbose: bool | None = None,
    progress_every_rows: int | None = None,
) -> Dict[str, Any]:
    """Train selected models; keyword arguments can override the config.

    ``model_types`` accepts ``("xgboost",)``, ``("lstm",)`` or both.
    ``prediction_model`` selects ``xgboost``, ``lstm`` or ``ensemble`` as the
    generic prediction returned later by notebook inference.
    """
    # Friendly call-time names map onto the explicit persisted config fields.
    xgb_map = {
        "n_estimators": "n_estimators", "max_depth": "xgb_max_depth",
        "learning_rate": "xgb_learning_rate", "subsample": "xgb_subsample",
        "colsample_bytree": "xgb_colsample_bytree", "n_jobs": "n_jobs",
        "min_child_weight": "xgb_min_child_weight",
        "reg_alpha": "xgb_reg_alpha", "reg_lambda": "xgb_reg_lambda",
    }
    lstm_map = {
        "sequence_length": "lstm_sequence_length", "hidden_size": "lstm_hidden_size",
        "layers": "lstm_layers", "dropout": "lstm_dropout", "epochs": "lstm_epochs",
        "batch_size": "lstm_batch_size", "learning_rate": "lstm_learning_rate",
        "train_stride": "lstm_train_stride",
        "max_batch_feature_values": "lstm_max_batch_feature_values",
    }

    def mapped(values, mapping, family):
        values = values or {}
        unknown = set(values).difference(mapping)
        if unknown:
            raise ValueError(f"Unknown {family} parameters: {sorted(unknown)}")
        return {mapping[key]: value for key, value in values.items()}

    overrides = mapped(xgboost_params, xgb_map, "XGBoost")
    overrides.update(mapped(lstm_params, lstm_map, "LSTM"))
    if ensemble_xgboost_weight is not None:
        overrides["ensemble_xgboost_weight"] = float(ensemble_xgboost_weight)
    for field, value in {
        "train_start_date": train_start_date, "train_end_date": train_end_date,
        "validation_start_date": validation_start_date,
        "validation_end_date": validation_end_date,
        "validation_fraction": validation_fraction,
        "test_start_date": test_start_date, "test_end_date": test_end_date,
        "chan_window_bars": chan_window_bars,
        "use_standard_technical_features": use_standard_technical_features,
        "use_chan_bsp_features": use_chan_bsp_features,
        "regular_session_only": regular_session_only,
        "regular_session_start": regular_session_start,
        "regular_session_end": regular_session_end,
        "verbose": verbose, "progress_every_rows": progress_every_rows,
    }.items():
        if value is not None:
            overrides[field] = value
    # ``None`` means no call-time override, so sampling can be configured as
    # either 30/60/etc. in this function or left on the config object.
    if sample_every_minutes is not None:
        overrides["sample_every_minutes"] = int(sample_every_minutes)
    config = replace(config, **overrides)
    if not 0.0 <= config.ensemble_xgboost_weight <= 1.0:
        raise ValueError("ensemble_xgboost_weight must be between 0 and 1")
    if not 0.0 < config.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not 0.0 < config.direction_probability_threshold < 1.0:
        raise ValueError("direction_probability_threshold must be between 0 and 1")

    selected_models = tuple(str(x).lower() for x in (model_types or config.model_types))
    invalid = set(selected_models).difference({"xgboost", "lstm"})
    if invalid or not selected_models:
        raise ValueError(f"model_types must contain xgboost and/or lstm; got {selected_models}")
    if config.target_mode == "up_direction" and "lstm" in selected_models:
        raise ValueError("target_mode='up_direction' currently supports model_types=('xgboost',) only")
    selected_prediction = str(prediction_model or config.prediction_model).lower()
    available_predictions = set(selected_models)
    if set(selected_models) == {"xgboost", "lstm"}:
        available_predictions.add("ensemble")
    if selected_prediction not in available_predictions:
        raise ValueError(f"prediction_model={selected_prediction!r} is unavailable for {selected_models}")
    out = config.output_path
    out.mkdir(parents=True, exist_ok=True)
    run_started = perf_counter()
    if config.verbose:
        print(f"[Run] Models={selected_models}; default prediction={selected_prediction}", flush=True)
        print(f"[Run] Standard technical features={'enabled' if config.use_standard_technical_features else 'disabled'}; "
              f"Chan features={'enabled' if config.enable_chan else 'disabled'}; "
              f"Chan BSP features={'enabled' if config.use_chan_bsp_features else 'disabled'}", flush=True)
        print(f"[Run] Train={config.train_start_date or 'fractional start'} to {config.train_end_date or 'fractional end'}; "
              f"validation={config.validation_start_date or f'last {config.validation_fraction:.0%}'} to "
              f"{config.validation_end_date or 'training end'}; "
              f"test={config.test_start_date or 'fractional start'} to {config.test_end_date or 'fractional end'}", flush=True)
    targets = target_columns(config.target_mode)
    df = build_dataset(config)
    technical_warmup = technical_warmup_bars(config)
    labeled = df.dropna(subset=targets).iloc[
        max(0, config.min_history_bars, technical_warmup):
    ].copy()
    labeled = _select_model_rows(labeled, config)
    features = _feature_columns(
        labeled,
        use_standard_technical_features=config.use_standard_technical_features,
        use_chan_bsp_features=config.use_chan_bsp_features,
    )
    if not features or len(labeled) < 100:
        raise ValueError("Not enough labeled rows/features to train (need at least 100 rows)")
    timestamps = pd.to_datetime(labeled["timestamp"])

    def in_dates(start, end):
        mask = pd.Series(True, index=labeled.index)
        if start:
            mask &= timestamps >= pd.Timestamp(start)
        if end:
            # A date-only end value includes every intraday bar on that date.
            end_ts = pd.Timestamp(end)
            if end_ts == end_ts.normalize():
                end_ts += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
            mask &= timestamps <= end_ts
        return mask

    explicit_dates = any((config.train_start_date, config.train_end_date,
                          config.validation_start_date, config.validation_end_date,
                          config.test_start_date, config.test_end_date))
    if explicit_dates:
        if not config.test_start_date:
            raise ValueError("test_start_date is required when explicit date splitting is used")
        test = labeled.loc[in_dates(config.test_start_date, config.test_end_date)].copy()
        explicit_validation = bool(config.validation_start_date or config.validation_end_date)
        if explicit_validation:
            if not (config.validation_start_date and config.validation_end_date):
                raise ValueError("validation_start_date and validation_end_date must be provided together")
            train = labeled.loc[in_dates(config.train_start_date, config.train_end_date)].copy()
            validation = labeled.loc[in_dates(
                config.validation_start_date, config.validation_end_date
            )].copy()
            if not config.train_end_date:
                train = train[pd.to_datetime(train["timestamp"]) < pd.Timestamp(config.validation_start_date)]
            if set(train.index).intersection(validation.index):
                raise ValueError("Explicit train and validation date ranges overlap")
            if set(train.index).intersection(test.index) or set(validation.index).intersection(test.index):
                raise ValueError("Explicit train, validation, and test date ranges overlap")
        else:
            train_pool = labeled.loc[in_dates(config.train_start_date, config.train_end_date)].copy()
            if not config.train_end_date:
                train_pool = train_pool[pd.to_datetime(train_pool["timestamp"]) < pd.Timestamp(config.test_start_date)]
            if set(train_pool.index).intersection(test.index):
                raise ValueError("Explicit train and test date ranges overlap")
    else:
        test_start_index = int(len(labeled) * (1.0 - config.test_fraction))
        train_pool, test = labeled.iloc[:test_start_index].copy(), labeled.iloc[test_start_index:].copy()

    if not (explicit_dates and (config.validation_start_date or config.validation_end_date)):
        validation_start_index = int(len(train_pool) * (1.0 - config.validation_fraction))
        validation = train_pool.iloc[validation_start_index:].copy()
        train = train_pool.iloc[:validation_start_index].copy()
    if explicit_dates and config.validation_start_date:
        if min(len(train), len(validation), len(test)) == 0:
            raise ValueError("An explicit train, validation, or test date range contains no labeled rows")
        train_last = pd.Timestamp(train["timestamp"].iloc[-1])
        validation_first = pd.Timestamp(validation["timestamp"].iloc[0])
        validation_last = pd.Timestamp(validation["timestamp"].iloc[-1])
        test_first = pd.Timestamp(test["timestamp"].iloc[0])
        if not (train_last < validation_first <= validation_last < test_first):
            raise ValueError("Date ranges must be chronological: train < validation < test")
    # Purge by the actual variable horizon end, not an assumed number of rows.
    if len(validation):
        train = train[pd.to_datetime(train[TARGET_METADATA_COLUMNS[0]]) < pd.Timestamp(validation["timestamp"].iloc[0])]
    if len(test):
        validation = validation[pd.to_datetime(validation[TARGET_METADATA_COLUMNS[0]]) < pd.Timestamp(test["timestamp"].iloc[0])]
    splits = {"train": train, "validation": validation, "test": test}
    if min(len(splits["train"]), len(splits["validation"]), len(splits["test"])) == 0:
        raise ValueError("Dataset is too short for chronological splits plus horizon purging")
    if config.verbose:
        sampling = "every bar" if config.sample_every_minutes is None else f"every {config.sample_every_minutes} minutes"
        session = (f"{config.regular_session_start}-{config.regular_session_end}"
                   if config.regular_session_only else "all sessions")
        print(f"[Rows] Modeling observations={session}, {sampling}", flush=True)
        print("[Split] " + ", ".join(f"{name}={len(part):,}" for name, part in splits.items()), flush=True)
        print(f"[Features] Selected {len(features):,} stationary model features", flush=True)
    models, report = {}, {"rows": {k: len(v) for k, v in splits.items()}, "models": {}}
    # XGBoost sees the current row's complete tabular state.
    if "xgboost" in selected_models:
        report["models"]["xgboost"] = {}
        for target in targets:
            target_started = perf_counter()
            if config.verbose:
                print(f"[XGBoost] Training {target} ({config.n_estimators} trees, depth={config.xgb_max_depth})", flush=True)
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", make_xgboost(config)),
            ])
            model.fit(splits["train"][features], _encode_target(target, splits["train"][target]))
            report["models"]["xgboost"][target] = {}
            for name, part in splits.items():
                prediction = _model_prediction(model, part[features], config.target_mode)
                report["models"]["xgboost"][target][name] = (
                    _classification_metrics(part[target].to_numpy(), prediction,
                                            config.direction_probability_threshold)
                    if config.target_mode == "up_direction"
                    else _metrics(part[target].to_numpy(), _restore_target(target, prediction))
                )
            models[target] = model
            if config.verbose:
                test_metrics = report["models"]["xgboost"][target]["test"]
                if config.target_mode == "up_direction":
                    print(f"[XGBoost] {target} complete in {perf_counter()-target_started:.1f}s | "
                          f"test accuracy={test_metrics['accuracy']:.4f}, AUC={test_metrics['roc_auc']:.4f}, "
                          f"log loss={test_metrics['log_loss']:.6f}", flush=True)
                else:
                    print(f"[XGBoost] {target} complete in {perf_counter()-target_started:.1f}s | "
                          f"test MAE={test_metrics['mae']:.6f}, RMSE={test_metrics['rmse']:.6f}, R2={test_metrics['r2']:.4f}", flush=True)

    # LSTM sees an ordered rolling window and predicts both targets jointly.
    to_x = lambda part: part[features].to_numpy(dtype=np.float32)
    def to_y(part):
        values = part[targets].to_numpy(dtype=np.float32)
        if "target_max_loss_2d" in targets:
            values[:, targets.index("target_max_loss_2d")] *= -1.0
        return values
    lstm = None
    split_lstm_predictions = {}
    if "lstm" in selected_models:
        report["models"]["lstm"] = {}
        if config.verbose:
            print(f"[LSTM] Training sequence={config.lstm_sequence_length}, hidden={config.lstm_hidden_size}, "
                  f"layers={config.lstm_layers}, epochs={config.lstm_epochs}", flush=True)
        lstm = fit_lstm(to_x(splits["train"]), to_y(splits["train"]),
                        to_x(splits["validation"]), to_y(splits["validation"]), config)
        report["lstm_training"] = {
            "requested_batch_size": int(config.lstm_batch_size),
            "effective_batch_size": int(lstm["effective_batch_size"]),
            "sequence_length": int(config.lstm_sequence_length),
            "feature_count": int(len(features)),
        }
        if "xgboost" in selected_models:
            report["models"]["ensemble"] = {}
        for name, part in splits.items():
            ends, prediction = predict_lstm(lstm, to_x(part), config.lstm_batch_size)
            split_lstm_predictions[name] = (ends, prediction)
            for target_index, target in enumerate(targets):
                truth = part[target].to_numpy()[ends]
                restored_lstm = _restore_target(target, prediction[:, target_index])
                report["models"]["lstm"].setdefault(target, {})[name] = _metrics(truth, restored_lstm)
                if "xgboost" in selected_models:
                    xgb_prediction = _restore_target(target, models[target].predict(part.iloc[ends][features]))
                    weight = config.ensemble_xgboost_weight
                    ensemble_prediction = weight * xgb_prediction + (1.0 - weight) * restored_lstm
                    report["models"]["ensemble"].setdefault(target, {})[name] = _metrics(truth, ensemble_prediction)

    saved_config = config.to_dict()
    saved_config.update({"model_types": list(selected_models), "prediction_model": selected_prediction})
    artifact = {"artifact_version": 5, "models": models,
                "features": features, "targets": targets, "config": saved_config}
    artifact["target_encoding"] = {"target_max_loss_2d": "positive_magnitude"}
    if lstm is not None:
        artifact["lstm"] = lstm
    joblib.dump(artifact, out / "model.joblib")
    test_predictions = splits["test"][["timestamp", *targets]].copy()
    if "xgboost" in selected_models:
        for target, model in models.items():
            test_predictions[f"xgboost_{target}"] = _restore_target(
                target, _model_prediction(model, splits["test"][features], config.target_mode)
            )
            if config.target_mode == "up_direction":
                test_predictions[f"xgboost_{target}_class"] = (
                    test_predictions[f"xgboost_{target}"] >= config.direction_probability_threshold
                ).astype(int)
    if "lstm" in selected_models:
        ends, lstm_test = split_lstm_predictions["test"]
        for target_index, target in enumerate(targets):
            test_predictions[f"lstm_{target}"] = np.nan
            test_predictions.iloc[ends, test_predictions.columns.get_loc(f"lstm_{target}")] = _restore_target(
                target, lstm_test[:, target_index]
            )
            if "xgboost" in selected_models:
                weight = config.ensemble_xgboost_weight
                test_predictions[f"ensemble_{target}"] = (
                    weight * test_predictions[f"xgboost_{target}"]
                    + (1.0 - weight) * test_predictions[f"lstm_{target}"]
                )
    test_predictions.to_csv(out / "test_predictions.csv", index=False)
    pd.DataFrame({"feature": features}).to_csv(out / "feature_manifest.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "config.json").write_text(json.dumps(saved_config, indent=2), encoding="utf-8")
    if config.save_enriched_csv:
        if config.verbose:
            print("[Save] Writing enriched_5m_dataset.csv", flush=True)
        df.to_csv(out / "enriched_5m_dataset.csv", index=False)
    if config.verbose:
        print(f"[Save] Model: {out / 'model.joblib'}", flush=True)
        print(f"[Run] Complete in {(perf_counter()-run_started)/60:.2f} minutes", flush=True)
    return {"model_path": str(out / "model.joblib"), "metrics": report, "feature_count": len(features)}
