"""Controlled feature-importance comparisons across Chan/BSP row cohorts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from .config import ForecastConfig
from .labels import TARGET_COLUMNS, TARGET_METADATA_COLUMNS
from .models import make_xgboost
from .pipeline import NONSTATIONARY_COLUMNS, build_dataset


def _importance_family(feature: str) -> str:
    name = str(feature).lower()
    if name.startswith("tech_"):
        return "standard_technical"
    if name.startswith("chan_tech_"):
        return "chan_technical"
    if "bsp" in name and name.startswith("chan_"):
        return "chan_bsp"
    if name.startswith("chan_"):
        return "chan_structure"
    if name.startswith("time_"):
        return "calendar"
    return "source_or_other"


def compare_chan_bsp_feature_importance(
    config: ForecastConfig,
    *,
    output_dir: str | Path | None = None,
    bspoint_path: str | Path = "outputs/bsp_trade_labels_TQQQ_5m_2012-2026_training.csv",
    bsp_timestamp_column: str = "bsp_timestamp",
    bsp_direction_column: str = "direction",
    separate_bsp_directions: bool = True,
    include_raw_ohlcv: bool = True,
    include_non_type_cohorts: bool = False,
    train_start_date: str | None = None,
    train_end_date: str | None = None,
    test_start_date: str | None = None,
    test_end_date: str | None = None,
    bsp_types: tuple[str, ...] = ("1", "1p", "2", "2s", "3a", "3b"),
    bsp_type_groups: dict[str, tuple[str, ...]] | None = None,
    validation_fraction: float | None = None,
    minimum_rows_per_split: int = 100,
    top_features: int = 30,
    xgboost_params: dict[str, Any] | None = None,
    show: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train comparable all-bar and direction-separated BSP importance models.

    The all-bar cohort and configurable BSP groups are trained for max gain and
    max loss. BSP groups can be split into independent buy/sell cohorts. Every
    model uses the same feature columns, split, labels and XGBoost settings.
    BSP membership is loaded from ``bspoint_path``; Chan is never recalculated.
    """
    out = Path(output_dir or (Path(config.output_dir) / "chan_bsp_importance_comparison"))
    out.mkdir(parents=True, exist_ok=True)
    model_dir = out / "models"
    prediction_dir = out / "test_predictions"
    plot_dir = out / "plots"
    for directory in (model_dir, prediction_dir, plot_dir):
        directory.mkdir(parents=True, exist_ok=True)

    overrides: dict[str, Any] = {"enable_chan": False, "use_chan_bsp_features": False}
    for field, value in {
        "train_start_date": train_start_date,
        "train_end_date": train_end_date,
        "test_start_date": test_start_date,
        "test_end_date": test_end_date,
        "validation_fraction": validation_fraction,
        "verbose": verbose,
    }.items():
        if value is not None:
            overrides[field] = value
    xgb_map = {
        "n_estimators": "n_estimators", "max_depth": "xgb_max_depth",
        "learning_rate": "xgb_learning_rate", "subsample": "xgb_subsample",
        "colsample_bytree": "xgb_colsample_bytree", "n_jobs": "n_jobs",
        "min_child_weight": "xgb_min_child_weight",
        "reg_alpha": "xgb_reg_alpha", "reg_lambda": "xgb_reg_lambda",
    }
    unknown = set(xgboost_params or {}).difference(xgb_map)
    if unknown:
        raise ValueError(f"Unknown XGBoost parameters: {sorted(unknown)}")
    overrides.update({xgb_map[key]: value for key, value in (xgboost_params or {}).items()})
    run_config = replace(config, **overrides)
    if not 0 < float(run_config.validation_fraction) < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if int(minimum_rows_per_split) < 20:
        raise ValueError("minimum_rows_per_split must be at least 20")

    if verbose:
        print("[Comparison] Building one shared OHLCV + technical K-line dataset", flush=True)
    data = build_dataset(run_config)
    labeled = data.dropna(subset=TARGET_COLUMNS).copy()
    timestamps = pd.to_datetime(labeled["timestamp"])

    def inclusive_end(value: str | None) -> pd.Timestamp | None:
        if value is None:
            return None
        stamp = pd.Timestamp(value)
        return (
            stamp + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
            if stamp == stamp.normalize() else stamp
        )

    train_end = inclusive_end(run_config.train_end_date)
    test_end = inclusive_end(run_config.test_end_date)
    train_pool_mask = pd.Series(True, index=labeled.index)
    if run_config.train_start_date:
        train_pool_mask &= timestamps.ge(pd.Timestamp(run_config.train_start_date))
    if train_end is not None:
        train_pool_mask &= timestamps.le(train_end)
    elif run_config.test_start_date:
        train_pool_mask &= timestamps.lt(pd.Timestamp(run_config.test_start_date))
    test_mask = pd.Series(True, index=labeled.index)
    if not run_config.test_start_date:
        raise ValueError("test_start_date is required for comparable experiments")
    test_mask &= timestamps.ge(pd.Timestamp(run_config.test_start_date))
    if test_end is not None:
        test_mask &= timestamps.le(test_end)
    train_pool = labeled.loc[train_pool_mask].sort_values("timestamp")
    if train_pool.empty:
        raise ValueError("The requested training period contains no labeled rows")
    validation_position = int(len(train_pool) * (1.0 - run_config.validation_fraction))
    if not 0 < validation_position < len(train_pool):
        raise ValueError("Training period is too short for the validation split")
    validation_start = pd.Timestamp(train_pool.iloc[validation_position]["timestamp"])

    common_excluded = {"timestamp", *TARGET_COLUMNS, *TARGET_METADATA_COLUMNS}
    raw_ohlcv = {"open", "high", "low", "close", "volume"}
    # Retaining raw OHLCV does not imply retaining other non-stationary
    # cumulative features such as OBV or absolute session VWAP.
    common_excluded.update(NONSTATIONARY_COLUMNS.difference(raw_ohlcv))
    if not include_raw_ohlcv:
        common_excluded.update(raw_ohlcv)
    normalized_excluded = {
        str(column).lower().replace(" ", "_") for column in common_excluded
    }
    numeric = list(labeled.select_dtypes(include=[np.number, "bool"]).columns)
    technical_features = [
        column for column in numeric
        if str(column).lower().replace(" ", "_") not in normalized_excluded
        and not str(column).lower().startswith("chan_")
        # normalize_ohlcv preserves source columns and also creates canonical
        # lowercase OHLCV. Keep only the canonical copy to avoid duplicate
        # predictors such as both ``Close`` and ``close``.
        and not (
            str(column).lower().replace(" ", "_") in raw_ohlcv
            and str(column) != str(column).lower().replace(" ", "_")
        )
    ]
    if not technical_features:
        raise ValueError("No comparable numeric features were generated")

    bsp_source = Path(bspoint_path)
    if not bsp_source.exists():
        raise FileNotFoundError(f"BSP training file not found: {bsp_source}")
    bsp_rows = pd.read_csv(bsp_source, low_memory=False)
    required_bsp_columns = {bsp_timestamp_column, "bsp_type"}
    if separate_bsp_directions:
        required_bsp_columns.add(bsp_direction_column)
    missing_bsp_columns = required_bsp_columns.difference(bsp_rows.columns)
    if missing_bsp_columns:
        raise KeyError(f"BSP training file is missing: {sorted(missing_bsp_columns)}")
    bsp_rows["_match_timestamp"] = pd.to_datetime(
        bsp_rows[bsp_timestamp_column], errors="coerce"
    )
    bsp_rows["_match_type"] = (
        bsp_rows["bsp_type"].astype(str).str.strip().str.lower()
        .str.replace(r"\.0$", "", regex=True)
    )
    if separate_bsp_directions:
        bsp_rows["_match_direction"] = (
            bsp_rows[bsp_direction_column].astype(str).str.strip().str.lower()
        )
    bsp_rows = bsp_rows.dropna(subset=["_match_timestamp"])
    bsp_timestamps = pd.DatetimeIndex(bsp_rows["_match_timestamp"].unique())
    bsp_mask = timestamps.isin(bsp_timestamps)

    groups = bsp_type_groups or {
        "type_1_1p": ("1", "1p"),
        "type_2_2s": ("2", "2s"),
        "type_3a": ("3a",),
    }
    requested_types = {str(value).lower() for values in groups.values() for value in values}
    unsupported = requested_types.difference({str(value).lower() for value in bsp_types})
    if unsupported:
        raise ValueError(
            "bsp_type_groups contains types excluded by bsp_types: "
            f"{sorted(unsupported)}"
        )
    cohorts: dict[str, tuple[pd.Series, list[str]]] = {
        "all_5min_technical": (pd.Series(True, index=labeled.index), technical_features),
    }
    for group_name, group_types in groups.items():
        normalized_types = {str(value).lower() for value in group_types}
        typed_rows = bsp_rows.loc[bsp_rows["_match_type"].isin(normalized_types)]
        if separate_bsp_directions:
            for direction in ("buy", "sell"):
                type_timestamps = pd.DatetimeIndex(
                    typed_rows.loc[
                        typed_rows["_match_direction"].eq(direction), "_match_timestamp"
                    ].unique()
                )
                group_mask = timestamps.isin(type_timestamps)
                cohort_mask = bsp_mask & group_mask
                cohorts[f"bsp_{group_name}_{direction}"] = (
                    cohort_mask, technical_features
                )
                if include_non_type_cohorts:
                    cohorts[f"other_klines_not_{group_name}_{direction}"] = (
                        ~cohort_mask, technical_features
                    )
        else:
            type_timestamps = pd.DatetimeIndex(typed_rows["_match_timestamp"].unique())
            group_mask = timestamps.isin(type_timestamps)
            cohort_mask = bsp_mask & group_mask
            cohorts[f"bsp_{group_name}"] = (cohort_mask, technical_features)
            if include_non_type_cohorts:
                cohorts[f"other_klines_not_{group_name}"] = (
                    ~cohort_mask, technical_features
                )

    def split_masks(cohort_mask: pd.Series) -> dict[str, pd.Series]:
        horizon_end = pd.to_datetime(labeled[TARGET_METADATA_COLUMNS[0]], errors="coerce")
        return {
            "train": (
                cohort_mask & train_pool_mask & timestamps.lt(validation_start)
                & horizon_end.lt(validation_start)
            ),
            "validation": (
                cohort_mask & train_pool_mask & timestamps.ge(validation_start)
                & (horizon_end.le(train_end) if train_end is not None else True)
            ),
            "test": (
                cohort_mask & test_mask
                & (horizon_end.le(test_end) if test_end is not None else True)
            ),
        }

    def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
        return {
            "rows": int(len(truth)),
            "mae": float(mean_absolute_error(truth, prediction)),
            "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
            "r2": float(r2_score(truth, prediction)),
        }

    report: dict[str, Any] = {"models": {}, "skipped": {}, "validation_start": str(validation_start)}
    importance_rows: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    for cohort_name, (cohort_mask, requested_features) in cohorts.items():
        masks = split_masks(cohort_mask)
        splits = {name: labeled.loc[mask].copy() for name, mask in masks.items()}
        counts = {name: int(len(part)) for name, part in splits.items()}
        if min(counts.values()) < int(minimum_rows_per_split):
            report["skipped"][cohort_name] = counts
            if verbose:
                print(f"[Comparison] Skip {cohort_name}: {counts}", flush=True)
            continue
        features = [
            feature for feature in requested_features
            if splits["train"][feature].notna().any()
        ]
        cohort_report: dict[str, Any] = {"rows": counts, "targets": {}}
        cohort_models: dict[str, Any] = {}
        prediction_frame = splits["test"][["timestamp", *TARGET_COLUMNS]].copy()
        for target in TARGET_COLUMNS:
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", make_xgboost(run_config)),
            ])
            y_train = splits["train"][target].to_numpy(float)
            encoded = -y_train if target == "target_max_loss_2d" else y_train
            model.fit(splits["train"][features], encoded)
            target_report: dict[str, Any] = {}
            for split_name, part in splits.items():
                prediction = model.predict(part[features])
                if target == "target_max_loss_2d":
                    prediction = -prediction
                target_report[split_name] = metrics(
                    part[target].to_numpy(float), prediction
                )
                if split_name == "test":
                    prediction_frame[f"predicted_{target}"] = prediction
            booster = model.named_steps["model"].get_booster()
            raw = booster.get_score(importance_type="gain")
            gains = np.asarray([
                float(raw.get(f"f{index}", raw.get(feature, 0.0)))
                for index, feature in enumerate(features)
            ])
            normalized = gains / gains.sum() if gains.sum() > 0 else gains
            order = np.argsort(-normalized)
            ranks = np.empty(len(features), dtype=int)
            ranks[order] = np.arange(1, len(features) + 1)
            for index, feature in enumerate(features):
                importance_rows.append({
                    "cohort": cohort_name, "target": target,
                    "feature": feature, "feature_family": _importance_family(feature),
                    "raw_gain": float(gains[index]),
                    "normalized_gain": float(normalized[index]),
                    "rank": int(ranks[index]),
                })
            cohort_report["targets"][target] = target_report
            cohort_models[target] = model
        prediction_frame.to_csv(prediction_dir / f"{cohort_name}.csv", index=False)
        report["models"][cohort_name] = cohort_report
        artifacts[cohort_name] = {"models": cohort_models, "features": features}
        if verbose:
            print(f"[Comparison] {cohort_name}: {counts}, features={len(features)}", flush=True)

    if not importance_rows:
        raise ValueError("Every comparison cohort was skipped; reduce minimum_rows_per_split")
    importance = pd.DataFrame(importance_rows)
    importance.to_csv(out / "feature_importance_long.csv", index=False)
    generated_plots: dict[str, str] = {}
    wide_tables: dict[str, pd.DataFrame] = {}
    for target in TARGET_COLUMNS:
        target_importance = importance.loc[importance["target"].eq(target)]
        wide = target_importance.pivot_table(
            index="feature", columns="cohort", values="normalized_gain", fill_value=0.0
        ).sort_index()
        wide.to_csv(out / f"feature_importance_{target}_comparison.csv")
        wide_tables[target] = wide
        for cohort_name in wide.columns:
            values = wide[cohort_name].nlargest(min(int(top_features), len(wide))).sort_values()
            values.rename("normalized_gain").to_csv(
                out / f"feature_importance_{target}_{cohort_name}.csv"
            )
            fig, ax = plt.subplots(
                figsize=(12, max(7, 0.32 * len(values))), constrained_layout=True
            )
            ax.barh(values.index, values.values, color="#2563EB")
            ax.set_xlim(0, max(float(values.max()) * 1.08, 1e-9))
            ax.set_title(f"{target}: {cohort_name}")
            ax.set_xlabel("Normalized XGBoost gain (sums to 1 per model)")
            ax.grid(axis="x", alpha=0.2)
            path = plot_dir / f"{target}_{cohort_name}_importance.png"
            fig.savefig(path, dpi=170, bbox_inches="tight")
            generated_plots[f"{target}_{cohort_name}"] = str(path.resolve())
            if show:
                plt.show()
            else:
                plt.close(fig)

    joblib.dump({
        "artifact_version": 1, "config": run_config.to_dict(),
        "targets": TARGET_COLUMNS, "cohorts": artifacts,
    }, model_dir / "comparison_models.joblib")
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "comparison_config.json").write_text(json.dumps({
        "config": run_config.to_dict(), "bsp_types": list(bsp_types),
        "bspoint_path": str(bsp_source.resolve()),
        "bsp_timestamp_column": bsp_timestamp_column,
        "bsp_direction_column": bsp_direction_column,
        "separate_bsp_directions": bool(separate_bsp_directions),
        "include_raw_ohlcv": bool(include_raw_ohlcv),
        "include_non_type_cohorts": bool(include_non_type_cohorts),
        "bsp_type_groups": {key: list(value) for key, value in groups.items()},
        "label_definition": {
            "window": (
                f"next bar through the same clock time "
                f"{run_config.horizon_days} future trading sessions later"
            ),
            "zero_clipped": False,
        },
        "minimum_rows_per_split": int(minimum_rows_per_split),
        "top_features": int(top_features), "xgboost_params": xgboost_params or {},
    }, indent=2), encoding="utf-8")
    return {
        "output_dir": str(out.resolve()), "metrics": report,
        "importance_long": importance, "importance_wide": wide_tables,
        "plot_paths": generated_plots,
        "model_path": str((model_dir / "comparison_models.joblib").resolve()),
        "skipped": report["skipped"],
    }


def compare_bsp_type_vs_other_klines(
    config: ForecastConfig,
    **kwargs: Any,
) -> dict[str, Any]:
    """Train BSP cohorts and their exact all-K-line complements.

    For example, ``bsp_type_2_2s_buy`` is compared with
    ``other_klines_not_type_2_2s_buy``. The negative cohort includes every
    labeled K-line that is not a Type-2/2s buy at that timestamp. All cohorts
    use the same features, targets, chronological splits and model parameters.
    """
    if "include_non_type_cohorts" in kwargs:
        raise TypeError(
            "compare_bsp_type_vs_other_klines controls "
            "include_non_type_cohorts internally"
        )
    return compare_chan_bsp_feature_importance(
        config, include_non_type_cohorts=True, **kwargs
    )


def create_paired_bsp_model_predictions(
    output_dir: str | Path,
    *,
    cohorts: tuple[str, ...] | list[str] | None = None,
    output_subdir: str = "paired_bsp_predictions",
) -> dict[str, Any]:
    """Compare all-bar and BSP-specific predictions on identical BSP test rows.

    This consumes the CSV predictions saved by the comparison training run; it
    does not rebuild features, rerun Chan, or retrain a model. One paired CSV is
    written per BSP cohort together with metrics for both models on the exact
    same timestamps.
    """
    out = Path(output_dir)
    prediction_dir = out / "test_predictions"
    all_path = prediction_dir / "all_5min_technical.csv"
    if not all_path.exists():
        raise FileNotFoundError(f"Missing all-bar predictions: {all_path}")
    available = sorted(
        path.stem for path in prediction_dir.glob("bsp_*.csv")
    )
    selected = available if cohorts is None else [str(value) for value in cohorts]
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise ValueError(f"Unknown BSP cohorts {unknown}. Available: {available}")
    if not selected:
        raise ValueError("No BSP-specific prediction cohorts were found")

    required_predictions = [f"predicted_{target}" for target in TARGET_COLUMNS]
    all_rows = pd.read_csv(all_path, parse_dates=["timestamp"])
    required_all = {"timestamp", *required_predictions}
    missing_all = required_all.difference(all_rows.columns)
    if missing_all:
        raise KeyError(f"All-bar prediction file is missing: {sorted(missing_all)}")
    all_rows = all_rows[["timestamp", *required_predictions]].rename(columns={
        column: f"all_bars_{column}" for column in required_predictions
    })
    if all_rows["timestamp"].duplicated().any():
        raise ValueError("All-bar prediction timestamps must be unique")

    paired_dir = out / output_subdir
    paired_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"cohorts": {}}
    output_paths: dict[str, str] = {}
    summary_rows: list[dict[str, Any]] = []

    def score(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
        return {
            "rows": int(len(truth)),
            "mae": float(mean_absolute_error(truth, prediction)),
            "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
            "r2": float(r2_score(truth, prediction)),
            "correlation": float(np.corrcoef(truth, prediction)[0, 1])
            if len(truth) > 1 and np.std(truth) > 0 and np.std(prediction) > 0
            else float("nan"),
        }

    for cohort in selected:
        specific_path = prediction_dir / f"{cohort}.csv"
        specific = pd.read_csv(specific_path, parse_dates=["timestamp"])
        required_specific = {"timestamp", *TARGET_COLUMNS, *required_predictions}
        missing = required_specific.difference(specific.columns)
        if missing:
            raise KeyError(f"{specific_path.name} is missing: {sorted(missing)}")
        if specific["timestamp"].duplicated().any():
            raise ValueError(f"{specific_path.name} contains duplicate timestamps")
        specific = specific[
            ["timestamp", *TARGET_COLUMNS, *required_predictions]
        ].rename(columns={
            column: f"bsp_specific_{column}" for column in required_predictions
        })
        paired = specific.merge(all_rows, on="timestamp", how="inner", validate="one_to_one")
        if len(paired) != len(specific):
            raise ValueError(
                f"Only {len(paired):,}/{len(specific):,} {cohort} rows matched "
                "the all-bar test predictions"
            )

        cohort_report: dict[str, Any] = {"rows": int(len(paired)), "targets": {}}
        for target in TARGET_COLUMNS:
            all_column = f"all_bars_predicted_{target}"
            bsp_column = f"bsp_specific_predicted_{target}"
            paired[f"all_bars_error_{target}"] = paired[all_column] - paired[target]
            paired[f"bsp_specific_error_{target}"] = paired[bsp_column] - paired[target]
            paired[f"prediction_difference_{target}"] = (
                paired[bsp_column] - paired[all_column]
            )
            valid = paired[[target, all_column, bsp_column]].dropna()
            truth = valid[target].to_numpy(float)
            all_metrics = score(truth, valid[all_column].to_numpy(float))
            bsp_metrics = score(truth, valid[bsp_column].to_numpy(float))
            comparison = {
                "mae_improvement_bsp_vs_all": float(
                    all_metrics["mae"] - bsp_metrics["mae"]
                ),
                "rmse_improvement_bsp_vs_all": float(
                    all_metrics["rmse"] - bsp_metrics["rmse"]
                ),
                "r2_improvement_bsp_vs_all": float(
                    bsp_metrics["r2"] - all_metrics["r2"]
                ),
            }
            cohort_report["targets"][target] = {
                "all_bars_model": all_metrics,
                "bsp_specific_model": bsp_metrics,
                "comparison": comparison,
            }
            summary_rows.append({
                "cohort": cohort, "target": target,
                "rows": int(len(valid)),
                "all_bars_mae": all_metrics["mae"],
                "bsp_specific_mae": bsp_metrics["mae"],
                "mae_improvement_bsp_vs_all": comparison["mae_improvement_bsp_vs_all"],
                "all_bars_rmse": all_metrics["rmse"],
                "bsp_specific_rmse": bsp_metrics["rmse"],
                "rmse_improvement_bsp_vs_all": comparison["rmse_improvement_bsp_vs_all"],
                "all_bars_r2": all_metrics["r2"],
                "bsp_specific_r2": bsp_metrics["r2"],
                "r2_improvement_bsp_vs_all": comparison["r2_improvement_bsp_vs_all"],
            })

        paired_path = paired_dir / f"{cohort}_paired_predictions.csv"
        paired.to_csv(paired_path, index=False)
        output_paths[cohort] = str(paired_path.resolve())
        report["cohorts"][cohort] = cohort_report

    summary = pd.DataFrame(summary_rows)
    summary_path = paired_dir / "paired_model_metrics.csv"
    report_path = paired_dir / "paired_model_metrics.json"
    summary.to_csv(summary_path, index=False)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {
        "output_dir": str(paired_dir.resolve()),
        "paired_prediction_paths": output_paths,
        "metrics": report,
        "metrics_table": summary,
        "metrics_csv": str(summary_path.resolve()),
        "metrics_json": str(report_path.resolve()),
    }
