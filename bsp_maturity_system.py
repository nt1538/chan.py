"""Causal Chan BSP maturity dataset generation and type-specific training.

Maturity is structural: a visible candidate must survive until its source Bi
(or segment for a segment BSP) becomes confirmed. If it disappears first, it
is invalidated. Candidates unresolved at timeout/end-of-data are censored and
are never silently used as negative training labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline

from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC
from realistic_chan_simulation import (
    RealisticSimulationConfig, _SimulationSlidingWindowChan, _build_klu,
    _kl_type, _load_ohlcv_csv,
)


_OUTCOME_COLUMNS = {
    "candidate_id", "candidate_key", "current_status", "is_mature",
    "is_invalidated", "is_censored", "resolution_timestamp",
    "bars_to_resolution", "snapshot_last_seen", "maturity_reason",
    "invalidation_reason", "eventual_is_mature",
}
_IDENTITY_COLUMNS = {
    "timestamp", "bsp_timestamp", "snapshot_timestamp",
    "first_seen_timestamp", "direction", "bsp_type", "bsp_types",
    "maturity_timestamp", "prediction_timestamp",
}
_NONSTATIONARY_COLUMNS = {
    "klu_idx", "klu_open", "klu_high", "klu_low", "klu_close", "klu_volume",
    "maturity_market_price",
    "snapshot_first_seen", "snapshot_age_bars", "revision_count",
}
_QUALITY_OUTCOME_COLUMNS = {
    "quality_status", "quality_target", "remained_valid", "was_covered",
    "quality_resolution_timestamp", "bars_to_quality_resolution",
    "coverage_timestamp", "post_maturity_reason", "coverage_boundary",
    "maturity_reference_price", "post_maturity_max_favorable_return",
    "post_maturity_max_adverse_return",
}


def _chan_engine(config: RealisticSimulationConfig, window_bars: int):
    chan_config = config.chan
    calculation_config = CChanConfig({
        "trigger_step": True,
        "cal_rsi": chan_config.cal_rsi,
        "cal_kdj": chan_config.cal_kdj,
        "cal_dmi": chan_config.cal_dmi,
    })
    return _SimulationSlidingWindowChan(
        code=chan_config.code,
        data_src=DATA_SRC.CSV,
        lv_list=[_kl_type(chan_config.frequency)],
        config=calculation_config,
        autype=AUTYPE.QFQ,
        max_klines=int(window_bars),
    )


def _active_bsp_snapshots(engine) -> dict[tuple, dict]:
    """Extract every BSP visible in the current point-in-time Chan state."""
    if engine.last_chan is None:
        return {}
    level = engine.lv_list[0]
    bsp_objects = engine.last_chan.kl_datas[level].bs_point_lst.getSortedBspList()
    active: dict[tuple, dict] = {}
    for bsp in bsp_objects:
        for bsp_type in bsp.type:
            type_name = str(bsp_type.value).lower()
            snapshot = engine._create_bsp_snapshot(
                bsp, type_name, engine.snapshot_count, chan=engine.last_chan,
            )
            key = (
                str(snapshot["timestamp"]), type_name,
                str(snapshot["direction"]), bool(snapshot.get("is_segbsp", False)),
            )
            active[key] = snapshot
    return active


def _is_structurally_sure(snapshot: dict) -> bool:
    if bool(snapshot.get("is_segbsp", False)):
        return bool(snapshot.get("segment_is_sure", False))
    return bool(snapshot.get("bi_is_sure", False))


def generate_bsp_maturity_dataset(
    config: RealisticSimulationConfig,
    *,
    path: str | Path = "outputs/bsp_maturity_dataset.xlsx",
    csv_path: str | Path | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    chan_window_bars: int | None = None,
    maximum_resolution_bars: int = 390,
    post_maturity_validation_bars: int = 156,
    coverage_buffer_pct: float = 0.0,
    include_types: tuple[str, ...] = ("1", "1p", "2", "2s", "3a", "3b"),
    verbose: bool = True,
    progress_every_rows: int = 5_000,
) -> dict[str, Any]:
    """Replay K-lines and save first-seen BSP features with lifecycle outcomes.

    The Excel ``BSP Points`` sheet is directly compatible with
    ``plot_mature_bspoints_from_excel``. The CSV contains the same model-ready
    rows and avoids Excel overhead during repeated training.
    """
    if int(maximum_resolution_bars) <= 0:
        raise ValueError("maximum_resolution_bars must be greater than zero")
    if int(post_maturity_validation_bars) <= 0:
        raise ValueError("post_maturity_validation_bars must be greater than zero")
    if float(coverage_buffer_pct) < 0:
        raise ValueError("coverage_buffer_pct cannot be negative")
    if int(progress_every_rows) <= 0:
        raise ValueError("progress_every_rows must be greater than zero")
    started = perf_counter()
    window = int(chan_window_bars or config.chan.max_klines)
    period_start = pd.Timestamp(start or config.chan.start)
    period_end = pd.Timestamp(end or config.chan.end)
    if period_end < period_start:
        raise ValueError("end must be on or after start")

    raw = _load_ohlcv_csv(config.data_path)
    first_position = int(raw["timestamp"].searchsorted(period_start, side="left"))
    warmup = max(window, int(config.chan.warmup_bars))
    feed = raw.iloc[max(0, first_position - warmup):].copy()
    feed = feed.loc[feed["timestamp"] <= period_end].reset_index(drop=True)
    allowed = {str(value).lower() for value in include_types}
    engine = _chan_engine(config, window)

    records: list[dict] = []
    quality_records: list[dict] = []
    open_candidates: dict[tuple, int] = {}
    post_maturity_candidates: dict[tuple, int] = {}
    ignored_until_absent: set[tuple] = set()
    next_candidate_number = 1

    def resolve(record_index: int, status: str, when, bar_index: int, reason: str):
        record = records[record_index]
        record["current_status"] = status
        record["is_mature"] = int(status == "mature")
        record["eventual_is_mature"] = int(status == "mature") if status != "censored" else np.nan
        record["is_invalidated"] = int(status == "invalidated")
        record["is_censored"] = int(status == "censored")
        record["resolution_timestamp"] = pd.Timestamp(when)
        record["bars_to_resolution"] = int(bar_index - record["_first_feed_index"])
        if status == "mature":
            record["maturity_reason"] = reason
        elif status == "invalidated":
            record["invalidation_reason"] = reason

    def start_quality_monitor(
        key: tuple, record_index: int, snapshot: dict, when, bar_index: int,
        maturity_market_price: float,
    ):
        """Capture only information available at the structural maturity time."""
        source = records[record_index]
        direction = str(snapshot["direction"]).lower()
        if direction == "buy":
            boundary = float(snapshot["klu_low"]) * (1.0 - float(coverage_buffer_pct))
        else:
            boundary = float(snapshot["klu_high"]) * (1.0 + float(coverage_buffer_pct))
        quality = dict(snapshot)
        quality.update({
            "candidate_id": source["candidate_id"],
            "candidate_key": source["candidate_key"],
            "timestamp": pd.to_datetime(snapshot["timestamp"]),
            "bsp_timestamp": pd.to_datetime(snapshot["timestamp"]),
            "first_seen_timestamp": source["first_seen_timestamp"],
            "maturity_timestamp": pd.Timestamp(when),
            "prediction_timestamp": pd.Timestamp(when),
            "maturity_age_bars": int(bar_index - source["_first_feed_index"]),
            "coverage_boundary": boundary,
            "maturity_reference_price": float(maturity_market_price),
            "maturity_market_price": float(maturity_market_price),
            "quality_status": "monitoring",
            "quality_target": np.nan,
            "remained_valid": np.nan,
            "was_covered": 0,
            "coverage_timestamp": pd.NaT,
            "quality_resolution_timestamp": pd.NaT,
            "bars_to_quality_resolution": np.nan,
            "post_maturity_reason": None,
            "post_maturity_max_favorable_return": 0.0,
            "post_maturity_max_adverse_return": 0.0,
            "_maturity_feed_index": int(bar_index),
        })
        quality_records.append(quality)
        post_maturity_candidates[key] = len(quality_records) - 1

    def resolve_quality(index: int, status: str, when, bar_index: int, reason: str):
        quality = quality_records[index]
        quality["quality_status"] = status
        quality["quality_target"] = 1 if status == "held" else (np.nan if status == "censored" else 0)
        quality["remained_valid"] = 1 if status == "held" else (np.nan if status == "censored" else 0)
        quality["was_covered"] = int(status == "covered")
        quality["quality_resolution_timestamp"] = pd.Timestamp(when)
        quality["bars_to_quality_resolution"] = int(bar_index - quality["_maturity_feed_index"])
        quality["post_maturity_reason"] = reason
        if status == "covered":
            quality["coverage_timestamp"] = pd.Timestamp(when)

    for feed_index, row in feed.iterrows():
        timestamp = pd.Timestamp(row["timestamp"])
        engine.process_new_kline(_build_klu(
            timestamp, row["_open"], row["_high"], row["_low"],
            row["_close"], row.get("_vol", 0.0),
        ))
        active = _active_bsp_snapshots(engine)
        active = {key: value for key, value in active.items() if value["bsp_type"] in allowed}
        active_keys = set(active)

        # Evaluate mature BSP quality starting with the bar after maturity.
        for key, quality_index in list(post_maturity_candidates.items()):
            quality = quality_records[quality_index]
            elapsed = int(feed_index - quality["_maturity_feed_index"])
            if elapsed <= 0:
                continue
            reference = float(quality["maturity_reference_price"])
            if quality["direction"] == "buy":
                favorable = float(row["_high"] / reference - 1.0)
                adverse = float(row["_low"] / reference - 1.0)
                covered = float(row["_low"]) <= float(quality["coverage_boundary"])
            else:
                favorable = float(1.0 - row["_low"] / reference)
                adverse = float(1.0 - row["_high"] / reference)
                covered = float(row["_high"]) >= float(quality["coverage_boundary"])
            quality["post_maturity_max_favorable_return"] = max(
                float(quality["post_maturity_max_favorable_return"]), favorable
            )
            quality["post_maturity_max_adverse_return"] = min(
                float(quality["post_maturity_max_adverse_return"]), adverse
            )
            if covered:
                resolve_quality(quality_index, "covered", timestamp, feed_index, "price_crossed_bsp_extreme")
                del post_maturity_candidates[key]
            elif key not in active_keys:
                resolve_quality(quality_index, "structurally_invalidated", timestamp, feed_index, "mature_bsp_disappeared")
                del post_maturity_candidates[key]
            elif elapsed >= int(post_maturity_validation_bars):
                resolve_quality(quality_index, "held", timestamp, feed_index, "survived_validation_horizon")
                del post_maturity_candidates[key]

        # A timed-out candidate remains ignored until it disappears, preventing
        # the same still-visible point from being opened as a new candidate.
        ignored_until_absent.intersection_update(active_keys)

        for key, record_index in list(open_candidates.items()):
            record = records[record_index]
            if key not in active_keys:
                resolve(record_index, "invalidated", timestamp, feed_index, "candidate_disappeared_before_confirmation")
                del open_candidates[key]
                continue
            snapshot = active[key]
            record["snapshot_last_seen"] = timestamp
            record["revision_count"] += 1
            if _is_structurally_sure(snapshot):
                resolve(record_index, "mature", timestamp, feed_index, "source_line_became_sure")
                start_quality_monitor(
                    key, record_index, snapshot, timestamp, feed_index, float(row["_close"])
                )
                del open_candidates[key]
                ignored_until_absent.add(key)
            elif feed_index - record["_first_feed_index"] >= int(maximum_resolution_bars):
                resolve(record_index, "censored", timestamp, feed_index, "maximum_resolution_bars_reached")
                del open_candidates[key]
                ignored_until_absent.add(key)

        if timestamp >= period_start:
            for key, snapshot in active.items():
                if key in open_candidates or key in ignored_until_absent:
                    continue
                candidate_id = f"{config.chan.code.upper()}-{next_candidate_number:09d}"
                next_candidate_number += 1
                record = dict(snapshot)
                record.update({
                    "candidate_id": candidate_id,
                    "candidate_key": "|".join(map(str, key)),
                    "timestamp": pd.to_datetime(snapshot["timestamp"]),
                    "bsp_timestamp": pd.to_datetime(snapshot["timestamp"]),
                    "snapshot_timestamp": timestamp,
                    "first_seen_timestamp": timestamp,
                    "snapshot_last_seen": timestamp,
                    "snapshot_first_seen": int(feed_index + 1),
                    "revision_count": 1,
                    "initial_source_is_sure": int(_is_structurally_sure(snapshot)),
                    "current_status": "pending",
                    "is_mature": 0,
                    "eventual_is_mature": np.nan,
                    "is_invalidated": 0,
                    "is_censored": 0,
                    "resolution_timestamp": pd.NaT,
                    "bars_to_resolution": np.nan,
                    "maturity_reason": None,
                    "invalidation_reason": None,
                    "_first_feed_index": int(feed_index),
                })
                records.append(record)
                record_index = len(records) - 1
                if _is_structurally_sure(snapshot):
                    resolve(record_index, "mature", timestamp, feed_index, "already_sure_when_first_seen")
                    start_quality_monitor(
                        key, record_index, snapshot, timestamp, feed_index, float(row["_close"])
                    )
                    ignored_until_absent.add(key)
                else:
                    open_candidates[key] = record_index

        if verbose and (feed_index + 1) % int(progress_every_rows) == 0:
            resolved = sum(r["current_status"] != "pending" for r in records)
            print(
                f"[Maturity] {feed_index + 1:,}/{len(feed):,} bars | "
                f"candidates={len(records):,}, resolved={resolved:,}, open={len(open_candidates):,}",
                flush=True,
            )

    final_timestamp = feed["timestamp"].iloc[-1] if len(feed) else period_end
    for key, record_index in list(open_candidates.items()):
        resolve(record_index, "censored", final_timestamp, len(feed) - 1, "end_of_data")
    for key, quality_index in list(post_maturity_candidates.items()):
        resolve_quality(quality_index, "censored", final_timestamp, len(feed) - 1, "end_of_data_before_quality_horizon")

    frame = pd.DataFrame(records)
    quality_frame = pd.DataFrame(quality_records)
    if not frame.empty:
        frame = frame.drop(columns=["_first_feed_index"], errors="ignore")
        frame = frame.sort_values(["snapshot_timestamp", "candidate_id"]).reset_index(drop=True)
    if not quality_frame.empty:
        quality_frame = quality_frame.drop(columns=["_maturity_feed_index"], errors="ignore")
        quality_frame = quality_frame.sort_values(["maturity_timestamp", "candidate_id"]).reset_index(drop=True)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    csv_target = Path(csv_path) if csv_path is not None else target.with_suffix(".csv")
    quality_csv_target = csv_target.with_name(f"{csv_target.stem}_quality{csv_target.suffix}")
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_target, index=False)
    quality_frame.to_csv(quality_csv_target, index=False)

    summary = (
        frame.groupby(["bsp_type", "current_status"]).size().unstack(fill_value=0)
        if not frame.empty else pd.DataFrame()
    )
    metadata = pd.DataFrame({
        "parameter": [
            "definition", "start", "end", "chan_window_bars",
            "maximum_resolution_bars", "post_maturity_validation_bars",
            "coverage_buffer_pct", "generated_rows", "quality_rows",
        ],
        "value": [
            "mature iff candidate survives until source Bi/segment is_sure",
            str(period_start), str(period_end), window, int(maximum_resolution_bars),
            int(post_maturity_validation_bars), float(coverage_buffer_pct),
            len(frame), len(quality_frame),
        ],
    })
    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="BSP Points", index=False)
        quality_frame.to_excel(writer, sheet_name="Mature BSP Quality", index=False)
        summary.to_excel(writer, sheet_name="Outcome Summary")
        metadata.to_excel(writer, sheet_name="Metadata", index=False)

    result = {
        "excel_path": str(target.resolve()), "csv_path": str(csv_target.resolve()),
        "quality_csv_path": str(quality_csv_target.resolve()),
        "rows": len(frame),
        "outcomes": frame["current_status"].value_counts().to_dict() if not frame.empty else {},
        "by_type": summary.to_dict() if not summary.empty else {},
        "quality_outcomes": quality_frame["quality_status"].value_counts().to_dict() if not quality_frame.empty else {},
    }
    if verbose:
        print(f"[Maturity] Saved {len(frame):,} candidates in {(perf_counter()-started)/60:.2f} minutes")
        print(f"[Maturity] Excel: {target.resolve()}")
        print(f"[Maturity] CSV:   {csv_target.resolve()}")
        print(f"[Quality]  CSV:   {quality_csv_target.resolve()}")
    return result


def _maturity_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = _OUTCOME_COLUMNS | _IDENTITY_COLUMNS | _NONSTATIONARY_COLUMNS
    return [
        column for column in frame.select_dtypes(include=[np.number, "bool"]).columns
        if column not in excluded and "next_bi_return" not in str(column).lower()
    ]


def _quality_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = (
        _OUTCOME_COLUMNS | _QUALITY_OUTCOME_COLUMNS | _IDENTITY_COLUMNS
        | _NONSTATIONARY_COLUMNS
    )
    return [
        column for column in frame.select_dtypes(include=[np.number, "bool"]).columns
        if column not in excluded and "next_bi_return" not in str(column).lower()
    ]


def train_bsp_maturity_models(
    dataset_path: str | Path,
    *,
    output_dir: str | Path = "outputs/bsp_maturity_models",
    train_start_date: str | None = None,
    train_end_date: str,
    validation_start_date: str,
    validation_end_date: str,
    test_start_date: str,
    test_end_date: str | None = None,
    bsp_types: tuple[str, ...] = ("1", "1p", "2", "2s", "3a", "3b"),
    xgboost_params: dict[str, Any] | None = None,
    minimum_rows_per_type: int = 200,
    probability_threshold: float = 0.5,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train one maturity classifier per exact BSP type using purged dates."""
    from xgboost import XGBClassifier

    source = Path(dataset_path)
    frame = (
        pd.read_excel(source, sheet_name="BSP Points")
        if source.suffix.lower() in {".xlsx", ".xls"}
        else pd.read_csv(source)
    )
    required = {"bsp_type", "snapshot_timestamp", "resolution_timestamp", "eventual_is_mature"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Maturity dataset is missing columns: {sorted(missing)}")
    frame["bsp_type"] = frame["bsp_type"].astype(str).str.lower().str.replace(r"\.0$", "", regex=True)
    frame["snapshot_timestamp"] = pd.to_datetime(frame["snapshot_timestamp"], errors="coerce")
    frame["resolution_timestamp"] = pd.to_datetime(frame["resolution_timestamp"], errors="coerce")
    frame["eventual_is_mature"] = pd.to_numeric(frame["eventual_is_mature"], errors="coerce")
    frame = frame.dropna(subset=["snapshot_timestamp", "resolution_timestamp", "eventual_is_mature"])
    # A point already confirmed when first observed needs no maturity forecast;
    # including it would let the model learn the answer directly from is_sure.
    if "initial_source_is_sure" in frame.columns:
        frame = frame.loc[pd.to_numeric(frame["initial_source_is_sure"], errors="coerce").fillna(0).eq(0)].copy()
    features = _maturity_feature_columns(frame)
    if not features:
        raise ValueError("No causal numeric maturity features were found")

    def inclusive_end(value: str | None):
        if value is None:
            return None
        stamp = pd.Timestamp(value)
        return stamp + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1) if stamp == stamp.normalize() else stamp

    train_end = inclusive_end(train_end_date)
    validation_end = inclusive_end(validation_end_date)
    test_end = inclusive_end(test_end_date)
    train_mask = frame["snapshot_timestamp"].le(train_end) & frame["resolution_timestamp"].le(train_end)
    if train_start_date:
        train_mask &= frame["snapshot_timestamp"].ge(pd.Timestamp(train_start_date))
    validation_mask = (
        frame["snapshot_timestamp"].ge(pd.Timestamp(validation_start_date))
        & frame["snapshot_timestamp"].le(validation_end)
        & frame["resolution_timestamp"].le(validation_end)
    )
    test_mask = frame["snapshot_timestamp"].ge(pd.Timestamp(test_start_date))
    if test_end is not None:
        test_mask &= frame["snapshot_timestamp"].le(test_end) & frame["resolution_timestamp"].le(test_end)

    parameters = {
        "n_estimators": 400, "max_depth": 5, "learning_rate": 0.03,
        "subsample": 0.8, "colsample_bytree": 0.8, "n_jobs": 1,
        "random_state": 42, "eval_metric": "logloss",
    }
    parameters.update(xgboost_params or {})
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"models": {}, "skipped": {}, "feature_count": len(features)}

    def metrics(truth, probability):
        prediction = (probability >= float(probability_threshold)).astype(int)
        result = {
            "rows": int(len(truth)), "positive_rate": float(np.mean(truth)),
            "accuracy": float(accuracy_score(truth, prediction)),
            "precision": float(precision_score(truth, prediction, zero_division=0)),
            "recall": float(recall_score(truth, prediction, zero_division=0)),
            "brier": float(brier_score_loss(truth, probability)),
        }
        if len(np.unique(truth)) == 2:
            result["roc_auc"] = float(roc_auc_score(truth, probability))
            result["pr_auc"] = float(average_precision_score(truth, probability))
        return result

    for bsp_type in map(str.lower, bsp_types):
        typed = frame["bsp_type"].eq(bsp_type)
        splits = {
            "train": frame.loc[typed & train_mask],
            "validation": frame.loc[typed & validation_mask],
            "test": frame.loc[typed & test_mask],
        }
        if min(map(len, splits.values())) < int(minimum_rows_per_type):
            report["skipped"][bsp_type] = {name: len(part) for name, part in splits.items()}
            continue
        y_train = splits["train"]["eventual_is_mature"].astype(int)
        if y_train.nunique() < 2:
            report["skipped"][bsp_type] = {"reason": "training split has one class only"}
            continue
        negative, positive = np.bincount(y_train, minlength=2)
        model_parameters = dict(parameters)
        model_parameters.setdefault("scale_pos_weight", float(negative / positive) if positive else 1.0)
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(**model_parameters)),
        ])
        model.fit(splits["train"][features], y_train)
        type_report = {}
        predictions = []
        for name, part in splits.items():
            truth = part["eventual_is_mature"].astype(int).to_numpy()
            probability = model.predict_proba(part[features])[:, 1]
            type_report[name] = metrics(truth, probability)
            if name == "test":
                prediction_frame = part[[
                    "candidate_id", "snapshot_timestamp", "bsp_timestamp",
                    "bsp_type", "direction", "eventual_is_mature",
                ]].copy()
                prediction_frame["maturity_probability"] = probability
                prediction_frame["predicted_mature"] = probability >= float(probability_threshold)
                predictions.append(prediction_frame)
        artifact = {
            "artifact_version": 1, "bsp_type": bsp_type, "model": model,
            "features": features, "probability_threshold": float(probability_threshold),
            "xgboost_params": model_parameters,
        }
        joblib.dump(artifact, out / f"maturity_model_type_{bsp_type}.joblib")
        if predictions:
            pd.concat(predictions).to_csv(out / f"test_predictions_type_{bsp_type}.csv", index=False)
        report["models"][bsp_type] = type_report
        if verbose:
            test_metrics = type_report["test"]
            print(
                f"[Maturity model {bsp_type}] test rows={test_metrics['rows']:,}, "
                f"precision={test_metrics['precision']:.3f}, recall={test_metrics['recall']:.3f}, "
                f"PR-AUC={test_metrics.get('pr_auc', float('nan')):.3f}", flush=True,
            )

    pd.DataFrame({"feature": features}).to_csv(out / "feature_manifest.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def train_bsp_quality_models(
    dataset_path: str | Path,
    *,
    output_dir: str | Path = "outputs/bsp_quality_models",
    train_start_date: str | None = None,
    train_end_date: str,
    validation_start_date: str,
    validation_end_date: str,
    test_start_date: str,
    test_end_date: str | None = None,
    bsp_types: tuple[str, ...] = ("1", "1p", "2", "2s", "3a", "3b"),
    xgboost_params: dict[str, Any] | None = None,
    minimum_rows_per_type: int = 200,
    probability_threshold: float = 0.5,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train per-type P(mature BSP remains valid and uncovered) models."""
    from xgboost import XGBClassifier

    source = Path(dataset_path)
    frame = (
        pd.read_excel(source, sheet_name="Mature BSP Quality")
        if source.suffix.lower() in {".xlsx", ".xls"}
        else pd.read_csv(source)
    )
    required = {
        "bsp_type", "maturity_timestamp", "quality_resolution_timestamp",
        "quality_target",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"BSP quality dataset is missing columns: {sorted(missing)}")
    frame["bsp_type"] = frame["bsp_type"].astype(str).str.lower().str.replace(r"\.0$", "", regex=True)
    frame["maturity_timestamp"] = pd.to_datetime(frame["maturity_timestamp"], errors="coerce")
    frame["quality_resolution_timestamp"] = pd.to_datetime(frame["quality_resolution_timestamp"], errors="coerce")
    frame["quality_target"] = pd.to_numeric(frame["quality_target"], errors="coerce")
    frame = frame.dropna(subset=[
        "maturity_timestamp", "quality_resolution_timestamp", "quality_target",
    ]).copy()
    features = _quality_feature_columns(frame)
    if not features:
        raise ValueError("No causal numeric BSP quality features were found")

    def inclusive_end(value: str | None):
        if value is None:
            return None
        stamp = pd.Timestamp(value)
        return stamp + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1) if stamp == stamp.normalize() else stamp

    train_end = inclusive_end(train_end_date)
    validation_end = inclusive_end(validation_end_date)
    test_end = inclusive_end(test_end_date)
    train_mask = frame["maturity_timestamp"].le(train_end) & frame["quality_resolution_timestamp"].le(train_end)
    if train_start_date:
        train_mask &= frame["maturity_timestamp"].ge(pd.Timestamp(train_start_date))
    validation_mask = (
        frame["maturity_timestamp"].ge(pd.Timestamp(validation_start_date))
        & frame["maturity_timestamp"].le(validation_end)
        & frame["quality_resolution_timestamp"].le(validation_end)
    )
    test_mask = frame["maturity_timestamp"].ge(pd.Timestamp(test_start_date))
    if test_end is not None:
        test_mask &= frame["maturity_timestamp"].le(test_end) & frame["quality_resolution_timestamp"].le(test_end)

    parameters = {
        "n_estimators": 400, "max_depth": 5, "learning_rate": 0.03,
        "subsample": 0.8, "colsample_bytree": 0.8, "n_jobs": 1,
        "random_state": 42, "eval_metric": "logloss",
    }
    parameters.update(xgboost_params or {})
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "target": "P(mature BSP remains structurally present and uncovered)",
        "models": {}, "skipped": {}, "feature_count": len(features),
    }

    def metrics(truth, probability):
        prediction = (probability >= float(probability_threshold)).astype(int)
        result = {
            "rows": int(len(truth)), "held_rate": float(np.mean(truth)),
            "accuracy": float(accuracy_score(truth, prediction)),
            "precision": float(precision_score(truth, prediction, zero_division=0)),
            "recall": float(recall_score(truth, prediction, zero_division=0)),
            "brier": float(brier_score_loss(truth, probability)),
        }
        if len(np.unique(truth)) == 2:
            result["roc_auc"] = float(roc_auc_score(truth, probability))
            result["pr_auc"] = float(average_precision_score(truth, probability))
        return result

    for bsp_type in map(str.lower, bsp_types):
        typed = frame["bsp_type"].eq(bsp_type)
        splits = {
            "train": frame.loc[typed & train_mask],
            "validation": frame.loc[typed & validation_mask],
            "test": frame.loc[typed & test_mask],
        }
        if min(map(len, splits.values())) < int(minimum_rows_per_type):
            report["skipped"][bsp_type] = {name: len(part) for name, part in splits.items()}
            continue
        y_train = splits["train"]["quality_target"].astype(int)
        if y_train.nunique() < 2:
            report["skipped"][bsp_type] = {"reason": "training split has one class only"}
            continue
        type_features = [
            feature for feature in features
            if splits["train"][feature].notna().any()
        ]
        if not type_features:
            report["skipped"][bsp_type] = {"reason": "no observed training features"}
            continue
        negative, positive = np.bincount(y_train, minlength=2)
        model_parameters = dict(parameters)
        model_parameters.setdefault("scale_pos_weight", float(negative / positive) if positive else 1.0)
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(**model_parameters)),
        ])
        model.fit(splits["train"][type_features], y_train)
        type_report = {}
        for name, part in splits.items():
            truth = part["quality_target"].astype(int).to_numpy()
            probability = model.predict_proba(part[type_features])[:, 1]
            type_report[name] = metrics(truth, probability)
            if name == "test":
                prediction_frame = part[[
                    "candidate_id", "maturity_timestamp", "bsp_timestamp",
                    "bsp_type", "direction", "quality_status", "quality_target",
                ]].copy()
                prediction_frame["hold_probability"] = probability
                prediction_frame["predicted_hold"] = probability >= float(probability_threshold)
                prediction_frame.to_csv(out / f"test_predictions_type_{bsp_type}.csv", index=False)
        joblib.dump({
            "artifact_version": 1, "target": "quality_target", "bsp_type": bsp_type,
            "model": model, "features": type_features,
            "probability_threshold": float(probability_threshold),
            "xgboost_params": model_parameters,
        }, out / f"quality_model_type_{bsp_type}.joblib")
        pd.DataFrame({"feature": type_features}).to_csv(
            out / f"feature_manifest_type_{bsp_type}.csv", index=False
        )
        type_report["feature_count"] = len(type_features)
        report["models"][bsp_type] = type_report
        if verbose:
            test_metrics = type_report["test"]
            print(
                f"[Quality model {bsp_type}] test rows={test_metrics['rows']:,}, "
                f"precision={test_metrics['precision']:.3f}, recall={test_metrics['recall']:.3f}, "
                f"PR-AUC={test_metrics.get('pr_auc', float('nan')):.3f}", flush=True,
            )

    pd.DataFrame({"feature": features}).to_csv(out / "feature_manifest.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


__all__ = [
    "generate_bsp_maturity_dataset", "train_bsp_maturity_models",
    "train_bsp_quality_models",
]
