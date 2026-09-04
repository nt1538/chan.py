"""Causal, precomputable multi-timeframe Chan feature architecture."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from Common.CEnum import KL_TYPE

from .chan_state import add_chan_features
from .features import normalize_ohlcv


@dataclass(frozen=True)
class ChanLevelSpec:
    """One independently calculated Chan timeframe."""

    name: str
    rule: str
    kl_type: KL_TYPE
    window_bars: int = 500


def default_chan_levels() -> tuple[ChanLevelSpec, ...]:
    return (
        ChanLevelSpec("5m", "5min", KL_TYPE.K_5M, 500),
        ChanLevelSpec("15m", "15min", KL_TYPE.K_15M, 500),
        ChanLevelSpec("30m", "30min", KL_TYPE.K_30M, 500),
        ChanLevelSpec("60m", "60min", KL_TYPE.K_60M, 500),
        ChanLevelSpec("1d", "1D", KL_TYPE.K_DAY, 500),
    )


@dataclass
class MultiLevelChanConfig:
    input_csv: str
    output_dir: str = "outputs/multilevel_chan"
    symbol: str = "TQQQ"
    levels: tuple[ChanLevelSpec, ...] = field(default_factory=default_chan_levels)
    base_bar_minutes: int = 5
    timestamp_semantics: Literal["bar_start", "bar_end"] = "bar_start"
    start: str | None = None
    end: str | None = None
    verbose: bool = True
    progress_every_rows: int = 2_500
    save_level_csvs: bool = True

    def to_dict(self) -> dict:
        result = asdict(self)
        for item, spec in zip(result["levels"], self.levels):
            item["kl_type"] = spec.kl_type.name
        return result


def _aggregate_ohlcv(part: pd.DataFrame, rule: str, semantics: str) -> pd.DataFrame:
    """Aggregate one trading date without allowing bins to cross sessions."""
    values = part.set_index("timestamp")
    if semantics == "bar_start":
        grouped = values.resample(
            rule, origin=part["timestamp"].iloc[0], closed="left", label="left"
        ).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        })
        grouped.index = grouped.index + pd.Timedelta(rule)
    else:
        grouped = values.resample(
            rule, origin=part["timestamp"].iloc[0], closed="right", label="right"
        ).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        })
    return grouped.dropna(subset=["open", "high", "low", "close"]).reset_index()


def resample_chan_level(
    frame: pd.DataFrame,
    spec: ChanLevelSpec,
    *,
    base_bar_minutes: int = 5,
    timestamp_semantics: Literal["bar_start", "bar_end"] = "bar_start",
) -> pd.DataFrame:
    """Create OHLCV bars labeled by the time at which they become observable."""
    if base_bar_minutes < 1:
        raise ValueError("base_bar_minutes must be positive")
    bars = normalize_ohlcv(frame)[
        ["timestamp", "open", "high", "low", "close", "volume"]
    ].copy()
    base_delta = pd.Timedelta(minutes=int(base_bar_minutes))
    rule_delta = pd.Timedelta(spec.rule) if spec.rule.lower() != "1d" else None

    if rule_delta == base_delta:
        result = bars.copy()
        if timestamp_semantics == "bar_start":
            result["timestamp"] += base_delta
        return result

    if spec.rule.lower() == "1d":
        records = []
        for _, part in bars.groupby(bars["timestamp"].dt.normalize(), sort=True):
            available = part["timestamp"].iloc[-1]
            if timestamp_semantics == "bar_start":
                available += base_delta
            records.append({
                "timestamp": available,
                "open": float(part["open"].iloc[0]),
                "high": float(part["high"].max()),
                "low": float(part["low"].min()),
                "close": float(part["close"].iloc[-1]),
                "volume": float(part["volume"].sum()),
            })
        return pd.DataFrame.from_records(records)

    chunks = [
        _aggregate_ohlcv(part, spec.rule, timestamp_semantics)
        for _, part in bars.groupby(bars["timestamp"].dt.normalize(), sort=True)
        if not part.empty
    ]
    return pd.concat(chunks, ignore_index=True) if chunks else bars.iloc[:0].copy()


def _add_cross_level_features(frame: pd.DataFrame, level_names: list[str]) -> pd.DataFrame:
    result = frame.copy()
    bi_directions = [
        f"mlchan_{name}_last_bi_direction" for name in level_names
        if f"mlchan_{name}_last_bi_direction" in result
    ]
    seg_directions = [
        f"mlchan_{name}_last_seg_direction" for name in level_names
        if f"mlchan_{name}_last_seg_direction" in result
    ]
    bottom_divergences = [
        f"mlchan_{name}_last_bi_area_divergence_strength" for name in level_names
        if f"mlchan_{name}_last_bi_area_divergence_strength" in result
    ]
    peak_divergences = [
        f"mlchan_{name}_last_bi_peak_divergence_strength" for name in level_names
        if f"mlchan_{name}_last_bi_peak_divergence_strength" in result
    ]
    if bi_directions:
        values = result[bi_directions].fillna(0.0)
        result["mlchan_bi_down_level_count"] = values.lt(0).sum(axis=1)
        result["mlchan_bi_up_level_count"] = values.gt(0).sum(axis=1)
        result["mlchan_bi_direction_balance"] = np.sign(values).sum(axis=1)
    if seg_directions:
        values = result[seg_directions].fillna(0.0)
        result["mlchan_seg_down_level_count"] = values.lt(0).sum(axis=1)
        result["mlchan_seg_up_level_count"] = values.gt(0).sum(axis=1)
        result["mlchan_seg_direction_balance"] = np.sign(values).sum(axis=1)
    if bottom_divergences:
        values = result[bottom_divergences].fillna(0.0)
        result["mlchan_area_divergence_level_count"] = values.gt(0).sum(axis=1)
        result["mlchan_area_divergence_strength_sum"] = values.sum(axis=1)
        result["mlchan_area_divergence_strength_max"] = values.max(axis=1)
    if peak_divergences:
        values = result[peak_divergences].fillna(0.0)
        result["mlchan_peak_divergence_level_count"] = values.gt(0).sum(axis=1)
        result["mlchan_peak_divergence_strength_sum"] = values.sum(axis=1)
    # A generic divergence strength is not a trade direction. Split it using
    # the direction of the Bi whose metric was compared.
    for metric in ("area", "peak", "full_area", "slope", "amplitude", "rsi"):
        bottom_columns: list[str] = []
        top_columns: list[str] = []
        for name in level_names:
            direction_column = f"mlchan_{name}_last_bi_direction"
            strength_column = f"mlchan_{name}_last_bi_{metric}_divergence_strength"
            if direction_column not in result or strength_column not in result:
                continue
            bottom_column = f"mlchan_{name}_bottom_{metric}_divergence_strength"
            top_column = f"mlchan_{name}_top_{metric}_divergence_strength"
            strength = result[strength_column].fillna(0.0)
            result[bottom_column] = strength.where(result[direction_column].lt(0), 0.0)
            result[top_column] = strength.where(result[direction_column].gt(0), 0.0)
            bottom_columns.append(bottom_column)
            top_columns.append(top_column)
        for side, columns in (("bottom", bottom_columns), ("top", top_columns)):
            if not columns:
                continue
            values = result[columns].fillna(0.0)
            result[f"mlchan_{side}_{metric}_divergence_level_count"] = (
                values.gt(0).sum(axis=1)
            )
            result[f"mlchan_{side}_{metric}_divergence_strength_sum"] = values.sum(axis=1)
            result[f"mlchan_{side}_{metric}_divergence_strength_max"] = values.max(axis=1)
    return result


def multilevel_chan_core_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return a compact structural manifest suitable for first-pass models."""
    structural_tokens = (
        "_age_minutes", "_merged_kline_count", "_bi_count", "_seg_count",
        "_zs_count", "_bsp_count", "_bars_since_bsp", "_last_bsp_direction",
        "_last_bi_direction", "_last_bi_sure", "_last_bi_amplitude_pct",
        "_last_bi_klu_count", "_last_bi_price_extension",
        "_last_bi_previous_end_distance", "_last_seg_direction",
        "_last_seg_sure", "_last_seg_amplitude_pct", "_last_seg_slope",
        "_last_seg_bi_count", "_last_zs_", "_latest_new_bsp_",
        "_current_last_bsp_", "_ratio", "_divergence_strength",
    )
    aggregate_prefixes = (
        "mlchan_bi_", "mlchan_seg_", "mlchan_area_", "mlchan_peak_",
        "mlchan_bottom_", "mlchan_top_",
    )
    return [
        column for column in frame.columns
        if str(column).startswith(aggregate_prefixes)
        or (
            str(column).startswith("mlchan_")
            and not str(column).endswith("_available_timestamp")
            and any(token in str(column) for token in structural_tokens)
        )
    ]


def build_multilevel_chan_features(config: MultiLevelChanConfig) -> dict:
    """Build and save causal per-level and base-bar-aligned Chan snapshots."""
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    level_dir = out / "levels"
    level_dir.mkdir(parents=True, exist_ok=True)

    raw = normalize_ohlcv(pd.read_csv(config.input_csv))
    if config.start is not None:
        raw = raw.loc[raw["timestamp"].ge(pd.Timestamp(config.start))]
    if config.end is not None:
        end = pd.Timestamp(config.end)
        if end == end.normalize():
            end += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        raw = raw.loc[raw["timestamp"].le(end)]
    raw = raw.reset_index(drop=True)
    if raw.empty:
        raise ValueError("No source K-lines matched the requested date range")

    base_available = raw["timestamp"].copy()
    if config.timestamp_semantics == "bar_start":
        base_available += pd.Timedelta(minutes=config.base_bar_minutes)
    aligned = pd.DataFrame({
        "timestamp": raw["timestamp"],
        "mlchan_base_available_timestamp": base_available,
    }).sort_values("mlchan_base_available_timestamp")
    level_paths: dict[str, str] = {}

    for spec in config.levels:
        if config.verbose:
            print(
                f"[MultiLevelChan] {spec.name}: resample={spec.rule}, "
                f"window={spec.window_bars}", flush=True,
            )
        bars = resample_chan_level(
            raw, spec,
            base_bar_minutes=config.base_bar_minutes,
            timestamp_semantics=config.timestamp_semantics,
        )
        prefix = f"mlchan_{spec.name}_"
        enriched = add_chan_features(
            bars,
            config.symbol,
            spec.window_bars,
            verbose=config.verbose,
            progress_every_rows=config.progress_every_rows,
            level=spec.kl_type,
            feature_prefix=prefix,
        )
        feature_columns = [
            column for column in enriched.columns if str(column).startswith(prefix)
        ]
        state = enriched[["timestamp", *feature_columns]].copy()
        available_column = f"mlchan_{spec.name}_available_timestamp"
        state = state.rename(columns={"timestamp": available_column}).sort_values(
            available_column
        )
        aligned = pd.merge_asof(
            aligned.sort_values("mlchan_base_available_timestamp"),
            state,
            left_on="mlchan_base_available_timestamp",
            right_on=available_column,
            direction="backward",
            allow_exact_matches=True,
        )
        aligned[f"mlchan_{spec.name}_age_minutes"] = (
            aligned["mlchan_base_available_timestamp"] - aligned[available_column]
        ).dt.total_seconds() / 60.0
        if config.save_level_csvs:
            path = level_dir / f"{spec.name}_chan_features.csv"
            state.to_csv(path, index=False)
            level_paths[spec.name] = str(path.resolve())

    aligned = _add_cross_level_features(
        aligned, [spec.name for spec in config.levels]
    )
    aligned_path = out / "multilevel_chan_aligned.csv"
    aligned.to_csv(aligned_path, index=False)
    core_columns = multilevel_chan_core_feature_columns(aligned)
    core_path = out / "multilevel_chan_core.csv"
    aligned[["timestamp", "mlchan_base_available_timestamp", *core_columns]].to_csv(
        core_path, index=False
    )
    config_path = out / "multilevel_chan_config.json"
    config_path.write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    return {
        "output_dir": str(out.resolve()),
        "aligned_path": str(aligned_path.resolve()),
        "core_path": str(core_path.resolve()),
        "level_paths": level_paths,
        "config_path": str(config_path.resolve()),
        "rows": int(len(aligned)),
        "feature_count": int(sum(str(c).startswith("mlchan_") for c in aligned.columns)),
        "core_feature_count": int(len(core_columns)),
        "frame": aligned,
    }


def attach_multilevel_chan_to_bsp(
    bspoint_path: str | Path,
    multilevel_path: str | Path,
    *,
    bsp_timestamp_column: str = "snapshot_timestamp",
    output_csv: str | Path | None = None,
) -> dict:
    """Attach the latest observable multi-level state to every BSP record."""
    bsp = pd.read_csv(bspoint_path, low_memory=False)
    if bsp_timestamp_column not in bsp:
        raise KeyError(f"BSP file has no {bsp_timestamp_column!r} column")
    features = pd.read_csv(multilevel_path, low_memory=False)
    if "timestamp" not in features:
        raise KeyError("Multi-level feature file has no 'timestamp' column")
    bsp["_ml_original_order"] = np.arange(len(bsp))
    bsp["_ml_join_timestamp"] = pd.to_datetime(
        bsp[bsp_timestamp_column], errors="coerce"
    )
    features["timestamp"] = pd.to_datetime(features["timestamp"], errors="coerce")
    feature_columns = [
        column for column in features if str(column).startswith("mlchan_")
    ]
    joined = pd.merge_asof(
        bsp.dropna(subset=["_ml_join_timestamp"]).sort_values("_ml_join_timestamp"),
        features[["timestamp", *feature_columns]].dropna(subset=["timestamp"]).sort_values("timestamp"),
        left_on="_ml_join_timestamp",
        right_on="timestamp",
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("_ml_original_order")
    joined = joined.drop(columns=["_ml_original_order", "_ml_join_timestamp"])
    path = Path(output_csv) if output_csv is not None else (
        Path(bspoint_path).with_name(f"{Path(bspoint_path).stem}_multilevel.csv")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(path, index=False)
    return {
        "output_csv": str(path.resolve()),
        "rows": int(len(joined)),
        "feature_count": int(len(feature_columns)),
        "frame": joined,
    }
