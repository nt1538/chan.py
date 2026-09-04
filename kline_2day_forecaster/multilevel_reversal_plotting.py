"""Causal plots for multi-level BSP reversal-entry confirmation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import normalize_ohlcv
from .multilevel_chan import default_chan_levels, resample_chan_level
from .multilevel_plotting import _attach_event_prices, _level_bsp_events


def _timeframe_bars(prices: pd.DataFrame, level: str) -> pd.DataFrame:
    specs = {spec.name: spec for spec in default_chan_levels()}
    if level not in specs:
        raise ValueError(f"Unsupported level {level!r}; choose from {sorted(specs)}")
    return resample_chan_level(
        prices, specs[level], base_bar_minutes=5, timestamp_semantics="bar_start"
    ).sort_values("timestamp").reset_index(drop=True)


def _add_atr_and_confirmed_pivots(
    bars: pd.DataFrame,
    *,
    atr_period: int,
    pivot_left_bars: int,
    pivot_right_bars: int,
) -> pd.DataFrame:
    result = bars.copy()
    previous_close = result["close"].shift(1)
    true_range = pd.concat(
        [
            result["high"] - result["low"],
            (result["high"] - previous_close).abs(),
            (result["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["atr"] = true_range.rolling(int(atr_period), min_periods=int(atr_period)).mean()

    width = int(pivot_left_bars) + int(pivot_right_bars) + 1
    highs = result["high"].rolling(width, center=True)
    lows = result["low"].rolling(width, center=True)
    result["pivot_high"] = result["high"].where(result["high"].eq(highs.max()))
    result["pivot_low"] = result["low"].where(result["low"].eq(lows.min()))
    result["pivot_confirmed_timestamp"] = result["timestamp"].shift(
        -int(pivot_right_bars)
    )
    return result


def _latest_confirmed_pivot(
    bars: pd.DataFrame,
    *,
    at: pd.Timestamp,
    side: str,
) -> pd.Series | None:
    column = "pivot_high" if side == "high" else "pivot_low"
    eligible = bars.loc[
        bars[column].notna() & bars["pivot_confirmed_timestamp"].le(at)
    ]
    return None if eligible.empty else eligible.iloc[-1]


def _first_execution_cross(
    execution: pd.DataFrame,
    *,
    after: pd.Timestamp,
    before: pd.Timestamp,
    direction: str,
    trigger_price: float,
    invalidation_price: float,
) -> tuple[pd.Series | None, pd.Series | None]:
    part = execution.loc[
        execution["timestamp"].gt(after) & execution["timestamp"].le(before)
    ]
    for _, row in part.iterrows():
        invalidated = (
            float(row["low"]) < invalidation_price
            if direction == "buy"
            else float(row["high"]) > invalidation_price
        )
        if invalidated:
            return None, row
        crossed = (
            float(row["close"]) > trigger_price
            if direction == "buy"
            else float(row["close"]) < trigger_price
        )
        if crossed:
            return row, None
    return None, None


def _evaluate_candidate(
    candidate: pd.Series,
    structure: pd.DataFrame,
    execution: pd.DataFrame,
    *,
    max_wait_minutes: int,
    invalidation_lookback_bars: int,
    break_buffer_atr: float,
    pullback_tolerance_atr: float,
    require_execution_rebreak: bool,
    enter_at_timeout_if_valid: bool,
    confirmation_rule: str,
    maximum_break_extension_atr: float,
    maximum_break_delay_bars: int,
    maximum_timeout_extension_atr: float,
) -> dict[str, Any]:
    timestamp = pd.Timestamp(candidate["event_timestamp"])
    direction = str(candidate["direction"])
    deadline = timestamp + pd.Timedelta(minutes=int(max_wait_minutes))
    history = structure.loc[structure["timestamp"].le(timestamp)].tail(
        int(invalidation_lookback_bars)
    )
    if history.empty:
        return {"status": "insufficient_history", "deadline": deadline}
    available_atr = history["atr"].dropna()
    candidate_atr = float(available_atr.iloc[-1]) if not available_atr.empty else np.nan

    if direction == "buy":
        reference = _latest_confirmed_pivot(structure, at=timestamp, side="high")
        invalidation = float(history["low"].min())
        reference_price = np.nan if reference is None else float(reference["pivot_high"])
    else:
        reference = _latest_confirmed_pivot(structure, at=timestamp, side="low")
        invalidation = float(history["high"].max())
        reference_price = np.nan if reference is None else float(reference["pivot_low"])

    center_boundary = np.nan
    if confirmation_rule == "center_retest":
        distance_name = "zs_high_distance" if direction == "buy" else "zs_low_distance"
        distance = candidate.get(distance_name, np.nan)
        if pd.notna(distance) and 1.0 + float(distance) != 0.0:
            center_boundary = float(candidate["price"]) / (1.0 + float(distance))
            # Keep ``candidate_atr`` as the scalar captured at the BSP time.
            # Reusing that name for this Series later makes comparisons such as
            # ``candidate_atr > 0`` ambiguous in pandas.
            candidate_atr_values = history["atr"].dropna()
            tolerance = (
                float(pullback_tolerance_atr) * float(candidate_atr_values.iloc[-1])
                if not candidate_atr_values.empty else 0.0
            )
            invalidation = (
                center_boundary - tolerance
                if direction == "buy"
                else center_boundary + tolerance
            )

    def timeout_fallback(base: dict[str, Any], fallback_status: str) -> dict[str, Any]:
        if not enter_at_timeout_if_valid:
            return {**base, "status": fallback_status}
        timeout_rows = execution.loc[execution["timestamp"].ge(deadline)]
        if timeout_rows.empty:
            return {**base, "status": fallback_status}
        timeout_row = timeout_rows.iloc[0]
        observed = execution.loc[
            execution["timestamp"].gt(timestamp)
            & execution["timestamp"].le(timeout_row["timestamp"])
        ]
        invalidated = (
            observed["low"].lt(invalidation)
            if direction == "buy"
            else observed["high"].gt(invalidation)
        )
        if invalidated.any():
            invalidated_row = observed.loc[invalidated].iloc[0]
            return {
                **base, "status": "invalidated_before_timeout_entry",
                "invalidation_timestamp": invalidated_row["timestamp"],
            }
        direction_sign = 1.0 if direction == "buy" else -1.0
        timeout_extension_atr = (
            direction_sign * (float(timeout_row["close"]) - float(candidate["price"]))
            / candidate_atr
            if pd.notna(candidate_atr) and candidate_atr > 0 else np.nan
        )
        if (
            pd.notna(timeout_extension_atr)
            and timeout_extension_atr > float(maximum_timeout_extension_atr)
        ):
            return {
                **base,
                "status": "timeout_valid_but_overextended",
                "timeout_extension_atr": timeout_extension_atr,
            }
        return {
            **base, "status": "entry_confirmed_timeout",
            "entry_reason": "timeout_still_valid",
            "timeout_extension_atr": timeout_extension_atr,
            "entry_timestamp": timeout_row["timestamp"],
            "entry_price": float(timeout_row["close"]),
        }

    if reference is None:
        return timeout_fallback({
            "deadline": deadline,
            "invalidation_price": invalidation,
            "candidate_atr": candidate_atr,
            "confirmation_rule": confirmation_rule,
            "center_boundary": center_boundary,
        }, "no_reference_pivot")

    future = structure.loc[
        structure["timestamp"].gt(timestamp) & structure["timestamp"].le(deadline)
    ].copy()
    if future.empty:
        return timeout_fallback({
            "deadline": deadline,
            "reference_price": reference_price, "invalidation_price": invalidation,
            "candidate_atr": candidate_atr,
            "confirmation_rule": confirmation_rule,
            "center_boundary": center_boundary,
        }, "no_future_bars")

    break_row = None
    invalidated_row = None
    for _, row in future.iterrows():
        if direction == "buy" and float(row["low"]) < invalidation:
            invalidated_row = row
            break
        if direction == "sell" and float(row["high"]) > invalidation:
            invalidated_row = row
            break
        atr = float(row["atr"]) if pd.notna(row["atr"]) else 0.0
        threshold = (
            reference_price + float(break_buffer_atr) * atr
            if direction == "buy"
            else reference_price - float(break_buffer_atr) * atr
        )
        crossed = float(row["close"]) > threshold if direction == "buy" else float(row["close"]) < threshold
        if crossed:
            break_row = row
            break
    if invalidated_row is not None:
        return {
            "status": "invalidated_before_break", "deadline": deadline,
            "reference_price": reference_price, "invalidation_price": invalidation,
            "candidate_atr": candidate_atr,
            "confirmation_rule": confirmation_rule,
            "center_boundary": center_boundary,
            "invalidation_timestamp": invalidated_row["timestamp"],
        }
    if break_row is None:
        return timeout_fallback({
            "deadline": deadline,
            "reference_price": reference_price, "invalidation_price": invalidation,
            "candidate_atr": candidate_atr,
            "confirmation_rule": confirmation_rule,
            "center_boundary": center_boundary,
        }, "no_structure_break")

    break_time = pd.Timestamp(break_row["timestamp"])
    break_price = float(break_row["close"])
    direction_sign = 1.0 if direction == "buy" else -1.0
    break_extension_atr = (
        direction_sign * (break_price - float(candidate["price"])) / candidate_atr
        if pd.notna(candidate_atr) and candidate_atr > 0 else np.nan
    )
    break_delay_bars = int(
        structure["timestamp"].gt(timestamp)
        .mul(structure["timestamp"].le(break_time)).sum()
    )
    extension_fresh = (
        pd.isna(break_extension_atr)
        or break_extension_atr <= float(maximum_break_extension_atr)
    )
    delay_fresh = break_delay_bars <= int(maximum_break_delay_bars)
    break_is_fresh = bool(extension_fresh and delay_fresh)
    stale_reasons: list[str] = []
    if not extension_fresh:
        stale_reasons.append("overextended")
    if not delay_fresh:
        stale_reasons.append("late")
    after_break = future.loc[future["timestamp"].gt(break_time)]
    pullback_row = None
    for _, row in after_break.iterrows():
        if direction == "buy" and float(row["low"]) < invalidation:
            invalidated_row = row
            break
        if direction == "sell" and float(row["high"]) > invalidation:
            invalidated_row = row
            break
        pivot_column = "pivot_low" if direction == "buy" else "pivot_high"
        if pd.isna(row[pivot_column]) or pd.Timestamp(row["pivot_confirmed_timestamp"]) > deadline:
            continue
        atr = float(row["atr"]) if pd.notna(row["atr"]) else 0.0
        if direction == "buy":
            held = (
                float(row["pivot_low"]) > invalidation
                and float(row["pivot_low"]) >= reference_price - float(pullback_tolerance_atr) * atr
            )
        else:
            held = (
                float(row["pivot_high"]) < invalidation
                and float(row["pivot_high"]) <= reference_price + float(pullback_tolerance_atr) * atr
            )
        if held:
            pullback_row = row
            break
    base = {
        "deadline": deadline,
        "reference_price": reference_price,
        "invalidation_price": invalidation,
        "confirmation_rule": confirmation_rule,
        "center_boundary": center_boundary,
        "candidate_atr": candidate_atr,
        "structure_break_timestamp": break_time,
        "structure_break_price": break_price,
        "break_extension_atr": break_extension_atr,
        "break_delay_bars": break_delay_bars,
        "break_is_fresh": break_is_fresh,
        "break_stale_reason": "+".join(stale_reasons),
    }
    if invalidated_row is not None:
        return {
            **base, "status": "invalidated_after_break",
            "invalidation_timestamp": invalidated_row["timestamp"],
        }
    if confirmation_rule in {"higher_low", "center_retest"} and break_is_fresh:
        entry_rows = execution.loc[execution["timestamp"].ge(break_time)]
        if entry_rows.empty:
            return timeout_fallback(base, "break_without_execution_bar")
        entry_row = entry_rows.iloc[0]
        return {
            **base,
            "status": "entry_confirmed",
            "entry_reason": (
                "higher_low_break" if confirmation_rule == "higher_low"
                else "center_retest_break"
            ),
            "entry_timestamp": entry_row["timestamp"],
            "entry_price": float(entry_row["close"]),
        }
    if pullback_row is None:
        return {
            **base,
            "status": (
                "break_without_valid_pullback" if break_is_fresh
                else "confirmed_but_stale"
            ),
        }

    pivot_time = pd.Timestamp(pullback_row["timestamp"])
    pullback_confirmed_time = pd.Timestamp(pullback_row["pivot_confirmed_timestamp"])
    pullback_price = float(
        pullback_row["pivot_low"] if direction == "buy" else pullback_row["pivot_high"]
    )
    impulse = execution.loc[
        execution["timestamp"].ge(break_time) & execution["timestamp"].le(pivot_time)
    ]
    if impulse.empty:
        return timeout_fallback({
            **base,
            "pullback_timestamp": pivot_time,
            "pullback_confirmed_timestamp": pullback_confirmed_time,
            "pullback_price": pullback_price,
        }, "pullback_held")
    trigger_price = float(
        impulse["high"].max() if direction == "buy" else impulse["low"].min()
    )
    entry_row = None
    entry_invalidation_row = None
    if require_execution_rebreak:
        entry_row, entry_invalidation_row = _first_execution_cross(
            execution, after=pullback_confirmed_time, before=deadline,
            direction=direction, trigger_price=trigger_price,
            invalidation_price=invalidation,
        )
    else:
        available = execution.loc[execution["timestamp"].ge(pullback_confirmed_time)]
        entry_row = None if available.empty else available.iloc[0]
    if entry_row is None:
        if entry_invalidation_row is not None:
            return {
                **base, "status": "invalidated_before_entry",
                "pullback_timestamp": pivot_time,
                "pullback_confirmed_timestamp": pullback_confirmed_time,
                "pullback_price": pullback_price,
                "entry_trigger_price": trigger_price,
                "invalidation_timestamp": entry_invalidation_row["timestamp"],
            }
        if not break_is_fresh:
            return {
                **base, "status": "confirmed_but_stale",
                "pullback_timestamp": pivot_time,
                "pullback_confirmed_timestamp": pullback_confirmed_time,
                "pullback_price": pullback_price,
                "entry_trigger_price": trigger_price,
            }
        if entry_invalidation_row is None and enter_at_timeout_if_valid:
            return timeout_fallback({
                **base,
                "pullback_timestamp": pivot_time,
                "pullback_confirmed_timestamp": pullback_confirmed_time,
                "pullback_price": pullback_price,
                "entry_trigger_price": trigger_price,
            }, "pullback_held_no_execution_rebreak")
        return {
            **base, "status": "pullback_held_no_execution_rebreak",
            "pullback_timestamp": pivot_time,
            "pullback_confirmed_timestamp": pullback_confirmed_time,
            "pullback_price": pullback_price,
            "entry_trigger_price": trigger_price,
            "invalidation_timestamp": (
                entry_invalidation_row["timestamp"]
                if entry_invalidation_row is not None else pd.NaT
            ),
        }
    return {
        **base, "status": "entry_confirmed",
        "entry_reason": (
            "delayed_retest_entry" if not break_is_fresh
            else (
                "execution_rebreak" if require_execution_rebreak
                else "pullback_confirmed"
            )
        ),
        "pullback_timestamp": pivot_time,
        "pullback_confirmed_timestamp": pullback_confirmed_time,
        "pullback_price": pullback_price,
        "entry_trigger_price": trigger_price,
        "entry_timestamp": entry_row["timestamp"],
        "entry_price": float(entry_row["close"]),
    }


def plot_multilevel_bsp_reversal_confirmation(
    multilevel_path: str | Path,
    price_csv: str | Path,
    *,
    start: str,
    end: str,
    large_level: str = "60m",
    large_levels: tuple[str, ...] | None = None,
    large_bsp_types: tuple[str, ...] = ("2", "2s"),
    direction: str = "buy",
    structure_level: str = "15m",
    execution_level: str = "5m",
    require_structure_level_bsp: bool = True,
    structure_level_bsp_types: tuple[str, ...] | None = None,
    structure_bsp_before_minutes: int = 60,
    structure_bsp_after_minutes: int | None = None,
    confirmation_mode: str = "type_aware",
    break_extension_limits_atr: dict[str, float] | None = None,
    break_delay_limits_bars: dict[str, int] | None = None,
    maximum_timeout_extension_atr: float = 1.00,
    max_wait_minutes: int = 780,
    invalidation_lookback_bars: int = 12,
    atr_period: int = 14,
    pivot_left_bars: int = 2,
    pivot_right_bars: int = 2,
    break_buffer_atr: float = 0.10,
    pullback_tolerance_atr: float = 0.25,
    require_execution_rebreak: bool = True,
    enter_at_timeout_if_valid: bool = False,
    show_invalidated: bool = True,
    annotate_max: int = 10,
    output_path: str | Path | None = None,
    signals_csv: str | Path | None = None,
    show: bool = True,
) -> dict[str, Any]:
    """Plot a causal BSP -> structure break -> pullback -> entry state chain."""
    if direction not in {"buy", "sell"}:
        raise ValueError("direction must be 'buy' or 'sell'")
    if min(atr_period, pivot_left_bars, pivot_right_bars, invalidation_lookback_bars) < 1:
        raise ValueError("ATR, pivot, and lookback parameters must be positive")
    if max_wait_minutes < 1 or break_buffer_atr < 0 or pullback_tolerance_atr < 0:
        raise ValueError("wait and ATR tolerance parameters must be non-negative")
    if structure_bsp_before_minutes < 0:
        raise ValueError("structure_bsp_before_minutes cannot be negative")
    if structure_bsp_after_minutes is not None and structure_bsp_after_minutes < 0:
        raise ValueError("structure_bsp_after_minutes cannot be negative")
    if confirmation_mode not in {"type_aware", "generic_reversal"}:
        raise ValueError("confirmation_mode must be 'type_aware' or 'generic_reversal'")
    extension_limits = {
        "trend_reversal": 1.50,
        "higher_low": 1.00,
        "center_retest": 0.75,
    }
    delay_limits = {
        "trend_reversal": 8,
        "higher_low": 4,
        "center_retest": 3,
    }
    if break_extension_limits_atr is not None:
        extension_limits.update({
            str(key): float(value)
            for key, value in break_extension_limits_atr.items()
        })
    if break_delay_limits_bars is not None:
        delay_limits.update({
            str(key): int(value)
            for key, value in break_delay_limits_bars.items()
        })
    required_rules = {"trend_reversal", "higher_low", "center_retest"}
    if required_rules.difference(extension_limits) or required_rules.difference(delay_limits):
        raise ValueError("break limit dictionaries must cover all confirmation rules")
    if any(value < 0 for value in extension_limits.values()):
        raise ValueError("break extension ATR limits cannot be negative")
    if any(value < 1 for value in delay_limits.values()):
        raise ValueError("break delay bar limits must be positive")
    if maximum_timeout_extension_atr < 0:
        raise ValueError("maximum_timeout_extension_atr cannot be negative")
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if end_ts <= start_ts:
        raise ValueError("end must be after start")
    candidate_levels = (
        tuple(dict.fromkeys(str(level) for level in large_levels))
        if large_levels is not None
        else (str(large_level),)
    )
    if not candidate_levels:
        raise ValueError("large_levels must contain at least one level")

    raw = normalize_ohlcv(pd.read_csv(price_csv))
    history_margin = pd.Timedelta(days=10)
    deadline_margin = pd.Timedelta(minutes=int(max_wait_minutes))
    working_prices = raw.loc[
        raw["timestamp"].between(start_ts - history_margin, end_ts + deadline_margin)
    ].copy()
    visible_prices = raw.loc[raw["timestamp"].between(start_ts, end_ts)].copy()
    if visible_prices.empty:
        raise ValueError("No price rows matched the requested plot period")

    # The aligned multi-level file can contain well over 1,000 columns.  This
    # plot only needs the large-level BSP event fields and two centre-distance
    # fields, so loading the full table wastes several GB and can exhaust the
    # notebook process.
    state_header = pd.read_csv(multilevel_path, nrows=0).columns
    bsp_levels = {*candidate_levels, structure_level}
    state_columns = [
        "timestamp",
        *[
            column
            for level in bsp_levels
            for column in (
                f"mlchan_{level}_available_timestamp",
                f"mlchan_{level}_latest_new_bsp_is_buy",
                f"mlchan_{level}_latest_new_bsp_is_sell",
                *(
                    f"mlchan_{level}_latest_new_bsp_type_{kind}"
                    for kind in ("1", "1p", "2", "2s", "3a", "3b")
                ),
            )
        ],
        *[
            column
            for level in candidate_levels
            for column in (
                f"mlchan_{level}_last_zs_low_distance",
                f"mlchan_{level}_last_zs_high_distance",
            )
        ],
    ]
    state_columns = [column for column in state_columns if column in state_header]
    states = pd.read_csv(
        multilevel_path,
        usecols=state_columns,
    )
    states["timestamp"] = pd.to_datetime(states["timestamp"], errors="coerce")
    state_margin = pd.Timedelta(minutes=int(structure_bsp_before_minutes))
    states = states.loc[
        states["timestamp"].between(start_ts - state_margin, end_ts)
    ].copy()
    candidate_frames: list[pd.DataFrame] = []
    for candidate_level in candidate_levels:
        level_candidates = _attach_event_prices(
            _level_bsp_events(states, candidate_level, large_bsp_types),
            working_prices,
        )
        level_candidates = level_candidates.loc[
            level_candidates["event_timestamp"].between(start_ts, end_ts)
            & level_candidates["direction"].eq(direction)
        ].copy()
        available_column = f"mlchan_{candidate_level}_available_timestamp"
        state_metadata = {
            "zs_low_distance": f"mlchan_{candidate_level}_last_zs_low_distance",
            "zs_high_distance": f"mlchan_{candidate_level}_last_zs_high_distance",
        }
        metadata_columns = [
            column for column in state_metadata.values() if column in states
        ]
        if available_column in states and metadata_columns:
            metadata = states[[available_column, *metadata_columns]].copy()
            metadata[available_column] = pd.to_datetime(
                metadata[available_column], errors="coerce"
            )
            metadata = (
                metadata.dropna(subset=[available_column])
                .groupby(available_column, as_index=False).last()
                .rename(columns={available_column: "event_timestamp", **{
                    source: target for target, source in state_metadata.items()
                    if source in metadata_columns
                }})
            )
            level_candidates = level_candidates.merge(
                metadata, on="event_timestamp", how="left"
            )
        level_candidates["large_level"] = candidate_level
        candidate_frames.append(level_candidates)
    candidates = (
        pd.concat(candidate_frames, ignore_index=True)
        .sort_values(["event_timestamp", "large_level"])
        .reset_index(drop=True)
    )

    structure_bsp_events = _attach_event_prices(
        _level_bsp_events(
            states,
            structure_level,
            structure_level_bsp_types,
        ),
        working_prices,
    )
    structure_bsp_events = structure_bsp_events.loc[
        structure_bsp_events["direction"].eq(direction)
    ].reset_index(drop=True)

    structure = _add_atr_and_confirmed_pivots(
        _timeframe_bars(working_prices, structure_level),
        atr_period=atr_period,
        pivot_left_bars=pivot_left_bars,
        pivot_right_bars=pivot_right_bars,
    )
    execution = _timeframe_bars(working_prices, execution_level)
    rows: list[dict[str, Any]] = []
    for candidate_id, candidate in candidates.iterrows():
        candidate_timestamp = pd.Timestamp(candidate["event_timestamp"])
        candidate_large_level = str(candidate["large_level"])
        structure_after = (
            int(max_wait_minutes)
            if structure_bsp_after_minutes is None
            else int(structure_bsp_after_minutes)
        )
        matching_structure_bsps = structure_bsp_events.loc[
            structure_bsp_events["event_timestamp"].between(
                candidate_timestamp - pd.Timedelta(
                    minutes=int(structure_bsp_before_minutes)
                ),
                candidate_timestamp + pd.Timedelta(minutes=structure_after),
            )
        ].copy()
        if matching_structure_bsps.empty and require_structure_level_bsp:
            rows.append({
                "candidate_id": candidate_id,
                "candidate_timestamp": candidate_timestamp,
                "large_level": candidate_large_level,
                "bsp_types": candidate["bsp_types"],
                "direction": direction,
                "candidate_price": candidate["price"],
                "status": f"no_{structure_level}_bsp_confirmation",
                "deadline": candidate_timestamp + pd.Timedelta(
                    minutes=int(max_wait_minutes)
                ),
                "confirmation_rule": "not_evaluated",
            })
            continue

        structure_bsp = None
        if not matching_structure_bsps.empty:
            after_candidate = matching_structure_bsps.loc[
                matching_structure_bsps["event_timestamp"].ge(candidate_timestamp)
            ]
            structure_bsp = (
                after_candidate.iloc[0]
                if not after_candidate.empty
                else matching_structure_bsps.iloc[-1]
            )

        evaluation_candidate = candidate.copy()
        if structure_bsp is not None:
            structure_bsp_timestamp = pd.Timestamp(
                structure_bsp["event_timestamp"]
            )
            # A later 30m BSP becomes the earliest causal time at which both
            # the 60m and 30m entry conditions are known.
            if structure_bsp_timestamp > candidate_timestamp:
                evaluation_candidate["event_timestamp"] = structure_bsp_timestamp
                evaluation_candidate["price"] = float(structure_bsp["price"])
        else:
            structure_bsp_timestamp = pd.NaT

        candidate_types = set(str(candidate["bsp_types"]).split("+"))
        if confirmation_mode == "generic_reversal":
            confirmation_rule = "trend_reversal"
        elif candidate_types.intersection({"1", "1p"}):
            confirmation_rule = "trend_reversal"
        elif candidate_types.intersection({"2", "2s"}):
            confirmation_rule = "higher_low"
        elif candidate_types.intersection({"3a", "3b"}):
            confirmation_rule = "center_retest"
        else:
            confirmation_rule = "trend_reversal"
        result = _evaluate_candidate(
            evaluation_candidate, structure, execution,
            max_wait_minutes=max_wait_minutes,
            invalidation_lookback_bars=invalidation_lookback_bars,
            break_buffer_atr=break_buffer_atr,
            pullback_tolerance_atr=pullback_tolerance_atr,
            require_execution_rebreak=require_execution_rebreak,
            enter_at_timeout_if_valid=enter_at_timeout_if_valid,
            confirmation_rule=confirmation_rule,
            maximum_break_extension_atr=extension_limits[confirmation_rule],
            maximum_break_delay_bars=delay_limits[confirmation_rule],
            maximum_timeout_extension_atr=maximum_timeout_extension_atr,
        )
        rows.append({
            "candidate_id": candidate_id,
            "candidate_timestamp": candidate["event_timestamp"],
            "large_level": candidate_large_level,
            "bsp_types": candidate["bsp_types"],
            "direction": direction,
            "candidate_price": candidate["price"],
            "structure_bsp_timestamp": structure_bsp_timestamp,
            "structure_bsp_types": (
                structure_bsp["bsp_types"] if structure_bsp is not None else np.nan
            ),
            "structure_bsp_price": (
                float(structure_bsp["price"])
                if structure_bsp is not None else np.nan
            ),
            **result,
        })
    summary = pd.DataFrame.from_records(rows)
    # Keep a stable signal schema even when every candidate is rejected before
    # an entry stage.  Downstream evaluation can then report zero confirmed
    # entries instead of treating optional entry fields as a malformed file.
    stable_entry_columns = {
        "entry_timestamp": pd.NaT,
        "entry_price": np.nan,
        "entry_reason": np.nan,
        "structure_break_timestamp": pd.NaT,
        "structure_break_price": np.nan,
        "pullback_confirmed_timestamp": pd.NaT,
        "pullback_price": np.nan,
        "invalidation_timestamp": pd.NaT,
    }
    for column, default in stable_entry_columns.items():
        if column not in summary:
            summary[column] = default
    invalidated_mask = (
        summary["status"].astype(str).str.startswith("invalidated")
        if not summary.empty else pd.Series(dtype=bool)
    )
    plot_summary = (
        summary if show_invalidated
        else summary.loc[~invalidated_mask].copy()
    )

    fig, (axis, state_axis) = plt.subplots(
        2, 1, figsize=(18, 10), sharex=True,
        gridspec_kw={"height_ratios": [4.2, 1.2]}, constrained_layout=True,
    )
    axis.plot(
        visible_prices["timestamp"], visible_prices["close"],
        color="#334155", linewidth=0.9, label="5m close",
    )
    stage_specs = (
        ("candidate_timestamp", "candidate_price", "^" if direction == "buy" else "v", "#F59E0B", "large BSP candidate"),
        ("structure_bsp_timestamp", "structure_bsp_price", "o", "#0891B2", f"{structure_level} BSP confirmation"),
        ("structure_break_timestamp", "structure_break_price", "D", "#2563EB", f"{structure_level} structure break"),
        ("pullback_confirmed_timestamp", "pullback_price", "s", "#7C3AED", f"{structure_level} pullback held"),
        ("invalidation_timestamp", "invalidation_price", "x", "#DC2626", "invalidated"),
    )
    for time_col, price_col, marker, color, label in stage_specs:
        if plot_summary.empty or time_col not in plot_summary or price_col not in plot_summary:
            continue
        part = plot_summary.dropna(subset=[time_col, price_col])
        part = part.loc[pd.to_datetime(part[time_col]).between(start_ts, end_ts)]
        if part.empty:
            continue
        axis.scatter(
            pd.to_datetime(part[time_col]), part[price_col], marker=marker,
            s=130 if marker == "*" else 75, color=color, label=label, zorder=5,
        )
    if not plot_summary.empty and {"entry_timestamp", "entry_price"}.issubset(plot_summary):
        reasons = plot_summary.get(
            "entry_reason", pd.Series("", index=plot_summary.index)
        ).fillna("")
        entry_styles = (
            ("execution_rebreak", "*", "#16A34A", f"{execution_level} entry confirmed"),
            ("pullback_confirmed", "*", "#16A34A", f"{structure_level} pullback entry"),
            ("higher_low_break", "*", "#16A34A", "type 2/2s higher-low break"),
            ("center_retest_break", "*", "#16A34A", "type 3 center-retest break"),
            ("delayed_retest_entry", "*", "#9333EA", "stale break + retest entry"),
            ("timeout_still_valid", "P", "#0D9488", "timeout-valid entry"),
        )
        for reason, marker, color, label in entry_styles:
            part = plot_summary.loc[reasons.eq(reason)].dropna(
                subset=["entry_timestamp", "entry_price"]
            )
            part = part.loc[
                pd.to_datetime(part["entry_timestamp"]).between(start_ts, end_ts)
            ]
            if part.empty:
                continue
            axis.scatter(
                pd.to_datetime(part["entry_timestamp"]), part["entry_price"],
                marker=marker, s=130, color=color, label=label, zorder=6,
            )
    if not plot_summary.empty and "break_is_fresh" in plot_summary:
        stale_breaks = plot_summary.loc[
            plot_summary["break_is_fresh"].eq(False)
        ].dropna(subset=["structure_break_timestamp", "structure_break_price"])
        stale_breaks = stale_breaks.loc[
            pd.to_datetime(stale_breaks["structure_break_timestamp"])
            .between(start_ts, end_ts)
        ]
        if not stale_breaks.empty:
            axis.scatter(
                pd.to_datetime(stale_breaks["structure_break_timestamp"]),
                stale_breaks["structure_break_price"], marker="D", s=95,
                facecolors="none", edgecolors="#F97316", linewidths=2.0,
                label="stale structure break", zorder=7,
            )
    if not plot_summary.empty:
        for row in plot_summary.head(max(0, int(annotate_max))).itertuples(index=False):
            axis.annotate(
                f"{row.large_level} {row.bsp_types} [{getattr(row, 'confirmation_rule', '')}]\n{row.status}",
                (row.candidate_timestamp, row.candidate_price), xytext=(5, 9),
                textcoords="offset points", fontsize=8,
            )
            if pd.notna(getattr(row, "reference_price", np.nan)):
                right = min(pd.Timestamp(row.deadline), end_ts)
                axis.hlines(
                    row.reference_price, row.candidate_timestamp, right,
                    colors="#2563EB", linestyles="dashed", linewidth=0.8, alpha=0.5,
                )
            if pd.notna(getattr(row, "invalidation_price", np.nan)):
                right = min(pd.Timestamp(row.deadline), end_ts)
                axis.hlines(
                    row.invalidation_price, row.candidate_timestamp, right,
                    colors="#DC2626", linestyles="dotted", linewidth=0.8, alpha=0.5,
                )

    stages = [
        ("candidate_timestamp", "candidate", "#F59E0B"),
        ("structure_bsp_timestamp", f"{structure_level} BSP", "#0891B2"),
        ("structure_break_timestamp", "structure break", "#2563EB"),
        ("pullback_confirmed_timestamp", "pullback held", "#7C3AED"),
        ("entry_timestamp", "entry", "#16A34A"),
    ]
    state_axis.set_yticks(range(len(stages)), [label for _, label, _ in stages])
    for y, (column, _, color) in enumerate(stages):
        if plot_summary.empty or column not in plot_summary:
            continue
        times = pd.to_datetime(plot_summary[column], errors="coerce").dropna()
        times = times.loc[times.between(start_ts, end_ts)]
        state_axis.scatter(times, np.full(len(times), y), s=45, color=color)
    state_axis.set_ylabel("Causal stage")
    state_axis.set_xlabel("Observable timestamp")
    state_axis.grid(axis="x", alpha=0.2)
    state_axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    axis.set_title(
        f"{'+'.join(candidate_levels)} {'/'.join(large_bsp_types)} {direction} reversal confirmation | "
        f"mode={confirmation_mode}, {structure_level} structure, {execution_level} execution | "
        f"timeout entry={'on' if enter_at_timeout_if_valid else 'off'} | "
        f"invalidated={'shown' if show_invalidated else 'hidden'}"
    )
    axis.set_ylabel("Price")
    axis.grid(alpha=0.18)
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(unique.values(), unique.keys(), ncol=3, fontsize=8, loc="best")

    output = Path(output_path) if output_path else (
        Path(multilevel_path).parent / "plots" /
        f"{'_'.join(candidate_levels)}_{direction}_reversal_{start_ts:%Y%m%d}_{end_ts:%Y%m%d}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    csv_path = Path(signals_csv) if signals_csv else output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(csv_path, index=False)
    status_counts = (
        summary["status"].value_counts(dropna=False).rename_axis("status")
        .reset_index(name="rows") if not summary.empty
        else pd.DataFrame(columns=["status", "rows"])
    )
    return {
        "plot_path": str(output.resolve()),
        "signals_csv": str(csv_path.resolve()),
        "candidate_rows": int(len(summary)),
        "plotted_candidate_rows": int(len(plot_summary)),
        "invalidated_rows": int(invalidated_mask.sum()) if not summary.empty else 0,
        "entry_confirmed_rows": int(summary["status"].astype(str).str.startswith("entry_confirmed").sum()) if not summary.empty else 0,
        "timeout_entry_rows": int(summary["status"].eq("entry_confirmed_timeout").sum()) if not summary.empty else 0,
        "stale_break_rows": int(summary.get("break_is_fresh", pd.Series(dtype=bool)).eq(False).sum()) if not summary.empty else 0,
        "timeout_overextended_rows": int(summary["status"].eq("timeout_valid_but_overextended").sum()) if not summary.empty else 0,
        "confirmation_mode": confirmation_mode,
        "large_levels": candidate_levels,
        "break_extension_limits_atr": extension_limits,
        "break_delay_limits_bars": delay_limits,
        "maximum_timeout_extension_atr": float(maximum_timeout_extension_atr),
        "require_structure_level_bsp": bool(require_structure_level_bsp),
        "structure_level_bsp_types": structure_level_bsp_types,
        "structure_bsp_before_minutes": int(structure_bsp_before_minutes),
        "structure_bsp_after_minutes": (
            int(max_wait_minutes)
            if structure_bsp_after_minutes is None
            else int(structure_bsp_after_minutes)
        ),
        "status_counts": status_counts,
        "summary": summary,
    }
