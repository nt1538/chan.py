"""Plots for inspecting confirmation of large-level BSPs by smaller levels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import normalize_ohlcv


_BSP_TYPES = ("1", "1p", "2", "2s", "3a", "3b")


def _level_bsp_events(
    frame: pd.DataFrame,
    level: str,
    selected_types: tuple[str, ...] | None,
) -> pd.DataFrame:
    available_column = f"mlchan_{level}_available_timestamp"
    buy_column = f"mlchan_{level}_latest_new_bsp_is_buy"
    sell_column = f"mlchan_{level}_latest_new_bsp_is_sell"
    required = {available_column, buy_column, sell_column}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Level {level!r} is missing BSP fields: {sorted(missing)}")
    type_columns = {
        kind: f"mlchan_{level}_latest_new_bsp_type_{kind}"
        for kind in _BSP_TYPES
        if f"mlchan_{level}_latest_new_bsp_type_{kind}" in frame
    }
    chosen = tuple(str(value).lower() for value in selected_types) if selected_types else tuple(type_columns)
    unavailable = sorted(set(chosen).difference(type_columns))
    if unavailable:
        raise ValueError(f"Level {level!r} has no BSP types: {unavailable}")

    columns = [available_column, buy_column, sell_column, *type_columns.values()]
    grouped = (
        frame[columns]
        .dropna(subset=[available_column])
        .groupby(available_column, as_index=False)
        .max(numeric_only=True)
    )
    grouped["event_timestamp"] = pd.to_datetime(grouped[available_column])
    selected_mask = grouped[[type_columns[kind] for kind in chosen]].fillna(0).gt(0).any(axis=1)
    records: list[dict[str, Any]] = []
    for row in grouped.loc[selected_mask].itertuples(index=False):
        values = row._asdict()
        kinds = [
            kind for kind in chosen
            if float(values.get(type_columns[kind], 0.0) or 0.0) > 0
        ]
        for direction, column in (("buy", buy_column), ("sell", sell_column)):
            if float(values.get(column, 0.0) or 0.0) <= 0:
                continue
            records.append({
                "event_timestamp": values["event_timestamp"],
                "level": level,
                "direction": direction,
                "bsp_types": "+".join(kinds),
            })
    return pd.DataFrame.from_records(
        records, columns=["event_timestamp", "level", "direction", "bsp_types"]
    ).sort_values("event_timestamp")


def _attach_event_prices(events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.assign(price=np.nan)
    reference = prices[["timestamp", "close"]].sort_values("timestamp")
    return pd.merge_asof(
        events.sort_values("event_timestamp"), reference,
        left_on="event_timestamp", right_on="timestamp",
        direction="backward", allow_exact_matches=True,
    ).rename(columns={"close": "price"}).drop(columns=["timestamp"])


def plot_selected_multilevel_bsp(
    multilevel_path: str | Path,
    price_csv: str | Path,
    *,
    start: str,
    end: str,
    bsp_selection: dict[str, tuple[str, ...]],
    directions: tuple[str, ...] = ("buy", "sell"),
    annotate_max: int = 30,
    output_path: str | Path | None = None,
    signals_csv: str | Path | None = None,
    show: bool = True,
) -> dict[str, Any]:
    """Plot only explicitly selected BSP levels, types, and directions.

    ``bsp_selection`` maps each timeframe to the BSP types to retain, for
    example ``{"60m": ("1",)}``. Events are placed at their point-in-time
    observable timestamp and priced using the latest 5-minute close available
    at or before that timestamp.
    """
    if not bsp_selection:
        raise ValueError("bsp_selection cannot be empty")
    normalized_directions = tuple(str(value).lower() for value in directions)
    invalid_directions = sorted(set(normalized_directions).difference({"buy", "sell"}))
    if invalid_directions or not normalized_directions:
        raise ValueError("directions must contain 'buy', 'sell', or both")

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if end_ts <= start_ts:
        raise ValueError("end must be after start")

    states = pd.read_csv(multilevel_path, low_memory=False)
    if "timestamp" not in states:
        raise KeyError("multilevel data has no 'timestamp' column")
    states["timestamp"] = pd.to_datetime(states["timestamp"], errors="coerce")
    states = states.loc[states["timestamp"].between(start_ts, end_ts)].copy()

    prices = normalize_ohlcv(pd.read_csv(price_csv))
    prices = prices.loc[prices["timestamp"].between(start_ts, end_ts)].copy()
    if prices.empty:
        raise ValueError("No price rows matched the requested plot period")

    selected_frames: list[pd.DataFrame] = []
    for level, bsp_types in bsp_selection.items():
        events = _level_bsp_events(states, str(level), tuple(bsp_types))
        events = events.loc[
            events["event_timestamp"].between(start_ts, end_ts)
            & events["direction"].isin(normalized_directions)
        ].copy()
        if not events.empty:
            selected_frames.append(_attach_event_prices(events, prices))

    signals = (
        pd.concat(selected_frames, ignore_index=True)
        if selected_frames
        else pd.DataFrame(
            columns=["event_timestamp", "level", "direction", "bsp_types", "price"]
        )
    )
    signals = signals.sort_values(["event_timestamp", "level", "direction"])

    fig, (price_axis, lane_axis) = plt.subplots(
        2, 1, figsize=(17, 9), sharex=True,
        gridspec_kw={"height_ratios": [4.0, 1.0]}, constrained_layout=True,
    )
    price_axis.plot(
        prices["timestamp"], prices["close"],
        color="#334155", linewidth=0.9, label="5m close",
    )
    levels = [str(level) for level in bsp_selection]
    level_colors = {
        level: plt.get_cmap("tab10")(index % 10)
        for index, level in enumerate(levels)
    }
    markers = {"buy": "^", "sell": "v"}
    for level in levels:
        level_signals = signals.loc[signals["level"].eq(level)]
        for side in normalized_directions:
            part = level_signals.loc[level_signals["direction"].eq(side)]
            if part.empty:
                continue
            price_axis.scatter(
                part["event_timestamp"], part["price"],
                marker=markers[side], s=100,
                facecolors=level_colors[level] if side == "buy" else "none",
                edgecolors=level_colors[level], linewidths=1.8,
                label=f"{level} {side}", zorder=5,
            )

    for row in signals.head(max(0, int(annotate_max))).itertuples(index=False):
        offset = (5, 9) if row.direction == "buy" else (5, -15)
        price_axis.annotate(
            f"{row.level} {row.bsp_types} {row.direction}",
            (row.event_timestamp, row.price), xytext=offset,
            textcoords="offset points", fontsize=8,
        )

    lane_positions = {level: index for index, level in enumerate(levels)}
    for level in levels:
        for side in normalized_directions:
            part = signals.loc[
                signals["level"].eq(level) & signals["direction"].eq(side)
            ]
            if part.empty:
                continue
            lane_axis.scatter(
                part["event_timestamp"],
                np.full(len(part), lane_positions[level]),
                marker=markers[side], s=55,
                facecolors=level_colors[level] if side == "buy" else "none",
                edgecolors=level_colors[level], linewidths=1.5,
            )
    lane_axis.set_yticks(range(len(levels)), levels)
    lane_axis.set_ylabel("BSP level")
    lane_axis.set_xlabel("Observable timestamp")
    lane_axis.grid(axis="x", alpha=0.2)
    lane_axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))

    selection_text = ", ".join(
        f"{level} {'/'.join(map(str, bsp_types))}"
        for level, bsp_types in bsp_selection.items()
    )
    price_axis.set_title(
        f"Selected BSP signals | {selection_text} | {'/'.join(normalized_directions)}"
    )
    price_axis.set_ylabel("Price")
    price_axis.grid(alpha=0.18)
    handles, labels = price_axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    price_axis.legend(unique.values(), unique.keys(), ncol=3, fontsize=8, loc="best")

    output = Path(output_path) if output_path else (
        Path(multilevel_path).parent / "plots" /
        f"selected_bsp_{start_ts:%Y%m%d}_{end_ts:%Y%m%d}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    csv_path = Path(signals_csv) if signals_csv else output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(csv_path, index=False)
    counts = (
        signals.groupby(["level", "direction", "bsp_types"], dropna=False)
        .size().rename("rows").reset_index()
    )
    return {
        "plot_path": str(output.resolve()),
        "signals_csv": str(csv_path.resolve()),
        "signal_rows": int(len(signals)),
        "counts": counts,
        "signals": signals,
    }


def plot_multilevel_bsp_confirmation(
    multilevel_path: str | Path,
    price_csv: str | Path,
    *,
    start: str,
    end: str,
    large_level: str = "60m",
    small_levels: tuple[str, ...] = ("5m", "15m", "30m"),
    large_bsp_types: tuple[str, ...] | None = None,
    small_bsp_types: tuple[str, ...] | None = None,
    direction: str | None = None,
    confirmation_before_minutes: int = 120,
    confirmation_after_minutes: int = 390,
    minimum_confirming_levels: int = 1,
    require_type_overlap: bool = False,
    annotate_max: int = 30,
    output_path: str | Path | None = None,
    confirmation_csv: str | Path | None = None,
    show: bool = True,
) -> dict[str, Any]:
    """Plot whether large-level BSP events receive same-direction lower-level confirmation.

    A lower-level confirmation is the nearest same-direction small-level BSP in
    ``[large_time-before, large_time+after]``. Matching is performed separately
    for each small level; a large BSP is confirmed when at least
    ``minimum_confirming_levels`` distinct levels match.
    """
    if direction not in (None, "buy", "sell"):
        raise ValueError("direction must be None, 'buy', or 'sell'")
    if int(confirmation_before_minutes) < 0 or int(confirmation_after_minutes) < 0:
        raise ValueError("confirmation windows cannot be negative")
    if not 1 <= int(minimum_confirming_levels) <= len(small_levels):
        raise ValueError("minimum_confirming_levels must fit within small_levels")
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if end_ts <= start_ts:
        raise ValueError("end must be after start")

    states = pd.read_csv(multilevel_path, low_memory=False)
    states["timestamp"] = pd.to_datetime(states["timestamp"], errors="coerce")
    # Include the matching margins so confirmations just outside the visible
    # large-event interval can still be found.
    margin_start = start_ts - pd.Timedelta(minutes=int(confirmation_before_minutes))
    margin_end = end_ts + pd.Timedelta(minutes=int(confirmation_after_minutes))
    states = states.loc[states["timestamp"].between(margin_start, margin_end)].copy()
    prices = normalize_ohlcv(pd.read_csv(price_csv))
    prices = prices.loc[prices["timestamp"].between(margin_start, margin_end)].copy()
    visible_prices = prices.loc[prices["timestamp"].between(start_ts, end_ts)].copy()
    if visible_prices.empty:
        raise ValueError("No price rows matched the requested plot period")

    large = _attach_event_prices(
        _level_bsp_events(states, large_level, large_bsp_types), prices
    )
    large = large.loc[large["event_timestamp"].between(start_ts, end_ts)].copy()
    if direction:
        large = large.loc[large["direction"].eq(direction)]
    small_by_level: dict[str, pd.DataFrame] = {}
    for level in small_levels:
        events = _attach_event_prices(
            _level_bsp_events(states, level, small_bsp_types), prices
        )
        if direction:
            events = events.loc[events["direction"].eq(direction)]
        small_by_level[level] = events

    match_rows: list[dict[str, Any]] = []
    large_summary: list[dict[str, Any]] = []
    before = pd.Timedelta(minutes=int(confirmation_before_minutes))
    after = pd.Timedelta(minutes=int(confirmation_after_minutes))
    for large_id, event in enumerate(large.itertuples(index=False)):
        matched_levels: list[str] = []
        for level, candidates in small_by_level.items():
            eligible = candidates.loc[
                candidates["direction"].eq(event.direction)
                & candidates["event_timestamp"].between(
                    event.event_timestamp - before, event.event_timestamp + after
                )
            ].copy()
            if require_type_overlap and not eligible.empty:
                large_types = set(str(event.bsp_types).split("+"))
                eligible = eligible.loc[
                    eligible["bsp_types"].map(
                        lambda value: bool(
                            large_types.intersection(str(value).split("+"))
                        )
                    )
                ]
            if eligible.empty:
                continue
            eligible["absolute_gap"] = (
                eligible["event_timestamp"] - event.event_timestamp
            ).abs()
            match = eligible.sort_values(["absolute_gap", "event_timestamp"]).iloc[0]
            gap_minutes = (
                match["event_timestamp"] - event.event_timestamp
            ).total_seconds() / 60.0
            matched_levels.append(level)
            match_rows.append({
                "large_id": large_id,
                "large_timestamp": event.event_timestamp,
                "large_level": large_level,
                "large_direction": event.direction,
                "large_bsp_types": event.bsp_types,
                "large_price": event.price,
                "small_timestamp": match["event_timestamp"],
                "small_level": level,
                "small_direction": match["direction"],
                "small_bsp_types": match["bsp_types"],
                "small_price": match["price"],
                "gap_minutes": gap_minutes,
            })
        large_summary.append({
            "large_id": large_id,
            "large_timestamp": event.event_timestamp,
            "large_level": large_level,
            "direction": event.direction,
            "bsp_types": event.bsp_types,
            "price": event.price,
            "confirming_level_count": len(matched_levels),
            "confirming_levels": "+".join(matched_levels),
            "is_confirmed": len(matched_levels) >= int(minimum_confirming_levels),
        })
    summary = pd.DataFrame.from_records(large_summary)
    matches = pd.DataFrame.from_records(match_rows)

    fig, (price_axis, lane_axis) = plt.subplots(
        2, 1, figsize=(17, 11), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.15]}, constrained_layout=True,
    )
    price_axis.plot(
        visible_prices["timestamp"], visible_prices["close"],
        color="#334155", linewidth=0.9, label="5m close",
    )
    level_colors = {
        level: plt.get_cmap("tab10")(index % 10)
        for index, level in enumerate((large_level, *small_levels))
    }
    direction_markers = {"buy": "^", "sell": "v"}

    # Small events are shown lightly; matched events are emphasized by links.
    for level, events in small_by_level.items():
        shown_events = events.loc[events["event_timestamp"].between(start_ts, end_ts)]
        for side in ("buy", "sell"):
            part = shown_events.loc[shown_events["direction"].eq(side)]
            if part.empty:
                continue
            price_axis.scatter(
                part["event_timestamp"], part["price"],
                marker=direction_markers[side], s=22,
                color=level_colors[level], alpha=0.35,
                label=f"{level} {side}",
            )

    if not matches.empty:
        for match in matches.itertuples(index=False):
            price_axis.plot(
                [match.large_timestamp, match.small_timestamp],
                [match.large_price, match.small_price],
                color=level_colors[match.small_level], alpha=0.45, linewidth=1.0,
            )

    if not summary.empty:
        for confirmed, edge, label in (
            (True, "#16A34A", f"{large_level} confirmed"),
            (False, "#DC2626", f"{large_level} unconfirmed"),
        ):
            part = summary.loc[summary["is_confirmed"].eq(confirmed)]
            if part.empty:
                continue
            for side in ("buy", "sell"):
                side_part = part.loc[part["direction"].eq(side)]
                if side_part.empty:
                    continue
                price_axis.scatter(
                    side_part["large_timestamp"], side_part["price"],
                    marker=direction_markers[side], s=115,
                    facecolors=level_colors[large_level] if confirmed else "none",
                    edgecolors=edge, linewidths=1.8,
                    label=f"{label} {side}", zorder=5,
                )
        for row in summary.head(max(0, int(annotate_max))).itertuples(index=False):
            status = "✓" if row.is_confirmed else "×"
            price_axis.annotate(
                f"{large_level} {row.bsp_types} {status}{row.confirming_level_count}",
                (row.large_timestamp, row.price), xytext=(4, 8),
                textcoords="offset points", fontsize=8,
            )

    lane_levels = [*small_levels, large_level]
    lane_positions = {level: index for index, level in enumerate(lane_levels)}
    for level in lane_levels:
        events = large if level == large_level else small_by_level[level]
        events = events.loc[events["event_timestamp"].between(start_ts, end_ts)]
        if events.empty:
            continue
        y = np.full(len(events), lane_positions[level])
        colors = ["#16A34A" if value == "buy" else "#DC2626" for value in events["direction"]]
        markers = [direction_markers[value] for value in events["direction"]]
        for marker in set(markers):
            mask = np.asarray(markers) == marker
            lane_axis.scatter(
                events.loc[mask, "event_timestamp"], y[mask],
                marker=marker, c=np.asarray(colors)[mask], s=35 if level != large_level else 75,
                alpha=0.75,
            )
    lane_axis.set_yticks(range(len(lane_levels)), lane_levels)
    lane_axis.set_ylabel("BSP level")
    lane_axis.set_xlabel("Observable timestamp")
    lane_axis.grid(axis="x", alpha=0.2)
    lane_axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    price_axis.set(
        title=(
            f"{large_level} BSP confirmation by {', '.join(small_levels)} | "
            f"window -{confirmation_before_minutes}/+{confirmation_after_minutes} min | "
            f"type overlap={'required' if require_type_overlap else 'not required'}"
        ),
        ylabel="Price",
    )
    handles, labels = price_axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    price_axis.legend(unique.values(), unique.keys(), ncol=3, fontsize=8, loc="best")
    price_axis.grid(alpha=0.18)

    output = Path(output_path) if output_path else (
        Path(multilevel_path).parent / "plots" /
        f"{large_level}_bsp_confirmation_{start_ts:%Y%m%d}_{end_ts:%Y%m%d}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    csv_path = Path(confirmation_csv) if confirmation_csv else output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(csv_path, index=False)
    return {
        "plot_path": str(output.resolve()),
        "confirmation_csv": str(csv_path.resolve()),
        "large_events": int(len(summary)),
        "confirmed_events": int(summary["is_confirmed"].sum()) if not summary.empty else 0,
        "confirmation_rate": float(summary["is_confirmed"].mean()) if not summary.empty else np.nan,
        "summary": summary,
        "matches": matches,
    }
