"""Evaluate and plot realized quality of multi-level reversal entries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import normalize_ohlcv
from .labels import (
    add_same_time_return_label,
    add_trading_day_extreme_labels,
)


# Keep these stable schema names local so importing this plotting module does
# not depend on which historical version of ``labels`` a notebook has cached.
TARGET_COLUMNS = ("target_max_gain_2d", "target_max_loss_2d")
TARGET_METADATA_COLUMNS = ("target_horizon_end_timestamp",)
EXACT_RETURN_TARGET = "target_exact_return"


def _entry_frame(source: dict[str, Any] | pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(source, dict):
        if "summary" not in source:
            raise KeyError("entry result dictionary has no 'summary' frame")
        frame = source["summary"].copy()
    elif isinstance(source, pd.DataFrame):
        frame = source.copy()
    else:
        frame = pd.read_csv(source, low_memory=False)
    required = {"status", "direction"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"entry signals are missing columns: {sorted(missing)}")
    if "entry_timestamp" not in frame:
        frame["entry_timestamp"] = pd.NaT
    if "entry_price" not in frame:
        frame["entry_price"] = np.nan
    frame["entry_timestamp"] = pd.to_datetime(frame["entry_timestamp"], errors="coerce")
    return frame.loc[
        frame["status"].astype(str).str.startswith("entry_confirmed")
        & frame["entry_timestamp"].notna()
        & pd.to_numeric(frame["entry_price"], errors="coerce").notna()
    ].copy()


def _first_hit_outcome(
    bars: pd.DataFrame,
    *,
    direction: str,
    entry_price: float,
    profit_target_pct: float,
    stop_loss_pct: float,
    same_bar_priority: str,
) -> dict[str, Any]:
    sign = 1.0 if direction == "buy" else -1.0
    target_price = entry_price * (
        1.0 + profit_target_pct if direction == "buy" else 1.0 - profit_target_pct
    )
    stop_price = entry_price * (
        1.0 - stop_loss_pct if direction == "buy" else 1.0 + stop_loss_pct
    )
    for row in bars.itertuples(index=False):
        target_hit = row.high >= target_price if direction == "buy" else row.low <= target_price
        stop_hit = row.low <= stop_price if direction == "buy" else row.high >= stop_price
        if not target_hit and not stop_hit:
            continue
        if target_hit and stop_hit:
            outcome = "target_first" if same_bar_priority == "target" else "stop_first"
        else:
            outcome = "target_first" if target_hit else "stop_first"
        exit_price = target_price if outcome == "target_first" else stop_price
        return {
            "outcome": outcome,
            "exit_timestamp": row.timestamp,
            "exit_price": exit_price,
            "realized_return": sign * (exit_price / entry_price - 1.0),
            "target_price": target_price,
            "stop_price": stop_price,
        }
    if bars.empty:
        return {
            "outcome": "no_horizon_data", "exit_timestamp": pd.NaT,
            "exit_price": np.nan, "realized_return": np.nan,
            "target_price": target_price, "stop_price": stop_price,
        }
    final = bars.iloc[-1]
    realized = sign * (float(final["close"]) / entry_price - 1.0)
    return {
        "outcome": "horizon_positive" if realized > 0 else "horizon_nonpositive",
        "exit_timestamp": final["timestamp"],
        "exit_price": float(final["close"]),
        "realized_return": realized,
        "target_price": target_price,
        "stop_price": stop_price,
    }


def plot_multilevel_entry_quality(
    entry_signals: dict[str, Any] | pd.DataFrame | str | Path,
    price_csv: str | Path,
    *,
    start: str | None = None,
    end: str | None = None,
    bsp_types: tuple[str, ...] | None = None,
    direction: str | None = None,
    entry_reasons: tuple[str, ...] | None = None,
    horizon_trading_days: int = 2,
    profit_target_pct: float = 0.03,
    stop_loss_pct: float = 0.015,
    minimum_favorable_excursion_pct: float = 0.02,
    maximum_adverse_excursion_pct: float = 0.015,
    same_bar_priority: str = "stop",
    annotate_max: int = 20,
    output_path: str | Path = "outputs/multilevel_entry_quality.png",
    evaluation_csv: str | Path | None = None,
    show: bool = True,
) -> dict[str, Any]:
    """Plot independent two-session quality diagnostics for confirmed entries."""
    if direction not in {None, "buy", "sell"}:
        raise ValueError("direction must be None, 'buy', or 'sell'")
    if same_bar_priority not in {"stop", "target"}:
        raise ValueError("same_bar_priority must be 'stop' or 'target'")
    if horizon_trading_days < 1:
        raise ValueError("horizon_trading_days must be positive")
    if min(
        profit_target_pct, stop_loss_pct,
        minimum_favorable_excursion_pct, maximum_adverse_excursion_pct,
    ) < 0:
        raise ValueError("return and risk thresholds cannot be negative")

    entries = _entry_frame(entry_signals)
    if start is not None:
        entries = entries.loc[entries["entry_timestamp"].ge(pd.Timestamp(start))]
    if end is not None:
        entries = entries.loc[entries["entry_timestamp"].le(pd.Timestamp(end))]
    if direction is not None:
        entries = entries.loc[entries["direction"].eq(direction)]
    if bsp_types is not None:
        selected = set(map(str, bsp_types))
        entries = entries.loc[
            entries["bsp_types"].astype(str).map(
                lambda value: bool(selected.intersection(value.split("+")))
            )
        ]
    if entry_reasons is not None:
        entries = entries.loc[entries["entry_reason"].isin(entry_reasons)]
    entries = entries.sort_values("entry_timestamp").reset_index(drop=True)
    if entries.empty:
        raise ValueError("No confirmed entries matched the requested filters")

    raw = normalize_ohlcv(pd.read_csv(price_csv)).sort_values("timestamp")
    # Raw timestamps are bar starts. Shift to the point at which each 5-minute
    # OHLC bar is fully observable so it aligns with reversal entry timestamps.
    evaluation_bars = raw.copy()
    evaluation_bars["timestamp"] += pd.Timedelta(minutes=5)
    labeled = add_trading_day_extreme_labels(
        evaluation_bars, int(horizon_trading_days)
    )
    labeled = add_same_time_return_label(labeled, int(horizon_trading_days))
    label_columns = [
        "timestamp", *TARGET_COLUMNS, EXACT_RETURN_TARGET,
        *TARGET_METADATA_COLUMNS,
    ]
    evaluated = entries.merge(
        labeled[label_columns], left_on="entry_timestamp", right_on="timestamp",
        how="left",
    ).drop(columns=["timestamp"])

    records: list[dict[str, Any]] = []
    for row in evaluated.itertuples(index=False):
        sign = 1.0 if row.direction == "buy" else -1.0
        raw_gain = getattr(row, TARGET_COLUMNS[0])
        raw_loss = getattr(row, TARGET_COLUMNS[1])
        favorable = raw_gain if row.direction == "buy" else -raw_loss
        adverse = raw_loss if row.direction == "buy" else -raw_gain
        horizon_return = sign * getattr(row, EXACT_RETURN_TARGET)
        horizon_end = getattr(row, TARGET_METADATA_COLUMNS[0])
        path = evaluation_bars.loc[
            evaluation_bars["timestamp"].gt(row.entry_timestamp)
            & evaluation_bars["timestamp"].le(horizon_end)
        ] if pd.notna(horizon_end) else evaluation_bars.iloc[:0]
        first_hit = _first_hit_outcome(
            path, direction=row.direction, entry_price=float(row.entry_price),
            profit_target_pct=float(profit_target_pct),
            stop_loss_pct=float(stop_loss_pct),
            same_bar_priority=same_bar_priority,
        )
        is_good = (
            pd.notna(favorable) and pd.notna(adverse)
            and favorable >= float(minimum_favorable_excursion_pct)
            and adverse >= -float(maximum_adverse_excursion_pct)
        )
        records.append({
            **row._asdict(),
            "favorable_excursion": favorable,
            "adverse_excursion": adverse,
            "directional_horizon_return": horizon_return,
            "is_good_entry": bool(is_good),
            **first_hit,
        })
    result = pd.DataFrame.from_records(records)

    view_start = pd.Timestamp(start) if start is not None else result["entry_timestamp"].min()
    view_end = pd.Timestamp(end) if end is not None else result["exit_timestamp"].max()
    if pd.isna(view_end):
        view_end = result["entry_timestamp"].max()
    visible_prices = evaluation_bars.loc[
        evaluation_bars["timestamp"].between(view_start, view_end)
    ]
    fig, (price_axis, excursion_axis) = plt.subplots(
        2, 1, figsize=(18, 11), sharex=True,
        gridspec_kw={"height_ratios": [3.5, 1.5]}, constrained_layout=True,
    )
    price_axis.plot(
        visible_prices["timestamp"], visible_prices["close"],
        color="#334155", linewidth=0.9, label="5m close",
    )
    outcome_styles = {
        "target_first": ("#16A34A", "target first"),
        "stop_first": ("#DC2626", "stop first"),
        "horizon_positive": ("#2563EB", "horizon positive"),
        "horizon_nonpositive": ("#F97316", "horizon nonpositive"),
        "no_horizon_data": ("#64748B", "no horizon data"),
    }
    for outcome, (color, label) in outcome_styles.items():
        part = result.loc[result["outcome"].eq(outcome)]
        if part.empty:
            continue
        for side, marker in (("buy", "^"), ("sell", "v")):
            side_part = part.loc[part["direction"].eq(side)]
            if side_part.empty:
                continue
            price_axis.scatter(
                side_part["entry_timestamp"], side_part["entry_price"], marker=marker,
                s=95, color=color, label=f"{label} {side}", zorder=5,
            )
        for trade in part.itertuples(index=False):
            if pd.notna(trade.exit_timestamp) and pd.notna(trade.exit_price):
                price_axis.plot(
                    [trade.entry_timestamp, trade.exit_timestamp],
                    [trade.entry_price, trade.exit_price],
                    color=color, linewidth=1.0, alpha=0.45,
                )
    for trade in result.head(max(0, int(annotate_max))).itertuples(index=False):
        price_axis.annotate(
            f"{trade.bsp_types} {trade.entry_reason}\n"
            f"MFE {trade.favorable_excursion:.1%} / MAE {trade.adverse_excursion:.1%}",
            (trade.entry_timestamp, trade.entry_price), xytext=(5, 8),
            textcoords="offset points", fontsize=8,
        )

    excursion_axis.scatter(
        result["entry_timestamp"], result["favorable_excursion"],
        marker="^", s=45, color="#16A34A", label="MFE",
    )
    excursion_axis.scatter(
        result["entry_timestamp"], result["adverse_excursion"],
        marker="v", s=45, color="#DC2626", label="MAE",
    )
    excursion_axis.scatter(
        result["entry_timestamp"], result["realized_return"],
        marker="o", s=28, color="#2563EB", label="realized",
    )
    excursion_axis.axhline(
        minimum_favorable_excursion_pct, color="#16A34A",
        linestyle="dashed", linewidth=0.9, label="good-entry MFE threshold",
    )
    excursion_axis.axhline(
        -maximum_adverse_excursion_pct, color="#DC2626",
        linestyle="dashed", linewidth=0.9, label="good-entry MAE threshold",
    )
    excursion_axis.axhline(0.0, color="#64748B", linewidth=0.8)
    excursion_axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    excursion_axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    excursion_axis.set_ylabel("Directional return")
    excursion_axis.set_xlabel("Entry timestamp")
    excursion_axis.grid(alpha=0.18)
    excursion_axis.legend(ncol=3, fontsize=8, loc="best")
    price_axis.set_title(
        f"Entry quality | {horizon_trading_days} trading days | "
        f"target {profit_target_pct:.1%}, stop {stop_loss_pct:.1%}"
    )
    price_axis.set_ylabel("Price")
    price_axis.grid(alpha=0.18)
    handles, labels = price_axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    price_axis.legend(unique.values(), unique.keys(), ncol=3, fontsize=8, loc="best")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    csv_path = Path(evaluation_csv) if evaluation_csv else output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(csv_path, index=False)
    valid = result["realized_return"].notna()
    return {
        "plot_path": str(output.resolve()),
        "evaluation_csv": str(csv_path.resolve()),
        "entry_rows": int(len(result)),
        "good_entry_rate": float(result["is_good_entry"].mean()),
        "positive_realized_rate": float(result.loc[valid, "realized_return"].gt(0).mean()) if valid.any() else np.nan,
        "target_first_rate": float(result["outcome"].eq("target_first").mean()),
        "stop_first_rate": float(result["outcome"].eq("stop_first").mean()),
        "mean_favorable_excursion": float(result["favorable_excursion"].mean()),
        "mean_adverse_excursion": float(result["adverse_excursion"].mean()),
        "mean_realized_return": float(result["realized_return"].mean()),
        "outcome_counts": result["outcome"].value_counts().rename_axis("outcome").reset_index(name="rows"),
        "entries": result,
    }
