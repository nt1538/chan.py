"""Create trading-oriented maturity labels from a BSP candidate dataset.

The labels in this module are strategy labels, not Chan structural definitions:

* A BSP discovered after its historical BSP timestamp is treated as mature.
* A lower subsequent same-type buy supersedes the previous buy.
* A higher subsequent same-type sell supersedes the previous sell.

Supersession is evaluated only within the same BSP type, direction, and
structural level (Bi BSP versus segment BSP).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LEAKAGE_COLUMNS = {
    "trade_maturity_target",
    "trade_maturity_status",
    "trade_label_reason",
    "trade_label_resolution_timestamp",
    "is_superseded",
    "invalidated_by_lower_buy",
    "invalidated_by_higher_sell",
    "next_candidate_id",
    "next_bsp_timestamp",
    "next_snapshot_timestamp",
    "next_bsp_low",
    "next_bsp_high",
    "was_unresolved_when_next_appeared",
    "structural_maturity_target",
    "eventual_is_mature",
    "is_mature",
    "is_invalidated",
    "is_censored",
    "current_status",
    "original_current_status",
    "resolution_timestamp",
    "bars_to_resolution",
    "maturity_reason",
    "invalidation_reason",
}


DEFAULT_BSP_TECHNICAL_WINDOWS = (20, 39, 78, 156, 200, 390)
DEFAULT_BSP_RSI_PERIODS = (14, 78, 156)
DEFAULT_BSP_ATR_PERIODS = (14, 78, 156)
DEFAULT_BSP_MACD_PERIODS = ((12, 26, 9), (39, 78, 20), (78, 156, 39))
LEGACY_EXPANDED_BSP_TECHNICAL_WINDOWS = (
    2, 3, 5, 10, 12, 20, 26, 39, 50, 78, 100, 156, 200, 234, 390, 780,
)
LEGACY_EXPANDED_BSP_RSI_PERIODS = (14, 39, 78, 156, 390)
LEGACY_EXPANDED_BSP_ATR_PERIODS = (14, 39, 78, 156, 390)
LEGACY_EXPANDED_BSP_MACD_PERIODS = (
    (12, 26, 9), (39, 78, 20), (78, 156, 39), (156, 390, 78),
)


def _read_bsp_points(path: Path, sheet_name: str) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported BSP dataset format: {path.suffix!r}")


def add_price_maturity_progress(
    bspoints: pd.DataFrame,
    price_csv: str | Path,
    *,
    reference_lookback_bars: int = 390,
    minimum_structure_pct: float = 0.002,
) -> pd.DataFrame:
    """Add causal price-recovery progress for each BSP candidate.

    Buy progress measures recovery from the candidate low toward the highest
    price in the lookback ending at the BSP bar. Sell progress is symmetric.
    Market price is observed at ``snapshot_timestamp``; no later data is used.
    The result is structural progress in [0, 1], not a probability.
    """
    if int(reference_lookback_bars) < 2:
        raise ValueError("reference_lookback_bars must be at least 2")
    if float(minimum_structure_pct) < 0:
        raise ValueError("minimum_structure_pct cannot be negative")
    frame = bspoints.copy()
    required = {"bsp_timestamp", "snapshot_timestamp", "direction", "klu_low", "klu_high"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"BSP dataset is missing price-progress columns: {sorted(missing)}")

    price = pd.read_csv(price_csv, low_memory=False)
    normalized = {str(column).strip().lower(): column for column in price.columns}
    timestamp_column = next(
        (normalized[name] for name in ("timestamp", "datetime", "date", "time") if name in normalized),
        price.columns[0],
    )

    def price_column(*names: str) -> str:
        column = next((normalized[name] for name in names if name in normalized), None)
        if column is None:
            raise KeyError(f"Price CSV is missing one of these columns: {names}")
        return column

    high_column = price_column("high", "_high", "klu_high")
    low_column = price_column("low", "_low", "klu_low")
    close_column = price_column("close", "_close", "klu_close")
    price[timestamp_column] = pd.to_datetime(price[timestamp_column], errors="coerce")
    for column in (high_column, low_column, close_column):
        price[column] = pd.to_numeric(price[column], errors="coerce")
    price = (
        price.dropna(subset=[timestamp_column, high_column, low_column, close_column])
        .sort_values(timestamp_column).drop_duplicates(timestamp_column, keep="last")
        .reset_index(drop=True)
    )
    if price.empty:
        raise ValueError("Price CSV has no usable OHLC rows")

    frame["bsp_timestamp"] = pd.to_datetime(frame["bsp_timestamp"], errors="coerce")
    frame["snapshot_timestamp"] = pd.to_datetime(frame["snapshot_timestamp"], errors="coerce")
    direction = frame["direction"].astype(str).str.lower().str.strip()
    candidate_low = pd.to_numeric(frame["klu_low"], errors="coerce").to_numpy(float)
    candidate_high = pd.to_numeric(frame["klu_high"], errors="coerce").to_numpy(float)
    timestamps = price[timestamp_column].to_numpy(dtype="datetime64[ns]")
    highs = price[high_column].to_numpy(float)
    lows = price[low_column].to_numpy(float)
    closes = price[close_column].to_numpy(float)
    bsp_indices = np.searchsorted(
        timestamps, frame["bsp_timestamp"].to_numpy(dtype="datetime64[ns]"), side="right"
    ) - 1
    snapshot_indices = np.searchsorted(
        timestamps, frame["snapshot_timestamp"].to_numpy(dtype="datetime64[ns]"), side="right"
    ) - 1

    reference = np.full(len(frame), np.nan)
    market = np.full(len(frame), np.nan)
    raw = np.full(len(frame), np.nan)
    lookback = int(reference_lookback_bars)
    min_pct = float(minimum_structure_pct)
    for position in range(len(frame)):
        bsp_index = int(bsp_indices[position])
        snapshot_index = int(snapshot_indices[position])
        if bsp_index < 0 or snapshot_index < bsp_index or snapshot_index >= len(price):
            continue
        start_index = max(0, bsp_index - lookback + 1)
        market[position] = closes[snapshot_index]
        if direction.iloc[position] == "buy" and np.isfinite(candidate_low[position]):
            reference[position] = np.nanmax(highs[start_index:bsp_index + 1])
            structure = reference[position] - candidate_low[position]
            if structure > abs(candidate_low[position]) * min_pct:
                raw[position] = (market[position] - candidate_low[position]) / structure
        elif direction.iloc[position] == "sell" and np.isfinite(candidate_high[position]):
            reference[position] = np.nanmin(lows[start_index:bsp_index + 1])
            structure = candidate_high[position] - reference[position]
            if structure > abs(candidate_high[position]) * min_pct:
                raw[position] = (candidate_high[position] - market[position]) / structure

    frame["price_maturity_market_price"] = market
    frame["price_maturity_reference_price"] = reference
    frame["price_maturity_progress_raw"] = raw
    frame["price_maturity_progress"] = np.clip(raw, 0.0, 1.0)
    return frame


def add_multihorizon_technical_features_to_bspoints(
    bspoints: pd.DataFrame,
    price_csv: str | Path,
    *,
    windows: tuple[int, ...] = DEFAULT_BSP_TECHNICAL_WINDOWS,
    rsi_periods: tuple[int, ...] = DEFAULT_BSP_RSI_PERIODS,
    atr_periods: tuple[int, ...] = DEFAULT_BSP_ATR_PERIODS,
    macd_periods: tuple[tuple[int, int, int], ...] = DEFAULT_BSP_MACD_PERIODS,
) -> pd.DataFrame:
    """Attach causal multi-horizon kline features at each BSP snapshot."""
    from kline_2day_forecaster.features import add_technical_features, normalize_ohlcv

    frame = bspoints.copy()
    if "snapshot_timestamp" not in frame.columns:
        raise KeyError("BSP dataset is missing snapshot_timestamp")
    price = normalize_ohlcv(pd.read_csv(price_csv, low_memory=False))
    technical = add_technical_features(
        price,
        windows=windows,
        rsi_periods=rsi_periods,
        atr_periods=atr_periods,
        macd_periods=macd_periods,
    )
    # Raw cumulative/absolute fields are deliberately omitted. Every attached
    # feature is computed with information available no later than the BSP's
    # observable snapshot timestamp.
    excluded = {"tech_obv", "tech_vwap_session", "tech_volume_log1p"}
    feature_columns = [
        column for column in technical.columns
        if str(column).startswith("tech_") and column not in excluded
    ]
    lookup = (
        technical[["timestamp", *feature_columns]]
        .sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        .set_index("timestamp")
    )
    snapshot = pd.to_datetime(frame["snapshot_timestamp"], errors="coerce")
    aligned = lookup.reindex(pd.DatetimeIndex(snapshot))
    aligned.index = frame.index
    # New calculations intentionally replace same-named stale columns.
    frame = frame.drop(columns=[c for c in feature_columns if c in frame.columns])
    return pd.concat([frame, aligned[feature_columns]], axis=1)


def label_bspoints_for_training(
    bspoint_path: str | Path,
    *,
    output_csv: str | Path = "outputs/bsp_trade_labels.csv",
    training_output_csv: str | Path | None = None,
    sheet_name: str = "BSP Points",
    include_sell_supersession: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Label BSP candidates and save full and model-ready CSV datasets.

    Parameters
    ----------
    bspoint_path:
        Excel workbook containing ``sheet_name``, or the equivalent CSV.
    output_csv:
        Destination containing every candidate, including censored rows.
    training_output_csv:
        Destination containing only binary labeled rows. When omitted,
        ``_training`` is appended to ``output_csv``.
    sheet_name:
        Excel worksheet containing candidate BSP records.
    include_sell_supersession:
        Apply the symmetric rule that a higher same-type sell supersedes the
        previous sell.

    Returns
    -------
    dict
        Paths, row counts, label counts, and a summary DataFrame.
    """
    source = Path(bspoint_path)
    if not source.exists():
        raise FileNotFoundError(f"BSP dataset does not exist: {source}")

    frame = _read_bsp_points(source, sheet_name).copy()
    required = {
        "candidate_id",
        "bsp_type",
        "direction",
        "is_segbsp",
        "klu_idx",
        "klu_low",
        "klu_high",
        "bsp_timestamp",
        "snapshot_timestamp",
        "snapshot_first_seen",
        "eventual_is_mature",
        "current_status",
        "resolution_timestamp",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"BSP dataset is missing columns: {sorted(missing)}")

    for column in (
        "timestamp",
        "bsp_timestamp",
        "snapshot_timestamp",
        "first_seen_timestamp",
        "resolution_timestamp",
    ):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")

    frame["bsp_type"] = (
        frame["bsp_type"].astype(str).str.lower().str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    frame["direction"] = frame["direction"].astype(str).str.lower().str.strip()
    frame["is_segbsp"] = (
        frame["is_segbsp"].astype(str).str.lower()
        .map({"true": True, "1": True, "false": False, "0": False})
        .fillna(False).astype(bool)
    )
    for column in (
        "klu_idx", "klu_low", "klu_high", "klu_close",
        "snapshot_first_seen", "eventual_is_mature", "is_mature",
        "is_censored",
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["discovery_delay_bars"] = frame["snapshot_first_seen"] - frame["klu_idx"]
    frame["discovery_delay_minutes"] = (
        frame["snapshot_timestamp"] - frame["bsp_timestamp"]
    ).dt.total_seconds() / 60.0
    frame["is_delayed_bsp"] = (
        frame["snapshot_timestamp"].notna()
        & frame["bsp_timestamp"].notna()
        & frame["snapshot_timestamp"].gt(frame["bsp_timestamp"])
    )

    frame["structural_maturity_target"] = pd.to_numeric(
        frame["eventual_is_mature"], errors="coerce"
    )
    frame["original_current_status"] = frame["current_status"]
    frame["trade_maturity_target"] = frame["structural_maturity_target"]
    frame.loc[frame["is_delayed_bsp"], "trade_maturity_target"] = 1.0
    frame["trade_label_reason"] = np.select(
        [
            frame["is_delayed_bsp"],
            frame["structural_maturity_target"].eq(1),
            frame["structural_maturity_target"].eq(0),
        ],
        [
            "delayed_bsp_treated_as_mature",
            "source_line_became_sure",
            "structurally_invalidated",
        ],
        default="unresolved_or_censored",
    )
    if "resolution_timestamp" in frame:
        frame["trade_label_resolution_timestamp"] = frame["resolution_timestamp"]
    else:
        frame["trade_label_resolution_timestamp"] = pd.NaT
    frame.loc[
        frame["is_delayed_bsp"], "trade_label_resolution_timestamp"
    ] = frame.loc[frame["is_delayed_bsp"], "snapshot_timestamp"]

    sort_columns = [
        "bsp_type", "is_segbsp", "direction", "snapshot_timestamp",
        "bsp_timestamp", "candidate_id",
    ]
    frame = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    groups = frame.groupby(
        ["bsp_type", "is_segbsp", "direction"], sort=False, dropna=False
    )
    frame["next_candidate_id"] = groups["candidate_id"].shift(-1)
    frame["next_bsp_timestamp"] = groups["bsp_timestamp"].shift(-1)
    frame["next_snapshot_timestamp"] = groups["snapshot_timestamp"].shift(-1)
    frame["next_bsp_low"] = groups["klu_low"].shift(-1)
    frame["next_bsp_high"] = groups["klu_high"].shift(-1)

    # A later point can supersede the previous candidate only while the
    # previous candidate is still structurally unresolved.  The replay loop
    # resolves existing candidates before adding new candidates on a bar, so
    # equality means the old candidate was already resolved and must not be
    # retroactively replaced.
    frame["was_unresolved_when_next_appeared"] = (
        frame["next_snapshot_timestamp"].notna()
        & frame["resolution_timestamp"].notna()
        & frame["next_snapshot_timestamp"].lt(frame["resolution_timestamp"])
    )

    frame["invalidated_by_lower_buy"] = (
        frame["was_unresolved_when_next_appeared"]
        &
        frame["direction"].eq("buy")
        & frame["next_candidate_id"].notna()
        & frame["klu_low"].notna()
        & frame["next_bsp_low"].notna()
        & frame["next_bsp_low"].lt(frame["klu_low"])
    )
    frame["invalidated_by_higher_sell"] = (
        bool(include_sell_supersession)
        & frame["was_unresolved_when_next_appeared"]
        & frame["direction"].eq("sell")
        & frame["next_candidate_id"].notna()
        & frame["klu_high"].notna()
        & frame["next_bsp_high"].notna()
        & frame["next_bsp_high"].gt(frame["klu_high"])
    )
    frame["is_superseded"] = (
        frame["invalidated_by_lower_buy"]
        | frame["invalidated_by_higher_sell"]
    )
    frame.loc[frame["is_superseded"], "trade_maturity_target"] = 0.0
    frame.loc[
        frame["invalidated_by_lower_buy"], "trade_label_reason"
    ] = "superseded_by_lower_same_type_buy"
    frame.loc[
        frame["invalidated_by_higher_sell"], "trade_label_reason"
    ] = "superseded_by_higher_same_type_sell"
    frame.loc[
        frame["is_superseded"], "trade_label_resolution_timestamp"
    ] = frame.loc[frame["is_superseded"], "next_snapshot_timestamp"]

    frame["trade_maturity_status"] = np.select(
        [
            frame["invalidated_by_lower_buy"],
            frame["invalidated_by_higher_sell"],
            frame["trade_maturity_target"].eq(1),
            frame["trade_maturity_target"].eq(0),
        ],
        [
            "superseded_lower_buy",
            "superseded_higher_sell",
            "mature",
            "immature",
        ],
        default="censored",
    )

    invalid_resolution = (
        frame["trade_label_resolution_timestamp"].notna()
        & frame["snapshot_timestamp"].notna()
        & frame["trade_label_resolution_timestamp"].lt(frame["snapshot_timestamp"])
    )
    if invalid_resolution.any():
        raise ValueError(
            f"{int(invalid_resolution.sum()):,} labels resolve before their "
            "candidate snapshot timestamp"
        )

    full_target = Path(output_csv)
    training_target = (
        Path(training_output_csv) if training_output_csv is not None
        else full_target.with_name(f"{full_target.stem}_training{full_target.suffix}")
    )
    full_target.parent.mkdir(parents=True, exist_ok=True)
    training_target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(full_target, index=False)
    training_frame = frame.loc[frame["trade_maturity_target"].isin([0, 1])].copy()
    training_frame.to_csv(training_target, index=False)

    summary = (
        training_frame.groupby(
            ["bsp_type", "direction", "trade_maturity_status"], dropna=False
        ).size().rename("rows").reset_index()
        .sort_values(["bsp_type", "direction", "trade_maturity_status"])
    )
    result = {
        "output_csv": str(full_target.resolve()),
        "training_output_csv": str(training_target.resolve()),
        "rows": int(len(frame)),
        "training_rows": int(len(training_frame)),
        "positive_rows": int(training_frame["trade_maturity_target"].eq(1).sum()),
        "negative_rows": int(training_frame["trade_maturity_target"].eq(0).sum()),
        "summary": summary,
        "leakage_columns": sorted(LEAKAGE_COLUMNS),
    }
    if verbose:
        print(f"[BSP labels] Full dataset:     {result['output_csv']}")
        print(f"[BSP labels] Training dataset: {result['training_output_csv']}")
        print(
            f"[BSP labels] rows={result['rows']:,}, "
            f"training={result['training_rows']:,}, "
            f"positive={result['positive_rows']:,}, "
            f"negative={result['negative_rows']:,}"
        )
        print(summary.to_string(index=False))
    return result


def plot_labeled_mature_bspoints(
    labeled_path: str | Path,
    price_csv: str | Path,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    bsp_types: tuple[str, ...] = ("1", "1p", "2", "2s", "3a", "3b"),
    direction: str | None = None,
    timestamp_basis: str = "bsp",
    annotate_max: int = 20,
    show_immature: bool = True,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (22, 9),
    show: bool = True,
) -> dict[str, Any]:
    """Plot price and BSP candidates, highlighting strategy-mature points.

    ``timestamp_basis='bsp'`` places a point at the historical bar where the
    BSP belongs.  ``timestamp_basis='snapshot'`` places it at the bar when the
    candidate first became observable, which is preferable for trading-time
    review.  At most ``annotate_max`` mature points are labeled, selected
    evenly across the requested period.
    """
    import matplotlib.pyplot as plt

    if direction is not None and str(direction).lower() not in {"buy", "sell"}:
        raise ValueError("direction must be 'buy', 'sell', or None")
    if timestamp_basis not in {"bsp", "snapshot"}:
        raise ValueError("timestamp_basis must be 'bsp' or 'snapshot'")
    if int(annotate_max) < 0:
        raise ValueError("annotate_max cannot be negative")

    labeled_source = Path(labeled_path)
    points = _read_bsp_points(labeled_source, "BSP Points")
    required = {
        "candidate_id", "bsp_type", "direction", "bsp_timestamp",
        "snapshot_timestamp", "klu_close", "trade_maturity_target",
    }
    missing = required.difference(points.columns)
    if missing:
        raise KeyError(
            "Labeled BSP dataset is missing columns: "
            f"{sorted(missing)}. Run label_bspoints_for_training first."
        )

    for column in ("bsp_timestamp", "snapshot_timestamp"):
        points[column] = pd.to_datetime(points[column], errors="coerce")
    points["klu_close"] = pd.to_numeric(points["klu_close"], errors="coerce")
    points["trade_maturity_target"] = pd.to_numeric(
        points["trade_maturity_target"], errors="coerce"
    )
    points["bsp_type"] = (
        points["bsp_type"].astype(str).str.lower().str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    points["direction"] = points["direction"].astype(str).str.lower().str.strip()
    plot_timestamp = (
        "bsp_timestamp" if timestamp_basis == "bsp" else "snapshot_timestamp"
    )
    period_start = pd.Timestamp(start)
    period_end = pd.Timestamp(end)
    if period_end < period_start:
        raise ValueError("end must be on or after start")
    allowed_types = {str(value).lower() for value in bsp_types}
    mask = (
        points[plot_timestamp].between(period_start, period_end, inclusive="both")
        & points["bsp_type"].isin(allowed_types)
        & points["klu_close"].notna()
    )
    if direction is not None:
        mask &= points["direction"].eq(str(direction).lower())
    points = points.loc[mask].sort_values(plot_timestamp).copy()

    price = pd.read_csv(price_csv)
    normalized_names = {
        str(column).lower().strip().replace(" ", "_"): column
        for column in price.columns
    }
    timestamp_column = next(
        (normalized_names[name] for name in ("timestamp", "datetime", "date")
         if name in normalized_names),
        None,
    )
    close_column = next(
        (normalized_names[name] for name in ("close", "_close", "klu_close")
         if name in normalized_names),
        None,
    )
    if timestamp_column is None or close_column is None:
        raise KeyError("Price CSV must contain timestamp/date and close columns")
    price["_plot_timestamp"] = pd.to_datetime(price[timestamp_column], errors="coerce")
    price["_plot_close"] = pd.to_numeric(price[close_column], errors="coerce")
    price = price.loc[
        price["_plot_timestamp"].between(period_start, period_end, inclusive="both")
        & price["_plot_close"].notna(),
        ["_plot_timestamp", "_plot_close"],
    ].sort_values("_plot_timestamp")
    if price.empty:
        raise ValueError("No price rows exist in the requested period")

    mature = points.loc[points["trade_maturity_target"].eq(1)].copy()
    immature = points.loc[points["trade_maturity_target"].eq(0)].copy()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        price["_plot_timestamp"], price["_plot_close"],
        color="#334155", linewidth=0.9, label="Close", zorder=1,
    )
    if show_immature and not immature.empty:
        ax.scatter(
            immature[plot_timestamp], immature["klu_close"],
            marker="x", s=22, color="#94A3B8", alpha=0.45,
            linewidths=0.8, label=f"Immature/superseded ({len(immature):,})",
            zorder=2,
        )
    for point_direction, color, marker in (
        ("buy", "#16A34A", "^"),
        ("sell", "#DC2626", "v"),
    ):
        group = mature.loc[mature["direction"].eq(point_direction)]
        if group.empty:
            continue
        ax.scatter(
            group[plot_timestamp], group["klu_close"], marker=marker,
            s=65, color=color, edgecolors="white", linewidths=0.5,
            alpha=0.92, label=f"Mature {point_direction} ({len(group):,})",
            zorder=4,
        )

    annotation_rows = mature
    if 0 < int(annotate_max) < len(annotation_rows):
        positions = np.linspace(0, len(annotation_rows) - 1, int(annotate_max))
        annotation_rows = annotation_rows.iloc[np.unique(positions.astype(int))]
    elif int(annotate_max) == 0:
        annotation_rows = annotation_rows.iloc[0:0]
    for row in annotation_rows.itertuples(index=False):
        row_direction = str(row.direction)
        color = "#15803D" if row_direction == "buy" else "#B91C1C"
        delay = getattr(row, "discovery_delay_bars", np.nan)
        delay_text = f" d={int(delay)}" if pd.notna(delay) else ""
        reason = str(getattr(row, "trade_label_reason", ""))
        reason_text = " delayed" if reason == "delayed_bsp_treated_as_mature" else ""
        label = f"{row.bsp_type} {row_direction}{delay_text}{reason_text}"
        ax.annotate(
            label,
            (getattr(row, plot_timestamp), float(row.klu_close)),
            xytext=(6, 10 if row_direction == "buy" else -15),
            textcoords="offset points", fontsize=8, color=color,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": color,
                  "alpha": 0.75, "lw": 0.5},
            zorder=5,
        )

    basis_label = "historical BSP time" if timestamp_basis == "bsp" else "first observable time"
    ax.set_title(
        f"Labeled Mature BSPs — {period_start} to {period_end}\n"
        f"Markers positioned at {basis_label}; mature={len(mature):,}, "
        f"other={len(immature):,}"
    )
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()

    saved_path = None
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=180, bbox_inches="tight")
        saved_path = str(target.resolve())
    if show:
        plt.show()
    else:
        plt.close(fig)
    return {
        "figure": fig,
        "axes": ax,
        "points": points,
        "mature_points": mature,
        "immature_points": immature,
        "mature_rows": int(len(mature)),
        "immature_rows": int(len(immature)),
        "output_path": saved_path,
    }


def train_bsp_trade_models(
    labeled_path: str | Path,
    *,
    output_dir: str | Path = "outputs/bsp_trade_models_by_type",
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
    include_delay_features: bool = False,
    price_csv: str | Path | None = None,
    price_progress_lookback_bars: int = 390,
    minimum_structure_pct: float = 0.002,
    include_multihorizon_technical_features: bool = False,
    technical_windows: tuple[int, ...] = DEFAULT_BSP_TECHNICAL_WINDOWS,
    technical_rsi_periods: tuple[int, ...] = DEFAULT_BSP_RSI_PERIODS,
    technical_atr_periods: tuple[int, ...] = DEFAULT_BSP_ATR_PERIODS,
    technical_macd_periods: tuple[tuple[int, int, int], ...] = DEFAULT_BSP_MACD_PERIODS,
    separate_buy_sell: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train one XGBoost trade-maturity classifier per exact BSP type.

    Splits are chronological and purged: a row is admitted to a split only
    when its label-resolution timestamp is no later than that split's end.
    This prevents a future-resolved outcome from leaking across date splits.
    Delay-derived inputs are excluded by default because delayed BSPs are
    already assigned maturity by a deterministic labeling rule.
    """
    import json

    import joblib
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        accuracy_score, average_precision_score, brier_score_loss,
        precision_score, recall_score, roc_auc_score,
    )
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    if not 0.0 <= float(probability_threshold) <= 1.0:
        raise ValueError("probability_threshold must be between 0 and 1")
    technical_windows = tuple(int(value) for value in technical_windows)
    technical_rsi_periods = tuple(int(value) for value in technical_rsi_periods)
    technical_atr_periods = tuple(int(value) for value in technical_atr_periods)
    technical_macd_periods = tuple(
        tuple(int(value) for value in periods)
        for periods in technical_macd_periods
    )
    source = Path(labeled_path)
    frame = _read_bsp_points(source, "BSP Points").copy()
    if price_csv is not None:
        frame = add_price_maturity_progress(
            frame, price_csv,
            reference_lookback_bars=price_progress_lookback_bars,
            minimum_structure_pct=minimum_structure_pct,
        )
        if include_multihorizon_technical_features:
            frame = add_multihorizon_technical_features_to_bspoints(
                frame, price_csv,
                windows=technical_windows,
                rsi_periods=technical_rsi_periods,
                atr_periods=technical_atr_periods,
                macd_periods=technical_macd_periods,
            )
    elif include_multihorizon_technical_features:
        raise ValueError(
            "include_multihorizon_technical_features=True requires price_csv"
        )
    required = {
        "candidate_id", "bsp_type", "direction", "snapshot_timestamp",
        "trade_label_resolution_timestamp", "trade_maturity_target",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(
            f"Labeled BSP dataset is missing columns: {sorted(missing)}. "
            "Run label_bspoints_for_training first."
        )
    frame["bsp_type"] = (
        frame["bsp_type"].astype(str).str.lower().str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    frame["direction"] = frame["direction"].astype(str).str.lower().str.strip()
    frame["snapshot_timestamp"] = pd.to_datetime(
        frame["snapshot_timestamp"], errors="coerce"
    )
    frame["trade_label_resolution_timestamp"] = pd.to_datetime(
        frame["trade_label_resolution_timestamp"], errors="coerce"
    )
    frame["trade_maturity_target"] = pd.to_numeric(
        frame["trade_maturity_target"], errors="coerce"
    )
    frame = frame.dropna(subset=[
        "snapshot_timestamp", "trade_label_resolution_timestamp",
        "trade_maturity_target",
    ]).copy()
    frame = frame.loc[frame["trade_maturity_target"].isin([0, 1])].copy()

    identity_columns = {
        "candidate_id", "candidate_key", "timestamp", "bsp_timestamp",
        "snapshot_timestamp", "first_seen_timestamp", "snapshot_last_seen",
        "direction", "bsp_type", "bsp_types",
        "bi_is_sure", "segment_is_sure", "initial_source_is_sure",
    }
    nonstationary_columns = {
        "klu_idx", "klu_open", "klu_high", "klu_low", "klu_close",
        "klu_volume", "snapshot_first_seen", "snapshot_age_bars",
        "revision_count",
        "price_maturity_market_price", "price_maturity_reference_price",
        "price_maturity_progress_raw",
    }
    excluded = LEAKAGE_COLUMNS | identity_columns | nonstationary_columns
    if separate_buy_sell:
        excluded |= {"is_buy", "direction_encoded"}
    if not include_delay_features:
        excluded |= {"is_delayed_bsp", "discovery_delay_bars", "discovery_delay_minutes"}
    features = [
        column
        for column in frame.select_dtypes(include=[np.number, "bool"]).columns
        if column not in excluded
        and "next_bi_return" not in str(column).lower()
    ]
    if not features:
        raise ValueError("No causal numeric BSP features were found")

    def inclusive_end(value: str | None) -> pd.Timestamp | None:
        if value is None:
            return None
        stamp = pd.Timestamp(value)
        if stamp == stamp.normalize():
            stamp += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        return stamp

    train_end = inclusive_end(train_end_date)
    validation_end = inclusive_end(validation_end_date)
    test_end = inclusive_end(test_end_date)
    train_mask = (
        frame["snapshot_timestamp"].le(train_end)
        & frame["trade_label_resolution_timestamp"].le(train_end)
    )
    if train_start_date:
        train_mask &= frame["snapshot_timestamp"].ge(pd.Timestamp(train_start_date))
    validation_mask = (
        frame["snapshot_timestamp"].ge(pd.Timestamp(validation_start_date))
        & frame["snapshot_timestamp"].le(validation_end)
        & frame["trade_label_resolution_timestamp"].le(validation_end)
    )
    test_mask = frame["snapshot_timestamp"].ge(pd.Timestamp(test_start_date))
    if test_end is not None:
        test_mask &= (
            frame["snapshot_timestamp"].le(test_end)
            & frame["trade_label_resolution_timestamp"].le(test_end)
        )

    parameters = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda": 2.0,
        "n_jobs": 1,
        "random_state": 42,
        "eval_metric": "logloss",
    }
    parameters.update(xgboost_params or {})
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "target": "trade_maturity_target",
        "models": {},
        "skipped": {},
        "feature_count": len(features),
        "include_delay_features": bool(include_delay_features),
        "uses_price_maturity_progress": "price_maturity_progress" in features,
        "price_progress_lookback_bars": int(price_progress_lookback_bars),
        "minimum_structure_pct": float(minimum_structure_pct),
        "include_multihorizon_technical_features": bool(
            include_multihorizon_technical_features
        ),
        "technical_windows": list(technical_windows),
        "technical_rsi_periods": list(technical_rsi_periods),
        "technical_atr_periods": list(technical_atr_periods),
        "technical_macd_periods": [list(values) for values in technical_macd_periods],
        "separate_buy_sell": bool(separate_buy_sell),
    }

    def calculate_metrics(truth: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
        prediction = probability >= float(probability_threshold)
        result: dict[str, float | int] = {
            "rows": int(len(truth)),
            "positive_rate": float(np.mean(truth)),
            "accuracy": float(accuracy_score(truth, prediction)),
            "precision": float(precision_score(truth, prediction, zero_division=0)),
            "recall": float(recall_score(truth, prediction, zero_division=0)),
            "brier": float(brier_score_loss(truth, probability)),
        }
        if len(np.unique(truth)) == 2:
            result["roc_auc"] = float(roc_auc_score(truth, probability))
            result["pr_auc"] = float(average_precision_score(truth, probability))
        return result

    model_groups = [
        (bsp_type, direction)
        for bsp_type in map(str.lower, bsp_types)
        for direction in (("buy", "sell") if separate_buy_sell else (None,))
    ]
    for bsp_type, model_direction in model_groups:
        model_key = (
            f"{bsp_type}_{model_direction}" if model_direction is not None else bsp_type
        )
        typed = frame["bsp_type"].eq(bsp_type)
        if model_direction is not None:
            typed &= frame["direction"].eq(model_direction)
        splits = {
            "train": frame.loc[typed & train_mask],
            "validation": frame.loc[typed & validation_mask],
            "test": frame.loc[typed & test_mask],
        }
        if min(map(len, splits.values())) < int(minimum_rows_per_type):
            report["skipped"][model_key] = {
                name: int(len(part)) for name, part in splits.items()
            }
            continue
        y_train = splits["train"]["trade_maturity_target"].astype(int)
        if y_train.nunique() < 2:
            report["skipped"][model_key] = {"reason": "training split has one class only"}
            continue
        type_features = [
            feature for feature in features
            if splits["train"][feature].notna().any()
        ]
        negative, positive = np.bincount(y_train, minlength=2)
        model_parameters = dict(parameters)
        model_parameters.setdefault(
            "scale_pos_weight", float(negative / positive) if positive else 1.0
        )
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(**model_parameters)),
        ])
        model.fit(splits["train"][type_features], y_train)
        type_report: dict[str, Any] = {}
        for split_name, part in splits.items():
            truth = part["trade_maturity_target"].astype(int).to_numpy()
            probability = model.predict_proba(part[type_features])[:, 1]
            type_report[split_name] = calculate_metrics(truth, probability)
            if split_name == "test":
                prediction_frame = part[[
                    "candidate_id", "snapshot_timestamp", "bsp_timestamp",
                    "bsp_type", "direction", "trade_maturity_target",
                    "trade_maturity_status", "trade_label_reason",
                ]].copy()
                prediction_frame["trade_maturity_probability"] = probability
                prediction_frame["predicted_trade_mature"] = (
                    probability >= float(probability_threshold)
                )
                prediction_frame.to_csv(
                    out / (
                        f"test_predictions_type_{bsp_type}_direction_{model_direction}.csv"
                        if model_direction is not None
                        else f"test_predictions_type_{bsp_type}.csv"
                    ), index=False
                )
        joblib.dump({
            "artifact_version": 1,
            "target": "trade_maturity_target",
            "bsp_type": bsp_type,
            "direction": model_direction,
            "model": model,
            "features": type_features,
            "probability_threshold": float(probability_threshold),
            "include_delay_features": bool(include_delay_features),
            "price_progress_lookback_bars": int(price_progress_lookback_bars),
            "minimum_structure_pct": float(minimum_structure_pct),
            "include_multihorizon_technical_features": bool(
                include_multihorizon_technical_features
            ),
            "technical_windows": technical_windows,
            "technical_rsi_periods": technical_rsi_periods,
            "technical_atr_periods": technical_atr_periods,
            "technical_macd_periods": technical_macd_periods,
            "xgboost_params": model_parameters,
        }, out / (
            f"trade_maturity_model_type_{bsp_type}_direction_{model_direction}.joblib"
            if model_direction is not None
            else f"trade_maturity_model_type_{bsp_type}.joblib"
        ))
        pd.DataFrame({"feature": type_features}).to_csv(
            out / (
                f"feature_manifest_type_{bsp_type}_direction_{model_direction}.csv"
                if model_direction is not None
                else f"feature_manifest_type_{bsp_type}.csv"
            ), index=False
        )
        type_report["feature_count"] = len(type_features)
        report["models"][model_key] = type_report
        if verbose:
            metrics = type_report["test"]
            print(
                f"[Trade maturity {model_key}] test rows={metrics['rows']:,}, "
                f"precision={metrics['precision']:.3f}, "
                f"recall={metrics['recall']:.3f}, "
                f"PR-AUC={metrics.get('pr_auc', float('nan')):.3f}",
                flush=True,
            )

    pd.DataFrame({"feature": features}).to_csv(out / "feature_manifest.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def create_predicted_mature_signals(
    bspoint_path: str | Path,
    model_dir: str | Path,
    *,
    output_csv: str | Path = "outputs/predicted_bsp_maturity.csv",
    signals_output_csv: str | Path | None = None,
    sheet_name: str = "BSP Points",
    bsp_types: tuple[str, ...] = ("1", "1p", "2", "2s", "3a", "3b"),
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    probability_threshold: float | None = None,
    apply_delayed_rule: bool = True,
    apply_already_sure_rule: bool = False,
    use_price_progress_as_maturity: bool = True,
    price_csv: str | Path | None = None,
    price_progress_lookback_bars: int | None = None,
    minimum_structure_pct: float | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Score BSP candidates and create predicted-mature trading signals.

    The function selects the artifact matching each exact BSP type.  Model
    probabilities are calculated first. By default, the trade maturity value
    and signal threshold use causal price recovery rather than ``is_sure``.
    Candidates
    without an available type model are retained in the full output but cannot
    become signals unless a deterministic rule applies.
    """
    import joblib

    if probability_threshold is not None and not 0.0 <= float(probability_threshold) <= 1.0:
        raise ValueError("probability_threshold must be between 0 and 1")
    source = Path(bspoint_path)
    models_path = Path(model_dir)
    frame = _read_bsp_points(source, sheet_name).copy()
    required = {
        "candidate_id", "bsp_type", "direction", "is_segbsp",
        "bsp_timestamp", "snapshot_timestamp",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"BSP candidate dataset is missing columns: {sorted(missing)}")

    frame["bsp_type"] = (
        frame["bsp_type"].astype(str).str.lower().str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    frame["direction"] = frame["direction"].astype(str).str.lower().str.strip()
    for column in ("bsp_timestamp", "snapshot_timestamp"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    if start is not None:
        frame = frame.loc[frame["snapshot_timestamp"].ge(pd.Timestamp(start))].copy()
    if end is not None:
        period_end = pd.Timestamp(end)
        if period_end == period_end.normalize():
            period_end += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        frame = frame.loc[frame["snapshot_timestamp"].le(period_end)].copy()

    def boolean_series(column: str) -> pd.Series:
        if column not in frame.columns:
            return pd.Series(False, index=frame.index, dtype=bool)
        return (
            frame[column].astype(str).str.strip().str.lower()
            .isin({"1", "1.0", "true", "yes", "y"})
        )

    frame["is_segbsp"] = boolean_series("is_segbsp")
    frame["is_delayed_bsp_live"] = (
        frame["snapshot_timestamp"].notna()
        & frame["bsp_timestamp"].notna()
        & frame["snapshot_timestamp"].gt(frame["bsp_timestamp"])
    )
    bi_sure = boolean_series("bi_is_sure")
    segment_sure = boolean_series("segment_is_sure")
    frame["source_line_is_sure_live"] = np.where(
        frame["is_segbsp"], segment_sure, bi_sure
    ).astype(bool)
    frame["model_maturity_probability"] = np.nan
    frame["trade_maturity_probability"] = np.nan
    frame["prediction_threshold"] = np.nan
    frame["prediction_source"] = "no_model"
    frame["predicted_mature_signal"] = False
    frame["model_available"] = False

    allowed_types = {str(value).lower() for value in bsp_types}
    artifacts: dict[tuple[str, str | None], dict[str, Any]] = {}
    for bsp_type in sorted(allowed_types):
        directional_paths = {
            direction: models_path / (
                f"trade_maturity_model_type_{bsp_type}_direction_{direction}.joblib"
            )
            for direction in ("buy", "sell")
        }
        available_directional = {
            direction: path for direction, path in directional_paths.items()
            if path.exists()
        }
        candidate_paths = (
            list(available_directional.items())
            if available_directional
            else [(None, models_path / f"trade_maturity_model_type_{bsp_type}.joblib")]
        )
        for model_direction, artifact_path in candidate_paths:
            if not artifact_path.exists():
                continue
            artifact = joblib.load(artifact_path)
            if str(artifact.get("bsp_type", "")).lower() != bsp_type:
                raise ValueError(
                    f"Artifact type mismatch in {artifact_path}: "
                    f"expected {bsp_type!r}, got {artifact.get('bsp_type')!r}"
                )
            artifact_direction = artifact.get("direction")
            if model_direction is not None and str(artifact_direction).lower() != model_direction:
                raise ValueError(
                    f"Artifact direction mismatch in {artifact_path}: "
                    f"expected {model_direction!r}, got {artifact_direction!r}"
                )
            artifacts[(bsp_type, model_direction)] = artifact

    needs_price_progress = any(
        "price_maturity_progress" in artifact.get("features", [])
        for artifact in artifacts.values()
    )
    if needs_price_progress:
        if price_csv is None:
            raise ValueError(
                "These models require price_maturity_progress; pass the OHLC price_csv."
            )
        artifact_lookbacks = {
            int(artifact.get("price_progress_lookback_bars", 390))
            for artifact in artifacts.values()
        }
        artifact_minimums = {
            float(artifact.get("minimum_structure_pct", 0.002))
            for artifact in artifacts.values()
        }
        if price_progress_lookback_bars is None and len(artifact_lookbacks) > 1:
            raise ValueError("Loaded artifacts use different price-progress lookbacks")
        if minimum_structure_pct is None and len(artifact_minimums) > 1:
            raise ValueError("Loaded artifacts use different minimum structure sizes")
        frame = add_price_maturity_progress(
            frame, price_csv,
            reference_lookback_bars=(
                int(price_progress_lookback_bars)
                if price_progress_lookback_bars is not None else next(iter(artifact_lookbacks))
            ),
            minimum_structure_pct=(
                float(minimum_structure_pct)
                if minimum_structure_pct is not None else next(iter(artifact_minimums))
            ),
        )

    needs_multihorizon_technical = any(
        bool(artifact.get("include_multihorizon_technical_features", False))
        for artifact in artifacts.values()
    )
    if needs_multihorizon_technical:
        if price_csv is None:
            raise ValueError(
                "These models require multi-horizon technical features; pass price_csv."
            )
        technical_configs = {
            (
                tuple(artifact.get(
                    "technical_windows", LEGACY_EXPANDED_BSP_TECHNICAL_WINDOWS
                )),
                tuple(artifact.get(
                    "technical_rsi_periods", LEGACY_EXPANDED_BSP_RSI_PERIODS
                )),
                tuple(artifact.get(
                    "technical_atr_periods", LEGACY_EXPANDED_BSP_ATR_PERIODS
                )),
                tuple(tuple(values) for values in artifact.get(
                    "technical_macd_periods", LEGACY_EXPANDED_BSP_MACD_PERIODS
                )),
            )
            for artifact in artifacts.values()
            if artifact.get("include_multihorizon_technical_features", False)
        }
        if len(technical_configs) != 1:
            raise ValueError("Loaded artifacts use different technical period settings")
        windows, rsi_periods, atr_periods, macd_periods = next(iter(technical_configs))
        frame = add_multihorizon_technical_features_to_bspoints(
            frame, price_csv,
            windows=windows,
            rsi_periods=rsi_periods,
            atr_periods=atr_periods,
            macd_periods=macd_periods,
        )

    for (bsp_type, model_direction), artifact in artifacts.items():
        type_mask = frame["bsp_type"].eq(bsp_type)
        if model_direction is not None:
            type_mask &= frame["direction"].eq(model_direction)
        if not type_mask.any():
            continue
        features = list(artifact.get("features", []))
        missing_features = [feature for feature in features if feature not in frame.columns]
        if missing_features:
            raise KeyError(
                f"BSP model {bsp_type!r}/{model_direction or 'combined'} is "
                f"missing features: {missing_features}"
            )
        model = artifact["model"]
        probability = model.predict_proba(frame.loc[type_mask, features])[:, 1]
        threshold = (
            float(probability_threshold)
            if probability_threshold is not None
            else float(artifact.get("probability_threshold", 0.5))
        )
        frame.loc[type_mask, "model_maturity_probability"] = probability
        frame.loc[type_mask, "trade_maturity_probability"] = probability
        frame.loc[type_mask, "prediction_threshold"] = threshold
        frame.loc[type_mask, "prediction_source"] = "xgboost"
        frame.loc[type_mask, "model_available"] = True

    if apply_delayed_rule:
        delayed = frame["is_delayed_bsp_live"]
        frame.loc[delayed, "trade_maturity_probability"] = 1.0
        frame.loc[delayed, "prediction_source"] = "delayed_rule"
        # A deterministic rule still needs a threshold for output clarity.
        frame.loc[delayed & frame["prediction_threshold"].isna(), "prediction_threshold"] = (
            float(probability_threshold) if probability_threshold is not None else 0.5
        )
    if apply_already_sure_rule:
        already_sure = frame["source_line_is_sure_live"]
        # Already-sure is more specific and takes display precedence if both
        # deterministic rules apply.
        frame.loc[already_sure, "trade_maturity_probability"] = 1.0
        frame.loc[already_sure, "prediction_source"] = "already_sure_rule"
        frame.loc[
            already_sure & frame["prediction_threshold"].isna(),
            "prediction_threshold",
        ] = float(probability_threshold) if probability_threshold is not None else 0.5

    if use_price_progress_as_maturity:
        if "price_maturity_progress" not in frame.columns:
            raise ValueError(
                "use_price_progress_as_maturity=True requires models trained with "
                "price_maturity_progress and a price_csv during prediction."
            )
        progress = pd.to_numeric(
            frame["price_maturity_progress"], errors="coerce"
        ).clip(0.0, 1.0)
        frame["trade_maturity_probability"] = progress
        frame["prediction_source"] = np.where(
            progress.notna(), "price_progress", frame["prediction_source"]
        )

    eligible_type = frame["bsp_type"].isin(allowed_types)
    frame["predicted_mature_signal"] = (
        eligible_type
        & frame["trade_maturity_probability"].notna()
        & frame["prediction_threshold"].notna()
        & frame["trade_maturity_probability"].ge(frame["prediction_threshold"])
    )
    frame["signal"] = np.select(
        [
            frame["predicted_mature_signal"] & frame["direction"].eq("buy"),
            frame["predicted_mature_signal"] & frame["direction"].eq("sell"),
        ],
        ["mature_buy", "mature_sell"],
        default="no_signal",
    )

    frame = frame.sort_values(
        ["snapshot_timestamp", "candidate_id"], kind="stable"
    ).reset_index(drop=True)
    signals = frame.loc[frame["predicted_mature_signal"]].copy()
    full_target = Path(output_csv)
    signal_target = (
        Path(signals_output_csv) if signals_output_csv is not None
        else full_target.with_name(f"{full_target.stem}_signals{full_target.suffix}")
    )
    full_target.parent.mkdir(parents=True, exist_ok=True)
    signal_target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(full_target, index=False)
    signals.to_csv(signal_target, index=False)

    by_type = (
        signals.groupby(["bsp_type", "direction", "prediction_source"])
        .size().rename("signals").reset_index()
        .sort_values(["bsp_type", "direction", "prediction_source"])
    )
    result = {
        "output_csv": str(full_target.resolve()),
        "signals_output_csv": str(signal_target.resolve()),
        "rows": int(len(frame)),
        "signal_rows": int(len(signals)),
        "models_loaded": sorted(
            f"{bsp_type}_{direction}" if direction else bsp_type
            for bsp_type, direction in artifacts
        ),
        "missing_models": sorted(
            bsp_type for bsp_type in allowed_types
            if not any(key_type == bsp_type for key_type, _ in artifacts)
        ),
        "summary": by_type,
        "predictions": frame,
        "signals": signals,
    }
    if verbose:
        print(f"[BSP prediction] Models loaded: {result['models_loaded']}")
        if result["missing_models"]:
            print(f"[BSP prediction] Missing models: {result['missing_models']}")
        print(
            f"[BSP prediction] candidates={result['rows']:,}, "
            f"signals={result['signal_rows']:,}"
        )
        print(f"[BSP prediction] All rows: {result['output_csv']}")
        print(f"[BSP prediction] Signals:  {result['signals_output_csv']}")
        if not by_type.empty:
            print(by_type.to_string(index=False))
    return result


def plot_predicted_mature_signals(
    predictions_path: str | Path,
    price_csv: str | Path,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    bsp_types: tuple[str, ...] = ("1", "1p", "2", "2s", "3a", "3b"),
    direction: str | None = None,
    timestamp_basis: str = "snapshot",
    minimum_probability: float | None = None,
    minimum_price_progress: float | None = None,
    annotate_max: int = 20,
    show_rejected: bool = True,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (22, 9),
    show: bool = True,
) -> dict[str, Any]:
    """Plot BSP signals with independent ML-probability and progress filters."""
    import matplotlib.pyplot as plt

    if direction is not None and str(direction).lower() not in {"buy", "sell"}:
        raise ValueError("direction must be 'buy', 'sell', or None")
    if timestamp_basis not in {"bsp", "snapshot"}:
        raise ValueError("timestamp_basis must be 'bsp' or 'snapshot'")
    if minimum_probability is not None and not 0.0 <= float(minimum_probability) <= 1.0:
        raise ValueError("minimum_probability must be between 0 and 1")
    if minimum_price_progress is not None and not 0.0 <= float(minimum_price_progress) <= 1.0:
        raise ValueError("minimum_price_progress must be between 0 and 1")
    if int(annotate_max) < 0:
        raise ValueError("annotate_max cannot be negative")

    predictions = _read_bsp_points(Path(predictions_path), "BSP Points")
    required = {
        "candidate_id", "bsp_type", "direction", "bsp_timestamp",
        "snapshot_timestamp", "klu_close", "predicted_mature_signal",
        "trade_maturity_probability", "price_maturity_progress", "prediction_source",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(
            f"Prediction dataset is missing columns: {sorted(missing)}. "
            "Run create_predicted_mature_signals first."
        )
    for column in ("bsp_timestamp", "snapshot_timestamp"):
        predictions[column] = pd.to_datetime(predictions[column], errors="coerce")
    predictions["klu_close"] = pd.to_numeric(
        predictions["klu_close"], errors="coerce"
    )
    predictions["trade_maturity_probability"] = pd.to_numeric(
        predictions["trade_maturity_probability"], errors="coerce"
    )
    predictions["price_maturity_progress"] = pd.to_numeric(
        predictions["price_maturity_progress"], errors="coerce"
    ).clip(0.0, 1.0)
    predictions["predicted_mature_signal"] = (
        predictions["predicted_mature_signal"].astype(str).str.lower()
        .isin({"1", "1.0", "true", "yes", "y"})
    )
    # Start from the signal decision saved during prediction. A supplied ML
    # threshold replaces that decision; progress is then an additional AND
    # filter rather than incorrectly replacing the model decision.
    if minimum_probability is not None:
        predictions["predicted_mature_signal"] = (
            predictions["trade_maturity_probability"].notna()
            & predictions["trade_maturity_probability"].ge(
                float(minimum_probability)
            )
        )
    if minimum_price_progress is not None:
        predictions["predicted_mature_signal"] &= (
            predictions["price_maturity_progress"].notna()
            & predictions["price_maturity_progress"].ge(
                float(minimum_price_progress)
            )
        )
    predictions["bsp_type"] = (
        predictions["bsp_type"].astype(str).str.lower().str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    predictions["direction"] = (
        predictions["direction"].astype(str).str.lower().str.strip()
    )
    plot_timestamp = (
        "bsp_timestamp" if timestamp_basis == "bsp" else "snapshot_timestamp"
    )
    period_start = pd.Timestamp(start)
    period_end = pd.Timestamp(end)
    if period_end < period_start:
        raise ValueError("end must be on or after start")
    if period_end == period_end.normalize():
        period_end += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    allowed_types = {str(value).lower() for value in bsp_types}
    mask = (
        predictions[plot_timestamp].between(period_start, period_end, inclusive="both")
        & predictions["bsp_type"].isin(allowed_types)
        & predictions["klu_close"].notna()
    )
    if direction is not None:
        mask &= predictions["direction"].eq(str(direction).lower())
    predictions = predictions.loc[mask].sort_values(plot_timestamp).copy()

    price = pd.read_csv(price_csv)
    normalized_names = {
        str(column).lower().strip().replace(" ", "_"): column
        for column in price.columns
    }
    timestamp_column = next(
        (normalized_names[name] for name in ("timestamp", "datetime", "date")
         if name in normalized_names), None,
    )
    close_column = next(
        (normalized_names[name] for name in ("close", "_close", "klu_close")
         if name in normalized_names), None,
    )
    if timestamp_column is None or close_column is None:
        raise KeyError("Price CSV must contain timestamp/date and close columns")
    price["_plot_timestamp"] = pd.to_datetime(price[timestamp_column], errors="coerce")
    price["_plot_close"] = pd.to_numeric(price[close_column], errors="coerce")
    price = price.loc[
        price["_plot_timestamp"].between(period_start, period_end, inclusive="both")
        & price["_plot_close"].notna(),
        ["_plot_timestamp", "_plot_close"],
    ].sort_values("_plot_timestamp")
    if price.empty:
        raise ValueError("No price rows exist in the requested period")

    if timestamp_basis == "snapshot":
        # At signal time, klu_close is still the price of the historical BSP
        # bar.  Plot against the actual close at the observable snapshot time
        # so markers sit on the displayed market-price line.
        close_by_timestamp = (
            price.drop_duplicates("_plot_timestamp", keep="last")
            .set_index("_plot_timestamp")["_plot_close"]
        )
        predictions["plot_price"] = predictions[plot_timestamp].map(
            close_by_timestamp
        )
    else:
        predictions["plot_price"] = predictions["klu_close"]
    missing_plot_price = predictions["plot_price"].isna()
    if missing_plot_price.any():
        raise ValueError(
            f"{int(missing_plot_price.sum()):,} BSP markers have no matching "
            f"price at {timestamp_basis} timestamp"
        )
    accepted = predictions.loc[predictions["predicted_mature_signal"]].copy()
    rejected = predictions.loc[~predictions["predicted_mature_signal"]].copy()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        price["_plot_timestamp"], price["_plot_close"],
        color="#334155", linewidth=0.9, label="Close", zorder=1,
    )
    if show_rejected and not rejected.empty:
        ax.scatter(
            rejected[plot_timestamp], rejected["plot_price"], marker="x",
            s=22, color="#94A3B8", alpha=0.4, linewidths=0.8,
            label=f"Rejected ({len(rejected):,})", zorder=2,
        )
    for point_direction, color, marker in (
        ("buy", "#16A34A", "^"), ("sell", "#DC2626", "v")
    ):
        group = accepted.loc[accepted["direction"].eq(point_direction)]
        if group.empty:
            continue
        ax.scatter(
            group[plot_timestamp], group["plot_price"], marker=marker,
            s=68, color=color, edgecolors="white", linewidths=0.5,
            alpha=0.94,
            label=f"Qualified {point_direction} ({len(group):,})",
            zorder=4,
        )

    annotation_rows = accepted
    if 0 < int(annotate_max) < len(annotation_rows):
        positions = np.linspace(0, len(annotation_rows) - 1, int(annotate_max))
        annotation_rows = annotation_rows.iloc[np.unique(positions.astype(int))]
    elif int(annotate_max) == 0:
        annotation_rows = annotation_rows.iloc[0:0]
    source_abbreviations = {
        "xgboost": "ML",
        "delayed_rule": "D",
        "already_sure_rule": "S",
        "price_progress": "P",
        "no_model": "NA",
    }
    for row in annotation_rows.itertuples(index=False):
        row_direction = str(row.direction)
        color = "#15803D" if row_direction == "buy" else "#B91C1C"
        progress = float(row.price_maturity_progress)
        probability = float(row.trade_maturity_probability)
        source = source_abbreviations.get(
            str(row.prediction_source), str(row.prediction_source)
        )
        label = (
            f"{row.bsp_type} {row_direction} "
            f"ML={probability:.2f} P={progress:.2f} {source}"
        )
        ax.annotate(
            label, (getattr(row, plot_timestamp), float(row.plot_price)),
            xytext=(6, 10 if row_direction == "buy" else -15),
            textcoords="offset points", fontsize=8, color=color,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": color,
                  "alpha": 0.78, "lw": 0.5}, zorder=5,
        )

    basis_label = "historical BSP time" if timestamp_basis == "bsp" else "signal time"
    threshold_parts = []
    if minimum_probability is not None:
        threshold_parts.append(f"ML >= {float(minimum_probability):.2f}")
    if minimum_price_progress is not None:
        threshold_parts.append(f"progress >= {float(minimum_price_progress):.2f}")
    threshold_label = (
        "; " + ", ".join(threshold_parts) if threshold_parts else ""
    )
    ax.set_title(
        f"BSP ML Probability and Price-Progress Signals — {period_start} to {period_end}\n"
        f"Markers positioned at {basis_label}; accepted={len(accepted):,}, "
        f"rejected={len(rejected):,}{threshold_label}"
    )
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()

    saved_path = None
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=180, bbox_inches="tight")
        saved_path = str(target.resolve())
    if show:
        plt.show()
    else:
        plt.close(fig)
    return {
        "figure": fig,
        "axes": ax,
        "predictions": predictions,
        "signals": accepted,
        "rejected": rejected,
        "signal_rows": int(len(accepted)),
        "rejected_rows": int(len(rejected)),
        "minimum_probability": minimum_probability,
        "minimum_price_progress": minimum_price_progress,
        "output_path": saved_path,
    }


def plot_bsp_trade_feature_importance(
    model_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    bsp_types: tuple[str, ...] = ("1", "1p", "2", "2s", "3a", "3b"),
    directions: tuple[str, ...] = ("buy", "sell"),
    importance_type: str = "gain",
    top_n: int | None = None,
    normalize: bool = True,
    figsize_width: float = 12.0,
    show: bool = True,
) -> dict[str, Any]:
    """Plot XGBoost feature importance for each saved BSP-type model.

    Supported XGBoost importance types are ``weight``, ``gain``, ``cover``,
    ``total_gain``, and ``total_cover``.  Features unused by a model are kept
    with zero importance.  An aggregate plot shows the mean normalized
    importance across every available BSP-type model.
    """
    import joblib
    import matplotlib.pyplot as plt

    allowed_importance = {"weight", "gain", "cover", "total_gain", "total_cover"}
    if importance_type not in allowed_importance:
        raise ValueError(
            f"importance_type must be one of {sorted(allowed_importance)}"
        )
    if top_n is not None and int(top_n) <= 0:
        raise ValueError("top_n must be greater than zero or None")
    if float(figsize_width) <= 0:
        raise ValueError("figsize_width must be greater than zero")

    models_path = Path(model_dir)
    target_dir = Path(output_dir) if output_dir is not None else models_path / "feature_importance"
    target_dir.mkdir(parents=True, exist_ok=True)
    importance_frames: dict[str, pd.DataFrame] = {}
    plot_paths: dict[str, str] = {}
    missing_models: list[str] = []

    normalized_directions = tuple(str(value).lower() for value in directions)
    if set(normalized_directions).difference({"buy", "sell"}):
        raise ValueError("directions may contain only 'buy' and/or 'sell'")
    model_specs: list[tuple[str, str, Path]] = []
    for bsp_type in map(str.lower, bsp_types):
        directional = [
            (
                f"{bsp_type}_{direction}", direction,
                models_path / f"trade_maturity_model_type_{bsp_type}_direction_{direction}.joblib",
            )
            for direction in normalized_directions
        ]
        existing = [spec for spec in directional if spec[2].exists()]
        if existing:
            model_specs.extend(existing)
        else:
            legacy_path = models_path / f"trade_maturity_model_type_{bsp_type}.joblib"
            if legacy_path.exists():
                model_specs.append((bsp_type, "combined", legacy_path))
            else:
                missing_models.append(bsp_type)

    for model_key, model_direction, artifact_path in model_specs:
        artifact = joblib.load(artifact_path)
        bsp_type = str(artifact.get("bsp_type", model_key)).lower()
        features = list(artifact.get("features", []))
        pipeline = artifact.get("model")
        estimator = getattr(pipeline, "named_steps", {}).get("model")
        if estimator is None or not hasattr(estimator, "get_booster"):
            raise TypeError(f"{artifact_path} does not contain an XGBoost pipeline")
        raw_scores = estimator.get_booster().get_score(importance_type=importance_type)
        scores = np.zeros(len(features), dtype=float)
        for index, feature in enumerate(features):
            # Pipeline imputation converts the DataFrame to an array, so
            # XGBoost normally stores feature keys as f0, f1, ... .  Retain a
            # fallback for artifacts that preserve actual feature names.
            scores[index] = float(raw_scores.get(f"f{index}", raw_scores.get(feature, 0.0)))
        if normalize and scores.sum() > 0:
            scores = scores / scores.sum()
        importance = pd.DataFrame({
            "feature": features,
            "importance": scores,
            "bsp_type": bsp_type,
            "direction": model_direction,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        importance["rank"] = np.arange(1, len(importance) + 1)
        importance_frames[model_key] = importance
        importance.to_csv(
            target_dir / f"feature_importance_{model_key}.csv", index=False
        )

        plotted = importance.head(int(top_n)) if top_n is not None else importance
        plotted = plotted.sort_values("importance", ascending=True)
        figure_height = max(6.0, 0.34 * len(plotted) + 1.8)
        fig, ax = plt.subplots(figsize=(float(figsize_width), figure_height))
        ax.barh(
            plotted["feature"], plotted["importance"],
            color="#2563EB", alpha=0.88,
        )
        ax.set_title(
            f"BSP {model_key} — XGBoost Feature Importance ({importance_type})"
        )
        ax.set_xlabel("Normalized importance" if normalize else importance_type)
        ax.set_ylabel("Feature")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        plot_path = target_dir / f"feature_importance_{model_key}.png"
        fig.savefig(plot_path, dpi=180, bbox_inches="tight")
        plot_paths[model_key] = str(plot_path.resolve())
        if show:
            plt.show()
        else:
            plt.close(fig)

    aggregate_path = None
    aggregate_csv_path = None
    aggregate = pd.DataFrame(columns=["feature", "mean_importance", "models_using"])
    if importance_frames:
        combined = pd.concat(importance_frames.values(), ignore_index=True)
        aggregate = (
            combined.groupby("feature", as_index=False)
            .agg(
                mean_importance=("importance", "mean"),
                models_using=("importance", lambda values: int((values > 0).sum())),
            )
            .sort_values("mean_importance", ascending=False)
            .reset_index(drop=True)
        )
        aggregate["rank"] = np.arange(1, len(aggregate) + 1)
        aggregate_csv = target_dir / "feature_importance_mean_across_types.csv"
        aggregate.to_csv(aggregate_csv, index=False)
        aggregate_csv_path = str(aggregate_csv.resolve())
        plotted = aggregate.head(int(top_n)) if top_n is not None else aggregate
        plotted = plotted.sort_values("mean_importance", ascending=True)
        figure_height = max(6.0, 0.34 * len(plotted) + 1.8)
        fig, ax = plt.subplots(figsize=(float(figsize_width), figure_height))
        ax.barh(
            plotted["feature"], plotted["mean_importance"],
            color="#0F766E", alpha=0.88,
        )
        ax.set_title(
            f"Mean XGBoost Feature Importance Across BSP Types ({importance_type})"
        )
        ax.set_xlabel("Mean normalized importance" if normalize else f"Mean {importance_type}")
        ax.set_ylabel("Feature")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        aggregate_plot = target_dir / "feature_importance_mean_across_types.png"
        fig.savefig(aggregate_plot, dpi=180, bbox_inches="tight")
        aggregate_path = str(aggregate_plot.resolve())
        if show:
            plt.show()
        else:
            plt.close(fig)

    return {
        "importance_by_type": importance_frames,
        "aggregate_importance": aggregate,
        "plot_paths": plot_paths,
        "aggregate_plot_path": aggregate_path,
        "aggregate_csv_path": aggregate_csv_path,
        "missing_models": missing_models,
        "output_dir": str(target_dir.resolve()),
    }


def backtest_and_plot_predicted_bsp_signals(
    predictions_path: str | Path,
    price_csv: str | Path,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    bsp_types: tuple[str, ...] = ("1", "1p", "2", "2s", "3a", "3b"),
    minimum_probability: float = 0.5,
    probability_column: str = "auto",
    minimum_price_progress: float = 0.0,
    minimum_reward_risk: float = 0.0,
    maximum_predicted_loss: float | None = None,
    require_risk_control_pass: bool = False,
    initial_equity: float = 100_000.0,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
    force_exit_at_end: bool = True,
    output_path: str | Path | None = None,
    trades_output_csv: str | Path | None = None,
    figsize: tuple[float, float] = (22, 12),
    show: bool = True,
) -> dict[str, Any]:
    """Backtest and plot a filtered long-only strategy from predicted BSPs.

    A predicted mature buy opens a full long position and a predicted mature
    sell closes it.  Orders execute at the next available bar's open, avoiding
    same-bar look-ahead. Repeated buys while long and sells while flat are
    ignored. If opposing signals share a timestamp, the higher-probability
    signal is used; exact probability ties are skipped.
    """
    import matplotlib.pyplot as plt

    if not 0.0 <= float(minimum_probability) <= 1.0:
        raise ValueError("minimum_probability must be between 0 and 1")
    if not 0.0 <= float(minimum_price_progress) <= 1.0:
        raise ValueError("minimum_price_progress must be between 0 and 1")
    if float(minimum_reward_risk) < 0:
        raise ValueError("minimum_reward_risk cannot be negative")
    if maximum_predicted_loss is not None and float(maximum_predicted_loss) <= 0:
        raise ValueError("maximum_predicted_loss must be a positive magnitude")
    if float(initial_equity) <= 0:
        raise ValueError("initial_equity must be greater than zero")
    if float(commission_bps) < 0 or float(slippage_bps) < 0:
        raise ValueError("commission_bps and slippage_bps cannot be negative")

    period_start = pd.Timestamp(start)
    period_end = pd.Timestamp(end)
    if period_end < period_start:
        raise ValueError("end must be on or after start")
    if period_end == period_end.normalize():
        period_end += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    allowed_types = {str(value).lower() for value in bsp_types}

    signals = _read_bsp_points(Path(predictions_path), "BSP Points")
    if probability_column == "auto":
        selected_probability_column = (
            "predicted_entry_quality_2d"
            if "predicted_entry_quality_2d" in signals.columns
            else (
                "price_maturity_progress"
                if "price_maturity_progress" in signals.columns
                else (
                    "predicted_maturity_rate"
                    if "predicted_maturity_rate" in signals.columns
                    else "trade_maturity_probability"
                )
            )
        )
    else:
        selected_probability_column = str(probability_column)
    required_signals = {
        "candidate_id", "snapshot_timestamp", "bsp_type", "direction",
        selected_probability_column,
    }
    missing = required_signals.difference(signals.columns)
    if missing:
        raise KeyError(f"Prediction dataset is missing columns: {sorted(missing)}")
    signals["snapshot_timestamp"] = pd.to_datetime(
        signals["snapshot_timestamp"], errors="coerce"
    )
    signals["bsp_type"] = (
        signals["bsp_type"].astype(str).str.lower().str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    signals["direction"] = signals["direction"].astype(str).str.lower().str.strip()
    signals["signal_probability_used"] = pd.to_numeric(
        signals[selected_probability_column], errors="coerce"
    )
    eligible = (
        signals["snapshot_timestamp"].between(period_start, period_end, inclusive="both")
        & signals["bsp_type"].isin(allowed_types)
        & signals["direction"].isin({"buy", "sell"})
        & signals["signal_probability_used"].ge(float(minimum_probability))
    )
    if float(minimum_price_progress) > 0:
        if "price_maturity_progress" not in signals.columns:
            raise KeyError("minimum_price_progress requires price_maturity_progress")
        signals["price_maturity_progress"] = pd.to_numeric(
            signals["price_maturity_progress"], errors="coerce"
        )
        eligible &= signals["price_maturity_progress"].ge(float(minimum_price_progress))
    if float(minimum_reward_risk) > 0:
        if "predicted_reward_risk_2d" not in signals.columns:
            raise KeyError("minimum_reward_risk requires predicted_reward_risk_2d")
        signals["predicted_reward_risk_2d"] = pd.to_numeric(
            signals["predicted_reward_risk_2d"], errors="coerce"
        )
        eligible &= signals["predicted_reward_risk_2d"].ge(float(minimum_reward_risk))
    if maximum_predicted_loss is not None:
        if "predicted_max_loss_2d" not in signals.columns:
            raise KeyError("maximum_predicted_loss requires predicted_max_loss_2d")
        signals["predicted_max_loss_2d"] = pd.to_numeric(
            signals["predicted_max_loss_2d"], errors="coerce"
        )
        eligible &= signals["predicted_max_loss_2d"].ge(-float(maximum_predicted_loss))
    if require_risk_control_pass:
        if "risk_control_pass" not in signals.columns:
            raise KeyError("require_risk_control_pass requires risk_control_pass")
        risk_pass = signals["risk_control_pass"].astype(str).str.lower().isin(
            {"1", "1.0", "true", "yes", "y"}
        )
        eligible &= risk_pass
    signals = signals.loc[eligible].copy()

    # Resolve multiple candidates at one timestamp without silently allowing
    # an arbitrary row order to decide the trade direction.
    resolved_events: list[pd.Series] = []
    for _, group in signals.groupby("snapshot_timestamp", sort=True):
        best_by_direction = (
            group.sort_values("signal_probability_used", ascending=False)
            .drop_duplicates("direction", keep="first")
        )
        if best_by_direction["direction"].nunique() == 1:
            resolved_events.append(best_by_direction.iloc[0])
            continue
        maximum = best_by_direction["signal_probability_used"].max()
        winners = best_by_direction.loc[
            best_by_direction["signal_probability_used"].eq(maximum)
        ]
        if len(winners) == 1:
            resolved_events.append(winners.iloc[0])
    events = pd.DataFrame(resolved_events)
    if not events.empty:
        events = events.sort_values("snapshot_timestamp").reset_index(drop=True)

    price = pd.read_csv(price_csv)
    normalized_names = {
        str(column).lower().strip().replace(" ", "_"): column
        for column in price.columns
    }
    timestamp_column = next(
        (normalized_names[name] for name in ("timestamp", "datetime", "date")
         if name in normalized_names), None,
    )
    open_column = next(
        (normalized_names[name] for name in ("open", "_open", "klu_open")
         if name in normalized_names), None,
    )
    close_column = next(
        (normalized_names[name] for name in ("close", "_close", "klu_close")
         if name in normalized_names), None,
    )
    if timestamp_column is None or open_column is None or close_column is None:
        raise KeyError("Price CSV must contain timestamp/date, open, and close columns")
    price["timestamp"] = pd.to_datetime(price[timestamp_column], errors="coerce")
    price["open"] = pd.to_numeric(price[open_column], errors="coerce")
    price["close"] = pd.to_numeric(price[close_column], errors="coerce")
    price = price.loc[
        price["timestamp"].between(period_start, period_end, inclusive="both")
        & price["open"].notna() & price["close"].notna(),
        ["timestamp", "open", "close"],
    ].sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    if price.empty:
        raise ValueError("No price rows exist in the requested period")

    price_times = price["timestamp"].to_numpy(dtype="datetime64[ns]")
    commission = float(commission_bps) / 10_000.0
    slippage = float(slippage_bps) / 10_000.0
    equity = float(initial_equity)
    entry: dict[str, Any] | None = None
    trades: list[dict[str, Any]] = []

    if not events.empty:
        for event in events.itertuples(index=False):
            signal_time = pd.Timestamp(event.snapshot_timestamp)
            execution_index = int(np.searchsorted(
                price_times, signal_time.to_datetime64(), side="right"
            ))
            if execution_index >= len(price):
                continue
            execution = price.iloc[execution_index]
            if event.direction == "buy" and entry is None:
                entry_price = float(execution["open"]) * (1.0 + slippage)
                entry = {
                    "entry_signal_timestamp": signal_time,
                    "entry_timestamp": pd.Timestamp(execution["timestamp"]),
                    "entry_price": entry_price,
                    "entry_candidate_id": event.candidate_id,
                    "entry_bsp_type": event.bsp_type,
                    "entry_probability": float(event.signal_probability_used),
                    "entry_price_progress": getattr(event, "price_maturity_progress", np.nan),
                    "entry_predicted_reward_risk": getattr(event, "predicted_reward_risk_2d", np.nan),
                    "entry_predicted_max_loss": getattr(event, "predicted_max_loss_2d", np.nan),
                    "equity_before": equity,
                }
            elif event.direction == "sell" and entry is not None:
                exit_price = float(execution["open"]) * (1.0 - slippage)
                gross_return = exit_price / float(entry["entry_price"]) - 1.0
                net_return = (1.0 + gross_return) * (1.0 - commission) ** 2 - 1.0
                equity_after = equity * (1.0 + net_return)
                trades.append({
                    **entry,
                    "exit_signal_timestamp": signal_time,
                    "exit_timestamp": pd.Timestamp(execution["timestamp"]),
                    "exit_price": exit_price,
                    "exit_candidate_id": event.candidate_id,
                    "exit_bsp_type": event.bsp_type,
                    "exit_probability": float(event.signal_probability_used),
                    "gross_return": gross_return,
                    "net_return": net_return,
                    "pnl": equity_after - equity,
                    "equity_after": equity_after,
                    "exit_reason": "mature_sell",
                })
                equity = equity_after
                entry = None

    if entry is not None and force_exit_at_end:
        execution = price.iloc[-1]
        exit_price = float(execution["close"]) * (1.0 - slippage)
        gross_return = exit_price / float(entry["entry_price"]) - 1.0
        net_return = (1.0 + gross_return) * (1.0 - commission) ** 2 - 1.0
        equity_after = equity * (1.0 + net_return)
        trades.append({
            **entry,
            "exit_signal_timestamp": pd.NaT,
            "exit_timestamp": pd.Timestamp(execution["timestamp"]),
            "exit_price": exit_price,
            "exit_candidate_id": None,
            "exit_bsp_type": None,
            "exit_probability": np.nan,
            "gross_return": gross_return,
            "net_return": net_return,
            "pnl": equity_after - equity,
            "equity_after": equity_after,
            "exit_reason": "end_of_period",
        })
        equity = equity_after

    trades_frame = pd.DataFrame(trades)
    equity_curve = pd.DataFrame({
        "timestamp": [period_start], "equity": [float(initial_equity)]
    })
    if not trades_frame.empty:
        equity_curve = pd.concat([
            equity_curve,
            trades_frame[["exit_timestamp", "equity_after"]].rename(
                columns={"exit_timestamp": "timestamp", "equity_after": "equity"}
            ),
        ], ignore_index=True).sort_values("timestamp")

    fig, (ax_price, ax_equity) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0]}, constrained_layout=True,
    )
    ax_price.plot(price["timestamp"], price["close"], color="#334155", lw=0.9, label="Close")
    if not trades_frame.empty:
        ax_price.scatter(
            trades_frame["entry_timestamp"], trades_frame["entry_price"],
            marker="^", s=85, color="#16A34A", edgecolors="white", lw=0.6,
            label=f"Long entry ({len(trades_frame):,})", zorder=4,
        )
        ax_price.scatter(
            trades_frame["exit_timestamp"], trades_frame["exit_price"],
            marker="v", s=85, color="#DC2626", edgecolors="white", lw=0.6,
            label=f"Long exit ({len(trades_frame):,})", zorder=4,
        )
        for row in trades_frame.itertuples(index=False):
            ax_price.plot(
                [row.entry_timestamp, row.exit_timestamp],
                [row.entry_price, row.exit_price],
                color="#16A34A" if row.net_return >= 0 else "#DC2626",
                alpha=0.35, lw=1.0, zorder=2,
            )
    total_return = equity / float(initial_equity) - 1.0
    win_rate = (
        float(trades_frame["net_return"].gt(0).mean())
        if not trades_frame.empty else float("nan")
    )
    ax_price.set_title(
        f"Predicted BSP Long-Only Trades — {selected_probability_column} >= "
        f"{float(minimum_probability):.2f}, price progress >= "
        f"{float(minimum_price_progress):.2f}\n"
        f"trades={len(trades_frame):,}, total return={total_return:.2%}, "
        f"win rate={win_rate:.2%}"
    )
    ax_price.set_ylabel("Price")
    ax_price.grid(alpha=0.2)
    ax_price.legend(loc="best")
    ax_equity.step(
        equity_curve["timestamp"], equity_curve["equity"], where="post",
        color="#2563EB", lw=1.5,
    )
    ax_equity.set_ylabel("Equity")
    ax_equity.set_xlabel("Timestamp")
    ax_equity.grid(alpha=0.2)

    saved_path = None
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=180, bbox_inches="tight")
        saved_path = str(target.resolve())
    trades_path = None
    if trades_output_csv is not None:
        target = Path(trades_output_csv)
        target.parent.mkdir(parents=True, exist_ok=True)
        trades_frame.to_csv(target, index=False)
        trades_path = str(target.resolve())
    if show:
        plt.show()
    else:
        plt.close(fig)
    return {
        "figure": fig,
        "axes": (ax_price, ax_equity),
        "events": events,
        "trades": trades_frame,
        "equity_curve": equity_curve,
        "trade_count": int(len(trades_frame)),
        "initial_equity": float(initial_equity),
        "final_equity": float(equity),
        "total_return": float(total_return),
        "win_rate": win_rate,
        "probability_column": selected_probability_column,
        "minimum_price_progress": float(minimum_price_progress),
        "minimum_reward_risk": float(minimum_reward_risk),
        "maximum_predicted_loss": maximum_predicted_loss,
        "output_path": saved_path,
        "trades_output_csv": trades_path,
    }


RISK_LABEL_COLUMNS = {
    "risk_entry_timestamp", "risk_entry_price", "risk_horizon_end_timestamp",
    "target_future_high_2d", "target_future_low_2d",
    "target_entry_quality_2d",
    "target_max_gain_2d", "target_max_loss_2d",
    "target_max_favorable_2d", "target_max_adverse_2d",
}


def add_bsp_two_day_gain_loss_labels(
    bspoint_path: str | Path,
    price_csv: str | Path,
    *,
    output_csv: str | Path = "outputs/bsp_trade_labels_with_2d_risk.csv",
    sheet_name: str = "BSP Points",
    horizon_days: int = 2,
    price_progress_lookback_bars: int = 390,
    minimum_structure_pct: float = 0.002,
    technical_windows: tuple[int, ...] = DEFAULT_BSP_TECHNICAL_WINDOWS,
    technical_rsi_periods: tuple[int, ...] = DEFAULT_BSP_RSI_PERIODS,
    technical_atr_periods: tuple[int, ...] = DEFAULT_BSP_ATR_PERIODS,
    technical_macd_periods: tuple[tuple[int, int, int], ...] = DEFAULT_BSP_MACD_PERIODS,
    verbose: bool = True,
) -> dict[str, Any]:
    """Add executable-entry two-session gain/loss labels to BSP candidates.

    Entry is the next available bar's open after ``snapshot_timestamp``. The
    horizon includes that entry bar, the remainder of its session, and enough
    following complete sessions to total ``horizon_days``. Favorable/adverse
    labels are direction-adjusted so they have consistent meaning for buys and
    sells: favorable is nonnegative and adverse is nonpositive.
    """
    if int(horizon_days) <= 0:
        raise ValueError("horizon_days must be greater than zero")
    frame = _read_bsp_points(Path(bspoint_path), sheet_name).copy()
    frame = add_price_maturity_progress(
        frame, price_csv,
        reference_lookback_bars=price_progress_lookback_bars,
        minimum_structure_pct=minimum_structure_pct,
    )
    frame = add_multihorizon_technical_features_to_bspoints(
        frame, price_csv,
        windows=technical_windows,
        rsi_periods=technical_rsi_periods,
        atr_periods=technical_atr_periods,
        macd_periods=technical_macd_periods,
    )
    required = {"candidate_id", "snapshot_timestamp", "direction"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"BSP dataset is missing columns: {sorted(missing)}")
    frame["snapshot_timestamp"] = pd.to_datetime(
        frame["snapshot_timestamp"], errors="coerce"
    )
    frame["direction"] = frame["direction"].astype(str).str.lower().str.strip()

    price = pd.read_csv(price_csv, low_memory=False)
    normalized = {
        str(column).lower().strip().replace(" ", "_"): column
        for column in price.columns
    }
    timestamp_column = next(
        (normalized[name] for name in ("timestamp", "datetime", "date")
         if name in normalized), None,
    )
    open_column = next(
        (normalized[name] for name in ("open", "_open", "klu_open")
         if name in normalized), None,
    )
    high_column = next(
        (normalized[name] for name in ("high", "_high", "klu_high")
         if name in normalized), None,
    )
    low_column = next(
        (normalized[name] for name in ("low", "_low", "klu_low")
         if name in normalized), None,
    )
    if None in {timestamp_column, open_column, high_column, low_column}:
        raise KeyError("Price CSV must contain timestamp/date, open, high, and low")
    price = pd.DataFrame({
        "timestamp": pd.to_datetime(price[timestamp_column], errors="coerce"),
        "open": pd.to_numeric(price[open_column], errors="coerce"),
        "high": pd.to_numeric(price[high_column], errors="coerce"),
        "low": pd.to_numeric(price[low_column], errors="coerce"),
    }).dropna().sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    if price.empty:
        raise ValueError("Price CSV contains no usable OHLC rows")

    sessions = price["timestamp"].dt.normalize()
    unique_sessions = pd.Index(sessions.drop_duplicates())
    session_positions = pd.Series(np.arange(len(unique_sessions)), index=unique_sessions)
    price_session_position = sessions.map(session_positions).to_numpy(dtype=int)
    session_last_index = (
        pd.Series(np.arange(len(price))).groupby(price_session_position).max().to_dict()
    )
    timestamps = price["timestamp"].to_numpy(dtype="datetime64[ns]")
    opens = price["open"].to_numpy(dtype=float)
    highs = price["high"].to_numpy(dtype=float)
    lows = price["low"].to_numpy(dtype=float)

    entry_timestamps: list[Any] = []
    entry_prices: list[float] = []
    horizon_ends: list[Any] = []
    maximum_gains: list[float] = []
    maximum_losses: list[float] = []
    future_highs: list[float] = []
    future_lows: list[float] = []
    entry_qualities: list[float] = []
    favorable: list[float] = []
    adverse: list[float] = []
    for row in frame[["snapshot_timestamp", "direction"]].itertuples(index=False):
        if pd.isna(row.snapshot_timestamp):
            entry_index = len(price)
        else:
            entry_index = int(np.searchsorted(
                timestamps, pd.Timestamp(row.snapshot_timestamp).to_datetime64(), side="right"
            ))
        if entry_index >= len(price):
            values = (pd.NaT, np.nan, pd.NaT, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        else:
            entry_session = int(price_session_position[entry_index])
            final_session = entry_session + int(horizon_days) - 1
            if final_session >= len(unique_sessions):
                values = (pd.NaT, np.nan, pd.NaT, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
            else:
                final_index = int(session_last_index[final_session])
                entry_price = float(opens[entry_index])
                future_high = float(np.max(highs[entry_index:final_index + 1]))
                future_low = float(np.min(lows[entry_index:final_index + 1]))
                long_gain = max(0.0, future_high / entry_price - 1.0)
                long_loss = min(0.0, future_low / entry_price - 1.0)
                if str(row.direction) == "sell":
                    fav = max(0.0, 1.0 - future_low / entry_price)
                    adv = min(0.0, 1.0 - future_high / entry_price)
                else:
                    fav, adv = long_gain, long_loss
                future_range = future_high - future_low
                if future_range > 0:
                    range_location = np.clip(
                        (entry_price - future_low) / future_range, 0.0, 1.0
                    )
                    entry_quality = (
                        range_location if str(row.direction) == "sell"
                        else 1.0 - range_location
                    )
                else:
                    entry_quality = 0.5
                values = (
                    pd.Timestamp(timestamps[entry_index]), entry_price,
                    pd.Timestamp(timestamps[final_index]), future_high, future_low,
                    float(entry_quality), long_gain, long_loss, fav, adv,
                )
        (
            entry_ts, entry_price, horizon_end, future_high, future_low,
            entry_quality, gain, loss, fav, adv,
        ) = values
        entry_timestamps.append(entry_ts)
        entry_prices.append(entry_price)
        horizon_ends.append(horizon_end)
        future_highs.append(future_high)
        future_lows.append(future_low)
        entry_qualities.append(entry_quality)
        maximum_gains.append(gain)
        maximum_losses.append(loss)
        favorable.append(fav)
        adverse.append(adv)

    frame["risk_entry_timestamp"] = entry_timestamps
    frame["risk_entry_price"] = entry_prices
    frame["risk_horizon_end_timestamp"] = horizon_ends
    frame["target_future_high_2d"] = future_highs
    frame["target_future_low_2d"] = future_lows
    frame["target_entry_quality_2d"] = entry_qualities
    frame["target_max_gain_2d"] = maximum_gains
    frame["target_max_loss_2d"] = maximum_losses
    frame["target_max_favorable_2d"] = favorable
    frame["target_max_adverse_2d"] = adverse
    target = Path(output_csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    labeled = frame[[
        "target_entry_quality_2d", "target_max_favorable_2d",
        "target_max_adverse_2d",
    ]].notna().all(axis=1)
    result = {
        "output_csv": str(target.resolve()),
        "rows": int(len(frame)),
        "labeled_rows": int(labeled.sum()),
        "unlabeled_rows": int((~labeled).sum()),
        "horizon_days": int(horizon_days),
    }
    if verbose:
        print(
            f"[BSP 2d labels] rows={result['rows']:,}, "
            f"labeled={result['labeled_rows']:,}, "
            f"unlabeled={result['unlabeled_rows']:,}"
        )
        print(f"[BSP 2d labels] {result['output_csv']}")
    return result


def train_bsp_two_day_gain_loss_models(
    labeled_path: str | Path,
    *,
    output_dir: str | Path = "outputs/bsp_2d_gain_loss_models_by_type",
    train_start_date: str | None = None,
    train_end_date: str,
    validation_start_date: str,
    validation_end_date: str,
    test_start_date: str,
    test_end_date: str | None = None,
    bsp_types: tuple[str, ...] = ("1", "1p", "2", "2s", "3a", "3b"),
    xgboost_params: dict[str, Any] | None = None,
    minimum_rows_per_type: int = 200,
    include_delay_features: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train per-type regressors for future entry quality and gain/loss."""
    import json
    import joblib
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline import Pipeline
    from xgboost import XGBRegressor

    frame = _read_bsp_points(Path(labeled_path), "BSP Points").copy()
    required = {
        "bsp_type", "snapshot_timestamp", "risk_horizon_end_timestamp",
        "target_entry_quality_2d", "target_max_favorable_2d",
        "target_max_adverse_2d",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"BSP risk dataset is missing columns: {sorted(missing)}")
    frame["bsp_type"] = (
        frame["bsp_type"].astype(str).str.lower().str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    frame["snapshot_timestamp"] = pd.to_datetime(frame["snapshot_timestamp"], errors="coerce")
    frame["risk_horizon_end_timestamp"] = pd.to_datetime(
        frame["risk_horizon_end_timestamp"], errors="coerce"
    )
    regression_targets = (
        "target_entry_quality_2d",
        "target_max_favorable_2d",
        "target_max_adverse_2d",
    )
    for target in regression_targets:
        frame[target] = pd.to_numeric(frame[target], errors="coerce")
    frame = frame.dropna(subset=[
        "snapshot_timestamp", "risk_horizon_end_timestamp",
        *regression_targets,
    ]).copy()
    identity = {
        "candidate_id", "candidate_key", "timestamp", "bsp_timestamp",
        "snapshot_timestamp", "first_seen_timestamp", "snapshot_last_seen",
        "direction", "bsp_type", "bsp_types",
        "bi_is_sure", "segment_is_sure", "initial_source_is_sure",
    }
    nonstationary = {
        "klu_idx", "klu_open", "klu_high", "klu_low", "klu_close", "klu_volume",
        "snapshot_first_seen", "snapshot_age_bars", "revision_count",
        "risk_entry_price",
        "price_maturity_market_price", "price_maturity_reference_price",
        "price_maturity_progress_raw",
    }
    excluded = LEAKAGE_COLUMNS | RISK_LABEL_COLUMNS | identity | nonstationary
    if not include_delay_features:
        excluded |= {"is_delayed_bsp", "discovery_delay_bars", "discovery_delay_minutes"}
    features = [
        column for column in frame.select_dtypes(include=[np.number, "bool"]).columns
        if column not in excluded and "next_bi_return" not in str(column).lower()
    ]
    if not features:
        raise ValueError("No causal numeric BSP risk features were found")

    def inclusive_end(value):
        if value is None:
            return None
        stamp = pd.Timestamp(value)
        return stamp + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1) if stamp == stamp.normalize() else stamp

    train_end = inclusive_end(train_end_date)
    validation_end = inclusive_end(validation_end_date)
    test_end = inclusive_end(test_end_date)
    train_mask = frame["snapshot_timestamp"].le(train_end) & frame["risk_horizon_end_timestamp"].le(train_end)
    if train_start_date:
        train_mask &= frame["snapshot_timestamp"].ge(pd.Timestamp(train_start_date))
    validation_mask = (
        frame["snapshot_timestamp"].ge(pd.Timestamp(validation_start_date))
        & frame["snapshot_timestamp"].le(validation_end)
        & frame["risk_horizon_end_timestamp"].le(validation_end)
    )
    test_mask = frame["snapshot_timestamp"].ge(pd.Timestamp(test_start_date))
    if test_end is not None:
        test_mask &= frame["snapshot_timestamp"].le(test_end) & frame["risk_horizon_end_timestamp"].le(test_end)

    parameters = {
        "objective": "reg:squarederror", "n_estimators": 350,
        "max_depth": 4, "learning_rate": 0.03, "min_child_weight": 5,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0,
        "n_jobs": 1, "random_state": 42,
    }
    parameters.update(xgboost_params or {})
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"models": {}, "skipped": {}, "feature_count": len(features)}

    def metrics(truth, prediction):
        return {
            "rows": int(len(truth)),
            "mae": float(mean_absolute_error(truth, prediction)),
            "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
            "r2": float(r2_score(truth, prediction)),
            "actual_mean": float(np.mean(truth)),
            "predicted_mean": float(np.mean(prediction)),
        }

    for bsp_type in map(str.lower, bsp_types):
        typed = frame["bsp_type"].eq(bsp_type)
        splits = {
            "train": frame.loc[typed & train_mask],
            "validation": frame.loc[typed & validation_mask],
            "test": frame.loc[typed & test_mask],
        }
        if min(map(len, splits.values())) < int(minimum_rows_per_type):
            report["skipped"][bsp_type] = {name: int(len(part)) for name, part in splits.items()}
            continue
        type_features = [feature for feature in features if splits["train"][feature].notna().any()]
        models = {}
        type_report: dict[str, Any] = {}
        test_predictions = splits["test"][[
            "candidate_id", "snapshot_timestamp", "bsp_timestamp", "bsp_type",
            "direction", "risk_entry_timestamp", "risk_entry_price",
            "target_entry_quality_2d", "target_max_favorable_2d",
            "target_max_adverse_2d",
        ]].copy()
        for target in regression_targets:
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", XGBRegressor(**parameters)),
            ])
            y_train = splits["train"][target].to_numpy(dtype=float)
            encoded_train = -y_train if target == "target_max_adverse_2d" else y_train
            model.fit(splits["train"][type_features], encoded_train)
            models[target] = model
            target_report = {}
            for split_name, part in splits.items():
                truth = part[target].to_numpy(dtype=float)
                prediction = model.predict(part[type_features])
                if target == "target_max_adverse_2d":
                    prediction = -np.maximum(prediction, 0.0)
                elif target == "target_entry_quality_2d":
                    prediction = np.clip(prediction, 0.0, 1.0)
                else:
                    prediction = np.maximum(prediction, 0.0)
                target_report[split_name] = metrics(truth, prediction)
                if split_name == "test":
                    test_predictions[f"predicted_{target}"] = prediction
            type_report[target] = target_report
        test_predictions["predicted_reward_risk_2d"] = (
            test_predictions["predicted_target_max_favorable_2d"]
            / test_predictions["predicted_target_max_adverse_2d"].abs().clip(lower=1e-6)
        )
        test_predictions.to_csv(out / f"test_predictions_type_{bsp_type}.csv", index=False)
        joblib.dump({
            "artifact_version": 1, "bsp_type": bsp_type, "models": models,
            "features": type_features, "targets": list(regression_targets),
            "xgboost_params": parameters,
        }, out / f"gain_loss_model_type_{bsp_type}.joblib")
        pd.DataFrame({"feature": type_features}).to_csv(
            out / f"feature_manifest_type_{bsp_type}.csv", index=False
        )
        report["models"][bsp_type] = type_report
        if verbose:
            gain = type_report["target_max_favorable_2d"]["test"]
            loss = type_report["target_max_adverse_2d"]["test"]
            quality = type_report["target_entry_quality_2d"]["test"]
            print(
                f"[BSP 2d {bsp_type}] rows={gain['rows']:,}, "
                f"quality MAE={quality['mae']:.3f}, "
                f"gain MAE={gain['mae']:.3%}, loss MAE={loss['mae']:.3%}, "
                f"gain R2={gain['r2']:.3f}, loss R2={loss['r2']:.3f}"
            )
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame({"feature": features}).to_csv(out / "feature_manifest.csv", index=False)
    return report


def add_maturity_rate_and_two_day_risk_predictions(
    predictions_path: str | Path,
    risk_model_dir: str | Path,
    *,
    output_csv: str | Path = "outputs/bsp_maturity_rate_and_2d_risk.csv",
    minimum_reward_risk: float = 1.0,
    maximum_adverse_loss: float | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Predict future two-day entry quality and favorable/adverse excursion."""
    import joblib

    frame = _read_bsp_points(Path(predictions_path), "BSP Points").copy()
    required = {"bsp_type", "model_maturity_probability"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Prediction dataset is missing columns: {sorted(missing)}")
    frame["bsp_type"] = (
        frame["bsp_type"].astype(str).str.lower().str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    # Price progress is a causal input. Future entry quality is predicted by a
    # model trained against forward two-session price ranges.
    if "price_maturity_progress" not in frame.columns:
        raise KeyError(
            "Prediction dataset is missing price_maturity_progress. Recreate "
            "predictions with the price-progress models and price_csv."
        )
    frame["price_maturity_progress"] = pd.to_numeric(
        frame["price_maturity_progress"], errors="coerce"
    ).clip(0.0, 1.0)
    frame["predicted_entry_quality_2d"] = np.nan
    frame["predicted_max_gain_2d"] = np.nan
    frame["predicted_max_loss_2d"] = np.nan
    loaded = []
    model_path = Path(risk_model_dir)
    for artifact_path in sorted(model_path.glob("gain_loss_model_type_*.joblib")):
        artifact = joblib.load(artifact_path)
        bsp_type = str(artifact["bsp_type"]).lower()
        mask = frame["bsp_type"].eq(bsp_type)
        if not mask.any():
            continue
        features = list(artifact["features"])
        missing_features = [feature for feature in features if feature not in frame.columns]
        if missing_features:
            raise KeyError(f"Type {bsp_type} is missing risk features: {missing_features}")
        gain = artifact["models"]["target_max_favorable_2d"].predict(frame.loc[mask, features])
        adverse_magnitude = artifact["models"]["target_max_adverse_2d"].predict(frame.loc[mask, features])
        quality_model = artifact["models"].get("target_entry_quality_2d")
        if quality_model is None:
            raise KeyError(
                f"Type {bsp_type} artifact has no entry-quality model; retrain "
                "with train_bsp_two_day_gain_loss_models."
            )
        quality = quality_model.predict(frame.loc[mask, features])
        frame.loc[mask, "predicted_entry_quality_2d"] = np.clip(quality, 0.0, 1.0)
        frame.loc[mask, "predicted_max_gain_2d"] = np.maximum(gain, 0.0)
        frame.loc[mask, "predicted_max_loss_2d"] = -np.maximum(adverse_magnitude, 0.0)
        loaded.append(bsp_type)
    # Compatibility alias for older notebook plotting calls. It now means
    # predicted future entry quality, not is_sure or elapsed maturity.
    frame["predicted_maturity_rate"] = frame["predicted_entry_quality_2d"]
    frame["predicted_reward_risk_2d"] = (
        frame["predicted_max_gain_2d"]
        / frame["predicted_max_loss_2d"].abs().clip(lower=1e-6)
    )
    risk_ok = frame["predicted_reward_risk_2d"].ge(float(minimum_reward_risk))
    if maximum_adverse_loss is not None:
        if float(maximum_adverse_loss) <= 0:
            raise ValueError("maximum_adverse_loss must be a positive magnitude")
        risk_ok &= frame["predicted_max_loss_2d"].ge(-float(maximum_adverse_loss))
    frame["risk_control_pass"] = risk_ok.fillna(False)
    target = Path(output_csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    result = {
        "output_csv": str(target.resolve()), "rows": int(len(frame)),
        "risk_control_pass_rows": int(frame["risk_control_pass"].sum()),
        "risk_models_loaded": sorted(loaded), "predictions": frame,
    }
    if verbose:
        print(f"[BSP maturity/risk] risk models={result['risk_models_loaded']}")
        print(
            f"[BSP maturity/risk] rows={result['rows']:,}, "
            f"risk-pass={result['risk_control_pass_rows']:,}"
        )
        print(f"[BSP maturity/risk] {result['output_csv']}")
    return result


def plot_bsp_maturity_rate_and_two_day_risk(
    predictions_path: str | Path,
    price_csv: str | Path,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    bsp_types: tuple[str, ...] = ("1", "1p", "2", "2s", "3a", "3b"),
    direction: str | None = None,
    minimum_maturity_rate: float | None = None,
    minimum_entry_quality: float = 0.0,
    minimum_price_progress: float = 0.0,
    minimum_reward_risk: float = 0.0,
    require_risk_control_pass: bool = False,
    show_rejected: bool = True,
    max_risk_bars: int = 150,
    annotate_max: int = 20,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (22, 13),
    show: bool = True,
) -> dict[str, Any]:
    """Plot price maturity progress and predicted two-session gain/loss."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    if direction is not None and str(direction).lower() not in {"buy", "sell"}:
        raise ValueError("direction must be 'buy', 'sell', or None")
    if minimum_maturity_rate is not None:
        minimum_entry_quality = float(minimum_maturity_rate)
    if not 0.0 <= float(minimum_entry_quality) <= 1.0:
        raise ValueError("minimum_entry_quality must be between 0 and 1")
    if not 0.0 <= float(minimum_price_progress) <= 1.0:
        raise ValueError("minimum_price_progress must be between 0 and 1")
    if float(minimum_reward_risk) < 0:
        raise ValueError("minimum_reward_risk cannot be negative")
    if int(max_risk_bars) < 0 or int(annotate_max) < 0:
        raise ValueError("max_risk_bars and annotate_max cannot be negative")

    period_start = pd.Timestamp(start)
    period_end = pd.Timestamp(end)
    if period_end < period_start:
        raise ValueError("end must be on or after start")
    if period_end == period_end.normalize():
        period_end += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    frame = _read_bsp_points(Path(predictions_path), "BSP Points").copy()
    required = {
        "snapshot_timestamp", "bsp_type", "direction",
        "price_maturity_progress", "predicted_entry_quality_2d",
        "predicted_max_gain_2d",
        "predicted_max_loss_2d", "predicted_reward_risk_2d",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(
            f"Combined prediction dataset is missing columns: {sorted(missing)}. "
            "Run add_maturity_rate_and_two_day_risk_predictions first."
        )
    frame["snapshot_timestamp"] = pd.to_datetime(
        frame["snapshot_timestamp"], errors="coerce"
    )
    frame["bsp_type"] = (
        frame["bsp_type"].astype(str).str.lower().str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    frame["direction"] = frame["direction"].astype(str).str.lower().str.strip()
    for column in (
        "price_maturity_progress", "predicted_entry_quality_2d",
        "predicted_max_gain_2d",
        "predicted_max_loss_2d", "predicted_reward_risk_2d",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "price_maturity_progress" in frame.columns:
        frame["price_maturity_progress"] = pd.to_numeric(
            frame["price_maturity_progress"], errors="coerce"
        ).clip(0.0, 1.0)
    if "risk_control_pass" not in frame.columns:
        frame["risk_control_pass"] = False
    else:
        frame["risk_control_pass"] = (
            frame["risk_control_pass"].astype(str).str.lower()
            .isin({"1", "1.0", "true", "yes", "y"})
        )
    allowed_types = {str(value).lower() for value in bsp_types}
    base_mask = (
        frame["snapshot_timestamp"].between(period_start, period_end, inclusive="both")
        & frame["bsp_type"].isin(allowed_types)
        & frame["direction"].isin({"buy", "sell"})
        & frame["price_maturity_progress"].notna()
        & frame["predicted_entry_quality_2d"].notna()
        & frame["predicted_max_gain_2d"].notna()
        & frame["predicted_max_loss_2d"].notna()
    )
    if direction is not None:
        base_mask &= frame["direction"].eq(str(direction).lower())
    frame = frame.loc[base_mask].sort_values("snapshot_timestamp").copy()
    accepted_mask = (
        frame["predicted_entry_quality_2d"].ge(float(minimum_entry_quality))
        & frame["price_maturity_progress"].ge(float(minimum_price_progress))
        & frame["predicted_reward_risk_2d"].ge(float(minimum_reward_risk))
    )
    if require_risk_control_pass:
        accepted_mask &= frame["risk_control_pass"]
    frame["plot_selected"] = accepted_mask

    price = pd.read_csv(price_csv, low_memory=False)
    normalized = {
        str(column).lower().strip().replace(" ", "_"): column
        for column in price.columns
    }
    timestamp_column = next(
        (normalized[name] for name in ("timestamp", "datetime", "date")
         if name in normalized), None,
    )
    close_column = next(
        (normalized[name] for name in ("close", "_close", "klu_close")
         if name in normalized), None,
    )
    if timestamp_column is None or close_column is None:
        raise KeyError("Price CSV must contain timestamp/date and close columns")
    price = pd.DataFrame({
        "timestamp": pd.to_datetime(price[timestamp_column], errors="coerce"),
        "close": pd.to_numeric(price[close_column], errors="coerce"),
    }).dropna()
    price = price.loc[
        price["timestamp"].between(period_start, period_end, inclusive="both")
    ].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    if price.empty:
        raise ValueError("No price rows exist in the requested period")
    close_by_timestamp = price.set_index("timestamp")["close"]
    frame["signal_price"] = frame["snapshot_timestamp"].map(close_by_timestamp)
    missing_prices = frame["signal_price"].isna()
    if missing_prices.any():
        raise ValueError(
            f"{int(missing_prices.sum()):,} BSP rows have no exact close at snapshot time"
        )

    buy = frame["direction"].eq("buy")
    frame["predicted_target_price"] = np.where(
        buy,
        frame["signal_price"] * (1.0 + frame["predicted_max_gain_2d"]),
        frame["signal_price"] * (1.0 - frame["predicted_max_gain_2d"]),
    )
    frame["predicted_adverse_price"] = np.where(
        buy,
        frame["signal_price"] * (1.0 + frame["predicted_max_loss_2d"]),
        frame["signal_price"] * (1.0 - frame["predicted_max_loss_2d"]),
    )
    selected = frame.loc[frame["plot_selected"]].copy()
    rejected = frame.loc[~frame["plot_selected"]].copy()

    fig, (ax_price, ax_metrics) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.2]}, constrained_layout=True,
    )
    ax_price.plot(price["timestamp"], price["close"], color="#334155", lw=0.9, label="Close")
    if show_rejected and not rejected.empty:
        ax_price.scatter(
            rejected["snapshot_timestamp"], rejected["signal_price"],
            marker="x", s=22, color="#94A3B8", alpha=0.4,
            label=f"Below filters ({len(rejected):,})", zorder=2,
        )
    norm = Normalize(vmin=0.0, vmax=1.0)
    scatter_for_colorbar = None
    for point_direction, marker, edgecolor in (
        ("buy", "^", "#15803D"), ("sell", "v", "#B91C1C")
    ):
        group = selected.loc[selected["direction"].eq(point_direction)]
        if group.empty:
            continue
        scatter_for_colorbar = ax_price.scatter(
            group["snapshot_timestamp"], group["signal_price"],
            c=group["predicted_entry_quality_2d"], cmap="viridis", norm=norm,
            marker=marker, s=78, edgecolors=edgecolor, linewidths=1.0,
            label=f"Selected {point_direction} ({len(group):,})", zorder=5,
        )
    if scatter_for_colorbar is not None:
        colorbar = fig.colorbar(scatter_for_colorbar, ax=ax_price, pad=0.01)
        colorbar.set_label("Predicted 2-day entry quality")

    risk_rows = selected
    if 0 < int(max_risk_bars) < len(risk_rows):
        positions = np.linspace(0, len(risk_rows) - 1, int(max_risk_bars))
        risk_rows = risk_rows.iloc[np.unique(positions.astype(int))]
    elif int(max_risk_bars) == 0:
        risk_rows = risk_rows.iloc[0:0]
    for row in risk_rows.itertuples(index=False):
        ax_price.vlines(
            row.snapshot_timestamp,
            min(row.predicted_adverse_price, row.predicted_target_price),
            max(row.predicted_adverse_price, row.predicted_target_price),
            color="#64748B", alpha=0.28, lw=1.0, zorder=2,
        )
        ax_price.scatter(
            [row.snapshot_timestamp], [row.predicted_target_price],
            marker="_", s=55, color="#16A34A", zorder=3,
        )
        ax_price.scatter(
            [row.snapshot_timestamp], [row.predicted_adverse_price],
            marker="_", s=55, color="#DC2626", zorder=3,
        )

    annotation_rows = selected
    if 0 < int(annotate_max) < len(annotation_rows):
        positions = np.linspace(0, len(annotation_rows) - 1, int(annotate_max))
        annotation_rows = annotation_rows.iloc[np.unique(positions.astype(int))]
    elif int(annotate_max) == 0:
        annotation_rows = annotation_rows.iloc[0:0]
    for row in annotation_rows.itertuples(index=False):
        label = (
            f"{row.bsp_type} {row.direction} "
            f"Q={row.predicted_entry_quality_2d:.2f} "
            f"P={row.price_maturity_progress:.2f} "
            f"G={row.predicted_max_gain_2d:.1%} "
            f"L={row.predicted_max_loss_2d:.1%}"
        )
        color = "#15803D" if row.direction == "buy" else "#B91C1C"
        ax_price.annotate(
            label, (row.snapshot_timestamp, row.signal_price),
            xytext=(6, 10 if row.direction == "buy" else -16),
            textcoords="offset points", fontsize=8, color=color,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": color,
                  "alpha": 0.78, "lw": 0.5}, zorder=6,
        )
    ax_price.set_title(
        f"Predicted 2-Day BSP Entry Quality and Risk — {period_start} to {period_end}\n"
        f"selected={len(selected):,}, quality >= {float(minimum_entry_quality):.2f}, "
        f"progress >= {float(minimum_price_progress):.2f}, "
        f"reward/risk >= {float(minimum_reward_risk):.2f}"
    )
    ax_price.set_ylabel("Price")
    ax_price.grid(alpha=0.2)
    ax_price.legend(loc="best")

    ax_metrics.scatter(
        frame["snapshot_timestamp"], frame["predicted_entry_quality_2d"],
        s=18, color="#7C3AED", alpha=0.62, label="Predicted entry quality",
    )
    ax_metrics.scatter(
        frame["snapshot_timestamp"], frame["price_maturity_progress"],
        s=16, color="#0284C7", alpha=0.60, label="Price maturity progress",
    )
    ax_metrics.axhline(
        float(minimum_entry_quality), color="#7C3AED", ls="--", lw=1.0,
        label="Entry-quality threshold",
    )
    ax_metrics.scatter(
        selected["snapshot_timestamp"], selected["predicted_max_gain_2d"],
        s=18, color="#16A34A", alpha=0.65, label="Predicted max gain",
    )
    ax_metrics.scatter(
        selected["snapshot_timestamp"], selected["predicted_max_loss_2d"].abs(),
        s=18, color="#DC2626", alpha=0.65, label="Predicted max loss magnitude",
    )
    ax_metrics.set_ylim(bottom=0)
    ax_metrics.set_ylabel("Rate / return")
    ax_metrics.set_xlabel("Timestamp")
    ax_metrics.grid(alpha=0.2)
    ax_metrics.legend(loc="best", ncol=2)

    saved_path = None
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=180, bbox_inches="tight")
        saved_path = str(target.resolve())
    if show:
        plt.show()
    else:
        plt.close(fig)
    return {
        "figure": fig, "axes": (ax_price, ax_metrics), "rows": frame,
        "selected": selected, "rejected": rejected,
        "selected_rows": int(len(selected)), "rejected_rows": int(len(rejected)),
        "output_path": saved_path,
    }


__all__ = [
    "add_price_maturity_progress",
    "add_multihorizon_technical_features_to_bspoints",
    "LEAKAGE_COLUMNS",
    "label_bspoints_for_training",
    "plot_labeled_mature_bspoints",
    "train_bsp_trade_models",
    "create_predicted_mature_signals",
    "plot_predicted_mature_signals",
    "plot_bsp_trade_feature_importance",
    "backtest_and_plot_predicted_bsp_signals",
    "RISK_LABEL_COLUMNS",
    "add_bsp_two_day_gain_loss_labels",
    "train_bsp_two_day_gain_loss_models",
    "add_maturity_rate_and_two_day_risk_predictions",
    "plot_bsp_maturity_rate_and_two_day_risk",
]
