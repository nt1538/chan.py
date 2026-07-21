from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from pipelineCurrent import build_klu, feed_chan_one, load_ohlcv_csv, normalize_bsp_row
from sliding_window_chan import SlidingWindowChan


class PlotSlidingWindowChan(SlidingWindowChan):
    """SlidingWindowChan variant that never loads CSV inside each window CChan."""

    def _create_window_chan(self):
        from Chan import CChan

        return CChan(
            code=self.code,
            begin_time=None,
            end_time=None,
            data_src=self.data_src,
            lv_list=self.lv_list,
            config=self.config,
            autype=self.autype,
        )


def _parse_dt(value: str | None) -> pd.Timestamp | None:
    if value is None or str(value).strip() == "":
        return None
    return pd.to_datetime(value)


def _kl_type(freq: str):
    freq = str(freq).lower()
    if freq in {"day", "daily", "d", "1d"}:
        return KL_TYPE.K_DAY
    if freq in {"5m", "5min", "5", "k_5m"}:
        return KL_TYPE.K_5M
    raise ValueError(f"Unsupported freq: {freq}. Use 'day' or '5m'.")


def _price_col(df: pd.DataFrame) -> str:
    for col in ("_close", "Close", "close", "price", "klu_close"):
        if col in df.columns:
            return col
    raise ValueError("Could not find a close/price column.")


def _filter_time(df: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"])
    if start is not None:
        out = out[out["timestamp"] >= start]
    if end is not None:
        out = out[out["timestamp"] <= end]
    return out.sort_values("timestamp").reset_index(drop=True)


def build_bspoints_from_kline(
    kline_csv: str | Path,
    *,
    code: str,
    freq: str,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    max_klines: int,
    warmup_bars: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recompute BSP points by feeding OHLCV rows into SlidingWindowChan.

    The returned price frame is filtered to [start, end].  The Chan feed includes
    warmup_bars before start, so BSP calculation has enough prior context.
    """
    raw = load_ohlcv_csv(str(kline_csv), freq.upper())
    raw = raw.sort_values("timestamp").reset_index(drop=True)

    if start is not None and warmup_bars > 0:
        before = raw[raw["timestamp"] < start].tail(int(warmup_bars))
        inside = raw[raw["timestamp"] >= start]
        feed_df = pd.concat([before, inside], ignore_index=True)
    else:
        feed_df = raw.copy()
    if end is not None:
        feed_df = feed_df[feed_df["timestamp"] <= end].copy()

    chan_config = CChanConfig({
        "trigger_step": True,
        "cal_rsi": True,
        "cal_kdj": True,
        "cal_dmi": True,
    })
    chan = PlotSlidingWindowChan(
        code=code,
        data_src=DATA_SRC.CSV,
        lv_list=[_kl_type(freq)],
        config=chan_config,
        autype=AUTYPE.QFQ,
        max_klines=int(max_klines),
    )

    rows = []
    for i, row in feed_df.reset_index(drop=True).iterrows():
        klu = build_klu(
            row["timestamp"],
            row["_open"],
            row["_high"],
            row["_low"],
            row["_close"],
            row.get("_vol", 0.0),
        )
        result = feed_chan_one(chan, klu)
        if isinstance(result, tuple) and len(result) == 2:
            _, new_rows = result
        else:
            new_rows = result or []
        for bsp in new_rows:
            rows.append(normalize_bsp_row(bsp))

    bsp_df = pd.DataFrame(rows)
    if not bsp_df.empty:
        bsp_df = _filter_time(bsp_df, start, end)

    price_df = _filter_time(raw, start, end)
    return price_df, bsp_df


def load_bspoints_from_signals(
    kline_csv: str | Path,
    signals_csv: str | Path,
    *,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load already-exported signal_decisions.csv as the point source."""
    price_df = _filter_time(load_ohlcv_csv(str(kline_csv), "KLINE"), start, end)
    sig = pd.read_csv(signals_csv)
    rename = {}
    if "ts" in sig.columns and "timestamp" not in sig.columns:
        rename["ts"] = "timestamp"
    if "side" in sig.columns and "direction" not in sig.columns:
        rename["side"] = "direction"
    if "price" in sig.columns and "klu_close" not in sig.columns:
        rename["price"] = "klu_close"
    sig = sig.rename(columns=rename)
    if "bsp_type" not in sig.columns:
        sig["bsp_type"] = sig.get("action", "?")
    sig = _filter_time(sig, start, end)
    return price_df, sig


def _scatter_group(
    ax,
    df: pd.DataFrame,
    *,
    direction: str,
    marker: str,
    color: str,
    label: str,
    annotate: bool,
):
    pts = df[df["direction"].astype(str).str.lower().eq(direction)].copy()
    if pts.empty:
        return
    y_col = "klu_close" if "klu_close" in pts.columns else _price_col(pts)
    ax.scatter(
        pts["timestamp"],
        pd.to_numeric(pts[y_col], errors="coerce"),
        marker=marker,
        s=70,
        color=color,
        edgecolor="black",
        linewidth=0.4,
        label=f"{label} ({len(pts)})",
        zorder=4,
    )
    if annotate:
        for _, row in pts.iterrows():
            labels = [str(row.get("bsp_type", "")), direction]
            bi_direction = row.get("bi_direction")
            segment_direction = row.get("segment_direction")
            if pd.notna(bi_direction) and str(bi_direction).strip():
                labels.append(f"Bi:{bi_direction}")
            if pd.notna(segment_direction) and str(segment_direction).strip():
                labels.append(f"Seg:{segment_direction}")
            text = " | ".join(labels)
            ax.annotate(
                text,
                (row["timestamp"], float(row[y_col])),
                xytext=(7, 8 if direction == "buy" else -12),
                textcoords="offset points",
                ha="left",
                fontsize=8,
                color=color,
            )


def plot_bspoints(
    price_df: pd.DataFrame,
    bsp_df: pd.DataFrame,
    *,
    title: str,
    output: str | Path,
    annotate: bool,
    show_volume: bool,
):
    if price_df.empty:
        raise ValueError("No price rows in the requested period.")

    price_col = _price_col(price_df)
    if show_volume and "_vol" in price_df.columns:
        fig, (ax, ax_vol) = plt.subplots(
            2,
            1,
            figsize=(15, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [4, 1]},
        )
    else:
        fig, ax = plt.subplots(figsize=(15, 7))
        ax_vol = None

    ax.plot(
        price_df["timestamp"],
        pd.to_numeric(price_df[price_col], errors="coerce"),
        color="#1f2937",
        linewidth=1.4,
        label="Close",
        zorder=2,
    )

    if not bsp_df.empty:
        _scatter_group(
            ax,
            bsp_df,
            direction="buy",
            marker="^",
            color="#16a34a",
            label="Buy BSP",
            annotate=annotate,
        )
        _scatter_group(
            ax,
            bsp_df,
            direction="sell",
            marker="v",
            color="#dc2626",
            label="Sell BSP",
            annotate=annotate,
        )

    ax.set_title(title)
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    if ax_vol is not None:
        ax_vol.bar(price_df["timestamp"], pd.to_numeric(price_df["_vol"], errors="coerce"), color="#94a3b8")
        ax_vol.set_ylabel("Volume")
        ax_vol.grid(True, alpha=0.2)

    x_axis = ax_vol if ax_vol is not None else ax
    x_axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\\n%H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_bsp_csv(bsp_df: pd.DataFrame, output: str | Path | None):
    if not output:
        return
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bsp_df.to_csv(output, index=False)


def main(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Plot all BSP points created in a time period on day or 5min K-lines."
    )
    parser.add_argument("--kline-csv", required=True, help="OHLCV CSV, e.g. DataAPI/data/TQQQ_5M.csv")
    parser.add_argument("--code", default="UNKNOWN", help="Ticker/code label used by Chan and chart title.")
    parser.add_argument("--freq", choices=["day", "5m"], default="5m", help="K-line frequency.")
    parser.add_argument("--start", help="Start datetime, e.g. 2026-06-01 or '2026-06-01 09:30'.")
    parser.add_argument("--end", help="End datetime.")
    parser.add_argument("--source", choices=["chan", "signals"], default="chan")
    parser.add_argument("--signals-csv", help="Existing signal_decisions.csv, used when --source signals.")
    parser.add_argument("--max-klines", type=int, default=500, help="SlidingWindowChan window size.")
    parser.add_argument("--warmup-bars", type=int, default=500, help="Bars before start used for Chan context.")
    parser.add_argument("--output", default="outputs/bspoints_plot.png", help="Output PNG path.")
    parser.add_argument("--bsp-csv-out", help="Optional CSV output for plotted BSP points.")
    parser.add_argument("--annotate", action="store_true", help="Annotate points with bsp_type.")
    parser.add_argument("--volume", action="store_true", help="Add volume subplot when volume exists.")
    args = parser.parse_args(argv)

    start = _parse_dt(args.start)
    end = _parse_dt(args.end)

    if args.source == "chan":
        price_df, bsp_df = build_bspoints_from_kline(
            args.kline_csv,
            code=args.code,
            freq=args.freq,
            start=start,
            end=end,
            max_klines=args.max_klines,
            warmup_bars=args.warmup_bars,
        )
    else:
        if not args.signals_csv:
            raise ValueError("--signals-csv is required when --source signals")
        price_df, bsp_df = load_bspoints_from_signals(
            args.kline_csv,
            args.signals_csv,
            start=start,
            end=end,
        )

    title = f"{args.code} {args.freq} BSP points"
    if start is not None or end is not None:
        title += f" ({start or ''} to {end or ''})"
    plot_bspoints(price_df, bsp_df, title=title, output=args.output, annotate=args.annotate, show_volume=args.volume)
    save_bsp_csv(bsp_df, args.bsp_csv_out)
    print(f"Saved plot: {Path(args.output).resolve()}")
    print(f"BSP points plotted: {len(bsp_df)}")
    if args.bsp_csv_out:
        print(f"Saved BSP CSV: {Path(args.bsp_csv_out).resolve()}")


if __name__ == "__main__":
    main()
