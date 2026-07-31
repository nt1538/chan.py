from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from pipelineCurrent import build_klu, load_ohlcv_csv


def plot_chan_bi_seg(
    data_path: str | Path,
    output_path: str | Path,
    *,
    code: str = "TQQQ",
    start: str = "2024-01-01",
    end: str = "2024-12-31 23:59:59",
    warmup_bars: int = 2000,
) -> Path:
    """Plot close, Chan Bi, and Chan segments without BSP markers."""
    raw = load_ohlcv_csv(str(data_path), "5M").sort_values("timestamp").reset_index(drop=True)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    warmup = raw.loc[raw["timestamp"] < start_ts].tail(warmup_bars)
    visible = raw.loc[raw["timestamp"].between(start_ts, end_ts)].copy()
    feed = pd.concat([warmup, visible], ignore_index=True)
    if visible.empty:
        raise ValueError(f"No price data between {start_ts} and {end_ts}")

    config = CChanConfig(
        {
            "trigger_step": True,
            "bi_strict": True,
            "bi_fx_check": "strict",
            "cal_rsi": False,
            "cal_kdj": False,
            "cal_dmi": False,
        }
    )
    chan = CChan(
        code=code,
        data_src=DATA_SRC.CSV,
        lv_list=[KL_TYPE.K_5M],
        config=config,
        autype=AUTYPE.QFQ,
    )
    for _, row in feed.iterrows():
        chan.trigger_load(
            {
                KL_TYPE.K_5M: [
                    build_klu(
                        row["timestamp"],
                        row["_open"],
                        row["_high"],
                        row["_low"],
                        row["_close"],
                        row.get("_vol", 0.0),
                    )
                ]
            }
        )

    kl = chan.kl_datas[KL_TYPE.K_5M]
    idx_to_time = {
        klu.idx: pd.Timestamp(str(klu.time))
        for klc in kl
        for klu in klc.lst
    }

    fig, ax = plt.subplots(figsize=(22, 9))
    ax.plot(visible["timestamp"], visible["_close"], color="#59636f", linewidth=0.65, label="Close")

    bi_count = 0
    for bi in kl.bi_list:
        x0 = idx_to_time.get(bi.get_begin_klu().idx)
        x1 = idx_to_time.get(bi.get_end_klu().idx)
        if x0 is None or x1 is None or x1 < start_ts or x0 > end_ts:
            continue
        ax.plot(
            [x0, x1],
            [bi.get_begin_val(), bi.get_end_val()],
            color="#3b82f6",
            linewidth=0.8,
            alpha=0.75,
            label="Bi" if bi_count == 0 else None,
            zorder=2,
        )
        bi_count += 1

    seg_count = 0
    for seg in kl.seg_list:
        x0 = idx_to_time.get(seg.get_begin_klu().idx)
        x1 = idx_to_time.get(seg.get_end_klu().idx)
        if x0 is None or x1 is None or x1 < start_ts or x0 > end_ts:
            continue
        ax.plot(
            [x0, x1],
            [seg.get_begin_val(), seg.get_end_val()],
            color="#f97316",
            linewidth=2.1,
            alpha=0.95,
            label="Segment" if seg_count == 0 else None,
            zorder=3,
        )
        seg_count += 1

    ax.set_xlim(start_ts, end_ts)
    ax.set_title(f"{code} 5-minute Chan structure — {start_ts.year}")
    ax.set_ylabel("Price")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output} with {bi_count} Bis and {seg_count} segments")
    return output


if __name__ == "__main__":
    plot_chan_bi_seg(
        "DataAPI/data/TQQQ_5M.csv",
        "outputs/TQQQ_2024_chan_bi_seg.png",
    )
