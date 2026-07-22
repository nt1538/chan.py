from __future__ import annotations

import numpy as np
import pandas as pd

KLINE_KEYS = ["ret1", "ret_5", "ret_10", "ret_20", "ret_40", "vol_5", "vol_10", "vol_20", "vol_40", "atr_14", "range_over_atr", "close_pos", "gap", "above_ma_20", "above_ma_50", "above_ma_100", "slope40"]


def _safe_div(a, b, eps=1e-12):
    return a / (b + eps)


def compute_daily_kline_features(df_day: pd.DataFrame) -> pd.DataFrame:
    d = df_day.copy().sort_values("timestamp").reset_index(drop=True)
    o, h, low, close = (d[name].astype(float) for name in ("_open", "_high", "_low", "_close"))
    previous = close.shift(1)
    d["ret1"] = (_safe_div(close, previous) - 1).replace([np.inf, -np.inf], np.nan).fillna(0)
    d["tr"] = pd.concat([(h-low).abs(), (h-previous).abs(), (low-previous).abs()], axis=1).max(axis=1).fillna(0)
    d["atr_14"] = d["tr"].rolling(14).mean().bfill().fillna(0)
    d["range"] = (h-low).fillna(0)
    d["range_over_atr"] = _safe_div(d["range"], d["atr_14"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    d["close_pos"] = ((close-low)/(h-low).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(.5)
    d["gap"] = (_safe_div(o, previous)-1).replace([np.inf, -np.inf], np.nan).fillna(0)
    for window in (5, 10, 20, 40):
        d[f"ret_{window}"] = (_safe_div(close, close.shift(window))-1).replace([np.inf, -np.inf], np.nan).fillna(0)
        d[f"vol_{window}"] = d["ret1"].rolling(window).std().replace([np.inf, -np.inf], np.nan).fillna(0)
    for window in (20, 50, 100):
        d[f"above_ma_{window}"] = (_safe_div(close, close.rolling(window).mean())-1).replace([np.inf, -np.inf], np.nan).fillna(0)
    def slope(values):
        values = np.log(np.maximum(np.asarray(values, dtype=float), 1e-12)); time = np.arange(len(values), dtype=float)
        time -= time.mean(); values -= values.mean(); denominator = (time*time).sum()
        return 0.0 if denominator <= 0 else float((time*values).sum()/denominator)
    d["slope40"] = close.rolling(40).apply(slope, raw=False).fillna(0)
    return d


def make_kline_dict(row: pd.Series) -> dict[str, float]:
    return {f"k_{key}": float(row[key]) if key in row.index and np.isfinite(row[key]) else 0.0 for key in KLINE_KEYS}
