from __future__ import annotations

import os
import numpy as np
import pandas as pd

from FeaturePipeline.DailyFeatures import KLINE_KEYS, compute_daily_kline_features
from .OHLCVLoader import load_ohlcv_csv


def load_macro_features_from_folder(folder: str, files: dict[str, str], start: str) -> pd.DataFrame:
    output = None
    for prefix, filename in files.items():
        path = os.path.join(folder, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Macro file not found: {path}")
        prefix = str(prefix).lower().rstrip("_") + "_"
        frame = load_ohlcv_csv(path, prefix.upper())
        frame = frame[frame["timestamp"] >= pd.to_datetime(start)].reset_index(drop=True)
        frame["ts_norm"] = frame["timestamp"].dt.normalize()
        features = compute_daily_kline_features(frame)
        features[f"{prefix}level"] = features["_close"].astype(float)
        features = features[["ts_norm", f"{prefix}level", *KLINE_KEYS]].rename(columns={key: f"{prefix}{key}" for key in KLINE_KEYS})
        output = features if output is None else output.merge(features, on="ts_norm", how="outer")
    output = output if output is not None else pd.DataFrame(columns=["ts_norm"])
    for long_name, short_name, curve in (("us10y", "us5y", "yc_10y5y"), ("us10y", "us2y", "yc_10y2y")):
        long_col, short_col = f"{long_name}_level", f"{short_name}_level"
        if long_col in output and short_col in output:
            output[f"{curve}_level"] = output[long_col] - output[short_col]
            previous = output[f"{curve}_level"].shift(1)
            output[f"{curve}_ret1"] = (output[f"{curve}_level"] / (previous + 1e-12) - 1).replace([np.inf, -np.inf], np.nan).fillna(0)
            for window in (5, 10, 20, 40):
                output[f"{curve}_chg_{window}"] = (output[f"{curve}_level"] - output[f"{curve}_level"].shift(window)).replace([np.inf, -np.inf], np.nan).fillna(0)
    return output.sort_values("ts_norm").reset_index(drop=True)
