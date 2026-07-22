from __future__ import annotations

import pandas as pd


def _pick_col(df: pd.DataFrame, candidates):
    lookup = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        if str(candidate).lower() in lookup:
            return lookup[str(candidate).lower()]
    return None


def load_ohlcv_csv(path: str, freq_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts_col = _pick_col(df, ["timestamp", "date", "datetime", "time"]) or df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True).rename(columns={ts_col: "timestamp"})
    columns = {name: _pick_col(df, choices) for name, choices in {
        "open": ["open", "o"], "high": ["high", "h"], "low": ["low", "l"],
        "close": ["close", "adj_close", "adj close", "adjclose", "c"], "volume": ["volume", "vol", "v"],
    }.items()}
    if columns["open"] is None or columns["close"] is None:
        raise ValueError(f"{freq_name} CSV must contain open/close columns.")
    columns["high"] = columns["high"] or columns["close"]
    columns["low"] = columns["low"] or columns["close"]
    for target, source in (("_open", columns["open"]), ("_high", columns["high"]), ("_low", columns["low"]), ("_close", columns["close"])):
        df[target] = pd.to_numeric(df[source], errors="coerce").astype(float)
    df["_vol"] = pd.to_numeric(df[columns["volume"]], errors="coerce").astype(float) if columns["volume"] else 0.0
    df = df.dropna(subset=["_open", "_high", "_low", "_close"]).reset_index(drop=True)
    df["day"] = df["timestamp"].dt.normalize()
    return df
