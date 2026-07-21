from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

import adaptive_reward_checkpoint_fresh as _fresh


LEGACY_DAILY_LOG_COLUMNS = [
    "date",
    "equity",
    "cash",
    "pos",
    "qty",
    "entry_px",
    "buy_th",
    "sell_th",
    "p_day",
    "p_day_source_date",
    "daily_action",
    "daily_buy_level",
    "daily_sell_level",
    "next_date_preview",
    "next_daily_action_preview",
    "next_p_day_preview",
    "next_p_day_source_date_preview",
    "next_daily_buy_level_preview",
    "next_daily_sell_level_preview",
    "next_decision_preview_note",
]

PREVIEW_NOTE = "Provisional: later daily/preopen data can change this next-day decision."


def _fmt_day(value: Any) -> Any:
    # Convert a value to a string in M/D/YYYY format if it can be parsed as a date, otherwise return the original value.
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return value
    return f"{ts.month}/{ts.day}/{ts.year}"


def _bundle_p_by_day(bundle: dict) -> dict[pd.Timestamp, float]:
    # Extracts the "p_by_day_str" dictionary from the bundle and converts its keys to normalized pd.Timestamp and values to float, filtering out any invalid dates.
    raw = bundle.get("p_by_day_str", {}) or {}
    return {
        pd.to_datetime(k).normalize(): float(v)
        for k, v in raw.items()
        if pd.notna(pd.to_datetime(k, errors="coerce"))
    }


def _latest_p_day_before(p_by_day: dict[pd.Timestamp, float], day: Any) -> tuple[float, Optional[pd.Timestamp]]:
    # Returns the latest p_day value and its source date from p_by_day that is strictly before the given day. If no such date exists, returns (np.nan, None).
    day_ts = pd.to_datetime(day, errors="coerce")
    if pd.isna(day_ts):
        return np.nan, None
    day_ts = day_ts.normalize()
    candidates = [
        pd.to_datetime(k).normalize()
        for k, value in (p_by_day or {}).items()
        if pd.to_datetime(k).normalize() < day_ts and np.isfinite(float(value))
    ]
    if not candidates:
        return np.nan, None
    source_day = max(candidates)
    return float(p_by_day[source_day]), source_day


def _last_preview_from_bundle(
    *,
    bundle: dict,
    next_date: pd.Timestamp,
    prev_buy_level: float,
    prev_sell_level: float,
) -> dict[str, Any]:
    # Computes the next-day action preview based on the bundle's p_by_day data and previous buy/sell levels. Returns a dictionary with the next action, p_day, source date, and buy/sell levels.
    p_by_day = _bundle_p_by_day(bundle)
    p_day, source_day = _latest_p_day_before(p_by_day, next_date)
    if not np.isfinite(float(p_day)):
        return {
            "next_daily_action_preview": np.nan,
            "next_p_day_preview": np.nan,
            "next_p_day_source_date_preview": None,
            "next_daily_buy_level_preview": np.nan,
            "next_daily_sell_level_preview": np.nan,
        }
    buy_level = (
        float(prev_buy_level)
        if np.isfinite(float(prev_buy_level))
        else float(bundle.get("static_buy_level", 0.20))
    )
    sell_level = (
        float(prev_sell_level)
        if np.isfinite(float(prev_sell_level))
        else float(bundle.get("static_sell_level", 0.30))
    )
    if float(p_day) <= buy_level:
        gate = "FORCE_BUY"
    elif float(p_day) >= sell_level:
        gate = "FORCE_SELL"
    else:
        gate = "FREE"
    return {
        "next_daily_action_preview": gate,
        "next_p_day_preview": float(p_day),
        "next_p_day_source_date_preview": source_day,
        "next_daily_buy_level_preview": buy_level,
        "next_daily_sell_level_preview": sell_level,
    }


def _legacy_daily_log_df(daily_log_df: pd.DataFrame, bundle: Optional[dict]) -> pd.DataFrame:
    # Converts the daily log DataFrame to the legacy 20-column format with next-day previews. If the input DataFrame is None or empty, returns an empty DataFrame with the legacy columns.
    if daily_log_df is None or daily_log_df.empty:
        return pd.DataFrame(columns=LEGACY_DAILY_LOG_COLUMNS)

    out = daily_log_df.copy()
    for col in LEGACY_DAILY_LOG_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    dates = pd.to_datetime(out["date"], errors="coerce")
    next_dates = dates + BDay(1)

    out["next_date_preview"] = next_dates
    out["next_daily_action_preview"] = out["daily_action"].shift(-1)
    out["next_p_day_preview"] = out["p_day"].shift(-1)
    out["next_p_day_source_date_preview"] = out["p_day_source_date"].shift(-1)
    out["next_daily_buy_level_preview"] = out["daily_buy_level"].shift(-1)
    out["next_daily_sell_level_preview"] = out["daily_sell_level"].shift(-1)

    if bundle is not None and len(out) > 0:
        last_idx = out.index[-1]
        last_preview = _last_preview_from_bundle(
            bundle=bundle,
            next_date=pd.to_datetime(next_dates.iloc[-1]),
            prev_buy_level=pd.to_numeric(out.loc[last_idx, "daily_buy_level"], errors="coerce"),
            prev_sell_level=pd.to_numeric(out.loc[last_idx, "daily_sell_level"], errors="coerce"),
        )
        for key, value in last_preview.items():
            out.loc[last_idx, key] = value

    out["next_decision_preview_note"] = PREVIEW_NOTE
    out = out[LEGACY_DAILY_LOG_COLUMNS].copy()

    for col in ["date", "p_day_source_date", "next_date_preview", "next_p_day_source_date_preview"]:
        out[col] = out[col].map(_fmt_day)

    return out


def _load_bundle(path: Optional[str]) -> Optional[dict]:
    # Loads a bundle dictionary from the given path using joblib. Returns None if the path is None, the file does not exist, or if any error occurs during loading.
    if not path:
        return None
    try:
        p = Path(path)
        if p.exists():
            bundle = _fresh.load_joblib(str(p))
            return bundle if isinstance(bundle, dict) else None
    except Exception:
        return None
    return None


def _rewrite_legacy_daily_log(
    *,
    res: dict,
    csv_path: str,
    bundle_path: Optional[str],
) -> dict:
    # Rewrites the daily log DataFrame in the result dictionary to the legacy 20-column format and saves it as a CSV file at the specified path. Returns the updated result dictionary.
    bundle = _load_bundle(bundle_path)
    legacy_df = _legacy_daily_log_df(res.get("daily_log_df"), bundle)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    legacy_df.to_csv(csv_path, index=False)
    res["daily_log_df"] = legacy_df
    return res


def build_adaptive_reward_snapshot(*, output_dir: str = "output_adaptive_reward_snapshot_build", snapshot_path: str, **kwargs) -> dict:
    """
    Legacy-output wrapper for the non-chronological snapshot builder.

    It calls adaptive_reward_checkpoint_fresh.build_adaptive_reward_snapshot and
    rewrites snapshot_daily_log.csv to the older 20-column preview schema.
    """
    res = _fresh.build_adaptive_reward_snapshot(
        output_dir=output_dir,
        snapshot_path=snapshot_path,
        **kwargs,
    )
    return _rewrite_legacy_daily_log(
        res=res,
        csv_path=os.path.join(output_dir, "snapshot_daily_log.csv"),
        bundle_path=res.get("snapshot_path", snapshot_path),
    )


def run_adaptive_reward_from_snapshot(*, output_dir: str = "output_adaptive_reward_resumed_fresh", snapshot_path: str, **kwargs) -> dict:
    """
    Legacy-output wrapper for the non-chronological snapshot resume.

    It calls adaptive_reward_checkpoint_fresh.run_adaptive_reward_from_snapshot and
    rewrites daily_log.csv to the older 20-column preview schema.
    """
    res = _fresh.run_adaptive_reward_from_snapshot(
        output_dir=output_dir,
        snapshot_path=snapshot_path,
        **kwargs,
    )
    return _rewrite_legacy_daily_log(
        res=res,
        csv_path=os.path.join(output_dir, "daily_log.csv"),
        bundle_path=res.get("continued_snapshot_path"),
    )


load_joblib = _fresh.load_joblib
save_joblib = _fresh.save_joblib
