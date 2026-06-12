from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from adaptive_threshold_raw_walkforward import (
    _make_chan_config,
    evaluate_three_day_rewards_for_logging,
)
from adaptive_trade_extensions import (
    RollingThresholdConfig,
    make_threshold_grid,
    select_oracle_thresholds_from_daily_rewards,
)
from pipelineCurrent import (
    AUTYPE,
    CChanConfig,
    DATA_SRC,
    KL_TYPE,
    DailyProbState,
    ExecutionEngine,
    RetModelPack,
    SlidingWindowChan,
    build_klu,
    compute_buy_hold_equity,
    compute_chain_endpoints,
    compute_daily_kline_features,
    extract_bsp_rows_from_chan,
    feature_importance_from_lr,
    feed_chan_one,
    fit_prob_model_dicts,
    get_feature_columns,
    label_bestlookahead_for_ready_points,
    label_confirm_extreme,
    latest_bsp_dir_up_to,
    load_5m_index,
    load_macro_features_from_folder,
    load_ohlcv_csv,
    make_daily_features_one_model,
    make_ret_grid,
    normalize_bsp_row,
    pack_ret_modelpack_for_save,
    predict_prob,
    predict_ret,
    prepare_ml_dataset,
    regime_for_day_from_ends,
    save_joblib,
    load_joblib,
    train_models_two_sided_ret_only,
    unpack_ret_modelpack_from_load,
    choose_thresholds_global_realized,
)


SNAPSHOT_SCHEMA = "adaptive_reward_fresh_start_v1"


def _standardize_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV data into timestamp/Open/High/Low/Close/Volume columns."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            str(next((part for part in c if str(part).lower() not in {"", "none"}), c[0]))
            for c in out.columns
        ]
    out = out.reset_index()

    ts_col = None
    for c in out.columns:
        if str(c).lower() in {"timestamp", "datetime", "date"}:
            ts_col = c
            break
    if ts_col is None:
        ts_col = out.columns[0]

    rename = {ts_col: "timestamp"}
    for src, dst in [
        ("open", "Open"),
        ("high", "High"),
        ("low", "Low"),
        ("close", "Close"),
        ("adj close", "Close"),
        ("volume", "Volume"),
    ]:
        for c in out.columns:
            if str(c).strip().lower() == src:
                rename[c] = dst
                break

    out = out.rename(columns=rename)
    out = out.loc[:, ~out.columns.duplicated(keep="first")]
    if "Volume" not in out.columns:
        out["Volume"] = 0.0
    out = out[["timestamp", "Open", "High", "Low", "Close", "Volume"]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["timestamp", "Open", "High", "Low", "Close"])

    ts = out["timestamp"]
    try:
        if getattr(ts.dt, "tz", None) is not None:
            out["timestamp"] = ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
    except Exception:
        try:
            out["timestamp"] = ts.dt.tz_localize(None)
        except Exception:
            pass

    return out.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def fetch_yfinance_ohlcv(
    *,
    code: str,
    start: str,
    end: Optional[str] = None,
    interval: str = "5m",
) -> pd.DataFrame:
    """Download OHLCV bars from yfinance and normalize them for this pipeline."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance to fetch realtime Yahoo Finance bars.") from exc

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end) if end is not None else pd.Timestamp.now()
    if end_ts <= start_ts:
        return _standardize_ohlcv_frame(pd.DataFrame())

    raw = yf.download(
        code,
        start=start_ts,
        end=end_ts + pd.Timedelta(days=1),
        interval=interval,
        progress=False,
        auto_adjust=False,
        prepost=False,
    )
    return _standardize_ohlcv_frame(raw)


def _merge_ohlcv_csv_with_new_bars(csv_path: str, new_bars: pd.DataFrame, out_path: str) -> str:
    """Merge historical CSV bars with newly fetched bars, replacing overlapping fetched days."""
    old = _standardize_ohlcv_frame(pd.read_csv(csv_path))
    new = _standardize_ohlcv_frame(new_bars)
    if not new.empty:
        replace_days = set(pd.to_datetime(new["timestamp"]).dt.normalize())
        old = old[~pd.to_datetime(old["timestamp"]).dt.normalize().isin(replace_days)].copy()
    merged = (
        pd.concat([old, new], ignore_index=True)
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.stem}.{os.getpid()}.{pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')}.tmp")
    merged.to_csv(tmp_path, index=False)
    try:
        os.replace(tmp_path, target)
        return str(target)
    except OSError:
        fallback = target.with_name(f"{target.stem}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S_%f')}{target.suffix}")
        os.replace(tmp_path, fallback)
        return str(fallback)


def _with_signal_decision_date(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a date string to each signal decision so outputs can be split per day."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "ts" in out.columns:
        ts = pd.to_datetime(out["ts"], errors="coerce")
        out["date"] = ts.dt.strftime("%Y-%m-%d")
    return out


def _filter_frame_by_date_range(
    df: pd.DataFrame,
    *,
    column: str,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
) -> pd.DataFrame:
    """Filter a DataFrame by an inclusive normalized date range."""
    if df is None or df.empty or column not in df.columns:
        return pd.DataFrame() if df is None else df.copy()
    out = df.copy()
    ts = pd.to_datetime(out[column], errors="coerce").dt.normalize()
    mask = (ts >= pd.to_datetime(start_day).normalize()) & (ts <= pd.to_datetime(end_day).normalize())
    return out[mask].copy().reset_index(drop=True)


def _filter_frame_by_timestamp_range(
    df: pd.DataFrame,
    *,
    column: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    """Filter a DataFrame by an inclusive timestamp range."""
    if df is None or df.empty or column not in df.columns:
        return pd.DataFrame() if df is None else df.copy()
    out = df.copy()
    ts = pd.to_datetime(out[column], errors="coerce")
    mask = (ts >= pd.to_datetime(start_ts)) & (ts <= pd.to_datetime(end_ts))
    return out[mask].copy().reset_index(drop=True)


def _save_signal_decision_outputs(output_dir: str, signal_decisions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Persist 5m signal decisions cumulatively.

    Each realtime poll only returns newly processed signals. For live review we
    keep a cumulative output file plus one file per signal date.
    """
    current = _with_signal_decision_date(signal_decisions_df)
    cumulative_path = os.path.join(output_dir, "signal_decisions.csv")

    frames = []
    if os.path.exists(cumulative_path):
        try:
            existing = pd.read_csv(cumulative_path)
            if not existing.empty:
                frames.append(_with_signal_decision_date(existing))
        except Exception:
            pass
    if not current.empty:
        frames.append(current)

    if frames:
        out = pd.concat(frames, ignore_index=True, sort=False)
        key_cols = [c for c in ["ts", "side", "action", "pred", "th", "gate", "reason"] if c in out.columns]
        if key_cols:
            out = out.drop_duplicates(subset=key_cols, keep="last")
        out = out.sort_values([c for c in ["ts", "side", "action"] if c in out.columns]).reset_index(drop=True)
    else:
        out = current

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out.to_csv(cumulative_path, index=False)

    if not out.empty and "date" in out.columns:
        for day, day_df in out.groupby("date", dropna=True):
            safe_day = str(day).replace("-", "")
            day_df.to_csv(os.path.join(output_dir, f"signal_decisions_{safe_day}.csv"), index=False)

    return out


def make_execution_state_from_position(
    *,
    initial_capital: float,
    fee_pct: float = 0.0,
    qty: float = 0.0,
    entry_px: Optional[float] = None,
    cash: Optional[float] = None,
    entry_idx: Optional[int] = None,
    trades: Optional[List[Dict[str, Any]]] = None,
) -> dict:
    """Create an ExecutionEngine-compatible state from a flat account or known position."""
    qty = float(qty or 0.0)
    entry_px_val = None if entry_px is None else float(entry_px)
    if qty <= 0:
        return {
            "cash": float(initial_capital if cash is None else cash),
            "fee_pct": float(fee_pct),
            "pos": 0,
            "qty": 0.0,
            "entry_px": None,
            "entry_idx": None,
            "pending_order": None,
            "trades": copy.deepcopy(trades or []),
        }
    if entry_px_val is None:
        raise ValueError("entry_px is required when qty is positive.")
    cash_val = float(initial_capital) - qty * entry_px_val if cash is None else float(cash)
    return {
        "cash": cash_val,
        "fee_pct": float(fee_pct),
        "pos": 1,
        "qty": qty,
        "entry_px": entry_px_val,
        "entry_idx": entry_idx,
        "pending_order": None,
        "trades": copy.deepcopy(trades or []),
    }


def _read_csv_if_exists(path: str) -> pd.DataFrame:
    """Read a CSV when it exists; otherwise return an empty frame for comparisons."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _compare_csv_frames(left: pd.DataFrame, right: pd.DataFrame, sort_cols: Optional[List[str]] = None) -> dict:
    """Compare two CSV-shaped DataFrames after aligning common columns."""
    if left is None:
        left = pd.DataFrame()
    if right is None:
        right = pd.DataFrame()

    common_cols = [c for c in left.columns if c in right.columns]
    left_cmp = left[common_cols].copy() if common_cols else pd.DataFrame(index=left.index)
    right_cmp = right[common_cols].copy() if common_cols else pd.DataFrame(index=right.index)

    if sort_cols:
        cols = [c for c in sort_cols if c in common_cols]
        if cols:
            left_cmp = left_cmp.sort_values(cols).reset_index(drop=True)
            right_cmp = right_cmp.sort_values(cols).reset_index(drop=True)
        else:
            left_cmp = left_cmp.reset_index(drop=True)
            right_cmp = right_cmp.reset_index(drop=True)
    else:
        left_cmp = left_cmp.reset_index(drop=True)
        right_cmp = right_cmp.reset_index(drop=True)

    same_shape = left_cmp.shape == right_cmp.shape
    same_values = bool(same_shape and left_cmp.fillna("__NA__").astype(str).equals(right_cmp.fillna("__NA__").astype(str)))
    return {
        "matches": same_values,
        "left_rows": int(len(left)),
        "right_rows": int(len(right)),
        "common_columns": common_cols,
        "left_only_columns": [c for c in left.columns if c not in right.columns],
        "right_only_columns": [c for c in right.columns if c not in left.columns],
    }


def _last_finite_value(rows: List[Dict[str, Any]], key: str, default: float) -> float:
    """Return the newest finite value for key from a list of log dictionaries."""
    for row in reversed(rows or []):
        try:
            value = float(row.get(key, np.nan))
        except Exception:
            continue
        if np.isfinite(value):
            return value
    return float(default)


def _latest_p_day_before(p_by_day: Dict[pd.Timestamp, float], day: pd.Timestamp) -> tuple[float, Optional[pd.Timestamp]]:
    """Return the latest available prior-day p_day to prevent same-day leakage."""
    day = pd.to_datetime(day).normalize()
    candidates = [
        pd.to_datetime(k).normalize()
        for k, value in (p_by_day or {}).items()
        if pd.to_datetime(k).normalize() < day and np.isfinite(float(value))
    ]
    if not candidates:
        return np.nan, None
    source_day = max(candidates)
    return float(p_by_day[source_day]), source_day


def _default_daily_threshold_config() -> RollingThresholdConfig:
    """Return the default dynamic daily threshold search configuration."""
    # Daily model gate defaults: p_day below buy_level forces buy; above sell_level forces sell.
    return RollingThresholdConfig(
        lookback_days=15,
        buy_grid=make_threshold_grid(0.05, 0.35, 0.005),
        sell_grid=make_threshold_grid(0.15, 0.60, 0.005),
        min_gap=0.02,
        max_gap=0.05,
        min_obs=60,
        switch_penalty=0.0,
    )


def _normalize_daily_threshold_config(
    config: Optional[RollingThresholdConfig],
    *,
    lookback_days_override: Optional[int] = None,
    max_gap_override: Optional[float] = None,
) -> RollingThresholdConfig:
    """Copy a daily threshold config and apply live-safe defaults/overrides."""
    out = copy.deepcopy(config or _default_daily_threshold_config())
    if not hasattr(out, "max_gap") or getattr(out, "max_gap", None) is None:
        out.max_gap = 0.05
    if lookback_days_override is not None:
        out.lookback_days = int(lookback_days_override)
    if max_gap_override is not None:
        out.max_gap = float(max_gap_override)
    if int(out.lookback_days) < 1:
        raise ValueError("daily_threshold_lookback_days_override must be >= 1")
    if getattr(out, "max_gap", None) is not None and float(out.max_gap) < float(out.min_gap):
        raise ValueError("daily_threshold_max_gap_override must be >= min_gap")
    return out


def _default_5m_ret_threshold_grid() -> List[float]:
    """Return the default percent-return threshold grid for 5m model predictions."""
    # `best_return_pct` and model predictions are percent returns, not fractions.
    return make_ret_grid(-0.5, 2.5, 0.005)


def _coerce_5m_ret_threshold_grid(threshold_ret_grid) -> List[float]:
    """Normalize an optional caller/snapshot threshold grid into a non-empty float list."""
    # Keep caller-provided grids exactly, including the legacy -0.5..2.5 grid.
    # Only missing or empty grids fall back to the newer percent-return default.
    if threshold_ret_grid is None:
        return _default_5m_ret_threshold_grid()
    grid = [float(x) for x in threshold_ret_grid]
    if not grid:
        return _default_5m_ret_threshold_grid()
    return grid


def _copy_list_of_dicts(rows: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Deep-ish copy a list of row dictionaries before storing them in checkpoints."""
    if not rows:
        return []
    return [dict(r) for r in rows]


def _set_to_list(items) -> list:
    """Serialize set-like checkpoint fields into lists."""
    if items is None:
        return []
    return list(items)


def _list_to_set(items) -> set:
    """Deserialize checkpoint list fields back into sets, preserving tuple keys."""
    if items is None:
        return set()
    out = set()
    for item in items:
        if isinstance(item, list):
            out.add(tuple(item))
        else:
            out.add(item)
    return out


def _reward_action_for_gate(gate: str) -> str:
    """Convert non-forced daily gates to FREE for reward-map lookup."""
    if gate in {"FORCE_BUY", "FORCE_SELL"}:
        return gate
    return "FREE"


def _warm_chan_from_bars(chan_obj, bars: pd.DataFrame) -> Optional[pd.Timestamp]:
    """Replay saved warmup bars into a Chan object and return the final timestamp."""
    if bars is None or bars.empty:
        return None
    bars = bars.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], errors="coerce")
    bars = (
        bars.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )
    last_ts = None
    for _, rr in bars.iterrows():
        ts = pd.to_datetime(rr["timestamp"])
        feed_chan_one(
            chan_obj,
            build_klu(ts, rr["_open"], rr["_high"], rr["_low"], rr["_close"], rr.get("_vol", 0.0)),
        )
        last_ts = ts
    return last_ts


def _find_next_index_after(df: pd.DataFrame, ts: Optional[pd.Timestamp]) -> int:
    """Find the first bar index strictly after a checkpoint timestamp."""
    if ts is None or df.empty:
        return 0
    idx = df.index[pd.to_datetime(df["timestamp"]) > pd.to_datetime(ts)]
    return int(idx[0]) if len(idx) else len(df)


def _rows_up_to(rows: List[Dict[str, Any]], ts: pd.Timestamp, key: str = "timestamp") -> List[Dict[str, Any]]:
    """Copy saved event rows whose timestamp is at or before ts."""
    out = []
    cutoff = pd.to_datetime(ts)
    for row in rows or []:
        row_ts = pd.to_datetime(row.get(key), errors="coerce")
        if pd.isna(row_ts) or row_ts <= cutoff:
            out.append(dict(row))
    return out


def _daily_reward_log_up_to(rows: List[Dict[str, Any]], ts: pd.Timestamp) -> List[Dict[str, Any]]:
    """Copy daily reward rows through the checkpoint day."""
    out = []
    cutoff = pd.to_datetime(ts).normalize()
    for row in rows or []:
        row_date = pd.to_datetime(row.get("date"), errors="coerce")
        if pd.isna(row_date) or row_date.normalize() <= cutoff:
            out.append(dict(row))
    return _dedupe_daily_reward_log(out)


def _dedupe_daily_reward_log(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep one daily reward row per date, preserving the latest row for duplicates."""
    by_date: dict[str, Dict[str, Any]] = {}
    no_date: List[Dict[str, Any]] = []
    for row in rows or []:
        rr = dict(row)
        row_date = pd.to_datetime(rr.get("date"), errors="coerce")
        if pd.isna(row_date):
            no_date.append(rr)
            continue
        by_date[row_date.normalize().strftime("%Y-%m-%d")] = rr
    ordered = sorted(
        by_date.values(),
        key=lambda r: pd.to_datetime(r.get("date"), errors="coerce"),
    )
    return no_date + ordered


def _append_or_replace_daily_reward_log(rows: List[Dict[str, Any]], row: Dict[str, Any]) -> None:
    """Append a daily reward row, replacing any existing row for the same date."""
    row_date = pd.to_datetime(row.get("date"), errors="coerce")
    if pd.isna(row_date):
        rows.append(dict(row))
        return
    key = row_date.normalize()
    for idx in range(len(rows) - 1, -1, -1):
        existing_date = pd.to_datetime(rows[idx].get("date"), errors="coerce")
        if not pd.isna(existing_date) and existing_date.normalize() == key:
            rows[idx] = dict(row)
            return
    rows.append(dict(row))


def _seen_daily_from_rows(rows: List[Dict[str, Any]]) -> set:
    """Rebuild daily duplicate-detection keys from saved buy/sell point rows."""
    out = set()
    for row in rows or []:
        row_ts = pd.to_datetime(row.get("timestamp"), errors="coerce")
        if pd.isna(row_ts):
            continue
        out.add((row_ts.strftime("%Y-%m-%d"), row.get("direction"), row.get("bsp_type", "?")))
    return out


def _seen_5m_from_rows(rows: List[Dict[str, Any]]) -> set:
    """Rebuild 5m duplicate-detection keys from saved buy/sell point rows."""
    return {
        (int(row.get("klu_idx", -1)), str(row.get("direction")), str(row.get("bsp_type")))
        for row in rows or []
    }


def _make_adaptive_reward_snapshot_bundle(
    *,
    source_bundle: Optional[dict],
    code: str,
    snapshot_ts: pd.Timestamp,
    daily_csv_path: str,
    k5m_csv_path: str,
    daily_chan_start: str,
    accumulation_start: str,
    macro_files: dict,
    df_day_raw: pd.DataFrame,
    df_5m_raw: pd.DataFrame,
    daily_prob_model,
    daily_prob_trained_n: int,
    X_days: List[Dict[str, float]],
    y_days: List[int],
    pending_idx: List[int],
    p_by_day: Dict[pd.Timestamp, float],
    p_series: np.ndarray,
    dp_vs_minK_series: np.ndarray,
    dp_vs_maxK_series: np.ndarray,
    bsp_rows_daily: List[Dict[str, Any]],
    seen_bsp_daily: set,
    buy_pack: Optional[RetModelPack],
    sell_pack: Optional[RetModelPack],
    buy_ret_th_live: float,
    sell_ret_th_live: float,
    bsp_rows_5m: List[Dict[str, Any]],
    seen_keys_5m: set,
    daily_reward_log: List[Dict[str, Any]],
    daily_chan_max_klines: int,
    five_chan_max_klines: int,
    daily_threshold_config: RollingThresholdConfig,
    threshold_window_days: float,
    threshold_ret_grid,
    threshold_min_open_signals: int,
    lookahead_days_5m: float,
    retrain_every_days_5m: int,
    min_samples_total_5m: int,
    N_confirm: int,
    min_labeled_days_to_train: int,
    retrain_every_new_labels: int,
    dp_lookback: int,
    static_buy_level: float,
    static_sell_level: float,
    execution_engine_state: Optional[dict] = None,
) -> dict:
    """Package model state, learned rows, warmup bars, thresholds, and execution state into one checkpoint."""
    snapshot_ts = pd.to_datetime(snapshot_ts)
    warm_daily = df_day_raw[df_day_raw["timestamp"] <= snapshot_ts].tail(int(daily_chan_max_klines)).copy()
    warm_5m = df_5m_raw[df_5m_raw["timestamp"] <= snapshot_ts].tail(int(five_chan_max_klines)).copy()

    bundle = dict(source_bundle or {})
    bundle.update(
        {
            "schema": SNAPSHOT_SCHEMA,
            "code": code,
            "snapshot_time": str(snapshot_ts),
            "daily_csv_path": daily_csv_path,
            "k5m_csv_path": k5m_csv_path,
            "original_daily_chan_start": str(daily_chan_start),
            "original_accumulation_start": str(accumulation_start),
            "macro_files": copy.deepcopy(macro_files),
            "daily_prob_model": daily_prob_model,
            "daily_prob_trained_n": int(daily_prob_trained_n),
            "X_days": copy.deepcopy(X_days),
            "y_days": copy.deepcopy(y_days),
            "pending_idx": copy.deepcopy(pending_idx),
            "p_by_day_str": {str(pd.to_datetime(k)): float(v) for k, v in p_by_day.items()},
            "p_series": np.asarray(p_series, dtype=float),
            "dp_vs_minK_series": np.asarray(dp_vs_minK_series, dtype=float),
            "dp_vs_maxK_series": np.asarray(dp_vs_maxK_series, dtype=float),
            "bsp_rows_daily": _copy_list_of_dicts(bsp_rows_daily),
            "seen_bsp_daily_list": _set_to_list(seen_bsp_daily),
            "buy_pack": pack_ret_modelpack_for_save(buy_pack),
            "sell_pack": pack_ret_modelpack_for_save(sell_pack),
            "buy_ret_th_live": float(buy_ret_th_live),
            "sell_ret_th_live": float(sell_ret_th_live),
            "bsp_rows_5m": _copy_list_of_dicts(bsp_rows_5m),
            "seen_keys_5m_list": _set_to_list(seen_keys_5m),
            "daily_reward_log": _dedupe_daily_reward_log(daily_reward_log),
            "warmup_daily_bars": warm_daily[["timestamp", "_open", "_high", "_low", "_close", "_vol"]].reset_index(drop=True),
            "warmup_5m_bars": warm_5m[["timestamp", "_open", "_high", "_low", "_close", "_vol"]].reset_index(drop=True),
            "daily_chan_max_klines": int(daily_chan_max_klines),
            "five_chan_max_klines": int(five_chan_max_klines),
            "daily_threshold_config": copy.deepcopy(daily_threshold_config),
            "threshold_window_days": float(threshold_window_days),
            "threshold_ret_grid": _coerce_5m_ret_threshold_grid(threshold_ret_grid),
            "threshold_min_open_signals": int(threshold_min_open_signals),
            "lookahead_days_5m": float(lookahead_days_5m),
            "retrain_every_days_5m": int(retrain_every_days_5m),
            "min_samples_total_5m": int(min_samples_total_5m),
            "N_confirm": int(N_confirm),
            "min_labeled_days_to_train": int(min_labeled_days_to_train),
            "retrain_every_new_labels": int(retrain_every_new_labels),
            "dp_lookback": int(dp_lookback),
            "static_buy_level": float(static_buy_level),
            "static_sell_level": float(static_sell_level),
            "execution_engine_state": copy.deepcopy(execution_engine_state),
        }
    )
    return bundle


def _autosave_year_start_snapshots(
    *,
    output_dir: str,
    source_path: str,
    source_bundle: Optional[dict],
    data: dict,
    code: str,
    daily_csv_path: str,
    k5m_csv_path: str,
    daily_chan_start: str,
    accumulation_start: str,
    daily_prob_model,
    daily_prob_trained_n: int,
    X_days: List[Dict[str, float]],
    y_days: List[int],
    pending_idx: List[int],
    p_by_day: Dict[pd.Timestamp, float],
    p_series: np.ndarray,
    dp_vs_minK_series: np.ndarray,
    dp_vs_maxK_series: np.ndarray,
    bsp_rows_daily: List[Dict[str, Any]],
    buy_pack: Optional[RetModelPack],
    sell_pack: Optional[RetModelPack],
    buy_ret_th_live: float,
    sell_ret_th_live: float,
    bsp_rows_5m: List[Dict[str, Any]],
    daily_reward_log: List[Dict[str, Any]],
    daily_chan_max_klines: int,
    five_chan_max_klines: int,
    daily_threshold_config: RollingThresholdConfig,
    threshold_window_days: float,
    threshold_ret_grid,
    threshold_min_open_signals: int,
    lookahead_days_5m: float,
    retrain_every_days_5m: int,
    min_samples_total_5m: int,
    N_confirm: int,
    min_labeled_days_to_train: int,
    retrain_every_new_labels: int,
    dp_lookback: int,
    static_buy_level: float,
    static_sell_level: float,
    start_after: pd.Timestamp,
    end_time: str,
    verbose: bool,
) -> List[str]:
    """Avoid saving forward-trained year-start checkpoints from a completed run."""
    if verbose:
        print(
            "[CHECKPOINT] skipped year-start autosaves: a full-run model cannot be "
            "rewound safely without leaking later training data into earlier checkpoints"
        )
    return []


def _build_data_views(
    daily_csv_path: str,
    k5m_csv_path: str,
    daily_chan_start: str,
    accumulation_start: str,
    end_time: str,
    macro_files: Optional[dict],
):
    """Load raw CSVs and derive the daily/5m views consumed by the adaptive phases."""
    if macro_files is None:
        macro_files = {"vix_": "VIX.csv"}

    df_day_raw = load_ohlcv_csv(daily_csv_path, "DAILY")
    df_5m_raw = load_ohlcv_csv(k5m_csv_path, "5M")

    daily_s = pd.to_datetime(daily_chan_start)
    acc_s = pd.to_datetime(accumulation_start)
    end_t = pd.to_datetime(end_time)

    df_day = df_day_raw[(df_day_raw["timestamp"] >= daily_s) & (df_day_raw["timestamp"] <= end_t)].copy().reset_index(drop=True)
    if df_day.empty:
        raise ValueError("No daily bars in requested range.")

    df_day_feat = compute_daily_kline_features(df_day)
    df_day_feat["ts_norm"] = pd.to_datetime(df_day_feat["timestamp"]).dt.normalize()

    macro_folder = os.path.dirname(os.path.abspath(daily_csv_path))
    macro_feat = load_macro_features_from_folder(folder=macro_folder, files=macro_files, start=daily_chan_start)
    df_day_feat = df_day_feat.merge(macro_feat, on="ts_norm", how="left").sort_values("timestamp").reset_index(drop=True)
    macro_cols = [c for c in df_day_feat.columns if any(c.startswith(pref) for pref in macro_files.keys())]

    df_5m = df_5m_raw[(df_5m_raw["timestamp"] >= acc_s) & (df_5m_raw["timestamp"] <= end_t + pd.Timedelta(days=1))].copy()
    df_5m = df_5m.sort_values("timestamp").reset_index(drop=True)
    df_5m_idx, next_open_by_idx, _, closes, highs, lows, day_close_map, all_days = load_5m_index(
        df_5m,
        accumulation_start,
        end_time,
    )

    return {
        "df_day_raw": df_day_raw,
        "df_5m_raw": df_5m_raw,
        "df_day_feat": df_day_feat,
        "df_5m_idx": df_5m_idx,
        "next_open_by_idx": next_open_by_idx,
        "closes": closes,
        "highs": highs,
        "lows": lows,
        "day_close_map": day_close_map,
        "all_days": all_days,
        "macro_cols": macro_cols,
        "macro_files": macro_files,
    }


def _run_daily_phase(
    *,
    df_day_feat: pd.DataFrame,
    daily_chan,
    macro_cols: List[str],
    N_confirm: int,
    min_labeled_days_to_train: int,
    retrain_every_new_labels: int,
    dp_lookback: int,
    verbose: bool,
    st: DailyProbState,
    bsp_rows_daily: List[Dict[str, Any]],
    seen_bsp_daily: set,
    X_days: List[Dict[str, float]],
    y_days: List[int],
    pending_idx: List[int],
    p_series: np.ndarray,
    dp_vs_minK_series: np.ndarray,
    dp_vs_maxK_series: np.ndarray,
    p_by_day: Dict[pd.Timestamp, float],
    start_idx: int = 0,
) -> None:
    """Advance daily Chan/model state and emit p_day probabilities without lookahead leakage."""
    for i in range(start_idx, len(df_day_feat)):
        r = df_day_feat.loc[i]
        ts = pd.to_datetime(r["timestamp"])
        day = ts.normalize()
        daily_chan.process_new_kline(build_klu(ts, r["_open"], r["_high"], r["_low"], r["_close"], r["_vol"]))

        for rr0 in extract_bsp_rows_from_chan(daily_chan) or []:
            rr = normalize_bsp_row(dict(rr0))
            rr.setdefault("timestamp", ts)
            key = (pd.to_datetime(rr["timestamp"]).strftime("%Y-%m-%d"), rr["direction"], rr.get("bsp_type", "?"))
            if key not in seen_bsp_daily:
                seen_bsp_daily.add(key)
                bsp_rows_daily.append(rr)

        ends = compute_chain_endpoints(bsp_rows_daily)
        regime = regime_for_day_from_ends(day, ends)
        base_dir_today = latest_bsp_dir_up_to(bsp_rows_daily, ts)

        p_val = np.nan
        if st.model is not None:
            bsp_hist = [b for b in bsp_rows_daily if pd.to_datetime(b["timestamp"]) <= ts]
            feat_i = make_daily_features_one_model(
                kline_row=r,
                bsp_hist_up_to_day=bsp_hist,
                p_val=0.0,
                dp_minK=0.0,
                dp_maxK=0.0,
                regime=regime,
                base_dir=base_dir_today,
                macro_cols=macro_cols,
            )
            p_val = float(predict_prob(st.model, [feat_i])[0])
            p_series[i] = p_val
            p_by_day[day] = p_val

        lb = int(dp_lookback)
        if lb > 0 and i >= 1 and np.isfinite(p_val):
            prev = pd.Series(p_series[max(0, i - lb):i]).dropna()
            if len(prev) > 0:
                dp_vs_minK_series[i] = p_val - float(prev.min())
                dp_vs_maxK_series[i] = p_val - float(prev.max())

        pending_idx.append(i)
        # Labels are only attached after N_confirm future daily bars are available.
        # This keeps the daily classifier from seeing information from the current day.
        while pending_idx and i >= pending_idx[0] + int(N_confirm):
            j = pending_idx.pop(0)
            t0 = pd.to_datetime(df_day_feat.loc[j, "timestamp"])
            base_dir_j = latest_bsp_dir_up_to(bsp_rows_daily, t0)
            if base_dir_j not in ("buy", "sell"):
                continue
            y = label_confirm_extreme(df_day_feat, j, int(N_confirm), base_dir_j)
            if y is None:
                continue
            ends_j = compute_chain_endpoints([b for b in bsp_rows_daily if pd.to_datetime(b["timestamp"]) <= t0])
            regime_j = regime_for_day_from_ends(t0.normalize(), ends_j)
            bsp_hist_j = [b for b in bsp_rows_daily if pd.to_datetime(b["timestamp"]) <= t0]
            feat_j = make_daily_features_one_model(
                kline_row=df_day_feat.loc[j],
                bsp_hist_up_to_day=bsp_hist_j,
                p_val=float(p_series[j]) if np.isfinite(p_series[j]) else 0.0,
                dp_minK=float(dp_vs_minK_series[j]) if np.isfinite(dp_vs_minK_series[j]) else 0.0,
                dp_maxK=float(dp_vs_maxK_series[j]) if np.isfinite(dp_vs_maxK_series[j]) else 0.0,
                regime=regime_j,
                base_dir=base_dir_j,
                macro_cols=macro_cols,
            )
            X_days.append(feat_j)
            y_days.append(int(y))
            st.new_labels += 1

        if len(y_days) >= int(min_labeled_days_to_train) and (st.model is None or st.new_labels >= int(retrain_every_new_labels)):
            y_arr = np.asarray(y_days, dtype=int)
            if len(np.unique(y_arr)) >= 2:
                st.model = fit_prob_model_dicts(X_days, y_arr)
                st.trained_n = len(y_arr)
                st.new_labels = 0
                if verbose:
                    print(f"[TRAIN][DAILY-PROB] n={len(y_arr)} pos={int(y_arr.sum())} ({y_arr.mean():.2%})")


def _run_adaptive_reward_5m_phase(
    *,
    df_5m_raw: pd.DataFrame,
    df_5m_idx: pd.DataFrame,
    next_open_by_idx: np.ndarray,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    day_close_map: dict,
    p_by_day: Dict[pd.Timestamp, float],
    snapshot_or_end_time: str,
    chan_5m,
    buy_pack: Optional[RetModelPack],
    sell_pack: Optional[RetModelPack],
    bsp_rows_5m: List[Dict[str, Any]],
    seen_keys_5m: set,
    daily_reward_log: List[Dict[str, Any]],
    initial_capital: float,
    fee_pct: float,
    threshold_window_days: float,
    threshold_ret_grid,
    threshold_min_open_signals: int,
    lookahead_days_5m: float,
    retrain_every_days_5m: int,
    min_samples_total_5m: int,
    daily_threshold_config: RollingThresholdConfig,
    static_buy_level: float,
    static_sell_level: float,
    buy_ret_th_live: float,
    sell_ret_th_live: float,
    sim_start: Optional[str],
    verbose: bool,
    trade_start: Optional[str] = None,
    start_idx: int = 0,
    execution_engine_state: Optional[dict] = None,
) -> dict:
    """Run the 5m trading simulation while daily gates control which intraday signals can execute."""
    if threshold_ret_grid is None:
        threshold_ret_grid = _default_5m_ret_threshold_grid()
    else:
        threshold_ret_grid = _coerce_5m_ret_threshold_grid(threshold_ret_grid)
    timestamp_to_idx = {
        pd.to_datetime(ts): int(idx)
        for idx, ts in enumerate(pd.to_datetime(df_5m_idx["timestamp"]))
    }

    engine = ExecutionEngine(initial_capital=initial_capital, fee_pct=fee_pct)
    if execution_engine_state:
        engine.load_state_dict(execution_engine_state)
    last_train_day = None
    last_day_end_idx = None
    current_day = None
    day_gate = "FREE"
    allow_buy = True
    allow_sell = True
    must_trade_dir = None
    day_start_engine_state = None
    day_events_today: List[Dict[str, Any]] = []
    day_start_idx = None
    daily_log = []
    signal_decisions = []
    equity_peak = initial_capital
    oracle_equity = initial_capital
    current_buy_level = float(static_buy_level)
    current_sell_level = float(static_sell_level)
    current_p_day = np.nan
    current_p_day_source_date = None
    buy_ret_th_live = float(buy_ret_th_live)
    sell_ret_th_live = float(sell_ret_th_live)
    daily_reward_log = _dedupe_daily_reward_log(daily_reward_log)
    fallback_daily_decision = None
    sim_start_ts = None if sim_start is None else pd.to_datetime(sim_start)
    trade_start_ts = None if trade_start is None else pd.to_datetime(trade_start)
    if trade_start_ts is not None and getattr(trade_start_ts, "tzinfo", None) is not None:
        trade_start_ts = trade_start_ts.tz_convert(None)

    if trade_start_ts is not None and engine.pending_order is not None:
        pending_ts = pd.to_datetime((engine.pending_order.get("meta") or {}).get("ts"), errors="coerce")
        if not pd.isna(pending_ts) and getattr(pending_ts, "tzinfo", None) is not None:
            pending_ts = pending_ts.tz_convert(None)
        if pd.isna(pending_ts) or pending_ts < trade_start_ts:
            engine.pending_order = None

    def maybe_retrain_5m(day_ts: pd.Timestamp):
        """Retrain 5m return models when enough new labeled rows have accumulated."""
        nonlocal buy_pack, sell_pack, last_train_day
        if last_train_day is not None and (day_ts - last_train_day).days < int(retrain_every_days_5m):
            return
        dfb = pd.DataFrame(bsp_rows_5m)
        if dfb.empty:
            return
        dfb2 = prepare_ml_dataset(dfb)
        feat_cols = get_feature_columns(dfb2)
        bp, sp = train_models_two_sided_ret_only(dfb2, feat_cols, min_samples_total=min_samples_total_5m)
        if bp is not None:
            buy_pack = bp
        if sp is not None:
            sell_pack = sp
        if (bp is not None) or (sp is not None):
            last_train_day = day_ts
            if verbose:
                print(
                    f"[TRAIN][5M] asof={day_ts.date()} feats={len(feat_cols)} "
                    f"buy={'YES' if bp else 'NO'} sell={'YES' if sp else 'NO'} rows={len(dfb2)}"
                )

    def maybe_opt_5m_thresholds(asof_bar_idx: int):
        """Refresh live 5m buy/sell thresholds from recent realized signal outcomes."""
        nonlocal buy_ret_th_live, sell_ret_th_live
        if buy_pack is None or sell_pack is None:
            return
        # Optimize 5m buy/sell prediction thresholds from recent realized outcomes.
        out = choose_thresholds_global_realized(
            df_5m=df_5m_idx,
            bsp_rows=bsp_rows_5m,
            buy_pack=buy_pack,
            sell_pack=sell_pack,
            asof_bar_idx=asof_bar_idx,
            window_days=threshold_window_days,
            ret_grid=threshold_ret_grid,
            next_open_by_idx=next_open_by_idx,
            closes=closes,
            fee_pct=fee_pct,
            min_open_signals=threshold_min_open_signals,
        )
        if out is not None:
            buy_ret_th_live, sell_ret_th_live = out

    def choose_daily_gate_for_day(bar_day: pd.Timestamp) -> dict:
        """Fit rolling daily thresholds and choose today's daily gate from prior p_day."""
        nonlocal current_buy_level, current_sell_level
        p_day_val, p_day_source_date = _latest_p_day_before(p_by_day, bar_day)
        hist_df = pd.DataFrame(daily_reward_log)
        # Dynamic daily thresholds are fitted on the prior reward log, then
        # today's p_day is mapped into FORCE_BUY, FREE, or FORCE_SELL.
        out = select_oracle_thresholds_from_daily_rewards(
            history_df=hist_df,
            current_p_day=p_day_val,
            config=daily_threshold_config,
            prev_buy_level=current_buy_level,
            prev_sell_level=current_sell_level,
            objective="reward",
        )
        current_buy_level = out.buy_level
        current_sell_level = out.sell_level
        return {
            "gate": out.gate,
            "buy_level": out.buy_level,
            "sell_level": out.sell_level,
            "p_day": p_day_val,
            "p_day_source_date": p_day_source_date,
        }

    def begin_day(bar_day: pd.Timestamp, bar_idx: int):
        """Reset per-day state and apply the selected daily gate restrictions."""
        nonlocal day_gate, allow_buy, allow_sell, must_trade_dir
        nonlocal day_start_engine_state, day_events_today, day_start_idx
        nonlocal current_p_day, current_p_day_source_date
        info = choose_daily_gate_for_day(bar_day)
        day_gate = info["gate"]
        current_p_day = info["p_day"]
        current_p_day_source_date = info["p_day_source_date"]
        allow_buy = True
        allow_sell = True
        must_trade_dir = None
        if day_gate == "FORCE_BUY":
            # FORCE_BUY disables sells for the day and waits for the first acceptable buy signal if flat.
            allow_sell = False
            if engine.pos == 0:
                must_trade_dir = "buy"
        elif day_gate == "FORCE_SELL":
            # FORCE_SELL disables buys for the day and waits for the first acceptable sell signal if long.
            allow_buy = False
            if engine.pos == 1:
                must_trade_dir = "sell"
        day_start_engine_state = copy.deepcopy(engine.state_dict())
        day_events_today = []
        day_start_idx = int(bar_idx)
        return info

    def fallback_decision_from_last_5m_bar() -> Optional[dict]:
        """Build a status row when no new 5m bars were processed in a resume/live call."""
        if df_5m_idx.empty:
            return None
        fallback_idx = len(df_5m_idx) - 1
        fallback_ts = pd.to_datetime(df_5m_idx.loc[fallback_idx, "timestamp"])
        fallback_day = fallback_ts.normalize()
        decision_day = trade_start_ts.normalize() if trade_start_ts is not None else fallback_day
        prev_buy_level = _last_finite_value(daily_reward_log, "buy_level", current_buy_level)
        prev_sell_level = _last_finite_value(daily_reward_log, "sell_level", current_sell_level)
        p_day_val, p_day_source_date = _latest_p_day_before(p_by_day, decision_day)
        if not np.isfinite(p_day_val):
            p_day_val = 0.5
            p_day_source_date = None
        out = select_oracle_thresholds_from_daily_rewards(
            history_df=pd.DataFrame(daily_reward_log),
            current_p_day=p_day_val,
            config=daily_threshold_config,
            prev_buy_level=prev_buy_level,
            prev_sell_level=prev_sell_level,
            objective="reward",
        )
        return {
            "date": decision_day,
            "source": "last_5m_bar",
            "source_ts": fallback_ts,
            "source_day": fallback_day,
            "source_idx": int(fallback_idx),
            "daily_action": out.gate,
            "p_day": p_day_val,
            "p_day_source_date": p_day_source_date,
            "daily_buy_level": out.buy_level,
            "daily_sell_level": out.sell_level,
            "buy_th": buy_ret_th_live,
            "sell_th": sell_ret_th_live,
            "cash": engine.cash,
            "pos": engine.pos,
            "qty": engine.qty,
            "entry_px": engine.entry_px,
        }

    def predict_5m_signal(r: Dict[str, Any], direction: str) -> tuple[float, float, bool]:
        """Predict one 5m signal's return and return its active threshold."""
        direction = str(direction).lower()
        if direction == "buy":
            pack = buy_pack
            threshold = float(buy_ret_th_live)
        elif direction == "sell":
            pack = sell_pack
            threshold = float(sell_ret_th_live)
        else:
            return np.nan, np.nan, False
        if pack is None:
            return np.nan, threshold, False

        row_df = prepare_ml_dataset(pd.DataFrame([r]))
        for cc in pack.feature_cols:
            if cc not in row_df.columns:
                row_df[cc] = 0.0
        return float(predict_ret(pack, row_df)), threshold, True

    if start_idx < len(df_5m_idx):
        begin_day(pd.to_datetime(df_5m_idx.loc[start_idx, "timestamp"]).normalize(), start_idx)
        current_day = pd.to_datetime(df_5m_idx.loc[start_idx, "timestamp"]).normalize().date()
    else:
        fallback_daily_decision = fallback_decision_from_last_5m_bar()

    for i in range(start_idx, len(df_5m_idx)):
        bar_ts = pd.to_datetime(df_5m_idx.loc[i, "timestamp"])
        bar_day = bar_ts.normalize()
        in_sim = sim_start_ts is None or bar_ts >= sim_start_ts

        if current_day is not None and bar_day.date() != current_day:
            prev_day = pd.to_datetime(current_day)
            prev_day_ts = pd.to_datetime(current_day)

            if day_start_engine_state is not None and day_start_idx is not None and last_day_end_idx is not None:
                reward_map = evaluate_three_day_rewards_for_logging(
                    engine_state=day_start_engine_state,
                    day_events=day_events_today,
                    day_start_idx=day_start_idx,
                    day_end_idx=last_day_end_idx,
                    df_5m_idx=df_5m_idx,
                    next_open_by_idx=next_open_by_idx,
                    closes=closes,
                    buy_pack=buy_pack,
                    sell_pack=sell_pack,
                    buy_ret_th_live=buy_ret_th_live,
                    sell_ret_th_live=sell_ret_th_live,
                    fee_pct=fee_pct,
                )
                chosen_reward = reward_map[_reward_action_for_gate(day_gate)]["day_return"]
                best_action_ex_post = max(
                    ["FORCE_BUY", "FREE", "FORCE_SELL"],
                    key=lambda k: reward_map[k]["day_return"],
                )
                oracle_equity *= (1.0 + reward_map[best_action_ex_post]["day_return"])
                _append_or_replace_daily_reward_log(
                    daily_reward_log,
                    {
                        "date": prev_day,
                        "p_day": current_p_day,
                        "p_day_source_date": current_p_day_source_date,
                        "buy_level": current_buy_level,
                        "sell_level": current_sell_level,
                        "chosen_action": day_gate,
                        "reward_force_buy": reward_map["FORCE_BUY"]["day_return"],
                        "reward_free": reward_map["FREE"]["day_return"],
                        "reward_force_sell": reward_map["FORCE_SELL"]["day_return"],
                        "chosen_reward": chosen_reward,
                        "best_action_ex_post": best_action_ex_post,
                        "oracle_equity": oracle_equity,
                        "close": float(day_close_map.get(prev_day.date(), np.nan)),
                        "buy_th_5m": buy_ret_th_live,
                        "sell_th_5m": sell_ret_th_live,
                    }
                )

            label_bestlookahead_for_ready_points(
                bsp_rows=bsp_rows_5m,
                highs=highs,
                lows=lows,
                closes=closes,
                lookahead_days=lookahead_days_5m,
                bar_interval_minutes=5,
                current_bar_idx=i,
            )
            maybe_retrain_5m(prev_day_ts)
            if last_day_end_idx is not None:
                maybe_opt_5m_thresholds(last_day_end_idx)

            day_close = day_close_map.get(prev_day.date())
            equity = engine.mark_to_market(day_close) if day_close is not None else engine.cash
            equity_peak = max(equity_peak, equity)
            daily_log.append(
                {
                    "date": prev_day,
                    "equity": equity,
                    "cash": engine.cash,
                    "pos": engine.pos,
                    "qty": engine.qty,
                    "entry_px": engine.entry_px,
                    "buy_th": buy_ret_th_live,
                    "sell_th": sell_ret_th_live,
                    "p_day": current_p_day,
                    "p_day_source_date": current_p_day_source_date,
                    "daily_action": day_gate,
                    "daily_buy_level": current_buy_level,
                    "daily_sell_level": current_sell_level,
                }
            )
            current_day = bar_day.date()
            begin_day(bar_day, i)

        last_day_end_idx = i
        if in_sim:
            engine.maybe_execute_pending(next_open_by_idx)

        klu = build_klu(
            df_5m_idx.loc[i, "timestamp"],
            df_5m_idx.loc[i, "Open"],
            df_5m_idx.loc[i, "High"],
            df_5m_idx.loc[i, "Low"],
            df_5m_idx.loc[i, "Close"],
            df_5m_idx.loc[i, "Volume"],
        )
        feed_chan_one(chan_5m, klu)

        new_rows = extract_bsp_rows_from_chan(chan_5m)
        if not new_rows:
            continue

        for r0 in new_rows:
            r = dict(r0)
            event_ts = pd.to_datetime(r.get("timestamp", bar_ts), errors="coerce")
            if pd.isna(event_ts):
                event_ts = bar_ts
            if getattr(event_ts, "tzinfo", None) is not None:
                event_ts = event_ts.tz_convert(None)
            global_ki = timestamp_to_idx.get(pd.to_datetime(event_ts), int(i))
            r["timestamp"] = str(event_ts)
            r["klu_idx"] = int(global_ki)
            if "direction" not in r or r["direction"] is None:
                if r.get("is_buy", None) is not None:
                    r["direction"] = "buy" if bool(r["is_buy"]) else "sell"
                else:
                    r["direction"] = "buy"
            r["direction"] = str(r["direction"]).lower()
            if "bsp_type" in r and r["bsp_type"] is not None:
                r["bsp_type"] = str(r["bsp_type"]).lower()
            r.setdefault("best_return_pct", np.nan)

            k = (int(r.get("klu_idx", -1)), str(r.get("direction")), str(r.get("bsp_type")))
            if k in seen_keys_5m:
                continue
            seen_keys_5m.add(k)
            bsp_rows_5m.append(r)
            if in_sim:
                day_events_today.append(copy.deepcopy(r))

            if not in_sim:
                continue

            d = str(r.get("direction", "buy")).lower()
            ki = int(r.get("klu_idx", i))
            signal_px = float(df_5m_idx.loc[ki, "Close"]) if ki in df_5m_idx.index else np.nan
            signal_meta = {
                "detected_at": str(bar_ts),
                "processed_through": str(bar_ts),
                "delay_bars": int(max(0, int(i) - int(ki))),
                "is_delayed": bool(int(i) > int(ki)),
            }
            pr, th_live, model_available = predict_5m_signal(r, d)
            if trade_start_ts is not None and event_ts < trade_start_ts:
                # Chan can emit old turning-point signals after warmup/history changes.
                # They are added to seen_keys above so they do not recur, but live logs
                # should stay focused on signals from the scheduled session.
                continue
            if int(r["klu_idx"]) < int(start_idx):
                signal_decisions.append(
                        {
                            "ts": str(event_ts),
                            "side": d,
                            "price": signal_px,
                            "pred": pr,
                            "th": th_live,
                            "gate": day_gate,
                            "action": "SKIP",
                            "reason": "delayed signal before start_idx",
                            **signal_meta,
                        }
                )
                continue
            if d == "buy" and not allow_buy:
                signal_decisions.append(
                        {
                            "ts": str(event_ts),
                            "side": d,
                            "price": signal_px,
                            "pred": pr,
                            "th": th_live,
                            "gate": day_gate,
                            "action": "SKIP",
                            "reason": "daily gate blocks buy",
                            **signal_meta,
                        }
                )
                continue
            if d == "sell" and not allow_sell:
                signal_decisions.append(
                        {
                            "ts": str(event_ts),
                            "side": d,
                            "price": signal_px,
                            "pred": pr,
                            "th": th_live,
                            "gate": day_gate,
                            "action": "SKIP",
                            "reason": "daily gate blocks sell",
                            **signal_meta,
                        }
                )
                continue
            if must_trade_dir is not None and d != must_trade_dir:
                signal_decisions.append(
                        {
                            "ts": str(event_ts),
                            "side": d,
                            "price": signal_px,
                            "pred": pr,
                            "th": th_live,
                            "gate": day_gate,
                            "action": "SKIP",
                            "reason": f"waiting for required {must_trade_dir}",
                            **signal_meta,
                        }
                )
                continue

            if d == "buy" and engine.pos == 0 and model_available:
                action = "BUY" if pr >= float(buy_ret_th_live) else "HOLD"
                signal_decisions.append(
                    {
                        "ts": str(event_ts),
                        "side": d,
                        "price": signal_px,
                        "pred": float(pr),
                        "th": float(buy_ret_th_live),
                        "gate": day_gate,
                        "action": action,
                        "reason": "passes threshold" if action == "BUY" else "below buy threshold",
                        **signal_meta,
                    }
                )
                if pr >= float(buy_ret_th_live):
                    engine.place_order_for_next_bar(
                        side="buy",
                        seen_idx=ki,
                        reason=("ADAPTIVE_FORCE_BUY->first acceptable 5m signal" if day_gate == "FORCE_BUY" else "5m BUY signal"),
                        meta={"ts": str(event_ts), "pred": float(pr), "th": float(buy_ret_th_live), "gate": day_gate},
                    )
                    if must_trade_dir == "buy":
                        must_trade_dir = None

            elif d == "sell" and engine.pos == 1 and model_available:
                action = "SELL" if pr >= float(sell_ret_th_live) else "HOLD"
                signal_decisions.append(
                    {
                        "ts": str(event_ts),
                        "side": d,
                        "price": signal_px,
                        "pred": float(pr),
                        "th": float(sell_ret_th_live),
                        "gate": day_gate,
                        "action": action,
                        "reason": "passes threshold" if action == "SELL" else "below sell threshold",
                        **signal_meta,
                    }
                )
                if pr >= float(sell_ret_th_live):
                    engine.place_order_for_next_bar(
                        side="sell",
                        seen_idx=ki,
                        reason=("ADAPTIVE_FORCE_SELL->first acceptable 5m signal" if day_gate == "FORCE_SELL" else "5m SELL signal"),
                        meta={"ts": str(event_ts), "pred": float(pr), "th": float(sell_ret_th_live), "gate": day_gate},
                    )
                    if must_trade_dir == "sell":
                        must_trade_dir = None
            elif d == "buy":
                signal_decisions.append(
                    {
                        "ts": str(event_ts),
                        "side": d,
                        "price": signal_px,
                        "pred": pr,
                        "th": float(buy_ret_th_live),
                        "gate": day_gate,
                        "action": "SKIP",
                        "reason": "already long or buy model unavailable",
                        **signal_meta,
                    }
                )
            elif d == "sell":
                signal_decisions.append(
                    {
                        "ts": str(event_ts),
                        "side": d,
                        "price": signal_px,
                        "pred": pr,
                        "th": float(sell_ret_th_live),
                        "gate": day_gate,
                        "action": "SKIP",
                        "reason": "flat or sell model unavailable",
                        **signal_meta,
                    }
                )

    if current_day is not None and day_start_engine_state is not None and day_start_idx is not None and last_day_end_idx is not None:
        prev_day = pd.to_datetime(current_day)
        reward_map = evaluate_three_day_rewards_for_logging(
            engine_state=day_start_engine_state,
            day_events=day_events_today,
            day_start_idx=day_start_idx,
            day_end_idx=last_day_end_idx,
            df_5m_idx=df_5m_idx,
            next_open_by_idx=next_open_by_idx,
            closes=closes,
            buy_pack=buy_pack,
            sell_pack=sell_pack,
            buy_ret_th_live=buy_ret_th_live,
            sell_ret_th_live=sell_ret_th_live,
            fee_pct=fee_pct,
        )
        chosen_reward = reward_map[_reward_action_for_gate(day_gate)]["day_return"]
        best_action_ex_post = max(
            ["FORCE_BUY", "FREE", "FORCE_SELL"],
            key=lambda k: reward_map[k]["day_return"],
        )
        oracle_equity *= (1.0 + reward_map[best_action_ex_post]["day_return"])
        _append_or_replace_daily_reward_log(
            daily_reward_log,
            {
                "date": prev_day,
                "p_day": current_p_day,
                "p_day_source_date": current_p_day_source_date,
                "buy_level": current_buy_level,
                "sell_level": current_sell_level,
                "chosen_action": day_gate,
                "reward_force_buy": reward_map["FORCE_BUY"]["day_return"],
                "reward_free": reward_map["FREE"]["day_return"],
                "reward_force_sell": reward_map["FORCE_SELL"]["day_return"],
                "chosen_reward": chosen_reward,
                "best_action_ex_post": best_action_ex_post,
                "oracle_equity": oracle_equity,
                "close": float(day_close_map.get(prev_day.date(), np.nan)),
                "buy_th_5m": buy_ret_th_live,
                "sell_th_5m": sell_ret_th_live,
            }
        )

        day_close = day_close_map.get(prev_day.date())
        equity = engine.mark_to_market(day_close) if day_close is not None else engine.cash
        daily_log.append(
            {
                "date": prev_day,
                "equity": equity,
                "cash": engine.cash,
                "pos": engine.pos,
                "qty": engine.qty,
                "entry_px": engine.entry_px,
                "buy_th": buy_ret_th_live,
                "sell_th": sell_ret_th_live,
                "p_day": current_p_day,
                "p_day_source_date": current_p_day_source_date,
                "daily_action": day_gate,
                "daily_buy_level": current_buy_level,
                "daily_sell_level": current_sell_level,
            }
        )
    trades_df = pd.DataFrame(engine.trades)
    signal_decisions_df = pd.DataFrame(signal_decisions)
    if not trades_df.empty and "seen_idx" in trades_df.columns:
        exec_idx = pd.to_numeric(trades_df["seen_idx"], errors="coerce") + 1
        trades_df["exec_idx"] = exec_idx
        trades_df["exec_ts"] = [
            df_5m_idx.loc[int(i), "timestamp"] if pd.notna(i) and 0 <= int(i) < len(df_5m_idx) else pd.NaT
            for i in exec_idx
        ]

    return {
        "buy_pack": buy_pack,
        "sell_pack": sell_pack,
        "buy_ret_th_live": buy_ret_th_live,
        "sell_ret_th_live": sell_ret_th_live,
        "bsp_rows_5m": bsp_rows_5m,
        "seen_keys_5m": seen_keys_5m,
        "daily_reward_log": _dedupe_daily_reward_log(daily_reward_log),
        "execution_engine_state": engine.state_dict(),
        "daily_log": daily_log,
        "trades_df": trades_df,
        "signal_decisions_df": signal_decisions_df,
        "daily_log_df": pd.DataFrame(daily_log),
        "fallback_daily_decision": fallback_daily_decision,
    }


def build_adaptive_reward_snapshot(
    *,
    daily_csv_path: str,
    k5m_csv_path: str,
    snapshot_path: str,
    code: str = "QQQ",
    daily_chan_start: str = "2008-01-01",
    accumulation_start: str = "2010-01-01",
    snapshot_end_time: str = "2023-12-31",
    N_confirm: int = 5,
    min_labeled_days_to_train: int = 200,
    retrain_every_new_labels: int = 25,
    dp_lookback: int = 5,
    lookahead_days_5m: float = 2.0,
    retrain_every_days_5m: int = 5,
    min_samples_total_5m: int = 300,
    threshold_window_days: float = 2.0,
    threshold_ret_grid=None,
    threshold_min_open_signals: int = 10,
    initial_capital: float = 100000.0,
    fee_pct: float = 0.0,
    daily_chan_max_klines: int = 500,
    five_chan_max_klines: int = 500,
    macro_files: Optional[dict] = None,
    static_buy_level: float = 0.20,
    static_sell_level: float = 0.30,
    daily_threshold_config: Optional[RollingThresholdConfig] = None,
    daily_threshold_max_gap_override: Optional[float] = None,
    output_dir: str = "output_adaptive_reward_snapshot_build",
    autosave_year_start_checkpoints: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Build a long-history adaptive reward checkpoint from daily and 5m CSV data.

    The snapshot stores trained daily/5m models, threshold history, warmup bars,
    and optional execution state so later calls can resume without rebuilding.
    """
    os.makedirs(output_dir, exist_ok=True)
    daily_threshold_config = _normalize_daily_threshold_config(
        daily_threshold_config,
        max_gap_override=daily_threshold_max_gap_override,
    )
    data = _build_data_views(
        daily_csv_path=daily_csv_path,
        k5m_csv_path=k5m_csv_path,
        daily_chan_start=daily_chan_start,
        accumulation_start=accumulation_start,
        end_time=snapshot_end_time,
        macro_files=macro_files,
    )

    daily_chan = SlidingWindowChan(
        code=code,
        begin_time=None,
        end_time=None,
        data_src=getattr(DATA_SRC, "CSV", "CSV"),
        lv_list=[KL_TYPE.K_DAY],
        config=_make_chan_config(),
        autype=AUTYPE.QFQ,
        max_klines=int(daily_chan_max_klines),
    )
    chan_5m = SlidingWindowChan(
        code=code,
        begin_time=None,
        end_time=None,
        data_src=getattr(DATA_SRC, "CSV", "CSV"),
        lv_list=[KL_TYPE.K_5M],
        config=_make_chan_config(),
        autype=AUTYPE.QFQ,
        max_klines=int(five_chan_max_klines),
    )

    st = DailyProbState()
    bsp_rows_daily: List[Dict[str, Any]] = []
    seen_bsp_daily = set()
    X_days: List[Dict[str, float]] = []
    y_days: List[int] = []
    pending_idx: List[int] = []
    p_series = np.full(len(data["df_day_feat"]), np.nan, dtype=float)
    dp_vs_minK_series = np.full(len(data["df_day_feat"]), np.nan, dtype=float)
    dp_vs_maxK_series = np.full(len(data["df_day_feat"]), np.nan, dtype=float)
    p_by_day: Dict[pd.Timestamp, float] = {}

    _run_daily_phase(
        df_day_feat=data["df_day_feat"],
        daily_chan=daily_chan,
        macro_cols=data["macro_cols"],
        N_confirm=N_confirm,
        min_labeled_days_to_train=min_labeled_days_to_train,
        retrain_every_new_labels=retrain_every_new_labels,
        dp_lookback=dp_lookback,
        verbose=verbose,
        st=st,
        bsp_rows_daily=bsp_rows_daily,
        seen_bsp_daily=seen_bsp_daily,
        X_days=X_days,
        y_days=y_days,
        pending_idx=pending_idx,
        p_series=p_series,
        dp_vs_minK_series=dp_vs_minK_series,
        dp_vs_maxK_series=dp_vs_maxK_series,
        p_by_day=p_by_day,
        start_idx=0,
    )

    if st.model is not None:
        try:
            feature_importance_from_lr(st.model, top_n=120).to_csv(
                os.path.join(output_dir, "daily_lr_feature_importance.csv"),
                index=False,
            )
        except Exception:
            pass

    phase2 = _run_adaptive_reward_5m_phase(
        df_5m_raw=data["df_5m_raw"],
        df_5m_idx=data["df_5m_idx"],
        next_open_by_idx=data["next_open_by_idx"],
        closes=data["closes"],
        highs=data["highs"],
        lows=data["lows"],
        day_close_map=data["day_close_map"],
        p_by_day=p_by_day,
        snapshot_or_end_time=snapshot_end_time,
        chan_5m=chan_5m,
        buy_pack=None,
        sell_pack=None,
        bsp_rows_5m=[],
        seen_keys_5m=set(),
        daily_reward_log=[],
        initial_capital=initial_capital,
        fee_pct=fee_pct,
        threshold_window_days=threshold_window_days,
        threshold_ret_grid=threshold_ret_grid,
        threshold_min_open_signals=threshold_min_open_signals,
        lookahead_days_5m=lookahead_days_5m,
        retrain_every_days_5m=retrain_every_days_5m,
        min_samples_total_5m=min_samples_total_5m,
        daily_threshold_config=daily_threshold_config,
        static_buy_level=static_buy_level,
        static_sell_level=static_sell_level,
        buy_ret_th_live=0.30,
        sell_ret_th_live=0.30,
        sim_start=accumulation_start,
        verbose=verbose,
        start_idx=0,
    )

    snapshot_ts = pd.to_datetime(snapshot_end_time)
    bundle = _make_adaptive_reward_snapshot_bundle(
        source_bundle=None,
        code=code,
        snapshot_ts=snapshot_ts,
        daily_csv_path=daily_csv_path,
        k5m_csv_path=k5m_csv_path,
        daily_chan_start=daily_chan_start,
        accumulation_start=accumulation_start,
        macro_files=data["macro_files"],
        df_day_raw=data["df_day_raw"],
        df_5m_raw=data["df_5m_raw"],
        daily_prob_model=st.model,
        daily_prob_trained_n=st.trained_n,
        X_days=X_days,
        y_days=y_days,
        pending_idx=pending_idx,
        p_by_day=p_by_day,
        p_series=p_series,
        dp_vs_minK_series=dp_vs_minK_series,
        dp_vs_maxK_series=dp_vs_maxK_series,
        bsp_rows_daily=bsp_rows_daily,
        seen_bsp_daily=seen_bsp_daily,
        buy_pack=phase2["buy_pack"],
        sell_pack=phase2["sell_pack"],
        buy_ret_th_live=phase2["buy_ret_th_live"],
        sell_ret_th_live=phase2["sell_ret_th_live"],
        bsp_rows_5m=phase2["bsp_rows_5m"],
        seen_keys_5m=phase2["seen_keys_5m"],
        daily_reward_log=phase2["daily_reward_log"],
        daily_chan_max_klines=daily_chan_max_klines,
        five_chan_max_klines=five_chan_max_klines,
        daily_threshold_config=daily_threshold_config,
        threshold_window_days=threshold_window_days,
        threshold_ret_grid=threshold_ret_grid,
        threshold_min_open_signals=threshold_min_open_signals,
        lookahead_days_5m=lookahead_days_5m,
        retrain_every_days_5m=retrain_every_days_5m,
        min_samples_total_5m=min_samples_total_5m,
        N_confirm=N_confirm,
        min_labeled_days_to_train=min_labeled_days_to_train,
        retrain_every_new_labels=retrain_every_new_labels,
        dp_lookback=dp_lookback,
        static_buy_level=static_buy_level,
        static_sell_level=static_sell_level,
        execution_engine_state=phase2["execution_engine_state"],
    )
    save_joblib(snapshot_path, bundle)

    year_start_checkpoint_paths = []
    if autosave_year_start_checkpoints:
        year_start_checkpoint_paths = _autosave_year_start_snapshots(
            output_dir=output_dir,
            source_path=snapshot_path,
            source_bundle=bundle,
            data=data,
            code=code,
            daily_csv_path=daily_csv_path,
            k5m_csv_path=k5m_csv_path,
            daily_chan_start=daily_chan_start,
            accumulation_start=accumulation_start,
            daily_prob_model=st.model,
            daily_prob_trained_n=st.trained_n,
            X_days=X_days,
            y_days=y_days,
            pending_idx=pending_idx,
            p_by_day=p_by_day,
            p_series=p_series,
            dp_vs_minK_series=dp_vs_minK_series,
            dp_vs_maxK_series=dp_vs_maxK_series,
            bsp_rows_daily=bsp_rows_daily,
            buy_pack=phase2["buy_pack"],
            sell_pack=phase2["sell_pack"],
            buy_ret_th_live=phase2["buy_ret_th_live"],
            sell_ret_th_live=phase2["sell_ret_th_live"],
            bsp_rows_5m=phase2["bsp_rows_5m"],
            daily_reward_log=phase2["daily_reward_log"],
            daily_chan_max_klines=daily_chan_max_klines,
            five_chan_max_klines=five_chan_max_klines,
            daily_threshold_config=daily_threshold_config,
            threshold_window_days=threshold_window_days,
            threshold_ret_grid=threshold_ret_grid,
            threshold_min_open_signals=threshold_min_open_signals,
            lookahead_days_5m=lookahead_days_5m,
            retrain_every_days_5m=retrain_every_days_5m,
            min_samples_total_5m=min_samples_total_5m,
            N_confirm=N_confirm,
            min_labeled_days_to_train=min_labeled_days_to_train,
            retrain_every_new_labels=retrain_every_new_labels,
            dp_lookback=dp_lookback,
            static_buy_level=static_buy_level,
            static_sell_level=static_sell_level,
            start_after=pd.to_datetime(accumulation_start),
            end_time=snapshot_end_time,
            verbose=verbose,
        )

    trades_df = phase2["trades_df"]
    daily_log_df = phase2["daily_log_df"]
    trades_df.to_csv(os.path.join(output_dir, "snapshot_trades.csv"), index=False)
    daily_log_df.to_csv(os.path.join(output_dir, "snapshot_daily_log.csv"), index=False)
    pd.DataFrame(bundle["daily_reward_log"]).to_csv(os.path.join(output_dir, "snapshot_daily_reward_log.csv"), index=False)

    return {
        "snapshot_path": snapshot_path,
        "snapshot_time": snapshot_ts,
        "trades_df": trades_df,
        "daily_log_df": daily_log_df,
        "daily_reward_df": pd.DataFrame(bundle["daily_reward_log"]),
        "year_start_checkpoint_paths": year_start_checkpoint_paths,
        "output_dir": output_dir,
    }


def run_adaptive_reward_from_snapshot(
    *,
    snapshot_path: str,
    end_time: str,
    sim_start: Optional[str] = None,
    trade_start: Optional[str] = None,
    reset_execution_state: bool = True,
    initial_capital: float = 100000.0,
    fee_pct: float = 0.0,
    dp_lookback_override: Optional[int] = None,
    daily_threshold_lookback_days_override: Optional[int] = None,
    daily_threshold_max_gap_override: Optional[float] = None,
    threshold_ret_grid_override=None,
    execution_engine_state_override: Optional[dict] = None,
    output_dir: str = "output_adaptive_reward_resumed_fresh",
    save_snapshot_path: Optional[str] = None,
    autosave_year_start_checkpoints: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Resume a saved adaptive reward snapshot through end_time.

    Use reset_execution_state=True for a clean backtest from initial_capital.
    Use threshold_ret_grid_override to test a custom 5m return-threshold grid.
    """
    os.makedirs(output_dir, exist_ok=True)
    bundle = load_joblib(snapshot_path)
    if not isinstance(bundle, dict) or bundle.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError(f"Unexpected snapshot format in {snapshot_path}")

    daily_csv_path = bundle["daily_csv_path"]
    k5m_csv_path = bundle["k5m_csv_path"]
    daily_chan_start = bundle["original_daily_chan_start"]
    accumulation_start = bundle["original_accumulation_start"]
    snapshot_time = pd.to_datetime(bundle["snapshot_time"])
    sim_start = sim_start or str((snapshot_time + pd.Timedelta(days=1)).date())
    dp_lookback = int(bundle.get("dp_lookback", 5) if dp_lookback_override is None else dp_lookback_override)
    if dp_lookback < 1:
        raise ValueError("dp_lookback_override must be >= 1")

    daily_threshold_config = _normalize_daily_threshold_config(
        bundle.get("daily_threshold_config"),
        lookback_days_override=daily_threshold_lookback_days_override,
        max_gap_override=daily_threshold_max_gap_override,
    )

    # A resumed run can either reuse the grid saved in the snapshot or test a
    # caller-provided grid without rebuilding the whole checkpoint.
    threshold_ret_grid = _coerce_5m_ret_threshold_grid(
        threshold_ret_grid_override
        if threshold_ret_grid_override is not None
        else bundle.get("threshold_ret_grid")
    )

    data = _build_data_views(
        daily_csv_path=daily_csv_path,
        k5m_csv_path=k5m_csv_path,
        daily_chan_start=daily_chan_start,
        accumulation_start=accumulation_start,
        end_time=end_time,
        macro_files=bundle.get("macro_files"),
    )
    buy_hold = compute_buy_hold_equity(data["day_close_map"], data["all_days"], initial_capital)

    daily_chan = SlidingWindowChan(
        code=bundle.get("code", "QQQ"),
        begin_time=None,
        end_time=None,
        data_src=getattr(DATA_SRC, "CSV", "CSV"),
        lv_list=[KL_TYPE.K_DAY],
        config=_make_chan_config(),
        autype=AUTYPE.QFQ,
        max_klines=int(bundle.get("daily_chan_max_klines", 500)),
    )
    chan_5m = SlidingWindowChan(
        code=bundle.get("code", "QQQ"),
        begin_time=None,
        end_time=None,
        data_src=getattr(DATA_SRC, "CSV", "CSV"),
        lv_list=[KL_TYPE.K_5M],
        config=_make_chan_config(),
        autype=AUTYPE.QFQ,
        max_klines=int(bundle.get("five_chan_max_klines", 500)),
    )

    warmup_daily_bars = bundle.get("warmup_daily_bars")

    last_warm_day_ts = _warm_chan_from_bars(daily_chan, warmup_daily_bars)
    last_warm_5m_ts = _warm_chan_from_bars(chan_5m, bundle.get("warmup_5m_bars"))

    st = DailyProbState()
    st.model = bundle.get("daily_prob_model")
    st.trained_n = int(bundle.get("daily_prob_trained_n", 0))
    st.new_labels = 0

    X_days = bundle.get("X_days", []) or []
    y_days = bundle.get("y_days", []) or []
    pending_idx = bundle.get("pending_idx", []) or []
    bsp_rows_daily = bundle.get("bsp_rows_daily", []) or []
    seen_bsp_daily = _list_to_set(bundle.get("seen_bsp_daily_list", []))

    p_series = np.full(len(data["df_day_feat"]), np.nan, dtype=float)
    dp_vs_minK_series = np.full(len(data["df_day_feat"]), np.nan, dtype=float)
    dp_vs_maxK_series = np.full(len(data["df_day_feat"]), np.nan, dtype=float)
    loaded_p = bundle.get("p_series")
    loaded_dp_min = bundle.get("dp_vs_minK_series")
    loaded_dp_max = bundle.get("dp_vs_maxK_series")
    if loaded_p is not None:
        n = min(len(p_series), len(loaded_p))
        p_series[:n] = np.asarray(loaded_p[:n], dtype=float)
    if loaded_dp_min is not None:
        n = min(len(dp_vs_minK_series), len(loaded_dp_min))
        dp_vs_minK_series[:n] = np.asarray(loaded_dp_min[:n], dtype=float)
    if loaded_dp_max is not None:
        n = min(len(dp_vs_maxK_series), len(loaded_dp_max))
        dp_vs_maxK_series[:n] = np.asarray(loaded_dp_max[:n], dtype=float)
    p_by_day = {pd.to_datetime(k).normalize(): float(v) for k, v in (bundle.get("p_by_day_str", {}) or {}).items()}

    daily_i_start = _find_next_index_after(data["df_day_feat"], last_warm_day_ts or snapshot_time)
    five_i_start = _find_next_index_after(data["df_5m_idx"], last_warm_5m_ts or snapshot_time)

    _run_daily_phase(
        df_day_feat=data["df_day_feat"],
        daily_chan=daily_chan,
        macro_cols=data["macro_cols"],
        N_confirm=int(bundle.get("N_confirm", 5)),
        min_labeled_days_to_train=int(bundle.get("min_labeled_days_to_train", 200)),
        retrain_every_new_labels=int(bundle.get("retrain_every_new_labels", 25)),
        dp_lookback=dp_lookback,
        verbose=verbose,
        st=st,
        bsp_rows_daily=bsp_rows_daily,
        seen_bsp_daily=seen_bsp_daily,
        X_days=X_days,
        y_days=y_days,
        pending_idx=pending_idx,
        p_series=p_series,
        dp_vs_minK_series=dp_vs_minK_series,
        dp_vs_maxK_series=dp_vs_maxK_series,
        p_by_day=p_by_day,
        start_idx=daily_i_start,
    )

    execution_engine_state = (
        copy.deepcopy(execution_engine_state_override)
        if execution_engine_state_override is not None
        else (None if reset_execution_state else bundle.get("execution_engine_state"))
    )
    # reset_execution_state=True starts the resumed backtest flat at initial_capital.
    # Passing execution_engine_state_override is for live/replay cases that need a known position.

    phase2 = _run_adaptive_reward_5m_phase(
        df_5m_raw=data["df_5m_raw"],
        df_5m_idx=data["df_5m_idx"],
        next_open_by_idx=data["next_open_by_idx"],
        closes=data["closes"],
        highs=data["highs"],
        lows=data["lows"],
        day_close_map=data["day_close_map"],
        p_by_day=p_by_day,
        snapshot_or_end_time=end_time,
        chan_5m=chan_5m,
        buy_pack=unpack_ret_modelpack_from_load(bundle.get("buy_pack")),
        sell_pack=unpack_ret_modelpack_from_load(bundle.get("sell_pack")),
        bsp_rows_5m=bundle.get("bsp_rows_5m", []) or [],
        seen_keys_5m=_list_to_set(bundle.get("seen_keys_5m_list", [])),
        daily_reward_log=bundle.get("daily_reward_log", []) or [],
        initial_capital=initial_capital,
        fee_pct=fee_pct,
        threshold_window_days=float(bundle.get("threshold_window_days", 2.0)),
        threshold_ret_grid=threshold_ret_grid,
        threshold_min_open_signals=int(bundle.get("threshold_min_open_signals", 10)),
        lookahead_days_5m=float(bundle.get("lookahead_days_5m", 2.0)),
        retrain_every_days_5m=int(bundle.get("retrain_every_days_5m", 5)),
        min_samples_total_5m=int(bundle.get("min_samples_total_5m", 300)),
        daily_threshold_config=daily_threshold_config,
        static_buy_level=float(bundle.get("static_buy_level", 0.20)),
        static_sell_level=float(bundle.get("static_sell_level", 0.30)),
        buy_ret_th_live=float(bundle.get("buy_ret_th_live", 0.30)),
        sell_ret_th_live=float(bundle.get("sell_ret_th_live", 0.30)),
        sim_start=sim_start,
        verbose=verbose,
        trade_start=trade_start,
        start_idx=five_i_start,
        execution_engine_state=execution_engine_state,
    )

    trades_df = phase2["trades_df"]
    signal_decisions_df = phase2["signal_decisions_df"]
    signal_decisions_all_df = _save_signal_decision_outputs(output_dir, signal_decisions_df)
    daily_log_df = phase2["daily_log_df"]
    daily_reward_df = pd.DataFrame(phase2["daily_reward_log"])
    trades_df.to_csv(os.path.join(output_dir, "trades.csv"), index=False)
    daily_log_df.to_csv(os.path.join(output_dir, "daily_log.csv"), index=False)
    daily_reward_df.to_csv(os.path.join(output_dir, "daily_reward_log.csv"), index=False)

    out_eq = os.path.join(output_dir, "equity_vs_buyhold.png")
    out_px = os.path.join(output_dir, "price_with_trades.png")
    out_sig = os.path.join(output_dir, "price_with_signals.png")
    out_p = os.path.join(output_dir, "p_day.png")

    plt.figure(figsize=(12, 6))
    if not daily_log_df.empty and "date" in daily_log_df.columns and "equity" in daily_log_df.columns:
        plot_daily = daily_log_df.copy()
        plot_daily["date"] = pd.to_datetime(plot_daily["date"])
        plt.plot(plot_daily["date"], plot_daily["equity"], label="Strategy Equity")
    if len(buy_hold) > 0:
        plt.plot(buy_hold.index, buy_hold.values, label="Buy&Hold")
    plt.legend()
    plt.title("Equity vs Buy&Hold")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(out_eq, dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.plot(data["df_5m_idx"]["timestamp"], data["df_5m_idx"]["Close"], label="5m Close")
    if not trades_df.empty and "seen_idx" in trades_df.columns and "exec_px" in trades_df.columns:
        for _, tr in trades_df.iterrows():
            idx = int(tr["exec_idx"]) if "exec_idx" in trades_df.columns and pd.notna(tr.get("exec_idx")) else int(tr["seen_idx"]) + 1
            if idx in data["df_5m_idx"].index:
                t = pd.to_datetime(data["df_5m_idx"].loc[idx, "timestamp"])
                px = float(tr["exec_px"])
                if str(tr.get("side", "")).lower() == "buy":
                    plt.scatter([t], [px], marker="^")
                else:
                    plt.scatter([t], [px], marker="v")
    plt.title("Price with Trade Markers")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_px, dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(14, 7))
    plt.plot(data["df_5m_idx"]["timestamp"], data["df_5m_idx"]["Close"], label="5m Close", linewidth=1.2)
    if not signal_decisions_all_df.empty and {"ts", "price", "side", "action"}.issubset(signal_decisions_all_df.columns):
        plot_signals = signal_decisions_all_df.copy()
        plot_signals["ts"] = pd.to_datetime(plot_signals["ts"], errors="coerce")
        plot_signals["price"] = pd.to_numeric(plot_signals["price"], errors="coerce")
        plot_signals = plot_signals.dropna(subset=["ts", "price"])
        if not plot_signals.empty:
            x_min = pd.to_datetime(data["df_5m_idx"]["timestamp"]).min()
            x_max = pd.to_datetime(data["df_5m_idx"]["timestamp"]).max()
            plot_signals = plot_signals[(plot_signals["ts"] >= x_min) & (plot_signals["ts"] <= x_max)].copy()
        style_map = {
            ("BUY", "buy"): {"marker": "^", "color": "green", "s": 90, "label": "BUY"},
            ("SELL", "sell"): {"marker": "v", "color": "red", "s": 90, "label": "SELL"},
            ("HOLD", "buy"): {"marker": "o", "color": "tab:blue", "s": 45, "label": "HOLD buy"},
            ("HOLD", "sell"): {"marker": "o", "color": "tab:orange", "s": 45, "label": "HOLD sell"},
            ("SKIP", "buy"): {"marker": "x", "color": "tab:cyan", "s": 55, "label": "SKIP buy"},
            ("SKIP", "sell"): {"marker": "x", "color": "tab:pink", "s": 55, "label": "SKIP sell"},
        }
        for (action, side), style in style_map.items():
            mask = (
                plot_signals["action"].astype(str).str.upper().eq(action)
                & plot_signals["side"].astype(str).str.lower().eq(side)
            )
            pts = plot_signals[mask]
            if pts.empty:
                continue
            plt.scatter(
                pts["ts"],
                pts["price"],
                marker=style["marker"],
                color=style["color"],
                s=style["s"],
                label=style["label"],
                alpha=0.85,
            )
        if "is_delayed" in plot_signals.columns:
            delayed = plot_signals[plot_signals["is_delayed"].astype(str).str.lower().isin({"true", "1"})]
            if not delayed.empty:
                plt.scatter(
                    delayed["ts"],
                    delayed["price"],
                    marker="o",
                    facecolors="none",
                    edgecolors="black",
                    s=135,
                    linewidths=1.1,
                    label="delayed",
                )
    plt.title("Price with 5m Signal Decisions")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_sig, dpi=150, bbox_inches="tight")
    plt.close()

    days_sorted = sorted(p_by_day.keys())
    if days_sorted:
        plt.figure(figsize=(12, 5))
        xs = pd.to_datetime(days_sorted)
        ys = [p_by_day[d] for d in days_sorted]
        plt.plot(xs, ys, label="p_day")
        if not daily_log_df.empty and {"date", "daily_buy_level", "daily_sell_level"}.issubset(daily_log_df.columns):
            plot_daily = daily_log_df.copy()
            plot_daily["date"] = pd.to_datetime(plot_daily["date"])
            plt.plot(plot_daily["date"], plot_daily["daily_buy_level"], linestyle="--", label="adaptive buy level")
            plt.plot(plot_daily["date"], plot_daily["daily_sell_level"], linestyle="--", label="adaptive sell level")
        plt.legend()
        plt.title("Daily Probability p_day")
        plt.xlabel("Date")
        plt.ylabel("Probability")
        plt.tight_layout()
        plt.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close()

    feature_imp_path = os.path.join(output_dir, "daily_lr_feature_importance.csv")
    if st.model is not None:
        try:
            feature_importance_from_lr(st.model, top_n=120).to_csv(feature_imp_path, index=False)
        except Exception:
            pass

    if save_snapshot_path is None:
        source = Path(snapshot_path)
        save_snapshot_path = str(source.with_name(f"{source.stem}__continued{source.suffix or '.joblib'}"))

    final_ts = pd.to_datetime(data["df_5m_idx"]["timestamp"].iloc[-1]) if not data["df_5m_idx"].empty else pd.to_datetime(end_time)
    continued_bundle = _make_adaptive_reward_snapshot_bundle(
        source_bundle=bundle,
        code=bundle.get("code", "QQQ"),
        snapshot_ts=final_ts,
        daily_csv_path=daily_csv_path,
        k5m_csv_path=k5m_csv_path,
        daily_chan_start=daily_chan_start,
        accumulation_start=accumulation_start,
        macro_files=data["macro_files"],
        df_day_raw=data["df_day_raw"],
        df_5m_raw=data["df_5m_raw"],
        daily_prob_model=st.model,
        daily_prob_trained_n=st.trained_n,
        X_days=X_days,
        y_days=y_days,
        pending_idx=pending_idx,
        p_by_day=p_by_day,
        p_series=p_series,
        dp_vs_minK_series=dp_vs_minK_series,
        dp_vs_maxK_series=dp_vs_maxK_series,
        bsp_rows_daily=bsp_rows_daily,
        seen_bsp_daily=seen_bsp_daily,
        buy_pack=phase2["buy_pack"],
        sell_pack=phase2["sell_pack"],
        buy_ret_th_live=phase2["buy_ret_th_live"],
        sell_ret_th_live=phase2["sell_ret_th_live"],
        bsp_rows_5m=phase2["bsp_rows_5m"],
        seen_keys_5m=phase2["seen_keys_5m"],
        daily_reward_log=phase2["daily_reward_log"],
        daily_chan_max_klines=int(bundle.get("daily_chan_max_klines", 500)),
        five_chan_max_klines=int(bundle.get("five_chan_max_klines", 500)),
        daily_threshold_config=daily_threshold_config,
        threshold_window_days=float(bundle.get("threshold_window_days", 2.0)),
        threshold_ret_grid=threshold_ret_grid,
        threshold_min_open_signals=int(bundle.get("threshold_min_open_signals", 10)),
        lookahead_days_5m=float(bundle.get("lookahead_days_5m", 2.0)),
        retrain_every_days_5m=int(bundle.get("retrain_every_days_5m", 5)),
        min_samples_total_5m=int(bundle.get("min_samples_total_5m", 300)),
        N_confirm=int(bundle.get("N_confirm", 5)),
        min_labeled_days_to_train=int(bundle.get("min_labeled_days_to_train", 200)),
        retrain_every_new_labels=int(bundle.get("retrain_every_new_labels", 25)),
        dp_lookback=dp_lookback,
        static_buy_level=float(bundle.get("static_buy_level", 0.20)),
        static_sell_level=float(bundle.get("static_sell_level", 0.30)),
        execution_engine_state=phase2["execution_engine_state"],
    )
    save_joblib(save_snapshot_path, continued_bundle)

    year_start_checkpoint_paths = []
    if autosave_year_start_checkpoints:
        year_start_checkpoint_paths = _autosave_year_start_snapshots(
            output_dir=output_dir,
            source_path=save_snapshot_path,
            source_bundle=continued_bundle,
            data=data,
            code=bundle.get("code", "QQQ"),
            daily_csv_path=daily_csv_path,
            k5m_csv_path=k5m_csv_path,
            daily_chan_start=daily_chan_start,
            accumulation_start=accumulation_start,
            daily_prob_model=st.model,
            daily_prob_trained_n=st.trained_n,
            X_days=X_days,
            y_days=y_days,
            pending_idx=pending_idx,
            p_by_day=p_by_day,
            p_series=p_series,
            dp_vs_minK_series=dp_vs_minK_series,
            dp_vs_maxK_series=dp_vs_maxK_series,
            bsp_rows_daily=bsp_rows_daily,
            buy_pack=phase2["buy_pack"],
            sell_pack=phase2["sell_pack"],
            buy_ret_th_live=phase2["buy_ret_th_live"],
            sell_ret_th_live=phase2["sell_ret_th_live"],
            bsp_rows_5m=phase2["bsp_rows_5m"],
            daily_reward_log=phase2["daily_reward_log"],
            daily_chan_max_klines=int(bundle.get("daily_chan_max_klines", 500)),
            five_chan_max_klines=int(bundle.get("five_chan_max_klines", 500)),
            daily_threshold_config=daily_threshold_config,
            threshold_window_days=float(bundle.get("threshold_window_days", 2.0)),
            threshold_ret_grid=threshold_ret_grid,
            threshold_min_open_signals=int(bundle.get("threshold_min_open_signals", 10)),
            lookahead_days_5m=float(bundle.get("lookahead_days_5m", 2.0)),
            retrain_every_days_5m=int(bundle.get("retrain_every_days_5m", 5)),
            min_samples_total_5m=int(bundle.get("min_samples_total_5m", 300)),
            N_confirm=int(bundle.get("N_confirm", 5)),
            min_labeled_days_to_train=int(bundle.get("min_labeled_days_to_train", 200)),
            retrain_every_new_labels=int(bundle.get("retrain_every_new_labels", 25)),
            dp_lookback=dp_lookback,
            static_buy_level=float(bundle.get("static_buy_level", 0.20)),
            static_sell_level=float(bundle.get("static_sell_level", 0.30)),
            start_after=snapshot_time,
            end_time=end_time,
            verbose=verbose,
        )

    if verbose:
        print(f"[SAVED] {os.path.join(output_dir, 'trades.csv')}")
        print(f"[SAVED] {os.path.join(output_dir, 'signal_decisions.csv')}")
        print(f"[SAVED] {os.path.join(output_dir, 'daily_log.csv')}")
        print(f"[SAVED] {os.path.join(output_dir, 'daily_reward_log.csv')}")
        print(f"[SAVED] {out_eq}")
        print(f"[SAVED] {out_px}")
        print(f"[SAVED] {out_sig}")
        if days_sorted:
            print(f"[SAVED] {out_p}")
        print(f"[CHECKPOINT] saved continued snapshot -> {save_snapshot_path}")

    return {
        "snapshot_path": snapshot_path,
        "continued_snapshot_path": save_snapshot_path,
        "snapshot_time": snapshot_time,
        "resume_sim_start": pd.to_datetime(sim_start),
        "trades_df": trades_df,
        "signal_decisions_df": signal_decisions_df,
        "signal_decisions_all_df": signal_decisions_all_df,
        "daily_log_df": daily_log_df,
        "daily_reward_df": daily_reward_df,
        "execution_engine_state": phase2["execution_engine_state"],
        "fallback_daily_decision": phase2.get("fallback_daily_decision"),
        "year_start_checkpoint_paths": year_start_checkpoint_paths,
        "buy_hold": buy_hold,
        "dp_lookback": int(dp_lookback),
        "daily_threshold_lookback_days": int(daily_threshold_config.lookback_days),
        "daily_threshold_max_gap": getattr(daily_threshold_config, "max_gap", None),
        "threshold_ret_grid": threshold_ret_grid,
        "signals_plot_path": out_sig,
        "output_dir": output_dir,
    }


def run_adaptive_reward_realtime_from_checkpoint(
    *,
    checkpoint_path: str,
    output_dir: str = "output_adaptive_reward_realtime",
    end_time: Optional[str] = None,
    initial_capital: float = 100000.0,
    fee_pct: float = 0.0,
    save_snapshot_path: Optional[str] = None,
    autosave_year_start_checkpoints: bool = False,
    dp_lookback_override: Optional[int] = None,
    daily_threshold_lookback_days_override: Optional[int] = None,
    daily_threshold_max_gap_override: Optional[float] = None,
    threshold_ret_grid_override=None,
    trade_start: Optional[str] = None,
    reset_execution_state: bool = False,
    execution_engine_state_override: Optional[dict] = None,
    daily_csv_path_override: Optional[str] = None,
    k5m_csv_path_override: Optional[str] = None,
    refresh_data: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Resume the adaptive reward strategy from a saved checkpoint and advance it
    with newly fetched market bars. This is a paper/live simulation helper: it
    produces orders/trades from the strategy state but does not send broker orders.
    """
    os.makedirs(output_dir, exist_ok=True)
    bundle = load_joblib(checkpoint_path)
    if not isinstance(bundle, dict) or bundle.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError(f"Unexpected checkpoint format in {checkpoint_path}")

    code = str(bundle.get("code", "QQQ"))
    snapshot_time = pd.to_datetime(bundle["snapshot_time"])
    run_end = pd.to_datetime(end_time) if end_time is not None else pd.Timestamp.now()
    if run_end <= snapshot_time:
        raise ValueError(
            f"Checkpoint snapshot_time {snapshot_time} is not before requested realtime end {run_end}. "
            "Use a checkpoint from before the live period you want to simulate."
        )

    source_daily_csv = daily_csv_path_override or bundle["daily_csv_path"]
    source_5m_csv = k5m_csv_path_override or bundle["k5m_csv_path"]
    realtime_daily_csv = os.path.join(output_dir, f"{code}_realtime_merged_DAY.csv")
    realtime_5m_csv = os.path.join(output_dir, f"{code}_realtime_merged_5M.csv")

    if refresh_data and daily_csv_path_override is None and k5m_csv_path_override is None:
        fetch_start = str((snapshot_time - pd.Timedelta(days=5)).date())
        daily_bars = fetch_yfinance_ohlcv(code=code, start=fetch_start, end=str(run_end.date()), interval="1d")
        five_bars = fetch_yfinance_ohlcv(code=code, start=fetch_start, end=str(run_end), interval="5m")
        realtime_daily_csv = _merge_ohlcv_csv_with_new_bars(source_daily_csv, daily_bars, realtime_daily_csv)
        realtime_5m_csv = _merge_ohlcv_csv_with_new_bars(source_5m_csv, five_bars, realtime_5m_csv)
    else:
        realtime_daily_csv = source_daily_csv
        realtime_5m_csv = source_5m_csv

    realtime_bundle = dict(bundle)
    realtime_bundle["daily_csv_path"] = realtime_daily_csv
    realtime_bundle["k5m_csv_path"] = realtime_5m_csv
    source_macro_folder = os.path.dirname(os.path.abspath(source_daily_csv))
    macro_files = copy.deepcopy(realtime_bundle.get("macro_files") or {"vix_": "VIX.csv"})
    realtime_bundle["macro_files"] = {
        pref: fn if os.path.isabs(str(fn)) else os.path.join(source_macro_folder, str(fn))
        for pref, fn in macro_files.items()
    }

    bootstrap_checkpoint_path = os.path.join(output_dir, "realtime_bootstrap_checkpoint.joblib")
    save_joblib(bootstrap_checkpoint_path, realtime_bundle)

    if save_snapshot_path is None:
        save_snapshot_path = os.path.join(output_dir, "realtime_final_checkpoint.joblib")

    res = run_adaptive_reward_from_snapshot(
        snapshot_path=bootstrap_checkpoint_path,
        end_time=str(run_end),
        sim_start=str((snapshot_time + pd.Timedelta(minutes=5))),
        trade_start=trade_start,
        reset_execution_state=reset_execution_state,
        initial_capital=initial_capital,
        fee_pct=fee_pct,
        dp_lookback_override=dp_lookback_override,
        daily_threshold_lookback_days_override=daily_threshold_lookback_days_override,
        daily_threshold_max_gap_override=daily_threshold_max_gap_override,
        threshold_ret_grid_override=threshold_ret_grid_override,
        execution_engine_state_override=execution_engine_state_override,
        output_dir=output_dir,
        save_snapshot_path=save_snapshot_path,
        autosave_year_start_checkpoints=autosave_year_start_checkpoints,
        verbose=verbose,
    )
    res["input_checkpoint_path"] = checkpoint_path
    res["realtime_bootstrap_checkpoint_path"] = bootstrap_checkpoint_path
    res["realtime_daily_csv_path"] = realtime_daily_csv
    res["realtime_5m_csv_path"] = realtime_5m_csv
    res["realtime_end_time"] = run_end
    return res


def run_adaptive_reward_yfinance_intraday_loop(
    *,
    checkpoint_path: str,
    output_dir: str = "output_adaptive_reward_yfinance_intraday",
    initial_capital: float = 100000.0,
    fee_pct: float = 0.0,
    save_snapshot_path: Optional[str] = None,
    autosave_year_start_checkpoints: bool = False,
    dp_lookback_override: Optional[int] = None,
    daily_threshold_lookback_days_override: Optional[int] = None,
    daily_threshold_max_gap_override: Optional[float] = None,
    threshold_ret_grid_override=None,
    reset_execution_state: bool = False,
    poll_seconds: int = 60,
    market_close_time: str = "16:00",
    timezone: str = "America/New_York",
    verbose: bool = True,
) -> dict:
    """
    Run the realtime yfinance paper simulation through the current session.

    First pass: advance from checkpoint_path through all Yahoo bars currently
    available. Later passes: keep polling yfinance and resuming from the latest
    saved checkpoint until the market close time.
    """
    import time

    os.makedirs(output_dir, exist_ok=True)
    if save_snapshot_path is None:
        save_snapshot_path = os.path.join(output_dir, "live_checkpoint.joblib")

    poll_seconds = max(1, int(poll_seconds))
    current_checkpoint_path = checkpoint_path
    results = []
    seen_trade_keys = set()
    latest_status = {
        "last_trading_decision": None,
        "equity": np.nan,
        "cash": np.nan,
        "pos": np.nan,
        "date": None,
        "buy_th": np.nan,
        "sell_th": np.nan,
        "p_day": np.nan,
        "daily_action": None,
        "daily_buy_level": np.nan,
        "daily_sell_level": np.nan,
    }

    def ny_now() -> pd.Timestamp:
        """Current timestamp in the configured market timezone."""
        return pd.Timestamp.now(tz=timezone)

    live_trade_start_ts = ny_now().tz_convert(None)

    def today_close(now_ts: pd.Timestamp) -> pd.Timestamp:
        """Market close timestamp for the same local day as now_ts."""
        hh, mm = [int(x) for x in str(market_close_time).split(":", 1)]
        return now_ts.normalize() + pd.Timedelta(hours=hh, minutes=mm)

    def print_new_trades(res: dict) -> None:
        """Print only trades that have not already been shown in this live loop."""
        trades = res.get("trades_df")
        if trades is None or trades.empty:
            return
        for _, tr in trades.iterrows():
            key = (
                str(tr.get("side", "")),
                int(tr.get("seen_idx", -1)) if pd.notna(tr.get("seen_idx", np.nan)) else -1,
                str(tr.get("ts", "")),
            )
            if key in seen_trade_keys:
                continue
            seen_trade_keys.add(key)
            print(
                "[LIVE TRADE] "
                f"{tr.get('side')} seen_idx={tr.get('seen_idx')} "
                f"exec_px={tr.get('exec_px')} reason={tr.get('reason')}"
            )

    def latest_live_status(res: dict) -> dict:
        """Extract the latest account, gate, threshold, and trade status from a run result."""
        status = copy.deepcopy(latest_status)
        daily_log = res.get("daily_log_df")
        if daily_log is not None and not daily_log.empty:
            row = daily_log.iloc[-1]
            status.update(
                {
                    "date": row.get("date"),
                    "equity": row.get("equity", np.nan),
                    "cash": row.get("cash", np.nan),
                    "pos": row.get("pos", np.nan),
                    "buy_th": row.get("buy_th", np.nan),
                    "sell_th": row.get("sell_th", np.nan),
                    "p_day": row.get("p_day", np.nan),
                    "p_day_source_date": row.get("p_day_source_date"),
                    "daily_action": row.get("daily_action"),
                    "daily_buy_level": row.get("daily_buy_level", np.nan),
                    "daily_sell_level": row.get("daily_sell_level", np.nan),
                    "daily_decision_source": "daily_log",
                    "daily_decision_source_ts": row.get("date"),
                    "daily_decision_source_day": row.get("date"),
                }
            )

        trades = res.get("trades_df")
        if trades is not None and not trades.empty:
            tr = trades.iloc[-1].to_dict()
            status["last_trading_decision"] = {
                "side": tr.get("side"),
                "seen_idx": tr.get("seen_idx"),
                "exec_idx": tr.get("exec_idx"),
                "exec_ts": tr.get("exec_ts"),
                "exec_px": tr.get("exec_px"),
                "qty": tr.get("qty"),
                "fee": tr.get("fee"),
                "reason": tr.get("reason"),
                "signal_ts": tr.get("ts"),
                "pred": tr.get("pred"),
                "th": tr.get("th"),
                "gate": tr.get("gate"),
                "pnl": tr.get("pnl"),
            }
        return status

    def print_live_status(status: dict) -> None:
        """Print a compact live account and decision status block."""
        decision = status.get("last_trading_decision")
        print("[LIVE STATUS]")
        print(
            "  today's decision: "
            f"date={status.get('date')} "
            f"daily_action={status.get('daily_action')} "
            f"p_day={status.get('p_day')} "
            f"buy_level={status.get('daily_buy_level')} "
            f"sell_level={status.get('daily_sell_level')}"
        )
        print(
            "  account: "
            f"date={status.get('date')} "
            f"equity={status.get('equity')} "
            f"cash={status.get('cash')} "
            f"pos={status.get('pos')}"
        )
        print(
            "  5m model: "
            f"buy_th={status.get('buy_th')} "
            f"sell_th={status.get('sell_th')}"
        )
        print(f"  last trading decision: {decision if decision is not None else 'None'}")

    while True:
        now_ts = ny_now()
        close_ts = today_close(now_ts)
        run_end_ts = min(now_ts, close_ts)

        if verbose:
            print(f"[LIVE] advancing through {run_end_ts.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        try:
            res = run_adaptive_reward_realtime_from_checkpoint(
                checkpoint_path=current_checkpoint_path,
                output_dir=output_dir,
                end_time=str(run_end_ts.tz_convert(None)),
                initial_capital=initial_capital,
                fee_pct=fee_pct,
                save_snapshot_path=save_snapshot_path,
                autosave_year_start_checkpoints=autosave_year_start_checkpoints,
                dp_lookback_override=dp_lookback_override,
                daily_threshold_lookback_days_override=daily_threshold_lookback_days_override,
                daily_threshold_max_gap_override=daily_threshold_max_gap_override,
                threshold_ret_grid_override=threshold_ret_grid_override,
                trade_start=str(live_trade_start_ts),
                reset_execution_state=bool(reset_execution_state and not results),
                refresh_data=True,
                verbose=verbose,
            )
            results.append(res)
            current_checkpoint_path = res["continued_snapshot_path"]
            if verbose:
                print(f"[LIVE] checkpoint -> {current_checkpoint_path}")
            print_new_trades(res)
            latest_status = latest_live_status(res)
            print_live_status(latest_status)
        except ValueError as exc:
            msg = str(exc)
            if "is not before requested realtime end" not in msg:
                raise
            if verbose:
                print(f"[LIVE] no newer bars to process yet: {msg}")
                print_live_status(latest_status)

        now_ts = ny_now()
        close_ts = today_close(now_ts)
        if now_ts >= close_ts:
            if verbose:
                print(f"[LIVE] reached market close {close_ts.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            break

        sleep_for = min(poll_seconds, max(1, int((close_ts - now_ts).total_seconds())))
        if verbose:
            print(f"[LIVE] sleeping {sleep_for}s before next yfinance poll")
        time.sleep(sleep_for)

    last_res = results[-1] if results else {}
    return {
        "latest_checkpoint_path": current_checkpoint_path,
        "save_snapshot_path": save_snapshot_path,
        "results": results,
        "last_result": last_res,
        "latest_status": latest_status,
        "last_trading_decision": latest_status.get("last_trading_decision"),
        "trades_df": last_res.get("trades_df", pd.DataFrame()),
        "daily_log_df": last_res.get("daily_log_df", pd.DataFrame()),
    }


def run_adaptive_reward_yfinance_day_lookback(
    *,
    checkpoint_path: str,
    lookback_date: str,
    output_dir: str = "output_adaptive_reward_lookback",
    compare_live_output_dir: Optional[str] = None,
    market_open_time: str = "09:30",
    market_close_time: str = "16:00",
    timezone: str = "America/New_York",
    initial_capital: float = 100000.0,
    fee_pct: float = 0.0,
    starting_execution_state: Optional[dict] = None,
    starting_position_qty: float = 0.0,
    starting_position_entry_px: Optional[float] = None,
    starting_cash: Optional[float] = None,
    save_snapshot_path: Optional[str] = None,
    autosave_year_start_checkpoints: bool = False,
    dp_lookback_override: Optional[int] = None,
    daily_threshold_lookback_days_override: Optional[int] = None,
    daily_threshold_max_gap_override: Optional[float] = None,
    threshold_ret_grid_override=None,
    refresh_data: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Replay one trading day from a checkpoint before that day.

    Use this to verify whether the saved live outputs for a day are reproducible.
    The replay writes into a separate lookback folder and can compare the replayed
    signal/trade CSVs against a live output directory.
    """
    lookback_day = pd.to_datetime(str(lookback_date)).normalize()
    hh_open, mm_open = [int(x) for x in str(market_open_time).split(":", 1)]
    hh_close, mm_close = [int(x) for x in str(market_close_time).split(":", 1)]
    start_ts = (lookback_day + pd.Timedelta(hours=hh_open, minutes=mm_open)).tz_localize(timezone)
    close_ts = (lookback_day + pd.Timedelta(hours=hh_close, minutes=mm_close)).tz_localize(timezone)
    start_naive = str(start_ts.tz_localize(None))
    close_naive = str(close_ts.tz_localize(None))

    bundle = load_joblib(checkpoint_path)
    if not isinstance(bundle, dict) or bundle.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError(f"Unexpected checkpoint format in {checkpoint_path}")
    snapshot_time = pd.to_datetime(bundle["snapshot_time"])
    if snapshot_time >= pd.to_datetime(close_naive):
        raise ValueError(
            f"Checkpoint snapshot_time {snapshot_time} is not before lookback close {close_naive}. "
            "Use the checkpoint from before the day you want to replay."
        )

    code = str(bundle.get("code", "QQQ"))
    safe_day = lookback_day.strftime("%Y%m%d")
    run_output_dir = os.path.join(output_dir, f"{code}_lookback_{safe_day}")
    os.makedirs(run_output_dir, exist_ok=True)
    if save_snapshot_path is None:
        save_snapshot_path = os.path.join(run_output_dir, "lookback_checkpoint.joblib")

    if starting_execution_state is None:
        starting_execution_state = make_execution_state_from_position(
            initial_capital=initial_capital,
            fee_pct=fee_pct,
            qty=starting_position_qty,
            entry_px=starting_position_entry_px,
            cash=starting_cash,
        )
    else:
        starting_execution_state = copy.deepcopy(starting_execution_state)

    if verbose:
        print(
            "[LOOKBACK] "
            f"date={lookback_day.date()} start={start_naive} close={close_naive} "
            f"checkpoint={checkpoint_path}"
        )

    res = run_adaptive_reward_realtime_from_checkpoint(
        checkpoint_path=checkpoint_path,
        output_dir=run_output_dir,
        end_time=close_naive,
        initial_capital=initial_capital,
        fee_pct=fee_pct,
        save_snapshot_path=save_snapshot_path,
        autosave_year_start_checkpoints=autosave_year_start_checkpoints,
        dp_lookback_override=dp_lookback_override,
        daily_threshold_lookback_days_override=daily_threshold_lookback_days_override,
        daily_threshold_max_gap_override=daily_threshold_max_gap_override,
        threshold_ret_grid_override=threshold_ret_grid_override,
        trade_start=start_naive,
        reset_execution_state=True,
        execution_engine_state_override=starting_execution_state,
        refresh_data=refresh_data,
        verbose=verbose,
    )

    comparison = {}
    if compare_live_output_dir is not None:
        replay_signals = _read_csv_if_exists(os.path.join(run_output_dir, f"signal_decisions_{safe_day}.csv"))
        live_signals = _read_csv_if_exists(os.path.join(compare_live_output_dir, f"signal_decisions_{safe_day}.csv"))
        if live_signals.empty:
            live_all = _with_signal_decision_date(_read_csv_if_exists(os.path.join(compare_live_output_dir, "signal_decisions.csv")))
            if not live_all.empty and "date" in live_all.columns:
                live_signals = live_all[live_all["date"] == lookback_day.strftime("%Y-%m-%d")].copy()

        comparison["signal_decisions"] = _compare_csv_frames(
            replay_signals,
            live_signals,
            sort_cols=["ts", "side", "action", "pred", "th", "gate", "reason"],
        )
        comparison["trades"] = _compare_csv_frames(
            _read_csv_if_exists(os.path.join(run_output_dir, "trades.csv")),
            _read_csv_if_exists(os.path.join(compare_live_output_dir, "trades.csv")),
            sort_cols=["ts", "side", "exec_ts", "exec_px", "qty", "reason"],
        )
        comparison["daily_log"] = _compare_csv_frames(
            _read_csv_if_exists(os.path.join(run_output_dir, "daily_log.csv")),
            _read_csv_if_exists(os.path.join(compare_live_output_dir, "daily_log.csv")),
            sort_cols=["date"],
        )

        if verbose:
            print(f"[LOOKBACK COMPARE] {comparison}")

    res.update(
        {
            "status": "lookback_complete",
            "lookback_date": lookback_day.date(),
            "lookback_output_dir": run_output_dir,
            "lookback_checkpoint_path": save_snapshot_path,
            "comparison": comparison,
        }
    )
    return res


def run_adaptive_reward_yfinance_range_backtest(
    *,
    checkpoint_path: str,
    start_date: str,
    end_date: str,
    output_dir: str = "output_adaptive_reward_range_backtest",
    market_open_time: str = "09:30",
    market_close_time: str = "16:00",
    timezone: str = "America/New_York",
    initial_capital: float = 100000.0,
    fee_pct: float = 0.0,
    starting_execution_state: Optional[dict] = None,
    starting_position_qty: float = 0.0,
    starting_position_entry_px: Optional[float] = None,
    starting_cash: Optional[float] = None,
    save_snapshot_path: Optional[str] = None,
    autosave_year_start_checkpoints: bool = False,
    dp_lookback_override: Optional[int] = None,
    daily_threshold_lookback_days_override: Optional[int] = None,
    daily_threshold_max_gap_override: Optional[float] = None,
    threshold_ret_grid_override=None,
    refresh_data: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Replay a selected date range from a checkpoint before that range.

    Example:
        run_adaptive_reward_yfinance_range_backtest(
            checkpoint_path=".../final_checkpoint.joblib",
            start_date="2026-06-01",
            end_date="2026-06-05",
        )

    The full continuation is saved in the run folder. Range-only CSVs are also
    written so the selected period can be reviewed without warmup/prior days.
    """
    start_day = pd.to_datetime(str(start_date)).normalize()
    end_day = pd.to_datetime(str(end_date)).normalize()
    if end_day < start_day:
        raise ValueError("end_date must be on or after start_date.")

    hh_open, mm_open = [int(x) for x in str(market_open_time).split(":", 1)]
    hh_close, mm_close = [int(x) for x in str(market_close_time).split(":", 1)]
    start_ts = (start_day + pd.Timedelta(hours=hh_open, minutes=mm_open)).tz_localize(timezone)
    close_ts = (end_day + pd.Timedelta(hours=hh_close, minutes=mm_close)).tz_localize(timezone)
    start_naive = pd.to_datetime(start_ts.tz_localize(None))
    close_naive = pd.to_datetime(close_ts.tz_localize(None))

    bundle = load_joblib(checkpoint_path)
    if not isinstance(bundle, dict) or bundle.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError(f"Unexpected checkpoint format in {checkpoint_path}")
    snapshot_time = pd.to_datetime(bundle["snapshot_time"])
    if snapshot_time >= close_naive:
        raise ValueError(
            f"Checkpoint snapshot_time {snapshot_time} is not before range close {close_naive}. "
            "Use a checkpoint from before the period you want to backtest."
        )

    code = str(bundle.get("code", "QQQ"))
    safe_start = start_day.strftime("%Y%m%d")
    safe_end = end_day.strftime("%Y%m%d")
    run_output_dir = os.path.join(output_dir, f"{code}_range_{safe_start}_{safe_end}")
    os.makedirs(run_output_dir, exist_ok=True)
    if save_snapshot_path is None:
        save_snapshot_path = os.path.join(run_output_dir, "range_checkpoint.joblib")

    if starting_execution_state is None:
        starting_execution_state = make_execution_state_from_position(
            initial_capital=initial_capital,
            fee_pct=fee_pct,
            qty=starting_position_qty,
            entry_px=starting_position_entry_px,
            cash=starting_cash,
        )
    else:
        starting_execution_state = copy.deepcopy(starting_execution_state)

    if verbose:
        print(
            "[RANGE BACKTEST] "
            f"start={start_naive} close={close_naive} checkpoint={checkpoint_path}"
        )

    res = run_adaptive_reward_realtime_from_checkpoint(
        checkpoint_path=checkpoint_path,
        output_dir=run_output_dir,
        end_time=str(close_naive),
        initial_capital=initial_capital,
        fee_pct=fee_pct,
        save_snapshot_path=save_snapshot_path,
        autosave_year_start_checkpoints=autosave_year_start_checkpoints,
        dp_lookback_override=dp_lookback_override,
        daily_threshold_lookback_days_override=daily_threshold_lookback_days_override,
        daily_threshold_max_gap_override=daily_threshold_max_gap_override,
        threshold_ret_grid_override=threshold_ret_grid_override,
        trade_start=str(start_naive),
        reset_execution_state=True,
        execution_engine_state_override=starting_execution_state,
        refresh_data=refresh_data,
        verbose=verbose,
    )

    range_daily_log_df = _filter_frame_by_date_range(
        res.get("daily_log_df", pd.DataFrame()),
        column="date",
        start_day=start_day,
        end_day=end_day,
    )
    range_daily_reward_df = _filter_frame_by_date_range(
        res.get("daily_reward_df", pd.DataFrame()),
        column="date",
        start_day=start_day,
        end_day=end_day,
    )
    range_trades_df = _filter_frame_by_timestamp_range(
        res.get("trades_df", pd.DataFrame()),
        column="ts",
        start_ts=start_naive,
        end_ts=close_naive,
    )
    range_signal_decisions_df = _filter_frame_by_timestamp_range(
        res.get("signal_decisions_all_df", res.get("signal_decisions_df", pd.DataFrame())),
        column="ts",
        start_ts=start_naive,
        end_ts=close_naive,
    )

    range_daily_log_path = os.path.join(run_output_dir, "range_daily_log.csv")
    range_daily_reward_path = os.path.join(run_output_dir, "range_daily_reward_log.csv")
    range_trades_path = os.path.join(run_output_dir, "range_trades.csv")
    range_signal_decisions_path = os.path.join(run_output_dir, "range_signal_decisions.csv")
    range_daily_log_df.to_csv(range_daily_log_path, index=False)
    range_daily_reward_df.to_csv(range_daily_reward_path, index=False)
    range_trades_df.to_csv(range_trades_path, index=False)
    range_signal_decisions_df.to_csv(range_signal_decisions_path, index=False)

    if verbose:
        print(
            "[RANGE BACKTEST SAVED] "
            f"daily={range_daily_log_path} trades={range_trades_path} "
            f"signals={range_signal_decisions_path}"
        )

    res.update(
        {
            "status": "range_backtest_complete",
            "range_start_date": start_day.date(),
            "range_end_date": end_day.date(),
            "range_output_dir": run_output_dir,
            "range_checkpoint_path": save_snapshot_path,
            "range_daily_log_df": range_daily_log_df,
            "range_daily_reward_df": range_daily_reward_df,
            "range_trades_df": range_trades_df,
            "range_signal_decisions_df": range_signal_decisions_df,
            "range_daily_log_path": range_daily_log_path,
            "range_daily_reward_path": range_daily_reward_path,
            "range_trades_path": range_trades_path,
            "range_signal_decisions_path": range_signal_decisions_path,
        }
    )
    return res


def run_adaptive_reward_yfinance_scheduled_flat_start_loop(
    *,
    checkpoint_path: str,
    output_dir: str = "output_adaptive_reward_next_monday_live",
    start_date: str = "next_monday",
    market_open_time: str = "09:30",
    market_close_time: str = "16:00",
    timezone: str = "America/New_York",
    initial_capital: float = 100000.0,
    fee_pct: float = 0.0,
    starting_execution_state: Optional[dict] = None,
    starting_position_qty: float = 0.0,
    starting_position_entry_px: Optional[float] = None,
    starting_cash: Optional[float] = None,
    save_snapshot_path: Optional[str] = None,
    autosave_year_start_checkpoints: bool = False,
    dp_lookback_override: Optional[int] = None,
    daily_threshold_lookback_days_override: Optional[int] = None,
    daily_threshold_max_gap_override: Optional[float] = None,
    threshold_ret_grid_override=None,
    local_daily_csv_path: Optional[str] = None,
    local_5m_csv_path: Optional[str] = None,
    local_data_dir: Optional[str] = None,
    poll_seconds: int = 60,
    preopen_update: bool = True,
    wait_until_start: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Update data, seed the requested account state, then run live from a scheduled market open.

    By default the account starts flat. Pass starting_position_qty and
    starting_position_entry_px to carry a live position into the new day, or
    pass starting_execution_state for full control over the engine state.
    Historical and pre-start bars may update model state, but only events at or
    after the scheduled open can create trades.
    """
    import time

    def ny_now() -> pd.Timestamp:
        """Current timestamp in the configured market timezone."""
        return pd.Timestamp.now(tz=timezone)

    def local_time_on_day(day_ts: pd.Timestamp, hhmm: str) -> pd.Timestamp:
        """Build a timezone-aware timestamp on a given day from an HH:MM string."""
        hh, mm = [int(x) for x in str(hhmm).split(":", 1)]
        return day_ts.normalize() + pd.Timedelta(hours=hh, minutes=mm)

    def parse_start(now_ts: pd.Timestamp) -> pd.Timestamp:
        """Resolve explicit dates or 'next_monday' into the scheduled local start timestamp."""
        if str(start_date).strip().lower() in {"next_monday", "monday"}:
            days_ahead = (0 - int(now_ts.weekday())) % 7
            if days_ahead == 0:
                candidate_day = now_ts
            else:
                candidate_day = now_ts + pd.Timedelta(days=days_ahead)
            candidate = local_time_on_day(candidate_day, market_open_time)
            if candidate <= now_ts:
                candidate = local_time_on_day(candidate + pd.Timedelta(days=7), market_open_time)
            return candidate

        raw = str(start_date).strip()
        if len(raw) == 10 and raw.count("-") == 2:
            raw = f"{raw} {market_open_time}"
        ts = pd.to_datetime(raw)
        if getattr(ts, "tzinfo", None) is None:
            ts = ts.tz_localize(timezone)
        else:
            ts = ts.tz_convert(timezone)
        return ts

    def trade_start_str(ts: pd.Timestamp) -> str:
        """Convert a timezone-aware trade start into the naive string used by the backtest engine."""
        return str(ts.tz_localize(None))

    if starting_execution_state is None:
        starting_execution_state = make_execution_state_from_position(
            initial_capital=initial_capital,
            fee_pct=fee_pct,
            qty=starting_position_qty,
            entry_px=starting_position_entry_px,
            cash=starting_cash,
        )
    else:
        starting_execution_state = copy.deepcopy(starting_execution_state)
    starting_pos_label = (
        "flat"
        if int(starting_execution_state.get("pos", 0)) == 0
        else (
            f"long qty={starting_execution_state.get('qty')} "
            f"entry_px={starting_execution_state.get('entry_px')} "
            f"cash={starting_execution_state.get('cash')}"
        )
    )

    def latest_live_status(res: dict, previous: Optional[dict] = None) -> dict:
        """Merge the latest run result into the status displayed by the scheduled loop."""
        status = copy.deepcopy(previous or {})
        status.setdefault("last_trading_decision", None)
        if res.get("buy_ret_th_live") is not None:
            status["buy_th"] = res.get("buy_ret_th_live")
        if res.get("sell_ret_th_live") is not None:
            status["sell_th"] = res.get("sell_ret_th_live")
        engine_state = res.get("execution_engine_state")
        if isinstance(engine_state, dict):
            status.update(
                {
                    "cash": engine_state.get("cash", status.get("cash", np.nan)),
                    "pos": engine_state.get("pos", status.get("pos", np.nan)),
                    "qty": engine_state.get("qty", status.get("qty", np.nan)),
                    "entry_px": engine_state.get("entry_px", status.get("entry_px", np.nan)),
                }
            )
        daily_log = res.get("daily_log_df")
        if daily_log is not None and not daily_log.empty:
            daily_log_view = daily_log.copy()
            daily_log_view["_status_date"] = pd.to_datetime(daily_log_view["date"], errors="coerce")
            scheduled_day = start_ts.tz_localize(None).normalize()
            current_rows = daily_log_view[daily_log_view["_status_date"].dt.normalize() >= scheduled_day]
            if not current_rows.empty:
                row = current_rows.iloc[-1]
                status.update(
                    {
                        "date": row.get("date"),
                        "equity": row.get("equity", np.nan),
                        "cash": row.get("cash", np.nan),
                        "pos": row.get("pos", np.nan),
                        "qty": row.get("qty", np.nan),
                        "entry_px": row.get("entry_px", np.nan),
                        "buy_th": row.get("buy_th", np.nan),
                        "sell_th": row.get("sell_th", np.nan),
                        "p_day": row.get("p_day", np.nan),
                        "p_day_source_date": row.get("p_day_source_date"),
                        "daily_action": row.get("daily_action"),
                        "daily_buy_level": row.get("daily_buy_level", np.nan),
                        "daily_sell_level": row.get("daily_sell_level", np.nan),
                        "daily_decision_source": "daily_log",
                        "daily_decision_source_ts": row.get("date"),
                        "daily_decision_source_day": row.get("date"),
                    }
                )
        else:
            fallback = res.get("fallback_daily_decision")
            if isinstance(fallback, dict):
                fallback_date = pd.to_datetime(fallback.get("date"), errors="coerce")
                status_date = pd.to_datetime(status.get("date"), errors="coerce")
                have_real_decision = (
                    status.get("daily_decision_source") == "daily_log"
                    and not pd.isna(status_date)
                    and not pd.isna(fallback_date)
                    and status_date.normalize() == fallback_date.normalize()
                )
                if not have_real_decision:
                    status.update(
                        {
                            "date": fallback.get("date"),
                            "equity": fallback.get("equity", np.nan),
                            "cash": fallback.get("cash", status.get("cash", np.nan)),
                            "pos": fallback.get("pos", status.get("pos", np.nan)),
                            "qty": fallback.get("qty", status.get("qty", np.nan)),
                            "entry_px": fallback.get("entry_px", status.get("entry_px", np.nan)),
                            "buy_th": fallback.get("buy_th", status.get("buy_th", np.nan)),
                            "sell_th": fallback.get("sell_th", status.get("sell_th", np.nan)),
                            "p_day": fallback.get("p_day", np.nan),
                            "p_day_source_date": fallback.get("p_day_source_date"),
                            "daily_action": fallback.get("daily_action"),
                            "daily_buy_level": fallback.get("daily_buy_level", np.nan),
                            "daily_sell_level": fallback.get("daily_sell_level", np.nan),
                            "daily_decision_source": fallback.get("source"),
                            "daily_decision_source_ts": fallback.get("source_ts"),
                            "daily_decision_source_day": fallback.get("source_day"),
                        }
                    )

        trades = res.get("trades_df")
        if trades is not None and not trades.empty:
            tr = trades.iloc[-1].to_dict()
            status["last_trading_decision"] = {
                "side": tr.get("side"),
                "signal_ts": tr.get("ts"),
                "exec_ts": tr.get("exec_ts"),
                "exec_px": tr.get("exec_px"),
                "qty": tr.get("qty"),
                "reason": tr.get("reason"),
                "pred": tr.get("pred"),
                "th": tr.get("th"),
                "gate": tr.get("gate"),
            }
        return status

    def print_day_decision(status: dict) -> None:
        """Print the daily gate decision and relevant thresholds once per decision state."""
        print("[LIVE DAY DECISION]")
        if status.get("date") is None:
            print(
                "  unavailable: no daily_log row yet; "
                "waiting for the first scheduled 5m bar/day to initialize"
            )
            print(
                "  "
                f"account_start={starting_pos_label} initial_capital={initial_capital} "
                f"buy_th_5m={status.get('buy_th')} sell_th_5m={status.get('sell_th')}"
            )
            return
        print(
            "  "
            f"date={status.get('date')} "
            f"daily_action={status.get('daily_action')} "
            f"p_day={status.get('p_day')} "
            f"p_day_source_date={status.get('p_day_source_date')} "
            f"buy_level={status.get('daily_buy_level')} "
            f"sell_level={status.get('daily_sell_level')}"
        )
        if status.get("daily_decision_source"):
            print(
                "  "
                f"source={status.get('daily_decision_source')} "
                f"source_ts={status.get('daily_decision_source_ts')} "
                f"source_day={status.get('daily_decision_source_day')}"
            )
        print(
            "  "
            f"account_start={starting_pos_label} initial_capital={initial_capital} "
            f"buy_th_5m={status.get('buy_th')} sell_th_5m={status.get('sell_th')}"
        )

    def day_decision_print_key(status: dict) -> tuple:
        """Build a stable key used to avoid repeating the same daily decision print."""
        if status.get("date") is None:
            return ("unavailable",)
        return (
            str(pd.to_datetime(status.get("date"), errors="coerce")),
            str(status.get("daily_action")),
            str(status.get("p_day")),
            str(status.get("daily_buy_level")),
            str(status.get("daily_sell_level")),
            str(status.get("daily_decision_source")),
        )

    def should_print_day_decision(status: dict, printed_key: Optional[tuple]) -> bool:
        """Return True when the daily decision has changed enough to print again."""
        source = status.get("daily_decision_source")
        if status.get("date") is None:
            return printed_key is None
        if printed_key is None:
            return True
        current_key = day_decision_print_key(status)
        if current_key != printed_key:
            return True
        return False

    def print_new_trades(res: dict, seen_trade_keys: set) -> None:
        """Print newly observed trades from the scheduled loop."""
        trades = res.get("trades_df")
        if trades is None or trades.empty:
            return
        for _, tr in trades.iterrows():
            key = (
                str(tr.get("side", "")),
                int(tr.get("seen_idx", -1)) if pd.notna(tr.get("seen_idx", np.nan)) else -1,
                str(tr.get("ts", "")),
            )
            if key in seen_trade_keys:
                continue
            seen_trade_keys.add(key)
            print(
                "[LIVE TRADE] "
                f"{str(tr.get('side')).upper()} "
                f"signal_ts={tr.get('ts')} "
                f"exec_ts={tr.get('exec_ts')} "
                f"exec_px={tr.get('exec_px')} "
                f"qty={tr.get('qty')} "
                f"reason={tr.get('reason')}"
            )

    def print_new_signal_decisions(res: dict, seen_signal_keys: set) -> None:
        """Print newly observed signal decisions from the scheduled loop."""
        decisions = res.get("signal_decisions_df")
        if decisions is None or decisions.empty:
            return
        for _, sig in decisions.iterrows():
            key = (
                str(sig.get("ts", "")),
                str(sig.get("side", "")),
                str(sig.get("action", "")),
                str(sig.get("pred", "")),
            )
            if key in seen_signal_keys:
                continue
            seen_signal_keys.add(key)
            prefix = "[LIVE 5M DELAYED] " if bool(sig.get("is_delayed", False)) else "[LIVE 5M] "
            print(
                prefix +
                f"action={sig.get('action')} "
                f"side={sig.get('side')} "
                f"ts={sig.get('ts')} "
                f"detected_at={sig.get('detected_at')} "
                f"delay_bars={sig.get('delay_bars')} "
                f"price={sig.get('price')} "
                f"pred={sig.get('pred')} "
                f"th={sig.get('th')} "
                f"gate={sig.get('gate')} "
                f"reason={sig.get('reason')}"
            )

    def next_trading_day_after(day_value: Any) -> pd.Timestamp:
        """Return the next weekday after a completed market day."""
        day = pd.to_datetime(day_value).normalize() + pd.Timedelta(days=1)
        while int(day.weekday()) >= 5:
            day = day + pd.Timedelta(days=1)
        return day

    def next_day_prediction_from_checkpoint(path: str, source_day: pd.Timestamp) -> dict:
        """Predict the next session gate from a p_day already saved in the checkpoint."""
        source_day = pd.to_datetime(source_day).normalize()
        try:
            bundle = load_joblib(path)
            p_by_day = {
                pd.to_datetime(k).normalize(): float(v)
                for k, v in (bundle.get("p_by_day_str", {}) or {}).items()
            }
            p_day = float(p_by_day.get(source_day, np.nan))
            if not np.isfinite(p_day):
                return {
                    "available": False,
                    "source_day": source_day,
                    "reason": f"p_day[{source_day.date()}] is not saved in the checkpoint",
                }

            threshold_config = _normalize_daily_threshold_config(
                bundle.get("daily_threshold_config"),
                lookback_days_override=daily_threshold_lookback_days_override,
                max_gap_override=daily_threshold_max_gap_override,
            )
            reward_log = _dedupe_daily_reward_log(bundle.get("daily_reward_log", []) or [])
            prev_buy_level = _last_finite_value(
                reward_log,
                "buy_level",
                float(bundle.get("static_buy_level", 0.20)),
            )
            prev_sell_level = _last_finite_value(
                reward_log,
                "sell_level",
                float(bundle.get("static_sell_level", 0.30)),
            )
            out = select_oracle_thresholds_from_daily_rewards(
                history_df=pd.DataFrame(reward_log),
                current_p_day=p_day,
                config=threshold_config,
                prev_buy_level=prev_buy_level,
                prev_sell_level=prev_sell_level,
                objective="reward",
            )
            return {
                "available": True,
                "date": next_trading_day_after(source_day),
                "daily_action": out.gate,
                "p_day": p_day,
                "p_day_source_date": source_day,
                "daily_buy_level": out.buy_level,
                "daily_sell_level": out.sell_level,
                "buy_th": bundle.get("buy_ret_th_live"),
                "sell_th": bundle.get("sell_ret_th_live"),
                "source": "checkpoint_close_prediction",
            }
        except Exception as exc:
            return {
                "available": False,
                "source_day": source_day,
                "reason": str(exc),
            }

    def checkpoint_code(path: str) -> str:
        """Read the ticker code stored in a checkpoint."""
        try:
            bundle = load_joblib(path)
            return str(bundle.get("code", "QQQ"))
        except Exception:
            return "QQQ"

    def csv_has_day(path: Optional[str], day: pd.Timestamp) -> bool:
        """Return True when an OHLCV CSV has a row for the normalized day."""
        if not path or not os.path.exists(path):
            return False
        try:
            df = _standardize_ohlcv_frame(pd.read_csv(path))
            if df.empty or "timestamp" not in df.columns:
                return False
            days = pd.to_datetime(df["timestamp"], errors="coerce").dt.normalize()
            return bool((days == pd.to_datetime(day).normalize()).any())
        except Exception:
            return False

    def refresh_prediction_from_existing_merged_files(source_day: pd.Timestamp) -> Optional[dict]:
        """Use already-written merged CSVs to compute p_day for the close day, without fetching."""
        code = checkpoint_code(current_checkpoint_path)
        merged_daily_csv = os.path.join(output_dir, f"{code}_realtime_merged_DAY.csv")
        merged_5m_csv = os.path.join(output_dir, f"{code}_realtime_merged_5M.csv")
        if not csv_has_day(merged_daily_csv, source_day):
            return None
        try:
            res = run_adaptive_reward_realtime_from_checkpoint(
                checkpoint_path=current_checkpoint_path,
                output_dir=output_dir,
                end_time=trade_start_str(close_ts),
                initial_capital=initial_capital,
                fee_pct=fee_pct,
                save_snapshot_path=save_snapshot_path,
                autosave_year_start_checkpoints=autosave_year_start_checkpoints,
                dp_lookback_override=dp_lookback_override,
                daily_threshold_lookback_days_override=daily_threshold_lookback_days_override,
                daily_threshold_max_gap_override=daily_threshold_max_gap_override,
                threshold_ret_grid_override=threshold_ret_grid_override,
                trade_start=trade_start_str(start_ts),
                reset_execution_state=False,
                execution_engine_state_override=None,
                daily_csv_path_override=merged_daily_csv,
                k5m_csv_path_override=merged_5m_csv if os.path.exists(merged_5m_csv) else local_source_5m_csv,
                refresh_data=False,
                verbose=verbose,
            )
            results.append(res)
            return res
        except ValueError as exc:
            if "is not before requested realtime end" not in str(exc):
                raise
            return None

    def print_next_day_prediction(prediction: dict) -> None:
        """Print the next-session daily gate prediction, or why it is unavailable."""
        print("[LIVE NEXT DAY PREDICTION]")
        if not prediction.get("available"):
            print(
                "  unavailable: "
                f"{prediction.get('reason')} "
                f"source_day={prediction.get('source_day')}"
            )
            return
        print(
            "  "
            f"date={prediction.get('date')} "
            f"daily_action={prediction.get('daily_action')} "
            f"p_day={prediction.get('p_day')} "
            f"p_day_source_date={prediction.get('p_day_source_date')} "
            f"buy_level={prediction.get('daily_buy_level')} "
            f"sell_level={prediction.get('daily_sell_level')}"
        )
        print(
            "  "
            f"source={prediction.get('source')} "
            f"buy_th_5m={prediction.get('buy_th')} "
            f"sell_th_5m={prediction.get('sell_th')}"
        )

    os.makedirs(output_dir, exist_ok=True)
    if save_snapshot_path is None:
        save_snapshot_path = os.path.join(output_dir, "live_checkpoint.joblib")

    poll_seconds = max(1, int(poll_seconds))
    start_ts = parse_start(ny_now())
    close_ts = local_time_on_day(start_ts, market_close_time)
    current_checkpoint_path = checkpoint_path
    results = []
    latest_status: dict = {"last_trading_decision": None}
    latest_status.update(
        {
            "cash": starting_execution_state.get("cash"),
            "pos": starting_execution_state.get("pos"),
            "qty": starting_execution_state.get("qty"),
            "entry_px": starting_execution_state.get("entry_px"),
        }
    )
    local_source_daily_csv = local_daily_csv_path
    local_source_5m_csv = local_5m_csv_path
    try:
        input_bundle = load_joblib(current_checkpoint_path)
        if isinstance(input_bundle, dict):
            latest_status.update(
                {
                    "buy_th": input_bundle.get("buy_ret_th_live"),
                    "sell_th": input_bundle.get("sell_ret_th_live"),
                }
            )
            code_for_local_data = str(input_bundle.get("code", "QQQ"))
            if local_data_dir is not None:
                local_source_daily_csv = local_source_daily_csv or os.path.join(local_data_dir, f"{code_for_local_data}_DAY.csv")
                local_source_5m_csv = local_source_5m_csv or os.path.join(local_data_dir, f"{code_for_local_data}_5M.csv")
    except Exception:
        pass
    seen_trade_keys = set()
    seen_signal_keys = set()
    reset_needed = True
    printed_day_decision_key = None

    if verbose:
        print(
            "[LIVE PLAN] "
            f"scheduled_start={start_ts.strftime('%Y-%m-%d %H:%M:%S %Z')} "
            f"scheduled_close={close_ts.strftime('%Y-%m-%d %H:%M:%S %Z')} "
            f"starting_position={starting_pos_label}"
        )

    now_ts = ny_now()
    if preopen_update and now_ts < start_ts:
        if verbose:
            print(f"[LIVE PREP] updating prior data through {now_ts.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        try:
            prep = run_adaptive_reward_realtime_from_checkpoint(
                checkpoint_path=current_checkpoint_path,
                output_dir=output_dir,
                end_time=str(now_ts.tz_convert(None)),
                initial_capital=initial_capital,
                fee_pct=fee_pct,
                save_snapshot_path=save_snapshot_path,
                autosave_year_start_checkpoints=autosave_year_start_checkpoints,
                dp_lookback_override=dp_lookback_override,
                daily_threshold_lookback_days_override=daily_threshold_lookback_days_override,
                daily_threshold_max_gap_override=daily_threshold_max_gap_override,
                threshold_ret_grid_override=threshold_ret_grid_override,
                trade_start=trade_start_str(start_ts),
                reset_execution_state=True,
                execution_engine_state_override=starting_execution_state,
                refresh_data=True,
                daily_csv_path_override=local_source_daily_csv,
                k5m_csv_path_override=local_source_5m_csv,
                verbose=verbose,
            )
            results.append(prep)
            current_checkpoint_path = prep["continued_snapshot_path"]
            latest_status = latest_live_status(prep, latest_status)
            reset_needed = False
            if verbose:
                print(f"[LIVE PREP] flat checkpoint -> {current_checkpoint_path}")
            if should_print_day_decision(latest_status, printed_day_decision_key):
                print_day_decision(latest_status)
                printed_day_decision_key = day_decision_print_key(latest_status)
        except ValueError as exc:
            if "is not before requested realtime end" not in str(exc):
                raise
            if verbose:
                print(f"[LIVE PREP] checkpoint is already current enough: {exc}")

    now_ts = ny_now()
    if now_ts < start_ts:
        if not wait_until_start:
            if verbose:
                print(f"[LIVE WAIT] start is {start_ts.strftime('%Y-%m-%d %H:%M:%S %Z')}; returning before live loop")
            return {
                "status": "waiting_for_start",
                "scheduled_start": start_ts,
                "scheduled_close": close_ts,
                "latest_checkpoint_path": current_checkpoint_path,
                "save_snapshot_path": save_snapshot_path,
                "results": results,
                "latest_status": latest_status,
                "trades_df": results[-1].get("trades_df", pd.DataFrame()) if results else pd.DataFrame(),
                "signal_decisions_df": results[-1].get("signal_decisions_df", pd.DataFrame()) if results else pd.DataFrame(),
                "signal_decisions_all_df": results[-1].get("signal_decisions_all_df", pd.DataFrame()) if results else pd.DataFrame(),
                "daily_log_df": results[-1].get("daily_log_df", pd.DataFrame()) if results else pd.DataFrame(),
            }

        while now_ts < start_ts:
            sleep_for = min(poll_seconds, max(1, int((start_ts - now_ts).total_seconds())))
            if verbose:
                print(f"[LIVE WAIT] sleeping {sleep_for}s until {start_ts.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            time.sleep(sleep_for)
            now_ts = ny_now()

    while True:
        now_ts = ny_now()
        run_end_ts = min(now_ts, close_ts)
        if run_end_ts < start_ts:
            run_end_ts = start_ts

        if verbose:
            print(f"[LIVE] advancing through {run_end_ts.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        try:
            res = run_adaptive_reward_realtime_from_checkpoint(
                checkpoint_path=current_checkpoint_path,
                output_dir=output_dir,
                end_time=str(run_end_ts.tz_convert(None)),
                initial_capital=initial_capital,
                fee_pct=fee_pct,
                save_snapshot_path=save_snapshot_path,
                autosave_year_start_checkpoints=autosave_year_start_checkpoints,
                dp_lookback_override=dp_lookback_override,
                daily_threshold_lookback_days_override=daily_threshold_lookback_days_override,
                daily_threshold_max_gap_override=daily_threshold_max_gap_override,
                threshold_ret_grid_override=threshold_ret_grid_override,
                trade_start=trade_start_str(start_ts),
                reset_execution_state=reset_needed,
                execution_engine_state_override=starting_execution_state if reset_needed else None,
                refresh_data=True,
                daily_csv_path_override=local_source_daily_csv,
                k5m_csv_path_override=local_source_5m_csv,
                verbose=verbose,
            )
            reset_needed = False
            results.append(res)
            current_checkpoint_path = res["continued_snapshot_path"]
            latest_status = latest_live_status(res, latest_status)
            if verbose:
                print(f"[LIVE] checkpoint -> {current_checkpoint_path}")
            if should_print_day_decision(latest_status, printed_day_decision_key):
                print_day_decision(latest_status)
                printed_day_decision_key = day_decision_print_key(latest_status)
            print_new_signal_decisions(res, seen_signal_keys)
            print_new_trades(res, seen_trade_keys)
        except ValueError as exc:
            msg = str(exc)
            if "is not before requested realtime end" not in msg:
                raise
            if verbose:
                print(f"[LIVE] no newer bars to process yet: {msg}")

        now_ts = ny_now()
        if now_ts >= close_ts:
            if verbose:
                print(f"[LIVE] reached market close {close_ts.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            close_source_day = start_ts.tz_localize(None).normalize()
            next_day_prediction = next_day_prediction_from_checkpoint(current_checkpoint_path, close_source_day)
            if not next_day_prediction.get("available"):
                merged_res = refresh_prediction_from_existing_merged_files(close_source_day)
                if isinstance(merged_res, dict):
                    current_checkpoint_path = merged_res["continued_snapshot_path"]
                    latest_status = latest_live_status(merged_res, latest_status)
                    if verbose:
                        print(f"[LIVE] close prediction checkpoint -> {current_checkpoint_path}")
                    if should_print_day_decision(latest_status, printed_day_decision_key):
                        print_day_decision(latest_status)
                        printed_day_decision_key = day_decision_print_key(latest_status)
                    next_day_prediction = next_day_prediction_from_checkpoint(current_checkpoint_path, close_source_day)
            latest_status["next_day_prediction"] = next_day_prediction
            if verbose:
                print_next_day_prediction(next_day_prediction)
            break

        sleep_for = min(poll_seconds, max(1, int((close_ts - now_ts).total_seconds())))
        if verbose:
            print(f"[LIVE] sleeping {sleep_for}s before next yfinance poll")
        time.sleep(sleep_for)

    last_res = results[-1] if results else {}
    return {
        "status": "closed",
        "scheduled_start": start_ts,
        "scheduled_close": close_ts,
        "latest_checkpoint_path": current_checkpoint_path,
        "save_snapshot_path": save_snapshot_path,
        "results": results,
        "latest_status": latest_status,
        "next_day_prediction": latest_status.get("next_day_prediction"),
        "last_trading_decision": latest_status.get("last_trading_decision"),
        "trades_df": last_res.get("trades_df", pd.DataFrame()),
        "signal_decisions_df": last_res.get("signal_decisions_df", pd.DataFrame()),
        "signal_decisions_all_df": last_res.get("signal_decisions_all_df", pd.DataFrame()),
        "daily_log_df": last_res.get("daily_log_df", pd.DataFrame()),
    }
