from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


GATES = ("FORCE_BUY", "FREE", "FORCE_SELL")
REWARD_COLUMNS = {
    "FORCE_BUY": "reward_force_buy",
    "FREE": "reward_free",
    "FORCE_SELL": "reward_force_sell",
}


@dataclass
class NextDayGateConfig:
    """
    Notebook-friendly configuration for the next-day gate model.

    The model uses today's finalized information to predict tomorrow's gate.
    The realized reward is then taken from tomorrow's reward row.
    """

    rewards_csv: str = "trades_with_gate_rewards.csv"
    trade_signals_csv: Optional[str] = None
    trade_signal_csvs: Optional[list[str]] = None
    market_csv: str = "DataAPI/data/TQQQ_DAY.csv"
    output_dir: str = "output_next_day_gate_model"
    checkpoint_dir: str = "checkpoints/next_day_gate_model"
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    initial_capital: float = 100000.0
    min_train_days: int = 252
    train_lookback_days: Optional[int] = None
    retrain_every_n_days: int = 20
    model_type: str = "extra_trees"
    random_state: int = 42
    n_jobs: int = 1
    checkpoint_every_n_days: int = 20
    resume_from_checkpoint: Optional[str] = None
    save_year_start_checkpoints: bool = True
    feature_lags: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 60)
    feature_mode: str = "rich_daily"
    tft_sequence_length: int = 60
    tft_hidden_size: int = 64
    tft_attention_heads: int = 4
    tft_epochs: int = 30
    tft_batch_size: int = 64
    tft_learning_rate: float = 1e-3
    tft_weight_decay: float = 1e-4
    tft_patience: int = 5
    tft_device: str = "auto"
    daily_chan_start: Optional[str] = None
    daily_chan_max_klines: int = 500
    macro_files: dict[str, str] = field(default_factory=dict)
    macro_base_dir: str = "DataAPI/data"
    reward_tie_epsilon: float = 1e-9
    fee_per_day: float = 0.0
    fee_pct: float = 0.0


P_DAY_LOGISTIC_KWARGS = {
    "N_confirm",
    "min_labeled_days_to_train",
    "retrain_every_new_labels",
    "dp_lookback",
    "static_buy_level",
    "static_sell_level",
    "daily_threshold_config",
}


def _normal_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path)


def _first_present(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lower_to_original = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        found = lower_to_original.get(candidate.lower())
        if found is not None:
            return found
    return None


def _clean_prefix(prefix: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(prefix).strip())
    return out.strip("_") or "macro"


def next_day_gate_config_from_adaptive_kwargs(
    adaptive_kwargs: dict[str, Any],
    *,
    rewards_csv: str = "trades_with_gate_rewards.csv",
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    output_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    model_type: str = "extra_trees",
    min_train_days: int = 252,
    retrain_every_n_days: int = 20,
    checkpoint_every_n_days: int = 20,
    **overrides: Any,
) -> NextDayGateConfig:
    """
    Build this model's config from the existing adaptive-reward kwargs.

    The adapter intentionally does not consume the previous p_day/logistic
    regression settings. Those keys are listed in P_DAY_LOGISTIC_KWARGS.
    """

    adaptive_kwargs = dict(adaptive_kwargs or {})
    code = str(adaptive_kwargs.get("code", "MODEL"))
    requested_trade_signals_csv = overrides.pop("trade_signals_csv", _default_trade_signals_csv(code))
    requested_trade_signal_csvs = overrides.pop("trade_signal_csvs", None)
    if requested_trade_signal_csvs is None and requested_trade_signals_csv:
        build_path = Path(f"trade_signals_build_{code.lower()}.csv")
        if build_path.exists() and str(build_path) != str(requested_trade_signals_csv):
            requested_trade_signal_csvs = [str(build_path), str(requested_trade_signals_csv)]
    cfg_values = {
        "rewards_csv": rewards_csv,
        "trade_signals_csv": requested_trade_signals_csv,
        "trade_signal_csvs": requested_trade_signal_csvs,
        "market_csv": adaptive_kwargs.get("daily_csv_path", NextDayGateConfig.market_csv),
        "output_dir": output_dir or f"output_next_day_gate_model_{code}",
        "checkpoint_dir": checkpoint_dir or f"checkpoints/next_day_gate_model_{code}",
        "start_year": start_year,
        "end_year": end_year,
        "initial_capital": float(adaptive_kwargs.get("initial_capital", NextDayGateConfig.initial_capital)),
        "fee_pct": float(adaptive_kwargs.get("fee_pct", 0.0)),
        "min_train_days": int(min_train_days),
        "retrain_every_n_days": int(retrain_every_n_days),
        "model_type": model_type,
        "checkpoint_every_n_days": int(checkpoint_every_n_days),
        "macro_files": dict(adaptive_kwargs.get("macro_files") or {}),
        "daily_chan_start": adaptive_kwargs.get("daily_chan_start"),
        "daily_chan_max_klines": int(adaptive_kwargs.get("daily_chan_max_klines", 500)),
    }
    cfg_values.update(overrides)
    return NextDayGateConfig(**cfg_values)


def _default_trade_signals_csv(code: str) -> Optional[str]:
    path = Path(f"trade_signals_{str(code).lower()}.csv")
    return str(path) if path.exists() else None


def _combine_trade_signal_csvs(
    trade_signal_csvs: list[str],
    output_path: str | Path,
) -> str:
    """Combine multiple precomputed 5-minute trade-signal CSVs into one file."""

    if not trade_signal_csvs:
        raise ValueError("trade_signal_csvs must contain at least one path.")

    frames = []
    for path in trade_signal_csvs:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Missing trade signal CSV: {p}")
        df = pd.read_csv(p)
        df["_source_file"] = str(p)
        frames.append(df)
        print(f"[SIGNALS] loaded {p} rows={len(df):,}")

    out = pd.concat(frames, ignore_index=True, sort=False)
    for ts_col in ["exec_ts", "ts", "timestamp", "datetime"]:
        if ts_col in out.columns:
            out["_sort_ts"] = pd.to_datetime(out[ts_col], errors="coerce")
            out = out.sort_values(["_sort_ts", "_source_file"]).drop(columns=["_sort_ts"])
            break

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"[SIGNALS] combined rows={len(out):,} -> {output_path}")
    return str(output_path)


def _normalize_trade_signal_inputs(config: NextDayGateConfig) -> None:
    """Materialize multiple trade-signal files to one CSV for reward/replay helpers."""

    signal_files = config.trade_signal_csvs
    if signal_files is None and isinstance(config.trade_signals_csv, (list, tuple)):
        signal_files = list(config.trade_signals_csv)

    if signal_files:
        combined_path = Path(config.output_dir) / "combined_trade_signals.csv"
        config.trade_signals_csv = _combine_trade_signal_csvs(list(signal_files), combined_path)
        config.trade_signal_csvs = list(signal_files)


def load_daily_rewards(rewards_csv: str | Path, tie_epsilon: float = 1e-9) -> pd.DataFrame:
    """
    Collapse trade-level rows into one daily reward row and label the best gate.

    Required logical fields:
    - timestamp/date
    - reward_force_buy
    - reward_force_sell
    - reward_free_trade or reward_free
    - day_open and day_close for FORCE_BUY/FORCE_SELL ties
    """

    raw = _read_csv(rewards_csv)
    date_col = _first_present(raw.columns, ["date", "ts", "timestamp", "exec_ts"])
    if date_col is None:
        raise ValueError("rewards_csv needs one of: date, ts, timestamp, exec_ts")

    rename = {}
    free_col = _first_present(raw.columns, ["reward_free", "reward_free_trade"])
    if free_col is not None:
        rename[free_col] = "reward_free"
    for col in ["reward_force_buy", "reward_force_sell", "day_open", "day_close"]:
        found = _first_present(raw.columns, [col])
        if found is not None:
            rename[found] = col

    df = raw.rename(columns=rename).copy()
    df["date"] = _normal_date(df[date_col])
    needed = ["date", "reward_force_buy", "reward_force_sell", "reward_free", "day_open", "day_close"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"rewards_csv is missing required columns after normalization: {missing}")

    for col in needed:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Trade rows repeat the same daily rewards. Keep the last non-null snapshot per day.
    daily = (
        df.dropna(subset=["date"])
        .sort_values("date")
        .groupby("date", as_index=False)
        .agg(
            reward_force_buy=("reward_force_buy", "last"),
            reward_force_sell=("reward_force_sell", "last"),
            reward_free=("reward_free", "last"),
            day_open=("day_open", "last"),
            day_close=("day_close", "last"),
        )
    )

    labels = []
    best_rewards = []
    for row in daily.itertuples(index=False):
        rewards = {
            "FORCE_BUY": float(row.reward_force_buy),
            "FREE": float(row.reward_free),
            "FORCE_SELL": float(row.reward_force_sell),
        }
        max_reward = max(rewards.values())
        winners = [gate for gate, reward in rewards.items() if abs(reward - max_reward) <= tie_epsilon]
        if set(winners) == {"FORCE_BUY", "FORCE_SELL"}:
            gate = "FORCE_BUY" if float(row.day_close) >= float(row.day_open) else "FORCE_SELL"
        elif "FREE" in winners:
            gate = "FREE"
        elif "FORCE_BUY" in winners and "FORCE_SELL" in winners:
            gate = "FORCE_BUY" if float(row.day_close) >= float(row.day_open) else "FORCE_SELL"
        else:
            gate = winners[0]
        labels.append(gate)
        best_rewards.append(float(rewards[gate]))

    daily["best_gate"] = labels
    daily["best_reward"] = best_rewards
    return daily.sort_values("date").reset_index(drop=True)


def load_trade_signals(signals_csv: str | Path) -> pd.DataFrame:
    """Normalize precomputed 5-minute model trade signals."""

    raw = _read_csv(signals_csv)
    ts_col = _first_present(raw.columns, ["exec_ts", "ts", "timestamp", "datetime"])
    if ts_col is None:
        raise ValueError("trade_signals_csv needs one of: exec_ts, ts, timestamp, datetime")
    px_col = _first_present(raw.columns, ["exec_px", "price", "px", "close"])
    if px_col is None:
        raise ValueError("trade_signals_csv needs one of: exec_px, price, px, close")
    side_col = _first_present(raw.columns, ["side", "signal", "direction"])
    if side_col is None:
        raise ValueError("trade_signals_csv needs one of: side, signal, direction")

    out = raw.copy()
    out["exec_ts_norm"] = pd.to_datetime(out[ts_col], errors="coerce")
    out["date"] = out["exec_ts_norm"].dt.normalize()
    out["side_norm"] = out[side_col].astype(str).str.lower().str.strip()
    out["exec_px_norm"] = pd.to_numeric(out[px_col], errors="coerce")
    if "fee" in out.columns:
        out["fee_norm"] = pd.to_numeric(out["fee"], errors="coerce").fillna(0.0)
    else:
        out["fee_norm"] = 0.0
    out = out[out["side_norm"].isin(["buy", "sell"])].dropna(subset=["exec_ts_norm", "exec_px_norm", "date"])
    return out.sort_values("exec_ts_norm").reset_index(drop=True)


def _simulate_one_day_signal_reward(
    *,
    day_signals: pd.DataFrame,
    gate: str,
    day_open: float,
    day_close: float,
    initial_capital: float,
    fee_pct: float,
) -> float:
    """Approximate one independent day's reward for gate-label training."""

    cash = float(initial_capital)
    qty = 0.0
    gate = str(gate)
    if gate == "FORCE_BUY" and np.isfinite(day_open) and day_open > 0:
        qty = cash / float(day_open)
        cash = 0.0

    for sig in day_signals.itertuples(index=False):
        side = str(getattr(sig, "side_norm"))
        px = float(getattr(sig, "exec_px_norm"))
        if not np.isfinite(px) or px <= 0:
            continue
        if gate == "FORCE_BUY" and side == "sell":
            continue
        if gate == "FORCE_SELL" and side == "buy":
            continue
        if side == "buy" and qty <= 0:
            fee = cash * float(fee_pct)
            invest = max(0.0, cash - fee)
            qty = invest / px
            cash = 0.0
        elif side == "sell" and qty > 0:
            gross = qty * px
            fee = gross * float(fee_pct)
            cash = gross - fee
            qty = 0.0

    end_equity = cash + qty * float(day_close)
    return float(end_equity - float(initial_capital))


def build_daily_rewards_from_trade_signals(
    *,
    trade_signals_csv: str | Path,
    market_csv: str | Path,
    initial_capital: float,
    fee_pct: float = 0.0,
    tie_epsilon: float = 1e-9,
) -> pd.DataFrame:
    """
    Build FORCE_BUY/FREE/FORCE_SELL daily rewards from precomputed 5m signals.

    These rewards are used for supervised labels. The later walk-forward
    simulation replays the predicted gate sequentially and can carry positions
    across days.
    """

    signals = load_trade_signals(trade_signals_csv)
    market = load_market_features(market_csv, lags=(2,))[["date", "open", "close"]].copy()
    market = market.dropna(subset=["date", "open", "close"]).sort_values("date").reset_index(drop=True)
    grouped = {day: g.copy() for day, g in signals.groupby("date")}

    rows = []
    for row in market.itertuples(index=False):
        day = pd.Timestamp(row.date).normalize()
        day_open = float(row.open)
        day_close = float(row.close)
        day_signals = grouped.get(day, pd.DataFrame(columns=signals.columns))
        rewards = {
            "FORCE_BUY": _simulate_one_day_signal_reward(
                day_signals=day_signals,
                gate="FORCE_BUY",
                day_open=day_open,
                day_close=day_close,
                initial_capital=float(initial_capital),
                fee_pct=float(fee_pct),
            ),
            "FREE": _simulate_one_day_signal_reward(
                day_signals=day_signals,
                gate="FREE",
                day_open=day_open,
                day_close=day_close,
                initial_capital=float(initial_capital),
                fee_pct=float(fee_pct),
            ),
            "FORCE_SELL": _simulate_one_day_signal_reward(
                day_signals=day_signals,
                gate="FORCE_SELL",
                day_open=day_open,
                day_close=day_close,
                initial_capital=float(initial_capital),
                fee_pct=float(fee_pct),
            ),
        }
        max_reward = max(rewards.values())
        winners = [gate for gate, reward in rewards.items() if abs(float(reward) - max_reward) <= tie_epsilon]
        if set(winners) == {"FORCE_BUY", "FORCE_SELL"}:
            best_gate = "FORCE_BUY" if day_close >= day_open else "FORCE_SELL"
        elif "FREE" in winners:
            best_gate = "FREE"
        elif "FORCE_BUY" in winners and "FORCE_SELL" in winners:
            best_gate = "FORCE_BUY" if day_close >= day_open else "FORCE_SELL"
        else:
            best_gate = winners[0]
        rows.append(
            {
                "date": day,
                "reward_force_buy": float(rewards["FORCE_BUY"]),
                "reward_free": float(rewards["FREE"]),
                "reward_force_sell": float(rewards["FORCE_SELL"]),
                "day_open": day_open,
                "day_close": day_close,
                "signal_count": int(len(day_signals)),
                "best_gate": best_gate,
                "best_reward": float(rewards[best_gate]),
            }
        )

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _load_one_macro_file(
    *,
    prefix: str,
    path: str | Path,
    lags: tuple[int, ...],
) -> pd.DataFrame:
    raw = _read_csv(path)
    date_col = _first_present(raw.columns, ["date", "timestamp", "datetime", "time"])
    if date_col is None:
        raise ValueError(f"Macro file {path} needs one of: date, timestamp, datetime, time")

    value_col = _first_present(raw.columns, ["close", "value", "price", "last", "yield"])
    if value_col is None:
        numeric_cols = [
            c
            for c in raw.columns
            if c != date_col and pd.to_numeric(raw[c], errors="coerce").notna().any()
        ]
        if not numeric_cols:
            raise ValueError(f"Macro file {path} has no numeric value column.")
        value_col = numeric_cols[-1]

    clean = _clean_prefix(prefix)
    out = pd.DataFrame(
        {
            "date": _normal_date(raw[date_col]),
            f"{clean}_value": pd.to_numeric(raw[value_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    out[f"{clean}_chg_1"] = out[f"{clean}_value"].diff()
    out[f"{clean}_pct_1"] = out[f"{clean}_value"].pct_change()
    for lag in lags:
        lag = int(lag)
        if lag <= 1:
            continue
        out[f"{clean}_chg_{lag}"] = out[f"{clean}_value"].diff(lag)
        out[f"{clean}_pct_{lag}"] = out[f"{clean}_value"].pct_change(lag)
        out[f"{clean}_z_{lag}"] = (
            (out[f"{clean}_value"] - out[f"{clean}_value"].rolling(lag).mean())
            / out[f"{clean}_value"].rolling(lag).std().replace(0.0, np.nan)
        )
    return out.replace([np.inf, -np.inf], np.nan)


def load_macro_features(
    *,
    macro_files: dict[str, str],
    macro_base_dir: str | Path,
    lags: tuple[int, ...],
) -> pd.DataFrame:
    frames = []
    base = Path(macro_base_dir)
    for prefix, file_name in dict(macro_files or {}).items():
        path = Path(file_name)
        if not path.is_absolute():
            path = base / path
        frames.append(_load_one_macro_file(prefix=prefix, path=path, lags=lags))
    if not frames:
        return pd.DataFrame(columns=["date"])

    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="date", how="outer")
    return out.sort_values("date").ffill().reset_index(drop=True)


def load_rich_daily_features(
    *,
    market_csv: str | Path,
    macro_files: dict[str, str],
    daily_chan_start: Optional[str],
    daily_chan_max_klines: int,
    feature_end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Reuse the previous daily feature stack: kline indicators, macro features,
    Chan BSP context, regime flags, and base-direction flags.

    The old logistic p_day fields are kept as zero placeholders so the feature
    names stay compatible without leaking the previous p-value model.
    """

    from adaptive_reward_checkpoint_fresh import _make_chan_config
    from pipelineCurrent import (
        SlidingWindowChan,
        build_klu,
        compute_chain_endpoints,
        compute_daily_kline_features,
        extract_bsp_rows_from_chan,
        feed_chan_one,
        latest_bsp_dir_up_to,
        load_macro_features_from_folder,
        load_ohlcv_csv,
        make_daily_features_one_model,
        normalize_bsp_row,
        regime_for_day_from_ends,
    )

    df_raw = load_ohlcv_csv(str(market_csv), "DAILY")
    start = pd.to_datetime(daily_chan_start) if daily_chan_start else pd.to_datetime(df_raw["timestamp"].min())
    mask = df_raw["timestamp"] >= start
    if feature_end_date is not None:
        mask &= df_raw["timestamp"] <= pd.to_datetime(feature_end_date)
    df_day = df_raw[mask].copy().reset_index(drop=True)
    if df_day.empty:
        raise ValueError("No daily bars available for rich daily feature generation.")

    df_feat = compute_daily_kline_features(df_day)
    df_feat["ts_norm"] = pd.to_datetime(df_feat["timestamp"]).dt.normalize()
    macro_folder = str(Path(market_csv).resolve().parent)
    macro_feat = load_macro_features_from_folder(folder=macro_folder, files=macro_files or {}, start=str(start.date()))
    df_feat = df_feat.merge(macro_feat, on="ts_norm", how="left").sort_values("timestamp").reset_index(drop=True)
    macro_prefixes = [str(pref).lower() if str(pref).lower().endswith("_") else f"{str(pref).lower()}_" for pref in (macro_files or {})]
    macro_cols = [c for c in df_feat.columns if any(str(c).startswith(pref) for pref in macro_prefixes)]

    daily_chan = SlidingWindowChan(
        code="DAILY_GATE_TFT",
        lv_list=None,
        config=_make_chan_config(),
        max_klines=int(daily_chan_max_klines),
    )
    bsp_rows_daily: list[dict[str, Any]] = []
    seen_bsp_daily: set[tuple[str, str, str]] = set()
    rows: list[dict[str, Any]] = []

    for _, r in df_feat.iterrows():
        ts = pd.to_datetime(r["timestamp"])
        day = ts.normalize()
        feed_chan_one(daily_chan, build_klu(ts, r["_open"], r["_high"], r["_low"], r["_close"], r.get("_vol", 0.0)))

        for rr0 in extract_bsp_rows_from_chan(daily_chan) or []:
            rr = normalize_bsp_row(dict(rr0))
            rr.setdefault("timestamp", ts)
            key = (pd.to_datetime(rr["timestamp"]).strftime("%Y-%m-%d"), rr["direction"], rr.get("bsp_type", "?"))
            if key not in seen_bsp_daily:
                seen_bsp_daily.add(key)
                bsp_rows_daily.append(rr)

        bsp_hist = [b for b in bsp_rows_daily if pd.to_datetime(b["timestamp"]) <= ts]
        ends = compute_chain_endpoints(bsp_hist)
        regime = regime_for_day_from_ends(
            day,
            ends,
            bsp_rows=bsp_hist,
            current_close=r.get("_close", np.nan),
        )
        base_dir = latest_bsp_dir_up_to(bsp_hist, ts)
        feat = make_daily_features_one_model(
            kline_row=r,
            bsp_hist_up_to_day=bsp_hist,
            p_val=0.0,
            dp_minK=0.0,
            dp_maxK=0.0,
            regime=regime,
            base_dir=base_dir,
            macro_cols=macro_cols,
        )
        feat["date"] = day
        feat["open"] = float(r["_open"])
        feat["high"] = float(r["_high"])
        feat["low"] = float(r["_low"])
        feat["close"] = float(r["_close"])
        feat["volume"] = float(r.get("_vol", 0.0))
        rows.append(feat)

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True).replace([np.inf, -np.inf], np.nan)


def load_market_features(
    market_csv: str | Path,
    lags: tuple[int, ...],
    macro_files: Optional[dict[str, str]] = None,
    macro_base_dir: str | Path = "DataAPI/data",
    feature_mode: str = "basic",
    daily_chan_start: Optional[str] = None,
    daily_chan_max_klines: int = 500,
    feature_end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Build daily features that are known after that day's close."""

    if str(feature_mode).lower() in {"rich", "rich_daily", "chan", "chan_daily"}:
        return load_rich_daily_features(
            market_csv=market_csv,
            macro_files=macro_files or {},
            daily_chan_start=daily_chan_start,
            daily_chan_max_klines=int(daily_chan_max_klines),
            feature_end_date=feature_end_date,
        )

    raw = _read_csv(market_csv)
    date_col = _first_present(raw.columns, ["date", "timestamp", "datetime"])
    if date_col is None:
        raise ValueError("market_csv needs one of: date, timestamp, datetime")

    rename = {date_col: "date"}
    for src, dst in [
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
    ]:
        found = _first_present(raw.columns, [src])
        if found is not None:
            rename[found] = dst

    df = raw.rename(columns=rename).copy()
    df["date"] = _normal_date(df["date"])
    if feature_end_date is not None:
        df = df[df["date"] <= pd.to_datetime(feature_end_date).normalize()].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)

    out = df[["date", "open", "high", "low", "close", "volume"]].copy()
    out["ret_1"] = out["close"].pct_change()
    out["intraday_ret"] = out["close"] / out["open"] - 1.0
    out["range_pct"] = (out["high"] - out["low"]) / out["close"].replace(0.0, np.nan)
    out["gap_ret"] = out["open"] / out["close"].shift(1) - 1.0
    out["dollar_volume"] = out["close"] * out["volume"]

    for lag in lags:
        lag = int(lag)
        if lag <= 0:
            continue
        out[f"ret_{lag}"] = out["close"].pct_change(lag)
        out[f"vol_{lag}"] = out["ret_1"].rolling(lag).std()
        out[f"volume_z_{lag}"] = (
            (out["volume"] - out["volume"].rolling(lag).mean())
            / out["volume"].rolling(lag).std().replace(0.0, np.nan)
        )
        out[f"close_to_ma_{lag}"] = out["close"] / out["close"].rolling(lag).mean() - 1.0

    out["day_of_week"] = out["date"].dt.dayofweek.astype(float)
    out["month"] = out["date"].dt.month.astype(float)
    macro = load_macro_features(macro_files=macro_files or {}, macro_base_dir=macro_base_dir, lags=lags)
    if not macro.empty:
        out = out.merge(macro, on="date", how="left").ffill()
    return out.replace([np.inf, -np.inf], np.nan)


def build_supervised_frame(config: NextDayGateConfig) -> pd.DataFrame:
    """
    Merge market features with daily rewards and shift labels one session ahead.

    Row date D contains features known at the close of D. The target/reward fields
    are from the next available reward date, so a prediction made at D is scored
    on D+1 without using D+1 information in X.
    """

    if config.trade_signals_csv:
        rewards = build_daily_rewards_from_trade_signals(
            trade_signals_csv=config.trade_signals_csv,
            market_csv=config.market_csv,
            initial_capital=float(config.initial_capital),
            fee_pct=float(config.fee_pct),
            tie_epsilon=float(config.reward_tie_epsilon),
        )
    else:
        rewards = load_daily_rewards(config.rewards_csv, tie_epsilon=float(config.reward_tie_epsilon))
    feature_end_date = None
    if config.end_year is not None:
        feature_end_date = f"{int(config.end_year)}-12-31"
    features = load_market_features(
        config.market_csv,
        tuple(config.feature_lags),
        macro_files=config.macro_files,
        macro_base_dir=config.macro_base_dir,
        feature_mode=config.feature_mode,
        daily_chan_start=config.daily_chan_start,
        daily_chan_max_klines=config.daily_chan_max_klines,
        feature_end_date=feature_end_date,
    )
    df = features.merge(rewards, on="date", how="inner").sort_values("date").reset_index(drop=True)

    target_cols = [
        "date",
        "best_gate",
        "best_reward",
        "reward_force_buy",
        "reward_free",
        "reward_force_sell",
        "day_open",
        "day_close",
    ]
    next_target = df[target_cols].shift(-1)
    next_target = next_target.rename(
        columns={
            "date": "target_date",
            "best_gate": "target_gate",
            "best_reward": "target_best_reward",
            "reward_force_buy": "target_reward_force_buy",
            "reward_free": "target_reward_free",
            "reward_force_sell": "target_reward_force_sell",
            "day_open": "target_day_open",
            "day_close": "target_day_close",
        }
    )
    out = pd.concat([df, next_target], axis=1)
    out = out.dropna(subset=["target_date", "target_gate"]).reset_index(drop=True)
    out["year"] = pd.to_datetime(out["target_date"]).dt.year.astype(int)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    blocked = {
        "date",
        "target_date",
        "best_gate",
        "target_gate",
        "best_reward",
        "target_best_reward",
        "reward_force_buy",
        "reward_free",
        "reward_force_sell",
        "target_reward_force_buy",
        "target_reward_free",
        "target_reward_force_sell",
        "day_open",
        "day_close",
        "target_day_open",
        "target_day_close",
        "year",
    }
    cols = [
        c
        for c in df.columns
        if c not in blocked
        and pd.api.types.is_numeric_dtype(df[c])
        and pd.to_numeric(df[c], errors="coerce").notna().any()
    ]
    if not cols:
        raise ValueError("No numeric feature columns were found.")
    return cols


if nn is not None:
    class TinyTFTNet(nn.Module):
        def __init__(self, n_features: int, n_classes: int, hidden: int, heads: int):
            super().__init__()
            self.var_gate = nn.Sequential(
                nn.Linear(n_features, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_features),
            )
            self.input_proj = nn.Linear(n_features, hidden)
            self.encoder = nn.LSTM(hidden, hidden, batch_first=True)
            self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
            self.gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
            self.norm = nn.LayerNorm(hidden)
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(hidden, n_classes),
            )

        def forward(self, x):
            weights = torch.softmax(self.var_gate(x[:, -1, :]), dim=-1).unsqueeze(1)
            xw = x * weights
            z = self.input_proj(xw)
            enc, _ = self.encoder(z)
            query = enc[:, -1:, :]
            ctx, attn = self.attn(query, enc, enc, need_weights=True)
            fused = self.norm(ctx.squeeze(1) * self.gate(ctx.squeeze(1)) + enc[:, -1, :])
            return self.head(fused), weights.squeeze(1), attn.squeeze(1)
else:
    TinyTFTNet = None


class TemporalFusionGateClassifier:
    """
    Lightweight TFT-style classifier with a scikit-like API.

    Components:
    - variable-selection gate over input features
    - recurrent sequence encoder
    - multi-head temporal attention
    - classifier head for FORCE_BUY/FREE/FORCE_SELL
    """

    def __init__(self, config: NextDayGateConfig, feature_cols: list[str]):
        self.config = config
        self.feature_cols = list(feature_cols)
        self.sequence_length = int(config.tft_sequence_length)
        self.classes_ = np.array(GATES, dtype=object)
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.model_: Any = None
        self.history_: list[np.ndarray] = []

    def _device(self):
        import torch

        if str(self.config.tft_device).lower() == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(str(self.config.tft_device))

    def _build_net(self, n_features: int, n_classes: int):
        if TinyTFTNet is None:
            raise ImportError("PyTorch is required for model_type='tft'.")

        hidden = int(self.config.tft_hidden_size)
        heads = max(1, int(self.config.tft_attention_heads))
        if hidden % heads != 0:
            heads = 1
        return TinyTFTNet(n_features=n_features, n_classes=n_classes, hidden=hidden, heads=heads)

    def _to_matrix(self, X: pd.DataFrame) -> np.ndarray:
        mat = X.reindex(columns=self.feature_cols).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        mat = np.where(np.isfinite(mat), mat, np.nan)
        return mat

    def _fit_scaler(self, mat: np.ndarray) -> np.ndarray:
        mean = np.nanmean(mat, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        filled = np.where(np.isfinite(mat), mat, mean)
        scale = np.nanstd(filled, axis=0)
        scale = np.where((np.isfinite(scale)) & (scale > 1e-12), scale, 1.0)
        self.mean_ = mean.astype(np.float32)
        self.scale_ = scale.astype(np.float32)
        return ((filled - self.mean_) / self.scale_).astype(np.float32)

    def _transform(self, mat: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("TFT classifier is not fitted.")
        filled = np.where(np.isfinite(mat), mat, self.mean_)
        return ((filled - self.mean_) / self.scale_).astype(np.float32)

    def _make_sequences(self, mat: np.ndarray, labels: Optional[np.ndarray] = None):
        seq_len = max(2, int(self.sequence_length))
        xs = []
        ys = []
        for i in range(len(mat)):
            start = max(0, i - seq_len + 1)
            seq = mat[start : i + 1]
            if len(seq) < seq_len:
                pad = np.repeat(seq[:1], seq_len - len(seq), axis=0)
                seq = np.vstack([pad, seq])
            xs.append(seq)
            if labels is not None:
                ys.append(labels[i])
        x_arr = np.asarray(xs, dtype=np.float32)
        if labels is None:
            return x_arr
        return x_arr, np.asarray(ys, dtype=np.int64)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(int(self.config.random_state))
        np.random.seed(int(self.config.random_state))

        raw = self._to_matrix(X)
        mat = self._fit_scaler(raw)
        class_to_idx = {gate: i for i, gate in enumerate(GATES)}
        labels = np.asarray([class_to_idx.get(str(v), 1) for v in y], dtype=np.int64)
        x_seq, y_idx = self._make_sequences(mat, labels)

        device = self._device()
        net = self._build_net(n_features=len(self.feature_cols), n_classes=len(GATES)).to(device)
        counts = np.bincount(y_idx, minlength=len(GATES)).astype(np.float32)
        weights = counts.sum() / np.maximum(counts, 1.0)
        weights = weights / max(float(weights.mean()), 1e-12)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
        opt = torch.optim.AdamW(
            net.parameters(),
            lr=float(self.config.tft_learning_rate),
            weight_decay=float(self.config.tft_weight_decay),
        )
        ds = TensorDataset(torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_idx, dtype=torch.long))
        loader = DataLoader(ds, batch_size=int(self.config.tft_batch_size), shuffle=True)

        best_loss = np.inf
        best_state = None
        stale = 0
        epochs = max(1, int(self.config.tft_epochs))
        for _ in range(epochs):
            net.train()
            losses = []
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad(set_to_none=True)
                logits, _, _ = net(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach().cpu()))
            avg = float(np.mean(losses)) if losses else np.inf
            if avg + 1e-5 < best_loss:
                best_loss = avg
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= int(self.config.tft_patience):
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        self.model_ = net.to(device)
        self.history_ = [row.copy() for row in mat[-max(1, self.sequence_length - 1) :]]
        return self

    def _sequence_for_next(self, X: pd.DataFrame):
        mat = self._transform(self._to_matrix(X))
        rows = self.history_ + [mat[-1]]
        seq_len = max(2, int(self.sequence_length))
        seq = np.asarray(rows[-seq_len:], dtype=np.float32)
        if len(seq) < seq_len:
            pad = np.repeat(seq[:1], seq_len - len(seq), axis=0)
            seq = np.vstack([pad, seq])
        return seq.reshape(1, seq_len, -1), mat[-1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        if self.model_ is None:
            raise RuntimeError("TFT classifier is not fitted.")
        seq, _ = self._sequence_for_next(X)
        device = self._device()
        self.model_.eval()
        with torch.no_grad():
            xb = torch.tensor(seq, dtype=torch.float32, device=device)
            logits, var_weights, attn = self.model_(xb)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            self.last_variable_weights_ = var_weights.detach().cpu().numpy()[0]
            self.last_attention_ = attn.detach().cpu().numpy()[0]
        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def observe(self, X: pd.DataFrame) -> None:
        mat = self._transform(self._to_matrix(X))
        self.history_.append(mat[-1].copy())
        keep = max(1, int(self.sequence_length) - 1)
        self.history_ = self.history_[-keep:]


def make_model(config: NextDayGateConfig) -> Pipeline:
    model_type = str(config.model_type).lower()
    if model_type in {"tft", "temporal_fusion_transformer", "temporal_fusion"}:
        raise RuntimeError("TFT models are built with _fit_model because they need feature column metadata.")
    if model_type in {"extra_trees", "extratrees", "et"}:
        clf = ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=int(config.random_state),
            n_jobs=int(config.n_jobs),
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", clf)])
    if model_type in {"random_forest", "rf"}:
        clf = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=int(config.random_state),
            n_jobs=int(config.n_jobs),
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", clf)])
    if model_type in {"logistic", "logreg", "lr"}:
        clf = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=int(config.random_state))
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", clf),
            ]
        )
    raise ValueError(f"Unsupported model_type={config.model_type!r}. Use extra_trees, random_forest, or logistic.")


def _class_probabilities(model: Pipeline, x: pd.DataFrame) -> dict[str, float]:
    if not hasattr(model, "predict_proba"):
        return {gate: np.nan for gate in GATES}
    probs = model.predict_proba(x)[0]
    classes = list(model.classes_)
    return {gate: float(probs[classes.index(gate)]) if gate in classes else 0.0 for gate in GATES}


def reward_for_gate(row: pd.Series, gate: str) -> float:
    col = f"target_{REWARD_COLUMNS.get(str(gate), 'reward_free')}"
    return float(pd.to_numeric(row.get(col, 0.0), errors="coerce") or 0.0)


def _trade_record(
    *,
    row: pd.Series | dict[str, Any],
    year: int,
    target_date: str,
    gate: str,
    side: str,
    px: float,
    qty: float,
    fee: float,
    cash_after: float,
    qty_after: float,
    source: str,
    pnl: float = 0.0,
    signal: Optional[pd.Series] = None,
) -> dict[str, Any]:
    rec = {
        "year": int(year),
        "target_date": target_date,
        "predicted_gate": gate,
        "side": side,
        "exec_px": float(px),
        "qty": float(qty),
        "fee": float(fee),
        "pnl": float(pnl),
        "cash_after": float(cash_after),
        "position_qty_after": float(qty_after),
        "trade_source": source,
    }
    if signal is not None:
        for col in ["exec_ts_norm", "pred", "th", "reason", "seen_idx", "exec_idx"]:
            if col in signal.index:
                val = signal.get(col)
                rec[col.replace("_norm", "")] = str(val) if isinstance(val, pd.Timestamp) else val
    else:
        rec["exec_ts"] = f"{target_date} open"
        rec["reason"] = source
    return rec


def replay_predictions_with_trade_signals(
    *,
    predictions: pd.DataFrame,
    trade_signals_csv: str | Path,
    fee_pct: float,
    initial_capital: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Replay predicted daily gates against the precomputed 5m trade signals.

    The replay is sequential inside each target year. Each year starts flat with
    `initial_capital`; positions can carry overnight inside that same year.
    """

    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame()

    signals = load_trade_signals(trade_signals_csv)
    grouped = {day: g.copy() for day, g in signals.groupby("date")}
    daily_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for year, year_pred in predictions.sort_values("target_date").groupby("year", sort=True):
        cash = float(initial_capital)
        qty = 0.0
        entry_px = np.nan
        prev_end_capital = float(initial_capital)

        for row in year_pred.itertuples(index=False):
            target_day = pd.Timestamp(row.target_date).normalize()
            target_date = target_day.date().isoformat()
            gate = str(row.predicted_gate)
            day_open = float(row.target_day_open)
            day_close = float(row.target_day_close)
            start_capital = float(prev_end_capital)
            open_equity = cash + qty * day_open
            day_trades_before = len(trade_rows)
            realized_pnl = 0.0

            if gate == "FORCE_BUY" and qty <= 0 and np.isfinite(day_open) and day_open > 0:
                fee = cash * float(fee_pct)
                invest = max(0.0, cash - fee)
                buy_qty = invest / day_open
                cash = 0.0
                qty = buy_qty
                entry_px = day_open
                trade_rows.append(
                    _trade_record(
                        row={},
                        year=int(year),
                        target_date=target_date,
                        gate=gate,
                        side="buy",
                        px=day_open,
                        qty=buy_qty,
                        fee=fee,
                        cash_after=cash,
                        qty_after=qty,
                        source="gate_forced_open_buy",
                    )
                )
            elif gate == "FORCE_SELL" and qty > 0 and np.isfinite(day_open) and day_open > 0:
                gross = qty * day_open
                fee = gross * float(fee_pct)
                pnl = (day_open - float(entry_px)) * qty if np.isfinite(entry_px) else 0.0
                realized_pnl += pnl
                cash = gross - fee
                sold_qty = qty
                qty = 0.0
                entry_px = np.nan
                trade_rows.append(
                    _trade_record(
                        row={},
                        year=int(year),
                        target_date=target_date,
                        gate=gate,
                        side="sell",
                        px=day_open,
                        qty=sold_qty,
                        fee=fee,
                        cash_after=cash,
                        qty_after=qty,
                        source="gate_forced_open_sell",
                        pnl=pnl,
                    )
                )

            for _, sig in grouped.get(target_day, pd.DataFrame()).iterrows():
                side = str(sig["side_norm"])
                if gate == "FORCE_BUY" and side == "sell":
                    continue
                if gate == "FORCE_SELL" and side == "buy":
                    continue

                px = float(sig["exec_px_norm"])
                if not np.isfinite(px) or px <= 0:
                    continue
                if side == "buy" and qty <= 0:
                    fee = cash * float(fee_pct)
                    invest = max(0.0, cash - fee)
                    buy_qty = invest / px
                    cash = 0.0
                    qty = buy_qty
                    entry_px = px
                    trade_rows.append(
                        _trade_record(
                            row={},
                            year=int(year),
                            target_date=target_date,
                            gate=gate,
                            side="buy",
                            px=px,
                            qty=buy_qty,
                            fee=fee,
                            cash_after=cash,
                            qty_after=qty,
                            source="5m_signal",
                            signal=sig,
                        )
                    )
                elif side == "sell" and qty > 0:
                    gross = qty * px
                    fee = gross * float(fee_pct)
                    pnl = (px - float(entry_px)) * qty if np.isfinite(entry_px) else 0.0
                    realized_pnl += pnl
                    cash = gross - fee
                    sold_qty = qty
                    qty = 0.0
                    entry_px = np.nan
                    trade_rows.append(
                        _trade_record(
                            row={},
                            year=int(year),
                            target_date=target_date,
                            gate=gate,
                            side="sell",
                            px=px,
                            qty=sold_qty,
                            fee=fee,
                            cash_after=cash,
                            qty_after=qty,
                            source="5m_signal",
                            pnl=pnl,
                            signal=sig,
                        )
                    )

            end_capital = cash + qty * day_close
            daily_rows.append(
                {
                    "target_date": target_date,
                    "year": int(year),
                    "predicted_gate": gate,
                    "actual_best_gate": str(row.actual_best_gate),
                    "start_capital": float(start_capital),
                    "open_equity": float(open_equity),
                    "end_capital": float(end_capital),
                    "daily_pnl": float(end_capital - start_capital),
                    "realized_trade_pnl": float(realized_pnl),
                    "cash": float(cash),
                    "position_qty": float(qty),
                    "mark_price": float(day_close),
                    "day_open": day_open,
                    "day_close": day_close,
                    "trade_count": int(len(trade_rows) - day_trades_before),
                    "five_min_signal_trade_count": int(
                        sum(1 for r in trade_rows[day_trades_before:] if r.get("trade_source") == "5m_signal")
                    ),
                }
            )
            prev_end_capital = float(end_capital)

    return pd.DataFrame(daily_rows), pd.DataFrame(trade_rows)


def _fit_model(train_df: pd.DataFrame, cols: list[str], config: NextDayGateConfig) -> Optional[Pipeline]:
    train_df = train_df.dropna(subset=["target_gate"]).copy()
    if len(train_df) < int(config.min_train_days):
        return None
    if train_df["target_gate"].nunique() < 2:
        return None
    if str(config.model_type).lower() in {"tft", "temporal_fusion_transformer", "temporal_fusion"}:
        model = TemporalFusionGateClassifier(config=config, feature_cols=cols)
    else:
        model = make_model(config)
    model.fit(train_df[cols], train_df["target_gate"].astype(str))
    return model


def _checkpoint_payload(
    *,
    config: NextDayGateConfig,
    feature_cols: list[str],
    all_results: list[dict[str, Any]],
    completed_years: list[int],
    current_year: Optional[int],
    next_row_index: int,
    model: Optional[Pipeline],
) -> dict[str, Any]:
    return {
        "schema": "next_day_gate_model_checkpoint_v1",
        "config": asdict(config),
        "feature_cols": feature_cols,
        "all_results": all_results,
        "completed_years": completed_years,
        "current_year": current_year,
        "next_row_index": int(next_row_index),
        "model": model,
        "saved_at": pd.Timestamp.now().isoformat(),
    }


def _save_checkpoint(path: str | Path, payload: dict[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp")
    joblib.dump(payload, tmp)
    try:
        os.replace(tmp, path)
        return str(path)
    except OSError:
        fallback = path.with_name(f"{path.stem}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S_%f')}{path.suffix}")
        os.replace(tmp, fallback)
        return str(fallback)


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = joblib.load(path)
    if payload.get("schema") != "next_day_gate_model_checkpoint_v1":
        raise ValueError(f"Unsupported checkpoint schema in {path}")
    return payload


def _write_outputs(output_dir: str | Path, results: list[dict[str, Any]], config: NextDayGateConfig) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = pd.DataFrame(results)
    daily_log = pd.DataFrame()
    executed_trades = pd.DataFrame()

    if config.trade_signals_csv and not pred.empty:
        daily_log, executed_trades = replay_predictions_with_trade_signals(
            predictions=pred,
            trade_signals_csv=config.trade_signals_csv,
            fee_pct=float(config.fee_pct),
            initial_capital=float(config.initial_capital),
        )
        if not daily_log.empty:
            pred = pred.merge(
                daily_log[
                    [
                        "target_date",
                        "start_capital",
                        "open_equity",
                        "end_capital",
                        "daily_pnl",
                        "cash",
                        "position_qty",
                        "trade_count",
                        "five_min_signal_trade_count",
                    ]
                ],
                on="target_date",
                how="left",
            )

    pred_path = out_dir / "daily_predictions.csv"
    pred.to_csv(pred_path, index=False)
    daily_log_path = out_dir / "daily_log.csv"
    daily_log.to_csv(daily_log_path, index=False)
    executed_trades_path = out_dir / "executed_5m_trades.csv"
    executed_trades.to_csv(executed_trades_path, index=False)
    by_year_dir = out_dir / "by_year"
    by_year_dir.mkdir(parents=True, exist_ok=True)
    by_year_paths: dict[str, dict[str, str]] = {}
    if not pred.empty and "year" in pred.columns:
        for year, year_pred in pred.groupby("year", sort=True):
            year_key = str(int(year))
            year_paths = by_year_paths.setdefault(year_key, {})
            path = by_year_dir / f"daily_predictions_{year_key}.csv"
            year_pred.to_csv(path, index=False)
            year_paths["daily_predictions"] = str(path)
    if not daily_log.empty:
        daily_log_years = daily_log.copy()
        if "year" not in daily_log_years.columns and "target_date" in daily_log_years.columns:
            daily_log_years["year"] = pd.to_datetime(daily_log_years["target_date"], errors="coerce").dt.year
        if "year" in daily_log_years.columns:
            for year, year_log in daily_log_years.dropna(subset=["year"]).groupby("year", sort=True):
                year_key = str(int(year))
                year_paths = by_year_paths.setdefault(year_key, {})
                path = by_year_dir / f"daily_log_{year_key}.csv"
                year_log.to_csv(path, index=False)
                year_paths["daily_log"] = str(path)
    if not executed_trades.empty:
        trade_years = executed_trades.copy()
        if "year" not in trade_years.columns and "target_date" in trade_years.columns:
            trade_years["year"] = pd.to_datetime(trade_years["target_date"], errors="coerce").dt.year
        if "year" in trade_years.columns:
            for year, year_trades in trade_years.dropna(subset=["year"]).groupby("year", sort=True):
                year_key = str(int(year))
                year_paths = by_year_paths.setdefault(year_key, {})
                path = by_year_dir / f"executed_5m_trades_{year_key}.csv"
                year_trades.to_csv(path, index=False)
                year_paths["executed_trades"] = str(path)

    if pred.empty:
        summary = pd.DataFrame()
    elif config.trade_signals_csv and "end_capital" in pred.columns:
        grouped = pred.groupby("year", as_index=False)
        summary = grouped.agg(
            first_target_date=("target_date", "first"),
            last_target_date=("target_date", "last"),
            trading_days=("target_date", "count"),
            final_capital=("end_capital", "last"),
            total_reward=("daily_pnl", "sum"),
            accuracy=("is_correct", "mean"),
            oracle_reward=("oracle_reward", "sum"),
            executed_trades=("trade_count", "sum"),
            executed_5m_signal_trades=("five_min_signal_trade_count", "sum"),
        )
        summary["initial_capital"] = float(config.initial_capital)
        summary["return_pct"] = summary["final_capital"] / float(config.initial_capital) - 1.0
        summary["oracle_final_capital"] = float(config.initial_capital) + summary["oracle_reward"]
    else:
        grouped = pred.groupby("year", as_index=False)
        summary = grouped.agg(
            first_target_date=("target_date", "first"),
            last_target_date=("target_date", "last"),
            trading_days=("target_date", "count"),
            final_capital=("capital", "last"),
            total_reward=("realized_reward", "sum"),
            accuracy=("is_correct", "mean"),
            oracle_reward=("oracle_reward", "sum"),
        )
        summary["initial_capital"] = float(config.initial_capital)
        summary["return_pct"] = summary["final_capital"] / float(config.initial_capital) - 1.0
        summary["oracle_final_capital"] = float(config.initial_capital) + summary["oracle_reward"]
    summary_path = out_dir / "yearly_summary.csv"
    summary.to_csv(summary_path, index=False)

    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(asdict(config), indent=2, default=str), encoding="utf-8")
    return {
        "predictions": pred,
        "daily_log": daily_log,
        "executed_trades": executed_trades,
        "summary": summary,
        "predictions_path": str(pred_path),
        "daily_log_path": str(daily_log_path),
        "executed_trades_path": str(executed_trades_path),
        "summary_path": str(summary_path),
        "config_path": str(config_path),
        "by_year_dir": str(by_year_dir),
        "by_year_paths": by_year_paths,
    }


def run_next_day_gate_experiment(config: Optional[NextDayGateConfig] = None) -> dict[str, Any]:
    """
    Run the year-by-year no-leak simulation.

    Each target year starts with `initial_capital`. Training uses only rows whose
    target date is earlier than the day being predicted. If interrupted, resume
    from the checkpoint path stored in the config.
    """

    config = config or NextDayGateConfig()
    _normalize_trade_signal_inputs(config)
    df = build_supervised_frame(config)
    cols = feature_columns(df)
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    main_checkpoint = checkpoint_dir / "latest.joblib"
    latest_checkpoint_path = str(main_checkpoint)

    all_results: list[dict[str, Any]] = []
    completed_years: list[int] = []
    resume_year = None
    resume_row_index = 0
    model: Optional[Pipeline] = None

    if config.resume_from_checkpoint:
        payload = _load_checkpoint(config.resume_from_checkpoint)
        all_results = list(payload.get("all_results", []))
        completed_years = [int(y) for y in payload.get("completed_years", [])]
        resume_year = payload.get("current_year")
        resume_row_index = int(payload.get("next_row_index", 0))
        model = payload.get("model")

    years = sorted(df["year"].unique().tolist())
    if config.start_year is not None:
        years = [y for y in years if y >= int(config.start_year)]
    if resume_year is not None:
        years = [y for y in years if y >= int(resume_year)]
    if config.end_year is not None:
        years = [y for y in years if y <= int(config.end_year)]
    if not years:
        raise ValueError("No target years are available for the requested start/end range.")

    for year in years:
        if year in completed_years:
            continue

        year_df = df[df["year"] == year].copy().reset_index(drop=False).rename(columns={"index": "_global_index"})
        if year_df.empty:
            completed_years.append(year)
            continue

        if config.save_year_start_checkpoints and resume_year is None:
            train_start_df = df[df["target_date"] < year_df.iloc[0]["target_date"]]
            if config.train_lookback_days is not None:
                train_start_df = train_start_df.tail(int(config.train_lookback_days))
            year_start_model = _fit_model(train_start_df, cols, config)
            _save_checkpoint(
                checkpoint_dir / f"year_start_{year}.joblib",
                _checkpoint_payload(
                    config=config,
                    feature_cols=cols,
                    all_results=all_results,
                    completed_years=completed_years,
                    current_year=year,
                    next_row_index=0,
                    model=year_start_model,
                ),
            )

        start_i = resume_row_index if resume_year == year else 0
        capital = float(config.initial_capital)
        if start_i > 0 and all_results:
            prior_for_year = [r for r in all_results if int(r["year"]) == int(year)]
            if prior_for_year:
                capital = float(prior_for_year[-1]["capital"])

        model = model if resume_year == year and model is not None else None
        days_since_fit = int(config.retrain_every_n_days)

        for local_i in range(start_i, len(year_df)):
            row = year_df.iloc[local_i]
            train_df = df[df["target_date"] < row["target_date"]]
            if config.train_lookback_days is not None:
                train_df = train_df.tail(int(config.train_lookback_days))
            if model is None or days_since_fit >= int(config.retrain_every_n_days):
                fitted = _fit_model(train_df, cols, config)
                if fitted is not None:
                    model = fitted
                    days_since_fit = 0

            if model is None:
                predicted_gate = "FREE"
                probs = {gate: np.nan for gate in GATES}
            else:
                x = pd.DataFrame([row[cols].to_dict()])
                predicted_gate = str(model.predict(x)[0])
                probs = _class_probabilities(model, x)
                if hasattr(model, "observe"):
                    model.observe(x)

            reward = reward_for_gate(row, predicted_gate) - float(config.fee_per_day)
            oracle_gate = str(row["target_gate"])
            oracle_reward = reward_for_gate(row, oracle_gate)
            capital += float(reward)
            result = {
                "decision_date": pd.Timestamp(row["date"]).date().isoformat(),
                "target_date": pd.Timestamp(row["target_date"]).date().isoformat(),
                "year": int(year),
                "predicted_gate": predicted_gate,
                "actual_best_gate": oracle_gate,
                "is_correct": bool(predicted_gate == oracle_gate),
                "realized_reward": float(reward),
                "oracle_reward": float(oracle_reward),
                "capital": float(capital),
                "prob_force_buy": probs["FORCE_BUY"],
                "prob_free": probs["FREE"],
                "prob_force_sell": probs["FORCE_SELL"],
                "target_reward_force_buy": float(row["target_reward_force_buy"]),
                "target_reward_free": float(row["target_reward_free"]),
                "target_reward_force_sell": float(row["target_reward_force_sell"]),
                "target_day_open": float(row["target_day_open"]),
                "target_day_close": float(row["target_day_close"]),
                "train_rows": int(len(train_df)),
            }
            all_results.append(result)
            days_since_fit += 1

            if config.checkpoint_every_n_days > 0 and (local_i + 1) % int(config.checkpoint_every_n_days) == 0:
                latest_checkpoint_path = _save_checkpoint(
                    main_checkpoint,
                    _checkpoint_payload(
                        config=config,
                        feature_cols=cols,
                        all_results=all_results,
                        completed_years=completed_years,
                        current_year=year,
                        next_row_index=local_i + 1,
                        model=model,
                    ),
                )

        completed_years.append(year)
        resume_year = None
        resume_row_index = 0
        model = None
        latest_checkpoint_path = _save_checkpoint(
            main_checkpoint,
            _checkpoint_payload(
                config=config,
                feature_cols=cols,
                all_results=all_results,
                completed_years=completed_years,
                current_year=None,
                next_row_index=0,
                model=None,
            ),
        )

    outputs = _write_outputs(config.output_dir, all_results, config)
    outputs["checkpoint_path"] = latest_checkpoint_path
    outputs["feature_columns"] = cols
    return outputs


def parse_args() -> NextDayGateConfig:
    parser = argparse.ArgumentParser(description="Train and simulate a no-leak next-day gate model.")
    parser.add_argument("--rewards-csv", default=NextDayGateConfig.rewards_csv)
    parser.add_argument("--trade-signals-csv", default=None)
    parser.add_argument("--market-csv", default=NextDayGateConfig.market_csv)
    parser.add_argument("--output-dir", default=NextDayGateConfig.output_dir)
    parser.add_argument("--checkpoint-dir", default=NextDayGateConfig.checkpoint_dir)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--initial-capital", type=float, default=NextDayGateConfig.initial_capital)
    parser.add_argument("--min-train-days", type=int, default=NextDayGateConfig.min_train_days)
    parser.add_argument("--train-lookback-days", type=int, default=None)
    parser.add_argument("--retrain-every-n-days", type=int, default=NextDayGateConfig.retrain_every_n_days)
    parser.add_argument("--model-type", default=NextDayGateConfig.model_type)
    parser.add_argument("--n-jobs", type=int, default=NextDayGateConfig.n_jobs)
    parser.add_argument("--feature-mode", default=NextDayGateConfig.feature_mode)
    parser.add_argument("--tft-sequence-length", type=int, default=NextDayGateConfig.tft_sequence_length)
    parser.add_argument("--tft-hidden-size", type=int, default=NextDayGateConfig.tft_hidden_size)
    parser.add_argument("--tft-attention-heads", type=int, default=NextDayGateConfig.tft_attention_heads)
    parser.add_argument("--tft-epochs", type=int, default=NextDayGateConfig.tft_epochs)
    parser.add_argument("--tft-batch-size", type=int, default=NextDayGateConfig.tft_batch_size)
    parser.add_argument("--tft-learning-rate", type=float, default=NextDayGateConfig.tft_learning_rate)
    parser.add_argument("--tft-device", default=NextDayGateConfig.tft_device)
    parser.add_argument("--daily-chan-start", default=None)
    parser.add_argument("--daily-chan-max-klines", type=int, default=NextDayGateConfig.daily_chan_max_klines)
    parser.add_argument("--checkpoint-every-n-days", type=int, default=NextDayGateConfig.checkpoint_every_n_days)
    parser.add_argument("--resume-from-checkpoint", default=None)
    return NextDayGateConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    result = run_next_day_gate_experiment(parse_args())
    print("daily predictions:", result["predictions_path"])
    print("daily log:", result["daily_log_path"])
    print("executed trades:", result["executed_trades_path"])
    print("yearly summary:", result["summary_path"])
    print("checkpoint:", result["checkpoint_path"])
