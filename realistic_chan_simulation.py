"""Notebook-friendly, point-in-time Chan/BSP strategy simulation.

The event order is:

1. A kline closes.
2. Chan processes that kline and may reveal one or more new BSPs.
3. The strategy evaluates only those newly revealed BSPs.
4. An accepted order executes at the next kline's open.

This module can either regenerate BSPs from the source OHLCV file or replay a
saved BSP workbook that contains ``snapshot_first_seen``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_FIELD, DATA_SRC, KL_TYPE
from Common.CTime import CTime
from CustomBuySellPoint import SegBspStrategy, SegBspStrategyConfig
from KLine.KLine_Unit import CKLine_Unit
from ModelStrategy import BacktestChanConfig, run_bsp_backtest
from ModelStrategy.backtest import BacktestResult
from Trade import RiskConfig, RiskManager
from sliding_window_chan import SlidingWindowChan


ReplayMode = Literal["saved", "regenerate"]


class _SimulationSlidingWindowChan(SlidingWindowChan):
    """Sliding Chan instance that receives klines directly from this module."""

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


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {str(col).lower(): str(col) for col in df.columns}
    for candidate in candidates:
        found = lookup.get(candidate.lower())
        if found is not None:
            return found
    return None


def _load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts_col = _pick_column(df, ["timestamp", "date", "datetime", "time"]) or str(df.columns[0])
    open_col = _pick_column(df, ["open", "o"])
    high_col = _pick_column(df, ["high", "h"])
    low_col = _pick_column(df, ["low", "l"])
    close_col = _pick_column(df, ["close", "adj_close", "adj close", "adjclose", "c"])
    volume_col = _pick_column(df, ["volume", "vol", "v"])
    if open_col is None or close_col is None:
        raise ValueError("Price CSV must contain open and close columns")
    high_col = high_col or close_col
    low_col = low_col or close_col

    df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
    df["_open"] = pd.to_numeric(df[open_col], errors="coerce")
    df["_high"] = pd.to_numeric(df[high_col], errors="coerce")
    df["_low"] = pd.to_numeric(df[low_col], errors="coerce")
    df["_close"] = pd.to_numeric(df[close_col], errors="coerce")
    df["_vol"] = (
        pd.to_numeric(df[volume_col], errors="coerce") if volume_col is not None else 0.0
    )
    return (
        df.dropna(subset=["timestamp", "_open", "_high", "_low", "_close"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def _to_ctime(timestamp) -> CTime:
    dt = pd.Timestamp(timestamp).to_pydatetime()
    try:
        return CTime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, auto=False)
    except TypeError:
        return CTime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)


def _build_klu(timestamp, open_, high, low, close, volume=0.0) -> CKLine_Unit:
    time = _to_ctime(timestamp)
    return CKLine_Unit(
        {
            DATA_FIELD.FIELD_TIME: time,
            DATA_FIELD.FIELD_OPEN: float(open_),
            DATA_FIELD.FIELD_HIGH: float(high),
            DATA_FIELD.FIELD_LOW: float(low),
            DATA_FIELD.FIELD_CLOSE: float(close),
            DATA_FIELD.FIELD_VOLUME: float(volume),
        }
    )


def _normalize_bsp_row(row: dict) -> dict:
    out = dict(row)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="raise")
    if out.get("direction") is None:
        out["direction"] = "buy" if bool(out.get("is_buy", True)) else "sell"
    out["direction"] = str(out["direction"]).lower()
    out["bsp_type"] = str(out.get("bsp_type", "?")).lower()
    return out


@dataclass(frozen=True)
class ChanSimulationConfig:
    """Chan discovery/replay parameters."""

    code: str = "QQQ"
    frequency: str = "5m"
    start: str | pd.Timestamp = "2020-01-01"
    end: str | pd.Timestamp = "2026-05-30"
    max_klines: int = 500
    warmup_bars: int = 500
    trigger_step: bool = True
    cal_rsi: bool = True
    cal_kdj: bool = True
    cal_dmi: bool = True

    def __post_init__(self) -> None:
        if self.max_klines <= 0:
            raise ValueError("max_klines must be greater than zero")
        if self.warmup_bars < 0:
            raise ValueError("warmup_bars cannot be negative")
        if pd.Timestamp(self.end) < pd.Timestamp(self.start):
            raise ValueError("end must be on or after start")


def default_strategy_config() -> SegBspStrategyConfig:
    """Return the current segment/BSP strategy parameters."""

    bsp_types = frozenset({"1", "1p", "2", "2s", "3a", "3b"})
    return SegBspStrategyConfig(
        entry_segment_directions=frozenset({"up", "down"}),
        entry_bsp_types=bsp_types,
        sell_bsp_types=bsp_types,
        buy_lookback_bars_by_segment={
            "up": 10,
            "down": 15,
        },
        required_buy_signals_by_segment={
            "up": {
                "1": 1,
                "1p": 1,
                "2": 1,
                "2s": 1,
                "3a": 1,
                "3b": 1,
            },
            "down": {
                "1": 5,
                "1p": 10,
                "2": 5,
                "2s": 5,
                "3a": 10,
                "3b": 10,
            },
        },
        sell_lookback_bars_by_entry_segment={
            "up": 20,
            "down": 10,
        },
        required_sell_signals_by_entry_segment={
            "up": {
                "1": 15,
                "1p": 15,
                "2": 15,
                "2s": 15,
                "3a": 15,
                "3b": 15,
            },
            "down": {
                "1": 1,
                "1p": 1,
                "2": 1,
                "2s": 1,
                "3a": 1,
                "3b": 1,
            },
        },
        exit_segment_directions_by_entry_segment={
            "up": frozenset({"down"}),
            "down": frozenset(),
        },
        required_buy_signals=1,
        buy_lookback_bars=15,
        required_sell_signals=5,
        sell_lookback_bars=10,
        exit_on_down_segment=True,
        allow_unconfirmed_entry=False,
        reset_sell_count_on_buy=True,
        reset_buy_count_on_sell=True,
        position_fraction=1.0,
    )


@dataclass(frozen=True)
class RealisticSimulationConfig:
    """All notebook-editable inputs for one simulation."""

    data_path: Path = Path("DataAPI/data/QQQ_5M.csv")
    bsp_workbook_path: Path = Path("outputs/bspoints_QQQ_5m_2020-2026.xlsx")
    bsp_sheet_name: str = "BSP Points"
    replay_mode: ReplayMode = "saved"
    ignore_delayed_bspoints: bool = False
    max_delayed_price_gap_pct: float | None = None
    max_discovery_delay_minutes: float | None = None
    chan: ChanSimulationConfig = field(default_factory=ChanSimulationConfig)
    strategy: SegBspStrategyConfig = field(default_factory=default_strategy_config)
    risk: RiskConfig = field(
        default_factory=lambda: RiskConfig(
            stop_loss_pct=0.03,
            take_profit_pct=0.20,
            trailing_stop_pct=0.025,
            max_holding_bars=78 * 10,
        )
    )
    backtest: BacktestChanConfig = field(
        default_factory=lambda: BacktestChanConfig(
            initial_capital=100_000,
            # Preserve the current notebook assumptions. Change these in the
            # notebook when you want to add explicit execution friction.
            fee_pct=0.0,
            slippage_pct=0.0,
            close_at_end=True,
        )
    )

    def __post_init__(self) -> None:
        if self.replay_mode not in {"saved", "regenerate"}:
            raise ValueError("replay_mode must be 'saved' or 'regenerate'")
        if (
            self.max_delayed_price_gap_pct is not None
            and self.max_delayed_price_gap_pct < 0
        ):
            raise ValueError("max_delayed_price_gap_pct cannot be negative")
        if (
            self.max_discovery_delay_minutes is not None
            and self.max_discovery_delay_minutes < 0
        ):
            raise ValueError("max_discovery_delay_minutes cannot be negative")


@dataclass
class RealisticSimulationOutput:
    """Data frames and backtest result returned to a notebook."""

    result: BacktestResult
    price_df: pd.DataFrame
    bsp_df: pd.DataFrame
    chan_feed_df: pd.DataFrame
    config: RealisticSimulationConfig


@dataclass
class SimulationTradeAnalysis:
    """Completed trades and profitability breakdowns for a simulation."""

    completed_trades: pd.DataFrame
    overall: pd.DataFrame
    by_entry_type: pd.DataFrame
    by_entry_segment: pd.DataFrame
    by_entry_segment_type: pd.DataFrame
    by_exit_category: pd.DataFrame
    by_exit_type: pd.DataFrame
    by_entry_exit: pd.DataFrame
    by_year: pd.DataFrame
    by_session: pd.DataFrame
    by_holding_bucket: pd.DataFrame


def _prepare_price_frames(
    data_path: str | Path,
    chan_config: ChanSimulationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = _load_ohlcv_csv(data_path)
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

    start = pd.Timestamp(chan_config.start)
    end = pd.Timestamp(chan_config.end)
    warmup = raw.loc[raw["timestamp"] < start].tail(chan_config.warmup_bars)
    simulation = raw.loc[raw["timestamp"].between(start, end, inclusive="both")].copy()
    if simulation.empty:
        raise ValueError(f"No price rows found between {start} and {end}")

    feed = pd.concat([warmup, simulation], ignore_index=True)
    simulation["open"] = simulation["_open"]
    simulation["high"] = simulation["_high"]
    simulation["low"] = simulation["_low"]
    simulation["close"] = simulation["_close"]
    simulation = simulation.reset_index(drop=True)
    return simulation, feed


def attach_trigger_timestamps(
    bsp_df: pd.DataFrame,
    chan_feed_df: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct observable trigger times from one-based snapshot numbers."""

    required = {"timestamp", "snapshot_first_seen"}
    missing = required.difference(bsp_df.columns)
    if missing:
        raise KeyError(
            "Saved BSP data cannot be replayed realistically; missing columns: "
            f"{sorted(missing)}"
        )
    if "timestamp" not in chan_feed_df.columns:
        raise KeyError("chan_feed_df is missing timestamp")

    out = bsp_df.copy()
    out["bsp_timestamp"] = pd.to_datetime(out["timestamp"], errors="raise")
    snapshots = pd.to_numeric(out["snapshot_first_seen"], errors="raise").astype("int64")
    trigger_positions = snapshots - 1
    invalid = (trigger_positions < 0) | (trigger_positions >= len(chan_feed_df))
    if invalid.any():
        sample = snapshots.loc[invalid].head(5).tolist()
        raise ValueError(
            f"{int(invalid.sum())} snapshot_first_seen values are outside the reconstructed "
            f"Chan feed of {len(chan_feed_df)} rows; examples={sample}. Confirm start, end, "
            "warmup_bars, and the source price file used to generate the BSP workbook."
        )

    out["trigger_timestamp"] = (
        chan_feed_df.iloc[trigger_positions.to_numpy()]["timestamp"].to_numpy()
    )
    out["trigger_timestamp"] = pd.to_datetime(out["trigger_timestamp"], errors="raise")
    if (out["trigger_timestamp"] < out["bsp_timestamp"]).any():
        raise ValueError(
            "A reconstructed trigger occurs before its BSP timestamp. The saved BSP workbook "
            "and Chan feed parameters do not describe the same generation run."
        )

    # run_bsp_backtest consumes the event arrival time from the timestamp column.
    out["timestamp"] = out["trigger_timestamp"]
    return out


def _kl_type(frequency: str):
    value = str(frequency).lower()
    if value in {"5m", "5min", "5", "k_5m"}:
        return KL_TYPE.K_5M
    if value in {"day", "daily", "d", "1d"}:
        return KL_TYPE.K_DAY
    raise ValueError(f"Unsupported frequency: {frequency}")


def generate_trigger_timed_bsp(
    chan_feed_df: pd.DataFrame,
    chan_config: ChanSimulationConfig,
) -> pd.DataFrame:
    """Regenerate BSP snapshots and record the kline that revealed each one."""

    c = CChanConfig(
        {
            "trigger_step": chan_config.trigger_step,
            "cal_rsi": chan_config.cal_rsi,
            "cal_kdj": chan_config.cal_kdj,
            "cal_dmi": chan_config.cal_dmi,
        }
    )
    chan = _SimulationSlidingWindowChan(
        code=chan_config.code,
        data_src=DATA_SRC.CSV,
        lv_list=[_kl_type(chan_config.frequency)],
        config=c,
        autype=AUTYPE.QFQ,
        max_klines=chan_config.max_klines,
    )

    rows: list[dict] = []
    for feed_idx, row in chan_feed_df.reset_index(drop=True).iterrows():
        klu = _build_klu(
            row["timestamp"],
            row["_open"],
            row["_high"],
            row["_low"],
            row["_close"],
            row.get("_vol", 0.0),
        )
        emitted = chan.process_new_kline(klu)
        if isinstance(emitted, tuple) and len(emitted) == 2:
            _, new_rows = emitted
        else:
            new_rows = emitted or []

        for bsp in new_rows:
            normalized = _normalize_bsp_row(bsp)
            normalized["bsp_timestamp"] = pd.to_datetime(normalized["timestamp"])
            normalized["trigger_timestamp"] = pd.Timestamp(row["timestamp"])
            normalized["trigger_idx"] = int(feed_idx)
            normalized["timestamp"] = normalized["trigger_timestamp"]
            rows.append(normalized)

    return pd.DataFrame(rows)


def _prepare_bsp_for_replay(
    bsp_df: pd.DataFrame,
    price_df: pd.DataFrame,
) -> pd.DataFrame:
    required = {"timestamp", "direction", "bsp_type"}
    missing = required.difference(bsp_df.columns)
    if missing:
        raise KeyError(f"BSP data is missing required columns: {sorted(missing)}")

    out = bsp_df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="raise")
    out = out.loc[
        out["timestamp"].between(
            price_df["timestamp"].min(),
            price_df["timestamp"].max(),
            inclusive="both",
        )
    ].copy()
    out = out.sort_values(["timestamp", "direction", "bsp_type"])

    dedup = ["timestamp", "direction", "bsp_type"]
    if "bsp_timestamp" in out.columns:
        dedup.insert(1, "bsp_timestamp")
    out = out.drop_duplicates(dedup, keep="first").reset_index(drop=True)

    unmatched = ~out["timestamp"].isin(price_df["timestamp"])
    if unmatched.any():
        examples = out.loc[unmatched, "timestamp"].head(5).astype(str).tolist()
        raise ValueError(
            f"{int(unmatched.sum())} BSP trigger timestamps do not match a price kline; "
            f"examples={examples}"
        )
    return out


def prepare_simulation_data(
    config: RealisticSimulationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load price data and produce point-in-time BSP events."""

    price_df, chan_feed_df = _prepare_price_frames(config.data_path, config.chan)
    if config.replay_mode == "saved":
        saved = pd.read_excel(
            config.bsp_workbook_path,
            sheet_name=config.bsp_sheet_name,
            engine="openpyxl",
        )
        bsp_df = attach_trigger_timestamps(saved, chan_feed_df)
    else:
        bsp_df = generate_trigger_timed_bsp(chan_feed_df, config.chan)

    if config.ignore_delayed_bspoints:
        required_timing = {"bsp_timestamp", "trigger_timestamp"}
        missing_timing = required_timing.difference(bsp_df.columns)
        if missing_timing:
            raise KeyError(
                "Cannot remove delayed BSPs; missing timing columns: "
                f"{sorted(missing_timing)}"
            )
        bsp_timestamp = pd.to_datetime(bsp_df["bsp_timestamp"], errors="raise")
        trigger_timestamp = pd.to_datetime(bsp_df["trigger_timestamp"], errors="raise")
        bsp_df = bsp_df.loc[trigger_timestamp <= bsp_timestamp].copy()
    elif (
        config.max_delayed_price_gap_pct is not None
        or config.max_discovery_delay_minutes is not None
    ):
        required_timing = {"bsp_timestamp", "trigger_timestamp"}
        missing_timing = required_timing.difference(bsp_df.columns)
        if missing_timing:
            raise KeyError(
                "Cannot filter delayed BSPs; missing timing columns: "
                f"{sorted(missing_timing)}"
            )
        bsp_df = bsp_df.copy()
        bsp_timestamp = pd.to_datetime(bsp_df["bsp_timestamp"], errors="raise")
        trigger_timestamp = pd.to_datetime(bsp_df["trigger_timestamp"], errors="raise")
        is_immediate = trigger_timestamp <= bsp_timestamp
        keep = pd.Series(True, index=bsp_df.index)

        if config.max_discovery_delay_minutes is not None:
            delay_minutes = (
                trigger_timestamp - bsp_timestamp
            ).dt.total_seconds() / 60.0
            keep &= is_immediate | (
                delay_minutes <= float(config.max_discovery_delay_minutes)
            )

        if config.max_delayed_price_gap_pct is not None:
            if "klu_close" not in bsp_df.columns:
                raise KeyError(
                    "Cannot apply max_delayed_price_gap_pct because BSP data "
                    "is missing klu_close"
                )
            trigger_close = trigger_timestamp.map(
                price_df.set_index("timestamp")["close"]
            )
            if trigger_close.isna().any():
                raise ValueError(
                    "Some BSP trigger timestamps do not have a matching price close"
                )
            historical_close = pd.to_numeric(bsp_df["klu_close"], errors="coerce")
            if historical_close.isna().any() or (historical_close <= 0).any():
                raise ValueError("Some BSP klu_close values are missing or nonpositive")
            absolute_gap = (trigger_close / historical_close - 1.0).abs()
            keep &= is_immediate | (
                absolute_gap <= float(config.max_delayed_price_gap_pct)
            )

        bsp_df = bsp_df.loc[keep].copy()

    bsp_df = _prepare_bsp_for_replay(bsp_df, price_df)
    return price_df, bsp_df, chan_feed_df


def run_realistic_simulation(
    config: RealisticSimulationConfig | None = None,
) -> RealisticSimulationOutput:
    """Run a configured simulation and return notebook-friendly outputs."""

    cfg = config or RealisticSimulationConfig()
    price_df, bsp_df, chan_feed_df = prepare_simulation_data(cfg)
    result = run_bsp_backtest(
        price_df=price_df,
        bsp_df=bsp_df,
        strategy=SegBspStrategy(cfg.strategy),
        config=cfg.backtest,
        risk_manager=RiskManager(cfg.risk),
    )
    return RealisticSimulationOutput(
        result=result,
        price_df=price_df,
        bsp_df=bsp_df,
        chan_feed_df=chan_feed_df,
        config=cfg,
    )


def print_simulation_summary(output: RealisticSimulationOutput) -> None:
    """Print compact diagnostics and strategy metrics in a notebook."""

    print(f"Price rows: {len(output.price_df):,}")
    print(f"Chan feed rows (including warmup): {len(output.chan_feed_df):,}")
    print(f"Point-in-time BSP events: {len(output.bsp_df):,}")
    print(
        "Simulation period:",
        output.price_df["timestamp"].min(),
        "to",
        output.price_df["timestamp"].max(),
    )
    print("\nStrategy metrics:")
    for name, value in output.result.metrics.items():
        print(f"{name}: {value}")


def _reason_bsp_type(reason: object, side: str) -> str:
    match = re.search(
        rf"type-([^\s]+)\s+{'buy' if side == 'buy' else 'sell'}",
        str(reason),
    )
    return match.group(1) if match else "none"


def _entry_segment_from_reason(reason: object) -> str:
    match = re.search(r"BSP\s+[^\s]+\s+in\s+(\w+)\s+segment", str(reason))
    return match.group(1) if match else "unknown"


def _exit_category(reason: object) -> str:
    value = str(reason)
    if value in {
        "stop_loss",
        "take_profit",
        "trailing_stop",
        "max_holding_bars",
        "end_of_backtest",
    }:
        return value
    if value.startswith("entry segment up; exit segment down"):
        return "structural_down"
    if re.search(r"type-[^\s]+\s+sell", value):
        return "bsp_sell"
    return "other"


def _entry_session(timestamp: pd.Timestamp) -> str:
    minute = timestamp.hour * 60 + timestamp.minute
    if minute < 9 * 60 + 30:
        return "premarket"
    if minute < 16 * 60:
        return "regular"
    return "after_hours"


def _holding_bucket(minutes: float) -> str:
    if minutes <= 60:
        return "<=1h"
    if minutes <= 390:
        return "1-6.5h"
    if minutes <= 1_440:
        return "6.5-24h"
    if minutes <= 4_320:
        return "1-3d"
    return ">3d"


def _trade_group_summary(group: pd.DataFrame) -> dict:
    returns = pd.to_numeric(group["return_pct"], errors="coerce").dropna()
    positive = returns.loc[returns > 0]
    negative = returns.loc[returns < 0]
    gross_wins = positive.sum()
    gross_losses = -negative.sum()
    return {
        "trades": int(len(group)),
        "wins": int((returns > 0).sum()),
        "losses": int((returns < 0).sum()),
        "win_rate": float((returns > 0).mean()) if len(returns) else 0.0,
        "average_return": float(returns.mean()) if len(returns) else 0.0,
        "median_return": float(returns.median()) if len(returns) else 0.0,
        "average_winner": float(positive.mean()) if len(positive) else float("nan"),
        "average_loser": float(negative.mean()) if len(negative) else float("nan"),
        "arithmetic_return_sum": float(returns.sum()),
        "compounded_return": float((1.0 + returns).prod() - 1.0),
        "profit_factor": (
            float(gross_wins / gross_losses)
            if gross_losses > 0
            else float("inf") if gross_wins > 0 else float("nan")
        ),
        "total_pnl": float(pd.to_numeric(group["pnl"], errors="coerce").sum()),
        "average_pnl": float(pd.to_numeric(group["pnl"], errors="coerce").mean()),
        "average_holding_minutes": float(
            pd.to_numeric(group["holding_minutes"], errors="coerce").mean()
        ),
    }


def _summarize_trade_groups(
    trades: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    group_arg: str | list[str] = columns[0] if len(columns) == 1 else columns
    for key, group in trades.groupby(group_arg, dropna=False, sort=False):
        keys = (key,) if len(columns) == 1 else key
        row = {column: value for column, value in zip(columns, keys)}
        row.update(_trade_group_summary(group))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values(
        ["average_return", "trades"],
        ascending=[False, False],
    ).reset_index(drop=True)


def analyze_simulation_trades(
    output: RealisticSimulationOutput,
) -> SimulationTradeAnalysis:
    """Pair fills and calculate BSP profitability/loss breakdowns."""

    if output.result is None:
        raise ValueError("output.result is missing; run the simulation before analysis")
    fills = output.result.trades.copy().reset_index(drop=True)
    required = {"side", "exec_idx", "px", "reason"}
    missing = required.difference(fills.columns)
    if missing:
        raise KeyError(f"fills are missing columns: {sorted(missing)}")

    prices = output.price_df.reset_index(drop=True)
    exec_idx = pd.to_numeric(fills["exec_idx"], errors="raise").astype("int64")
    invalid = (exec_idx < 0) | (exec_idx >= len(prices))
    if invalid.any():
        raise ValueError(f"{int(invalid.sum())} fill indexes are outside price_df")
    fills["execution_timestamp"] = prices.iloc[exec_idx.to_numpy()][
        "timestamp"
    ].to_numpy()
    fills["execution_timestamp"] = pd.to_datetime(
        fills["execution_timestamp"],
        errors="raise",
    )

    completed: list[dict] = []
    entry = None
    for row in fills.to_dict("records"):
        side = str(row["side"]).lower()
        if side == "buy":
            entry = row
            continue
        if side != "sell" or entry is None:
            continue

        entry_time = pd.Timestamp(entry["execution_timestamp"])
        exit_time = pd.Timestamp(row["execution_timestamp"])
        return_pct = row.get("return_pct")
        if return_pct is None or pd.isna(return_pct):
            return_pct = float(row["px"]) / float(entry["px"]) - 1.0
        pnl = row.get("pnl")
        if pnl is None or pd.isna(pnl):
            pnl = 0.0
        entry_reason = str(entry["reason"])
        exit_reason = str(row["reason"])
        entry_type = _reason_bsp_type(entry_reason, "buy")
        exit_type = _reason_bsp_type(exit_reason, "sell")
        holding_minutes = (exit_time - entry_time).total_seconds() / 60.0
        completed.append(
            {
                "entry_timestamp": entry_time,
                "exit_timestamp": exit_time,
                "entry_price": float(entry["px"]),
                "exit_price": float(row["px"]),
                "return_pct": float(return_pct),
                "pnl": float(pnl),
                "holding_minutes": holding_minutes,
                "entry_type": entry_type,
                "entry_segment": _entry_segment_from_reason(entry_reason),
                "exit_category": _exit_category(exit_reason),
                "exit_type": exit_type,
                "entry_reason": entry_reason,
                "exit_reason": exit_reason,
                "year": entry_time.year,
                "weekday": entry_time.day_name(),
                "session": _entry_session(entry_time),
                "holding_bucket": _holding_bucket(holding_minutes),
            }
        )
        entry = None

    completed_df = pd.DataFrame(completed)
    if completed_df.empty:
        raise ValueError("No completed buy/sell trade pairs were found")
    overall = pd.DataFrame([_trade_group_summary(completed_df)])
    return SimulationTradeAnalysis(
        completed_trades=completed_df,
        overall=overall,
        by_entry_type=_summarize_trade_groups(completed_df, ["entry_type"]),
        by_entry_segment=_summarize_trade_groups(completed_df, ["entry_segment"]),
        by_entry_segment_type=_summarize_trade_groups(
            completed_df,
            ["entry_segment", "entry_type"],
        ),
        by_exit_category=_summarize_trade_groups(completed_df, ["exit_category"]),
        by_exit_type=_summarize_trade_groups(completed_df, ["exit_type"]),
        by_entry_exit=_summarize_trade_groups(
            completed_df,
            ["entry_segment", "entry_type", "exit_category"],
        ),
        by_year=_summarize_trade_groups(completed_df, ["year"]),
        by_session=_summarize_trade_groups(completed_df, ["session"]),
        by_holding_bucket=_summarize_trade_groups(
            completed_df,
            ["holding_bucket"],
        ),
    )


def display_simulation_analysis(analysis: SimulationTradeAnalysis) -> None:
    """Display the principal analysis tables in a Jupyter notebook."""

    try:
        from IPython.display import display
    except ImportError as exc:
        raise ImportError("IPython is required for notebook display") from exc

    tables = [
        ("Overall", analysis.overall),
        ("Profitability by entry BSP type", analysis.by_entry_type),
        ("Profitability by entry segment", analysis.by_entry_segment),
        ("Profitability by segment and BSP type", analysis.by_entry_segment_type),
        ("Results by exit cause", analysis.by_exit_category),
        ("Results by BSP exit type", analysis.by_exit_type),
        ("Results by year", analysis.by_year),
        ("Results by entry session", analysis.by_session),
        ("Results by holding period", analysis.by_holding_bucket),
    ]
    for heading, table in tables:
        print(f"\n{heading}")
        display(table)


def _excel_safe_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return an Excel-safe copy without changing the simulation data."""

    out = df.copy()
    for column in out.columns:
        if isinstance(out[column].dtype, pd.DatetimeTZDtype):
            out[column] = out[column].dt.tz_localize(None)
        elif out[column].dtype == "object":
            out[column] = out[column].map(
                lambda value: json.dumps(
                    value,
                    sort_keys=True,
                    default=str,
                )
                if isinstance(value, (dict, list, tuple, set, frozenset))
                else value
            )
    return out


def _flatten_config(prefix: str, value, rows: list[dict]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_config(child_prefix, child, rows)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        value = json.dumps(sorted(value) if isinstance(value, (set, frozenset)) else value, default=str)
    elif isinstance(value, Path):
        value = str(value)
    rows.append({"parameter": prefix, "value": value})


def save_simulation_to_excel(
    output: RealisticSimulationOutput,
    path: str | Path | None = None,
    *,
    include_price_data: bool = True,
    include_bsp_points: bool = True,
    include_chan_feed: bool = False,
) -> Path:
    """Save simulation results and parameters to one notebook-friendly workbook."""

    if path is None:
        code = output.config.chan.code.upper()
        path = Path("outputs/strategy_backtest") / f"{code}_realistic_strategy_result.xlsx"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame(
        output.result.metrics.items(),
        columns=["metric", "value"],
    )
    config_rows: list[dict] = []
    _flatten_config("", asdict(output.config), config_rows)
    config_df = pd.DataFrame(config_rows)

    sheets: list[tuple[str, pd.DataFrame]] = [
        ("Metrics", metrics),
        ("Fills", output.result.trades),
        ("Equity", output.result.equity),
        ("Configuration", config_df),
    ]
    if include_bsp_points:
        sheets.append(("BSP Points", output.bsp_df))
    if include_price_data:
        sheets.append(("Price Data", output.price_df))
    if include_chan_feed:
        sheets.append(("Chan Feed", output.chan_feed_df))

    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        for sheet_name, frame in sheets:
            safe = _excel_safe_frame(frame)
            if len(safe) + 1 > 1_048_576:
                raise ValueError(
                    f"{sheet_name!r} has {len(safe):,} rows and exceeds Excel's row limit"
                )
            safe.to_excel(writer, sheet_name=sheet_name, index=False)

        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            worksheet.sheet_view.showGridLines = False
            for cell in worksheet[1]:
                cell.font = cell.font.copy(bold=True, color="FFFFFF")
                cell.fill = cell.fill.copy(fill_type="solid", fgColor="1F4E78")
            for column_cells in worksheet.iter_cols(
                min_row=1,
                max_row=min(worksheet.max_row, 200),
            ):
                header = str(column_cells[0].value or "")
                sample_width = max(
                    [len(header)]
                    + [len(str(cell.value)) for cell in column_cells[1:] if cell.value is not None]
                )
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(
                    max(sample_width + 2, 10),
                    40,
                )

    return target.resolve()


def plot_delayed_bspoints(
    output: RealisticSimulationOutput,
    path: str | Path | None = None,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    include_not_delayed: bool = False,
    max_price_gap_pct: float | None = None,
    selection: Literal["all", "largest_time", "largest_price_gap"] = "all",
    top_n: int = 50,
    annotate_largest: int = 0,
) -> Path:
    """Plot selected BSPs at historical and first-observable locations.

    ``selection`` choices:
    - ``"all"``: show every delayed BSP in the requested period.
    - ``"largest_time"``: show the ``top_n`` longest discovery delays.
    - ``"largest_price_gap"``: show the ``top_n`` largest absolute percentage
      gaps between the historical BSP close and discovery-kline close.

    Set ``include_not_delayed=True`` to include immediately observable BSPs.
    ``max_price_gap_pct`` is an absolute decimal-return ceiling; for example,
    ``0.003`` keeps points whose historical-to-discovery gap is at most 0.3%.
    """

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    bsp = output.bsp_df.copy()
    required = {"bsp_timestamp", "trigger_timestamp", "direction", "bsp_type", "klu_close"}
    missing = required.difference(bsp.columns)
    if missing:
        raise KeyError(f"Trigger-timed BSP data is missing columns: {sorted(missing)}")

    bsp["bsp_timestamp"] = pd.to_datetime(bsp["bsp_timestamp"], errors="raise")
    bsp["trigger_timestamp"] = pd.to_datetime(bsp["trigger_timestamp"], errors="raise")
    bsp["discovery_delay"] = bsp["trigger_timestamp"] - bsp["bsp_timestamp"]
    if include_not_delayed:
        delayed = bsp.loc[bsp["discovery_delay"] >= pd.Timedelta(0)].copy()
    else:
        delayed = bsp.loc[bsp["discovery_delay"] > pd.Timedelta(0)].copy()
    if start is not None:
        delayed = delayed.loc[delayed["trigger_timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        delayed = delayed.loc[delayed["trigger_timestamp"] <= pd.Timestamp(end)]
    if delayed.empty:
        raise ValueError("No delayed BSPs exist in the requested period")

    trigger_prices = output.price_df[["timestamp", "close"]].rename(
        columns={"timestamp": "trigger_timestamp", "close": "trigger_close"}
    )
    delayed = delayed.merge(
        trigger_prices,
        on="trigger_timestamp",
        how="left",
        validate="many_to_one",
    )
    if delayed["trigger_close"].isna().any():
        raise ValueError("Some delayed BSP trigger timestamps do not have a matching close price")

    delayed["price_gap_pct"] = (
        delayed["trigger_close"] / pd.to_numeric(delayed["klu_close"], errors="coerce") - 1.0
    )
    delayed["absolute_price_gap_pct"] = delayed["price_gap_pct"].abs()
    if max_price_gap_pct is not None:
        max_gap = float(max_price_gap_pct)
        if max_gap < 0:
            raise ValueError("max_price_gap_pct cannot be negative")
        delayed = delayed.loc[delayed["absolute_price_gap_pct"] <= max_gap].copy()
        if delayed.empty:
            raise ValueError(
                f"No BSPs remain after applying max_price_gap_pct={max_gap:.4%}"
            )
    allowed_selections = {"all", "largest_time", "largest_price_gap"}
    if selection not in allowed_selections:
        raise ValueError(
            f"selection must be one of {sorted(allowed_selections)}; got {selection!r}"
        )
    if int(top_n) <= 0:
        raise ValueError("top_n must be greater than zero")
    if selection == "largest_time":
        delayed = delayed.nlargest(min(int(top_n), len(delayed)), "discovery_delay")
    elif selection == "largest_price_gap":
        delayed = delayed.nlargest(min(int(top_n), len(delayed)), "absolute_price_gap_pct")

    chart_start = min(delayed["bsp_timestamp"].min(), delayed["trigger_timestamp"].min())
    chart_end = max(delayed["bsp_timestamp"].max(), delayed["trigger_timestamp"].max())
    price = output.price_df.loc[
        output.price_df["timestamp"].between(chart_start, chart_end, inclusive="both"),
        ["timestamp", "close"],
    ].copy()
    # Keep the full delayed population while reducing only the background price line.
    stride = max(1, len(price) // 75_000)
    price_plot = price.iloc[::stride]

    colors = {"buy": "#16803C", "sell": "#C33C3C"}
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(
        price_plot["timestamp"],
        price_plot["close"],
        color="#62748A",
        linewidth=0.55,
        alpha=0.75,
        label="QQQ close",
        zorder=1,
    )

    for direction in ("buy", "sell"):
        group = delayed.loc[delayed["direction"].astype(str).str.lower() == direction]
        if group.empty:
            continue
        color = colors[direction]
        segments = [
            [
                (mdates.date2num(row.bsp_timestamp), float(row.klu_close)),
                (mdates.date2num(row.trigger_timestamp), float(row.trigger_close)),
            ]
            for row in group.itertuples(index=False)
        ]
        ax.add_collection(
            LineCollection(
                segments,
                colors=color,
                linewidths=0.45,
                alpha=0.20,
                zorder=2,
            )
        )
        ax.scatter(
            group["bsp_timestamp"],
            group["klu_close"],
            s=15,
            facecolors="none",
            edgecolors=color,
            linewidths=0.65,
            alpha=0.75,
            marker="o",
            label=f"{direction.title()} historical BSP",
            zorder=3,
        )
        ax.scatter(
            group["trigger_timestamp"],
            group["trigger_close"],
            s=18,
            color=color,
            linewidths=0.6,
            alpha=0.85,
            marker="x",
            label=f"{direction.title()} discovery",
            zorder=4,
        )

    if annotate_largest > 0:
        annotation_metric = (
            "absolute_price_gap_pct"
            if selection == "largest_price_gap"
            else "discovery_delay"
        )
        largest = delayed.nlargest(
            min(int(annotate_largest), len(delayed)),
            annotation_metric,
        )
        for row in largest.itertuples(index=False):
            hours = row.discovery_delay.total_seconds() / 3600
            annotation = (
                f"{row.direction} {row.bsp_type}: {row.price_gap_pct:+.1%}"
                if selection == "largest_price_gap"
                else f"{row.direction} {row.bsp_type}: {hours:.0f}h"
            )
            ax.annotate(
                annotation,
                xy=(row.trigger_timestamp, row.trigger_close),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=7,
                alpha=0.85,
            )

    population_label = "BSPs" if include_not_delayed else "delayed BSPs"
    selection_labels = {
        "all": f"all selected {population_label}",
        "largest_time": "largest discovery-time delays",
        "largest_price_gap": "largest absolute price gaps",
    }
    gap_label = (
        f", gap ≤ {float(max_price_gap_pct):.2%}"
        if max_price_gap_pct is not None
        else ""
    )
    ax.set_title(
        f"{output.config.chan.code.upper()} {selection_labels[selection]}: "
        f"historical location to discovery ({len(delayed):,} points{gap_label})"
    )
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="best", ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()

    if path is None:
        code = output.config.chan.code.upper()
        path = Path("outputs") / f"delayed_bspoints_{code}_5m.png"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return target.resolve()


def plot_bsp_delay_comparison(
    output: RealisticSimulationOutput,
    path: str | Path | None = None,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    show: bool = True,
    max_background_points: int = 75_000,
) -> Path | None:
    """Plot one period with all observable BSPs and with delayed BSPs removed.

    Both panels place BSP markers at ``trigger_timestamp`` and the matching
    trigger-bar close, which is the first point at which a live strategy could
    act.  The upper panel contains immediate and delayed BSPs.  The lower panel
    contains only BSPs whose trigger time is not later than their historical
    BSP time.  Shared axes make the two populations directly comparable.
    """

    import matplotlib.pyplot as plt

    if int(max_background_points) <= 0:
        raise ValueError("max_background_points must be greater than zero")

    price = output.price_df.copy()
    required_price = {"timestamp", "close"}
    missing_price = required_price.difference(price.columns)
    if missing_price:
        raise KeyError(f"price_df is missing columns: {sorted(missing_price)}")
    price["timestamp"] = pd.to_datetime(price["timestamp"], errors="raise")

    period_start = (
        pd.Timestamp(start) if start is not None else price["timestamp"].min()
    )
    period_end = pd.Timestamp(end) if end is not None else price["timestamp"].max()
    if period_end < period_start:
        raise ValueError("end must be greater than or equal to start")

    price = price.loc[
        price["timestamp"].between(period_start, period_end, inclusive="both"),
        ["timestamp", "close"],
    ].copy()
    if price.empty:
        raise ValueError("No price rows exist in the requested period")

    bsp = output.bsp_df.copy()
    required_bsp = {
        "bsp_timestamp",
        "trigger_timestamp",
        "direction",
        "bsp_type",
    }
    missing_bsp = required_bsp.difference(bsp.columns)
    if missing_bsp:
        raise KeyError(f"Trigger-timed BSP data is missing columns: {sorted(missing_bsp)}")
    bsp["bsp_timestamp"] = pd.to_datetime(bsp["bsp_timestamp"], errors="raise")
    bsp["trigger_timestamp"] = pd.to_datetime(
        bsp["trigger_timestamp"], errors="raise"
    )
    bsp = bsp.loc[
        bsp["trigger_timestamp"].between(
            period_start, period_end, inclusive="both"
        )
    ].copy()
    if bsp.empty:
        raise ValueError("No BSP events exist in the requested period")

    trigger_prices = price.rename(
        columns={"timestamp": "trigger_timestamp", "close": "trigger_close"}
    )
    bsp = bsp.merge(
        trigger_prices,
        on="trigger_timestamp",
        how="left",
        validate="many_to_one",
    )
    if bsp["trigger_close"].isna().any():
        raise ValueError(
            "Some BSP trigger timestamps do not have a matching price close "
            "inside the requested period"
        )

    bsp["is_delayed"] = bsp["trigger_timestamp"] > bsp["bsp_timestamp"]
    immediate = bsp.loc[~bsp["is_delayed"]].copy()
    delayed_count = int(bsp["is_delayed"].sum())

    stride = max(1, len(price) // int(max_background_points))
    price_plot = price.iloc[::stride]
    colors = {"buy": "#16803C", "sell": "#C33C3C"}
    markers = {"buy": "^", "sell": "v"}

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(20, 12),
        sharex=True,
        sharey=True,
    )

    panels = [
        (
            axes[0],
            bsp,
            f"All point-in-time BSPs: {len(bsp):,} "
            f"({delayed_count:,} delayed)",
        ),
        (
            axes[1],
            immediate,
            f"Immediate BSPs only: {len(immediate):,} "
            f"({delayed_count:,} delayed removed)",
        ),
    ]
    for ax, events, title in panels:
        ax.plot(
            price_plot["timestamp"],
            price_plot["close"],
            color="#62748A",
            linewidth=0.65,
            alpha=0.80,
            label=f"{output.config.chan.code.upper()} close",
            zorder=1,
        )
        for direction in ("buy", "sell"):
            group = events.loc[
                events["direction"].astype(str).str.lower() == direction
            ]
            if group.empty:
                continue
            ax.scatter(
                group["trigger_timestamp"],
                group["trigger_close"],
                s=22,
                color=colors[direction],
                marker=markers[direction],
                alpha=0.75,
                linewidths=0.35,
                label=f"{direction.title()} BSP",
                zorder=3,
            )
        ax.set_title(title)
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.18)
        ax.legend(loc="best", ncol=3)

    axes[1].set_xlabel("Observable trigger timestamp")
    fig.suptitle(
        f"{output.config.chan.code.upper()} BSP delay comparison: "
        f"{period_start} to {period_end}",
        fontsize=14,
    )
    fig.autofmt_xdate()
    fig.tight_layout()

    saved_path: Path | None = None
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=180, bbox_inches="tight")
        saved_path = target.resolve()
    if show:
        plt.show()
    else:
        plt.close(fig)
    return saved_path


def plot_bsp_delay_comparison_from_files(
    config: RealisticSimulationConfig | None = None,
    path: str | Path | None = None,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    show: bool = True,
    max_background_points: int = 75_000,
) -> Path | None:
    """Load price/BSP files and plot all BSPs versus immediate BSPs only.

    Delay filters on ``config`` are cleared for this read so the upper panel
    always contains the complete BSP population.  The supplied config is not
    mutated.  ``replay_mode='saved'`` reads the configured BSP workbook;
    ``replay_mode='regenerate'`` rebuilds point-in-time BSPs from price data.
    """

    cfg = config or RealisticSimulationConfig()
    comparison_cfg = replace(
        cfg,
        ignore_delayed_bspoints=False,
        max_delayed_price_gap_pct=None,
        max_discovery_delay_minutes=None,
    )
    price_df, bsp_df, chan_feed_df = prepare_simulation_data(comparison_cfg)
    plotting_output = RealisticSimulationOutput(
        result=None,  # type: ignore[arg-type]
        price_df=price_df,
        bsp_df=bsp_df,
        chan_feed_df=chan_feed_df,
        config=comparison_cfg,
    )
    return plot_bsp_delay_comparison(
        plotting_output,
        path=path,
        start=start,
        end=end,
        show=show,
        max_background_points=max_background_points,
    )


def plot_immediate_vs_delayed_bspoints_from_files(
    config: RealisticSimulationConfig | None = None,
    path: str | Path | None = None,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    show: bool = True,
    max_background_points: int = 75_000,
) -> Path | None:
    """Plot immediate-only and delayed-only BSPs at historical BSP locations.

    This view intentionally does not plot discovery locations or lines from a
    historical BSP to its later discovery.  The upper panel contains BSPs
    discovered on their own historical bar.  The lower panel contains only
    BSPs discovered later, with markers still placed at ``bsp_timestamp`` and
    ``klu_close``.
    """

    import matplotlib.pyplot as plt

    if int(max_background_points) <= 0:
        raise ValueError("max_background_points must be greater than zero")

    cfg = config or RealisticSimulationConfig()
    comparison_cfg = replace(
        cfg,
        ignore_delayed_bspoints=False,
        max_delayed_price_gap_pct=None,
        max_discovery_delay_minutes=None,
    )
    price, bsp, _ = prepare_simulation_data(comparison_cfg)
    required = {
        "bsp_timestamp",
        "trigger_timestamp",
        "direction",
        "bsp_type",
        "klu_close",
    }
    missing = required.difference(bsp.columns)
    if missing:
        raise KeyError(f"Trigger-timed BSP data is missing columns: {sorted(missing)}")

    price = price.copy()
    price["timestamp"] = pd.to_datetime(price["timestamp"], errors="raise")
    bsp = bsp.copy()
    bsp["bsp_timestamp"] = pd.to_datetime(bsp["bsp_timestamp"], errors="raise")
    bsp["trigger_timestamp"] = pd.to_datetime(
        bsp["trigger_timestamp"], errors="raise"
    )
    bsp["klu_close"] = pd.to_numeric(bsp["klu_close"], errors="raise")

    period_start = (
        pd.Timestamp(start) if start is not None else price["timestamp"].min()
    )
    period_end = pd.Timestamp(end) if end is not None else price["timestamp"].max()
    if period_end < period_start:
        raise ValueError("end must be greater than or equal to start")

    price = price.loc[
        price["timestamp"].between(period_start, period_end, inclusive="both"),
        ["timestamp", "close"],
    ].copy()
    if price.empty:
        raise ValueError("No price rows exist in the requested period")

    bsp = bsp.loc[
        bsp["bsp_timestamp"].between(period_start, period_end, inclusive="both")
    ].copy()
    if bsp.empty:
        raise ValueError("No BSP events exist in the requested period")

    bsp["is_delayed"] = bsp["trigger_timestamp"] > bsp["bsp_timestamp"]
    immediate = bsp.loc[~bsp["is_delayed"]].copy()
    delayed = bsp.loc[bsp["is_delayed"]].copy()

    stride = max(1, len(price) // int(max_background_points))
    price_plot = price.iloc[::stride]
    colors = {"buy": "#16803C", "sell": "#C33C3C"}
    markers = {"buy": "^", "sell": "v"}

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(20, 12),
        sharex=True,
        sharey=True,
    )
    panels = [
        (axes[0], immediate, f"Immediate BSPs only: {len(immediate):,}"),
        (axes[1], delayed, f"Delayed BSPs only: {len(delayed):,}"),
    ]
    for ax, events, title in panels:
        ax.plot(
            price_plot["timestamp"],
            price_plot["close"],
            color="#62748A",
            linewidth=0.65,
            alpha=0.80,
            label=f"{comparison_cfg.chan.code.upper()} close",
            zorder=1,
        )
        for direction in ("buy", "sell"):
            group = events.loc[
                events["direction"].astype(str).str.lower() == direction
            ]
            if group.empty:
                continue
            ax.scatter(
                group["bsp_timestamp"],
                group["klu_close"],
                s=25,
                color=colors[direction],
                marker=markers[direction],
                alpha=0.80,
                linewidths=0.35,
                label=f"{direction.title()} BSP",
                zorder=3,
            )
        ax.set_title(title)
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.18)
        ax.legend(loc="best", ncol=3)

    axes[1].set_xlabel("Historical BSP timestamp")
    fig.suptitle(
        f"{comparison_cfg.chan.code.upper()} immediate vs delayed BSPs: "
        f"{period_start} to {period_end}",
        fontsize=14,
    )
    fig.autofmt_xdate()
    fig.tight_layout()

    saved_path: Path | None = None
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=180, bbox_inches="tight")
        saved_path = target.resolve()
    if show:
        plt.show()
    else:
        plt.close(fig)
    return saved_path


def generate_virtual_bsp_candidates(
    chan_feed_df: pd.DataFrame,
    chan_config: ChanSimulationConfig,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    virtual_lookback_bars: int = 5,
) -> pd.DataFrame:
    """Generate point-in-time possible BSPs before Bi confirmation.

    A Buy candidate is emitted when the current Chan Bi is downward and a
    bullish bar makes the trailing-window low.  A Sell candidate is the
    symmetric case: an upward Bi plus a bearish bar at the trailing-window
    high.  This deliberately uses only bars available at that timestamp.  The
    candidates are provisional observations, not confirmed Chan BSPs, and may
    disappear as later bars reshape the structure.
    """

    required = {"timestamp", "_open", "_high", "_low", "_close"}
    missing = required.difference(chan_feed_df.columns)
    if missing:
        raise KeyError(f"chan_feed_df is missing columns: {sorted(missing)}")
    if int(virtual_lookback_bars) < 2:
        raise ValueError("virtual_lookback_bars must be at least 2")

    feed = chan_feed_df.copy()
    feed["timestamp"] = pd.to_datetime(feed["timestamp"], errors="raise")
    period_start = pd.Timestamp(start) if start is not None else feed["timestamp"].min()
    period_end = pd.Timestamp(end) if end is not None else feed["timestamp"].max()
    if period_end < period_start:
        raise ValueError("end must be greater than or equal to start")

    # A sliding Chan calculation needs only the configured window immediately
    # before the requested period to reconstruct its point-in-time state.
    before = feed.loc[feed["timestamp"] < period_start].tail(
        max(int(chan_config.warmup_bars), int(chan_config.max_klines))
    )
    during = feed.loc[
        feed["timestamp"].between(period_start, period_end, inclusive="both")
    ]
    replay_feed = pd.concat([before, during], ignore_index=True)
    if during.empty:
        raise ValueError("No Chan feed rows exist in the requested period")

    c = CChanConfig(
        {
            "trigger_step": chan_config.trigger_step,
            # Candidate-only relaxation: allow a possible virtual leg before
            # the production Chan rules have enough evidence to confirm it.
            "bi_strict": False,
            "bi_fx_check": "loss",
            "cal_rsi": chan_config.cal_rsi,
            "cal_kdj": chan_config.cal_kdj,
            "cal_dmi": chan_config.cal_dmi,
            # Save the complete technical snapshot for every candidate.
            "cal_demark": True,
            "cal_rsl": True,
            "cal_demand_index": True,
            "cal_adline": True,
            "cal_bb_vals": True,
            "cal_kc_vals": True,
            "cal_starc_vals": True,
        }
    )
    chan = _SimulationSlidingWindowChan(
        code=chan_config.code,
        data_src=DATA_SRC.CSV,
        lv_list=[_kl_type(chan_config.frequency)],
        config=c,
        autype=AUTYPE.QFQ,
        max_klines=chan_config.max_klines,
    )
    level = _kl_type(chan_config.frequency)
    seen: set[tuple[pd.Timestamp, str]] = set()
    rows: list[dict] = []

    for row_pos, row in replay_feed.iterrows():
        trigger_timestamp = pd.Timestamp(row["timestamp"])
        chan.process_new_kline(
            _build_klu(
                trigger_timestamp,
                row["_open"],
                row["_high"],
                row["_low"],
                row["_close"],
                row.get("_vol", 0.0),
            )
        )
        if trigger_timestamp < period_start or chan.last_chan is None:
            continue
        try:
            kl_data = chan.last_chan.kl_datas[level]
            bi_list = kl_data.bi_list
            if len(bi_list) == 0:
                continue
            last_bi = bi_list[-1]
            window_start = max(0, int(row_pos) - int(virtual_lookback_bars) + 1)
            trailing = replay_feed.iloc[window_start : int(row_pos) + 1]
            bar_open = float(row["_open"])
            bar_close = float(row["_close"])
            bar_low = float(row["_low"])
            bar_high = float(row["_high"])
            if last_bi.is_down() and bar_close > bar_open and bar_low <= float(
                trailing["_low"].min()
            ):
                direction = "buy"
                candidate_price = bar_low
            elif last_bi.is_up() and bar_close < bar_open and bar_high >= float(
                trailing["_high"].max()
            ):
                direction = "sell"
                candidate_price = bar_high
            else:
                continue
            bsp_timestamp = trigger_timestamp
            candidate_bi_idx = int(last_bi.idx)
            source = "trailing_extreme_reversal"
            key = (bsp_timestamp, direction)
            if key in seen:
                continue
            seen.add(key)
            current_klu = kl_data.lst[-1].lst[-1]
            macd = getattr(current_klu, "macd", None)
            kdj = getattr(current_klu, "kdj", None)
            dmi = getattr(current_klu, "dmi", None)
            previous_bi = bi_list[-2] if len(bi_list) > 1 else None
            seg_list = kl_data.seg_list
            last_seg = seg_list[-1] if len(seg_list) else None
            zs_list = kl_data.zs_list
            last_zs = zs_list[-1] if len(zs_list) else None
            begin_value = float(last_bi.get_begin_val())
            bi_amp = float(last_bi.amp())
            bi_sure_attr = getattr(last_bi, "is_sure", False)
            bi_is_sure = bool(
                bi_sure_attr() if callable(bi_sure_attr) else bi_sure_attr
            )
            event = {
                    "bsp_timestamp": bsp_timestamp,
                    "trigger_timestamp": trigger_timestamp,
                    "trigger_idx": int(row_pos),
                    "direction": direction,
                    "bsp_type": "virtual_bi",
                    "klu_close": candidate_price,
                    "klu_open": bar_open,
                    "klu_high": bar_high,
                    "klu_low": bar_low,
                    "klu_volume": float(row.get("_vol", 0.0)),
                    "price_change_pct": 100.0 * (bar_close - bar_open) / bar_open if bar_open else None,
                    "high_low_spread_pct": 100.0 * (bar_high - bar_low) / bar_low if bar_low else None,
                    "upper_shadow": bar_high - max(bar_open, bar_close),
                    "lower_shadow": min(bar_open, bar_close) - bar_low,
                    "body_size": abs(bar_close - bar_open),
                    "is_bullish_candle": int(bar_close > bar_open),
                    "bi_idx": candidate_bi_idx,
                    "bi_direction": "down" if last_bi.is_down() else "up",
                    "bi_is_sure": bi_is_sure,
                    "bi_begin_price": begin_value,
                    "bi_end_price": float(last_bi.get_end_val()),
                    "bi_amp": bi_amp,
                    "bi_amp_rate": bi_amp / begin_value if begin_value else None,
                    "bi_klu_cnt": int(last_bi.get_klu_cnt()),
                    "bi_count": int(len(bi_list)),
                    "previous_bi_amp": float(previous_bi.amp()) if previous_bi is not None else None,
                    "previous_bi_klu_cnt": int(previous_bi.get_klu_cnt()) if previous_bi is not None else None,
                    "seg_count": int(len(seg_list)),
                    "seg_idx": int(last_seg.idx) if last_seg is not None else None,
                    "seg_direction": ("down" if last_seg.is_down() else "up") if last_seg is not None else None,
                    "seg_is_sure": bool(last_seg.is_sure) if last_seg is not None else None,
                    "seg_amp": float(last_seg.amp()) if last_seg is not None else None,
                    "seg_bi_cnt": int(last_seg.cal_bi_cnt()) if last_seg is not None else None,
                    "zs_count": int(len(zs_list)),
                    "zs_low": float(last_zs.low) if last_zs is not None else None,
                    "zs_high": float(last_zs.high) if last_zs is not None else None,
                    "zs_mid": float(last_zs.mid) if last_zs is not None else None,
                    "zs_is_sure": bool(last_zs.is_sure) if last_zs is not None else None,
                    "macd_value": float(macd.macd) if macd is not None else None,
                    "macd_diff": float(macd.DIF) if macd is not None else None,
                    "macd_dea": float(macd.DEA) if macd is not None else None,
                    "feat_ppo": ((float(macd.fast_ema) - float(macd.slow_ema)) / float(macd.slow_ema)) if macd is not None and macd.slow_ema else None,
                    "rsi": getattr(current_klu, "rsi", None),
                    "kdj_k": getattr(kdj, "k", None),
                    "kdj_d": getattr(kdj, "d", None),
                    "kdj_j": getattr(kdj, "j", None),
                    "dmi_plus": getattr(dmi, "plus_di", None),
                    "dmi_minus": getattr(dmi, "minus_di", None),
                    "dmi_adx": getattr(dmi, "adx", None),
                    "demark": getattr(current_klu, "demark", None),
                    "rsl": getattr(current_klu, "rsl", None),
                    "demand_index": getattr(current_klu, "demand_index", None),
                    "advance_decline_line": getattr(current_klu, "ad_line", None),
                    "bb_upper": getattr(current_klu, "bb_upper", None),
                    "bb_middle": getattr(current_klu, "bb_middle", None),
                    "bb_lower": getattr(current_klu, "bb_lower", None),
                    "kc_upper": getattr(current_klu, "kc_upper", None),
                    "kc_middle": getattr(current_klu, "kc_middle", None),
                    "kc_lower": getattr(current_klu, "kc_lower", None),
                    "starc_upper": getattr(current_klu, "starc_upper", None),
                    "starc_middle": getattr(current_klu, "starc_middle", None),
                    "starc_lower": getattr(current_klu, "starc_lower", None),
                    "candidate_source": source,
                    "virtual_lookback_bars": int(virtual_lookback_bars),
                }
            rows.append(event)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Could not capture virtual BSP snapshot at {trigger_timestamp}"
            ) from exc

    return pd.DataFrame(rows)


def save_virtual_bsp_candidates_to_excel(
    config: RealisticSimulationConfig | None = None,
    path: str | Path | None = None,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    virtual_lookback_bars: int = 5,
) -> Path:
    """Generate and save point-in-time virtual BSP candidates and snapshots."""

    cfg = config or RealisticSimulationConfig()
    _, _, chan_feed = prepare_simulation_data(cfg)
    candidates = generate_virtual_bsp_candidates(
        chan_feed,
        cfg.chan,
        start=start,
        end=end,
        virtual_lookback_bars=virtual_lookback_bars,
    )
    if path is None:
        path = Path("outputs") / f"{cfg.chan.code.upper()}_virtual_bsp_candidates.xlsx"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    config_rows: list[dict] = []
    _flatten_config("", asdict(cfg), config_rows)
    metadata = pd.DataFrame(
        [
            {"parameter": "export.start", "value": start},
            {"parameter": "export.end", "value": end},
            {"parameter": "export.virtual_lookback_bars", "value": virtual_lookback_bars},
            {"parameter": "export.candidate_count", "value": len(candidates)},
            {"parameter": "export.point_in_time", "value": True},
        ]
        + config_rows
    )
    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        _excel_safe_frame(candidates).to_excel(
            writer, sheet_name="Virtual BSP Candidates", index=False
        )
        _excel_safe_frame(metadata).to_excel(writer, sheet_name="Metadata", index=False)
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
    return target.resolve()


def plot_virtual_bsp_candidates_from_excel(
    config: RealisticSimulationConfig | None = None,
    virtual_bsp_path: str | Path = "outputs/TQQQ_virtual_bsp_candidates.xlsx",
    path: str | Path | None = None,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    show: bool = True,
    count_frequency: str = "D",
    max_background_points: int = 75_000,
) -> Path | None:
    """Plot saved virtual BSP candidates without regenerating Chan data.

    The upper panel shows candidates on the underlying close-price series.  The
    lower panel shows Buy and Sell candidate counts by ``count_frequency``.
    """

    import matplotlib.pyplot as plt

    if int(max_background_points) <= 0:
        raise ValueError("max_background_points must be greater than zero")
    cfg = config or RealisticSimulationConfig()
    candidates = pd.read_excel(
        virtual_bsp_path, sheet_name="Virtual BSP Candidates"
    )
    required = {"bsp_timestamp", "direction", "klu_close"}
    missing = required.difference(candidates.columns)
    if missing:
        raise KeyError(
            f"Virtual BSP workbook is missing columns: {sorted(missing)}"
        )
    candidates["bsp_timestamp"] = pd.to_datetime(
        candidates["bsp_timestamp"], errors="raise"
    )
    candidates["direction"] = candidates["direction"].astype(str).str.lower()
    candidates["klu_close"] = pd.to_numeric(
        candidates["klu_close"], errors="raise"
    )

    raw = _load_ohlcv_csv(cfg.data_path)
    period_start = (
        pd.Timestamp(start)
        if start is not None
        else candidates["bsp_timestamp"].min()
    )
    period_end = (
        pd.Timestamp(end)
        if end is not None
        else candidates["bsp_timestamp"].max()
    )
    if pd.isna(period_start) or pd.isna(period_end):
        raise ValueError("The virtual BSP workbook contains no candidates")
    if period_end < period_start:
        raise ValueError("end must be greater than or equal to start")

    price = raw.loc[
        raw["timestamp"].between(period_start, period_end, inclusive="both"),
        ["timestamp", "_close"],
    ].rename(columns={"_close": "close"})
    candidates = candidates.loc[
        candidates["bsp_timestamp"].between(
            period_start, period_end, inclusive="both"
        )
    ].copy()
    if price.empty:
        raise ValueError("No price rows exist in the requested plot period")

    stride = max(1, len(price) // int(max_background_points))
    price_plot = price.iloc[::stride]
    colors = {"buy": "#16803C", "sell": "#C33C3C"}
    markers = {"buy": "^", "sell": "v"}
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(20, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axes[0].plot(
        price_plot["timestamp"],
        price_plot["close"],
        color="#62748A",
        linewidth=0.7,
        label=f"{cfg.chan.code.upper()} close",
        zorder=1,
    )
    for direction in ("buy", "sell"):
        group = candidates.loc[candidates["direction"] == direction]
        if group.empty:
            continue
        axes[0].scatter(
            group["bsp_timestamp"],
            group["klu_close"],
            color=colors[direction],
            marker=markers[direction],
            s=25,
            alpha=0.8,
            label=f"Virtual {direction.title()} ({len(group):,})",
            zorder=3,
        )
    axes[0].set_ylabel("Price")
    axes[0].set_title(f"Possible virtual BSPs: {len(candidates):,}")
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="best")

    counts = (
        candidates.groupby(
            [pd.Grouper(key="bsp_timestamp", freq=count_frequency), "direction"]
        )
        .size()
        .unstack("direction", fill_value=0)
    )
    for direction in ("buy", "sell"):
        if direction not in counts.columns:
            counts[direction] = 0
    bar_width = 0.8
    if len(counts.index) > 1:
        bar_width = max(
            0.01,
            0.8 * (counts.index[1] - counts.index[0]).total_seconds() / 86_400,
        )
    axes[1].bar(
        counts.index,
        counts["buy"],
        width=bar_width,
        color=colors["buy"],
        alpha=0.75,
        label="Buy count",
    )
    axes[1].bar(
        counts.index,
        -counts["sell"],
        width=bar_width,
        color=colors["sell"],
        alpha=0.75,
        label="Sell count",
    )
    axes[1].axhline(0, color="#62748A", linewidth=0.7)
    axes[1].set_ylabel(f"Count / {count_frequency}\nSell shown below zero")
    axes[1].set_xlabel("Candidate detection timestamp")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(loc="best")

    fig.suptitle(
        f"{cfg.chan.code.upper()} virtual BSP candidates: "
        f"{period_start} to {period_end}",
        fontsize=14,
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    saved_path: Path | None = None
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=180, bbox_inches="tight")
        saved_path = target.resolve()
    if show:
        plt.show()
    else:
        plt.close(fig)
    return saved_path


def plot_virtual_bspoints_from_excel(
    config: RealisticSimulationConfig,
    virtual_bsp_path: str | Path | None = None,
    path: str | Path | None = None,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    show: bool = True,
    annotate: bool = False,
    max_background_points: int = 75_000,
) -> Path | None:
    """Show every saved virtual Buy/Sell point in one requested period."""

    import matplotlib.pyplot as plt

    if int(max_background_points) <= 0:
        raise ValueError("max_background_points must be greater than zero")
    workbook_path = virtual_bsp_path or config.bsp_workbook_path
    candidates = pd.read_excel(
        workbook_path, sheet_name="Virtual BSP Candidates"
    )
    required = {"bsp_timestamp", "direction", "klu_close"}
    missing = required.difference(candidates.columns)
    if missing:
        raise KeyError(
            f"Virtual BSP workbook is missing columns: {sorted(missing)}"
        )
    candidates["bsp_timestamp"] = pd.to_datetime(
        candidates["bsp_timestamp"], errors="raise"
    )
    candidates["direction"] = candidates["direction"].astype(str).str.lower()
    candidates["klu_close"] = pd.to_numeric(
        candidates["klu_close"], errors="raise"
    )
    period_start = (
        pd.Timestamp(start)
        if start is not None
        else candidates["bsp_timestamp"].min()
    )
    period_end = (
        pd.Timestamp(end)
        if end is not None
        else candidates["bsp_timestamp"].max()
    )
    if pd.isna(period_start) or pd.isna(period_end):
        raise ValueError("The virtual BSP workbook contains no candidates")
    if period_end < period_start:
        raise ValueError("end must be greater than or equal to start")
    candidates = candidates.loc[
        candidates["bsp_timestamp"].between(
            period_start, period_end, inclusive="both"
        )
    ].copy()

    raw = _load_ohlcv_csv(config.data_path)
    price = raw.loc[
        raw["timestamp"].between(period_start, period_end, inclusive="both"),
        ["timestamp", "_close"],
    ].rename(columns={"_close": "close"})
    if price.empty:
        raise ValueError("No price rows exist in the requested plot period")
    stride = max(1, len(price) // int(max_background_points))
    price_plot = price.iloc[::stride]

    colors = {"buy": "#16803C", "sell": "#C33C3C"}
    markers = {"buy": "^", "sell": "v"}
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(
        price_plot["timestamp"],
        price_plot["close"],
        color="#62748A",
        linewidth=0.7,
        label=f"{config.chan.code.upper()} close",
        zorder=1,
    )
    for direction in ("buy", "sell"):
        group = candidates.loc[candidates["direction"] == direction]
        if group.empty:
            continue
        ax.scatter(
            group["bsp_timestamp"],
            group["klu_close"],
            color=colors[direction],
            marker=markers[direction],
            s=35,
            alpha=0.85,
            label=f"{direction.title()} BSP ({len(group):,})",
            zorder=3,
        )
        if annotate:
            for _, event in group.iterrows():
                ax.annotate(
                    direction[0].upper(),
                    (event["bsp_timestamp"], event["klu_close"]),
                    xytext=(0, 7 if direction == "buy" else -10),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color=colors[direction],
                )
    ax.set_title(
        f"{config.chan.code.upper()} virtual BSPs: {period_start} to {period_end} "
        f"({len(candidates):,} points)"
    )
    ax.set_xlabel("BSP timestamp")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()

    saved_path: Path | None = None
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=180, bbox_inches="tight")
        saved_path = target.resolve()
    if show:
        plt.show()
    else:
        plt.close(fig)
    return saved_path


def _chan_geometry_at_timestamp(
    raw: pd.DataFrame,
    chan_config: ChanSimulationConfig,
    timestamp: pd.Timestamp,
) -> dict[str, list]:
    """Rebuild a local point-in-time Chan snapshot for plotting geometry."""

    positions = raw.index[raw["timestamp"] <= timestamp]
    if len(positions) == 0:
        return {"bis": [], "segments": [], "zones": []}
    end_pos = int(positions[-1])
    replay = raw.iloc[
        max(0, end_pos - max(chan_config.warmup_bars, chan_config.max_klines)) : end_pos + 1
    ]
    c = CChanConfig(
        {
            "trigger_step": chan_config.trigger_step,
            "cal_rsi": chan_config.cal_rsi,
            "cal_kdj": chan_config.cal_kdj,
            "cal_dmi": chan_config.cal_dmi,
        }
    )
    chan = _SimulationSlidingWindowChan(
        code=chan_config.code,
        data_src=DATA_SRC.CSV,
        lv_list=[_kl_type(chan_config.frequency)],
        config=c,
        autype=AUTYPE.QFQ,
        max_klines=chan_config.max_klines,
    )
    for _, row in replay.iterrows():
        chan.process_new_kline(
            _build_klu(
                row["timestamp"], row["_open"], row["_high"], row["_low"],
                row["_close"], row.get("_vol", 0.0)
            )
        )
    if chan.last_chan is None:
        return {"bis": [], "segments": [], "zones": []}
    kl_data = chan.last_chan.kl_datas[_kl_type(chan_config.frequency)]
    bis = [
        (
            pd.Timestamp(str(bi.get_begin_klu().time)),
            float(bi.get_begin_val()),
            pd.Timestamp(str(bi.get_end_klu().time)),
            float(bi.get_end_val()),
            bool(bi.is_sure() if callable(getattr(bi, "is_sure", None)) else bi.is_sure),
        )
        for bi in kl_data.bi_list
    ]
    segments = [
        (
            pd.Timestamp(str(seg.get_begin_klu().time)),
            float(seg.get_begin_val()),
            pd.Timestamp(str(seg.get_end_klu().time)),
            float(seg.get_end_val()),
            bool(seg.is_sure),
        )
        for seg in kl_data.seg_list
    ]
    zone_objects = list(kl_data.zs_list)
    for seg in kl_data.seg_list:
        zone_objects.extend(seg.zs_lst)
    zones = []
    seen_zones = set()
    for zs in zone_objects:
        zone = (
            pd.Timestamp(str(zs.begin.time)),
            pd.Timestamp(str(zs.end.time)),
            float(zs.low),
            float(zs.high),
            bool(zs.is_sure),
        )
        key = zone[:4]
        if key not in seen_zones:
            seen_zones.add(key)
            zones.append(zone)
    return {"bis": bis, "segments": segments, "zones": zones}


_BSP_PLOT_THRESHOLD_COLUMNS: dict[str, dict[str, str]] = {
    "1": {
        "divergence_rate": "feat_divergence_rate", "bi_amp": "feat_bsp1_bi_amp",
        "bi_amp_rate": "feat_bsp1_bi_amp_rate", "bi_klu_cnt": "feat_bsp1_bi_klu_cnt",
        "zs_cnt": "feat_zs_cnt",
    },
    "1p": {
        "divergence_rate": "feat_divergence_rate", "bi_amp": "feat_bsp1_bi_amp",
        "bi_amp_rate": "feat_bsp1_bi_amp_rate", "bi_klu_cnt": "feat_bsp1_bi_klu_cnt",
    },
    "2": {
        "retrace_rate": "feat_bsp2_retrace_rate", "break_bi_amp": "feat_bsp2_break_bi_amp",
        "break_bi_amp_rate": "feat_bsp2_break_bi_amp_rate",
        "break_bi_klu_cnt": "feat_bsp2_break_bi_bi_klu_cnt",
        "bi_amp": "feat_bsp2_bi_amp", "bi_amp_rate": "feat_bsp2_bi_amp_rate",
        "bi_klu_cnt": "feat_bsp2_bi_klu_cnt",
    },
    "2s": {
        "retrace_rate": "feat_bsp2s_retrace_rate", "break_bi_amp": "feat_bsp2s_break_bi_amp",
        "break_bi_amp_rate": "feat_bsp2s_break_bi_amp_rate",
        "break_bi_klu_cnt": "feat_bsp2s_break_bi_klu_cnt",
        "bi_amp": "feat_bsp2s_bi_amp", "bi_amp_rate": "feat_bsp2s_bi_amp_rate",
        "bi_klu_cnt": "feat_bsp2s_bi_klu_cnt", "level": "feat_bsp2s_lv",
    },
    "3a": {
        "zs_height": "feat_bsp3_zs_height", "bi_amp": "feat_bsp3_bi_amp",
        "bi_amp_rate": "feat_bsp3_bi_amp_rate", "bi_klu_cnt": "feat_bsp3_bi_klu_cnt",
    },
    "3b": {
        "zs_height": "feat_bsp3_zs_height", "bi_amp": "feat_bsp3_bi_amp",
        "bi_amp_rate": "feat_bsp3_bi_amp_rate", "bi_klu_cnt": "feat_bsp3_bi_klu_cnt",
    },
}


def _apply_bsp_plot_type_thresholds(
    bsp: pd.DataFrame,
    type_thresholds: dict[str, dict] | None,
) -> tuple[pd.DataFrame, dict[str, tuple[int, int]]]:
    """Apply min/max rules only to their corresponding BSP type.

    Friendly rules use names such as ``min_divergence_rate``. Direct workbook
    columns can use ``{"feat_rsi": {"min": 20, "max": 80}}``.
    Numeric NaN values do not pass an active threshold.
    """
    if not type_thresholds:
        return bsp, {}
    result = bsp.copy()
    counts: dict[str, tuple[int, int]] = {}
    for raw_type, rules in type_thresholds.items():
        bsp_type = str(raw_type).lower()
        if not isinstance(rules, dict):
            raise TypeError(f"Thresholds for BSP {bsp_type!r} must be a dictionary")
        type_mask = result["bsp_type"] == bsp_type
        before = int(type_mask.sum())
        keep = pd.Series(True, index=result.index)
        aliases = _BSP_PLOT_THRESHOLD_COLUMNS.get(bsp_type, {})
        for raw_rule, raw_value in rules.items():
            rule = str(raw_rule)
            if rule.startswith("min_") or rule.startswith("max_"):
                bound, feature = rule.split("_", 1)
                column = aliases.get(feature, feature)
                bounds = {bound: raw_value}
            elif isinstance(raw_value, dict):
                column = aliases.get(rule, rule)
                bounds = raw_value
            else:
                raise ValueError(
                    f"Rule {rule!r} must start with min_/max_, or its value must be a min/max dictionary"
                )
            if column not in result.columns:
                raise KeyError(
                    f"BSP {bsp_type} threshold {rule!r} maps to missing workbook column {column!r}"
                )
            numeric = pd.to_numeric(result[column], errors="coerce")
            unknown_bounds = set(bounds).difference({"min", "max", "ranges"})
            if unknown_bounds:
                raise ValueError(f"Unsupported bounds for {rule!r}: {sorted(unknown_bounds)}")
            if "ranges" in bounds:
                ranges = bounds["ranges"]
                if not isinstance(ranges, (list, tuple)) or not ranges:
                    raise ValueError(f"{rule!r} ranges must be a non-empty list")
                range_keep = pd.Series(False, index=result.index)
                for index, accepted_range in enumerate(ranges, start=1):
                    if not isinstance(accepted_range, dict):
                        raise TypeError(f"{rule!r} range {index} must be a dictionary")
                    invalid = set(accepted_range).difference({"min", "max"})
                    if invalid:
                        raise ValueError(
                            f"Unsupported bounds for {rule!r} range {index}: {sorted(invalid)}"
                        )
                    accepted = numeric.notna()
                    if "min" in accepted_range:
                        accepted &= numeric >= float(accepted_range["min"])
                    if "max" in accepted_range:
                        accepted &= numeric <= float(accepted_range["max"])
                    range_keep |= accepted
                keep &= range_keep
            if "min" in bounds:
                keep &= numeric >= float(bounds["min"])
            if "max" in bounds:
                keep &= numeric <= float(bounds["max"])
        result = result.loc[~type_mask | keep].copy()
        counts[bsp_type] = (before, int((result["bsp_type"] == bsp_type).sum()))
    return result, counts


def plot_bsp_type_examples_from_excel(
    config: RealisticSimulationConfig,
    bsp_workbook_path: str | Path | None = None,
    path: str | Path | None = None,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    direction: str | None = None,
    example_types: tuple[str, ...] = ("1", "1p", "2", "2s", "3a", "3b"),
    context_bars: int = 60,
    selection: Literal["first", "largest"] = "largest",
    show_chan_structure: bool = True,
    show_discovery_time: bool = True,
    show: bool = True,
    all_points_by_type: bool = True,
    max_background_points: int = 75_000,
    include_delayed: bool = True,
    type_thresholds: dict[str, dict] | None = None,
) -> dict[str, Path | None] | Path | None:
    """Plot confirmed BSP types.

    By default, create one independent full-period plot per requested type and
    include every matching BSP. Set ``all_points_by_type=False`` to retain the
    legacy behavior: one selected example per type in a combined subplot grid.
    """

    import matplotlib.pyplot as plt

    if int(context_bars) <= 0:
        raise ValueError("context_bars must be greater than zero")
    if selection not in {"first", "largest"}:
        raise ValueError("selection must be 'first' or 'largest'")
    if direction is not None and str(direction).lower() not in {"buy", "sell"}:
        raise ValueError("direction must be 'buy', 'sell', or None")
    if int(max_background_points) <= 0:
        raise ValueError("max_background_points must be greater than zero")

    workbook = bsp_workbook_path or config.bsp_workbook_path
    bsp = pd.read_excel(workbook, sheet_name=config.bsp_sheet_name)
    required = {"timestamp", "direction", "bsp_type", "klu_close"}
    missing = required.difference(bsp.columns)
    if missing:
        raise KeyError(f"BSP workbook is missing columns: {sorted(missing)}")
    bsp["timestamp"] = pd.to_datetime(bsp["timestamp"], errors="raise")
    bsp["direction"] = bsp["direction"].astype(str).str.lower()
    bsp["bsp_type"] = (
        bsp["bsp_type"].astype(str).str.lower().str.replace(r"\.0$", "", regex=True)
    )
    bsp["klu_close"] = pd.to_numeric(bsp["klu_close"], errors="raise")
    if start is not None:
        bsp = bsp.loc[bsp["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        bsp = bsp.loc[bsp["timestamp"] <= pd.Timestamp(end)]
    if direction is not None:
        bsp = bsp.loc[bsp["direction"] == str(direction).lower()]

    raw = _load_ohlcv_csv(config.data_path).reset_index(drop=True)
    _, chan_feed = _prepare_price_frames(config.data_path, config.chan)
    # snapshot_first_seen is one-based and refers to the chronological Chan
    # feed position at which the historical BSP first became discoverable.
    bsp["_discovery_timestamp"] = bsp["timestamp"]
    if "snapshot_first_seen" in bsp.columns:
        snapshot_positions = pd.to_numeric(
            bsp["snapshot_first_seen"], errors="coerce"
        ) - 1
        valid = snapshot_positions.notna() & snapshot_positions.between(
            0, len(chan_feed) - 1
        )
        if valid.any():
            positions = snapshot_positions.loc[valid].astype(int).to_numpy()
            bsp.loc[valid, "_discovery_timestamp"] = pd.to_datetime(
                chan_feed.iloc[positions]["timestamp"].to_numpy()
            )
    bsp["_is_delayed"] = bsp["_discovery_timestamp"] > bsp["timestamp"]
    if not include_delayed:
        bsp = bsp.loc[~bsp["_is_delayed"]].copy()
    bsp, threshold_counts = _apply_bsp_plot_type_thresholds(bsp, type_thresholds)
    raw_index = pd.Series(raw.index.to_numpy(), index=raw["timestamp"])

    if all_points_by_type:
        period_start = pd.Timestamp(start) if start is not None else bsp["timestamp"].min()
        period_end = pd.Timestamp(end) if end is not None else bsp["timestamp"].max()
        price = raw.loc[
            raw["timestamp"].between(period_start, period_end, inclusive="both"),
            ["timestamp", "_close"],
        ]
        if price.empty:
            raise ValueError("No price rows exist in the requested plot period")
        stride = max(1, len(price) // int(max_background_points))
        price_plot = price.iloc[::stride]
        colors = {"buy": "#16803C", "sell": "#C33C3C"}
        markers = {"buy": "^", "sell": "v"}
        saved: dict[str, Path | None] = {}

        def type_path(value: str) -> Path | None:
            if path is None:
                return None
            requested = Path(path)
            safe_type = str(value).replace("/", "_").replace("\\", "_")
            if requested.suffix:
                return requested.with_name(f"{requested.stem}_{safe_type}{requested.suffix}")
            return requested / f"{config.chan.code.upper()}_bsp_type_{safe_type}.png"

        for requested_type in example_types:
            type_name = str(requested_type).lower()
            points = bsp.loc[bsp["bsp_type"] == type_name].sort_values("timestamp")
            fig, ax = plt.subplots(figsize=(22, 8))
            ax.plot(
                price_plot["timestamp"], price_plot["_close"],
                color="#62748A", linewidth=0.75,
                label=f"{config.chan.code.upper()} close", zorder=1,
            )
            for point_direction in ("buy", "sell"):
                group = points.loc[points["direction"] == point_direction]
                if group.empty:
                    continue
                ax.scatter(
                    group["timestamp"], group["klu_close"],
                    color=colors[point_direction], marker=markers[point_direction],
                    s=34, alpha=0.82, linewidths=0.3,
                    label=f"{point_direction.title()} ({len(group):,})", zorder=3,
                )
            ax.set_title(
                f"{config.chan.code.upper()} — all confirmed BSP type {requested_type} "
                f"({len(points):,} points; {'including delayed' if include_delayed else 'immediate only'})\n"
                f"{period_start} to {period_end}"
            )
            if type_name in threshold_counts:
                before, after = threshold_counts[type_name]
                ax.set_title(ax.get_title() + f" | threshold filter: {before:,} → {after:,}")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Price")
            ax.grid(alpha=0.2)
            ax.legend(loc="best")
            fig.autofmt_xdate()
            fig.tight_layout()
            target = type_path(type_name)
            saved_path: Path | None = None
            if target is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(target, dpi=180, bbox_inches="tight")
                saved_path = target.resolve()
            saved[type_name] = saved_path
            if show:
                plt.show()
            else:
                plt.close(fig)
        if not saved:
            raise ValueError("No requested BSP types exist in the selected period")
        return saved

    examples: list[tuple[str, pd.Series]] = []
    for bsp_type in example_types:
        group = bsp.loc[bsp["bsp_type"] == str(bsp_type).lower()].copy()
        if group.empty:
            continue
        if selection == "largest":
            amp_column = next(
                (
                    name
                    for name in (
                        "feat_bsp_bi_amp",
                        f"feat_bsp{bsp_type}_bi_amp",
                        "feat_bi_amp",
                    )
                    if name in group.columns and group[name].notna().any()
                ),
                None,
            )
            event = (
                group.loc[pd.to_numeric(group[amp_column], errors="coerce").idxmax()]
                if amp_column is not None
                else group.sort_values("timestamp").iloc[0]
            )
        else:
            event = group.sort_values("timestamp").iloc[0]
        examples.append((str(bsp_type), event))
    if not examples:
        raise ValueError("No requested BSP types exist in the selected period")

    columns = 2
    rows_count = (len(examples) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(20, max(4.5 * rows_count, 5)),
        squeeze=False,
    )
    colors = {"buy": "#16803C", "sell": "#C33C3C"}
    markers = {"buy": "^", "sell": "v"}
    for ax, (bsp_type, event) in zip(axes.flat, examples):
        timestamp = pd.Timestamp(event["timestamp"])
        trigger_timestamp = timestamp
        snapshot = event.get("snapshot_first_seen")
        if snapshot is not None and not pd.isna(snapshot):
            snapshot_pos = int(snapshot) - 1
            if 0 <= snapshot_pos < len(chan_feed):
                trigger_timestamp = pd.Timestamp(
                    chan_feed.iloc[snapshot_pos]["timestamp"]
                )
        if timestamp not in raw_index.index:
            ax.set_visible(False)
            continue
        center_value = raw_index.loc[timestamp]
        center = int(
            center_value.iloc[-1]
            if isinstance(center_value, pd.Series)
            else center_value
        )
        trigger_positions = raw.index[raw["timestamp"] <= trigger_timestamp]
        trigger_center = int(trigger_positions[-1]) if len(trigger_positions) else center
        left = max(0, min(center, trigger_center) - int(context_bars))
        right = min(len(raw), max(center, trigger_center) + int(context_bars) + 1)
        window = raw.iloc[left:right]
        event_direction = str(event["direction"]).lower()
        ax.plot(
            window["timestamp"],
            window["_close"],
            color="#62748A",
            linewidth=0.9,
        )
        ax.scatter(
            [timestamp],
            [float(event["klu_close"])],
            color=colors[event_direction],
            marker=markers[event_direction],
            s=100,
            zorder=4,
            label=f"{event_direction.title()} BSP {bsp_type}",
        )
        ax.axvline(timestamp, color=colors[event_direction], alpha=0.3, linewidth=0.8)
        if show_discovery_time and trigger_timestamp > timestamp:
            ax.axvline(
                trigger_timestamp,
                color="#A25AC4",
                linestyle="--",
                alpha=0.75,
                linewidth=1.0,
                label=f"discovered {trigger_timestamp}",
            )
        if show_chan_structure:
            geometry = _chan_geometry_at_timestamp(raw, config.chan, trigger_timestamp)
            bi_label_used = seg_label_used = zs_label_used = False
            for x1, y1, x2, y2, sure in geometry["bis"]:
                if x2 < window["timestamp"].min() or x1 > window["timestamp"].max():
                    continue
                ax.plot(
                    [x1, x2], [y1, y2], color="#E28A2B", linewidth=1.25,
                    linestyle="-" if sure else "--", alpha=0.9,
                    label="Bi" if not bi_label_used else None, zorder=2,
                )
                bi_label_used = True
            for x1, y1, x2, y2, sure in geometry["segments"]:
                if x2 < window["timestamp"].min() or x1 > window["timestamp"].max():
                    continue
                ax.plot(
                    [x1, x2], [y1, y2], color="#7651B5", linewidth=2.2,
                    linestyle="-" if sure else "--", alpha=0.9,
                    label="Segment" if not seg_label_used else None, zorder=2.5,
                )
                seg_label_used = True
            for x1, x2, low, high, sure in geometry["zones"]:
                if x2 < window["timestamp"].min() or x1 > window["timestamp"].max():
                    continue
                ax.fill_between(
                    [x1, x2], [low, low], [high, high], color="#4F9D8A",
                    alpha=0.16 if sure else 0.08,
                    label="Zhongshu" if not zs_label_used else None, zorder=1.5,
                )
                zs_label_used = True
        detail_parts = []
        for column, label in (
            ("segment_direction", "segment"),
            ("feat_divergence_rate", "divergence"),
            ("feat_bsp2_retrace_rate", "retrace"),
            ("feat_bsp2s_retrace_rate", "retrace"),
            ("feat_bsp3_zs_height", "ZS height"),
        ):
            value = event.get(column)
            if value is not None and not pd.isna(value):
                detail_parts.append(
                    f"{label}={value:.4f}" if isinstance(value, (int, float)) else f"{label}={value}"
                )
        ax.set_title(
            f"BSP {bsp_type} example — {event_direction.title()} — {timestamp}\n"
            + ", ".join(detail_parts)
        )
        ax.set_ylabel("Price")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
    for ax in axes.flat[len(examples):]:
        ax.set_visible(False)
    fig.suptitle(
        f"{config.chan.code.upper()} confirmed BSP type examples "
        f"({selection} amplitude selection)",
        fontsize=14,
    )
    fig.autofmt_xdate()
    fig.tight_layout()

    saved_path: Path | None = None
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=180, bbox_inches="tight")
        saved_path = target.resolve()
    if show:
        plt.show()
    else:
        plt.close(fig)
    return saved_path


def interactive_bsp_type_explorer_from_excel(
    config: RealisticSimulationConfig,
    bsp_workbook_path: str | Path | None = None,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    example_types: tuple[str, ...] = ("1", "1p", "2", "2s", "3a", "3b"),
    max_background_points: int = 75_000,
):
    """Display a Jupyter BSP threshold explorer with buttons and live Plotly output.

    Data is loaded once. Select a BSP type, direction and delayed-point policy,
    edit that type's Chan thresholds, then click ``Update Plot``. Thresholds
    use the same aliases and filtering implementation as the static plot.
    """
    try:
        import ipywidgets as widgets
        import plotly.graph_objects as go
        from IPython.display import HTML, clear_output, display
    except ImportError as exc:
        raise ImportError(
            "Interactive BSP controls require ipywidgets. In Jupyter run "
            "`%pip install ipywidgets`, restart the kernel, then call this function again."
        ) from exc
    if int(max_background_points) <= 0:
        raise ValueError("max_background_points must be greater than zero")

    workbook = bsp_workbook_path or config.bsp_workbook_path
    bsp = pd.read_excel(workbook, sheet_name=config.bsp_sheet_name)
    required = {"timestamp", "direction", "bsp_type", "klu_close"}
    missing = required.difference(bsp.columns)
    if missing:
        raise KeyError(f"BSP workbook is missing columns: {sorted(missing)}")
    bsp["timestamp"] = pd.to_datetime(bsp["timestamp"], errors="raise")
    bsp["direction"] = bsp["direction"].astype(str).str.lower()
    bsp["bsp_type"] = bsp["bsp_type"].astype(str).str.lower().str.replace(r"\.0$", "", regex=True)
    bsp["klu_close"] = pd.to_numeric(bsp["klu_close"], errors="raise")
    period_start = pd.Timestamp(start) if start is not None else bsp["timestamp"].min()
    period_end = pd.Timestamp(end) if end is not None else bsp["timestamp"].max()
    bsp = bsp.loc[bsp["timestamp"].between(period_start, period_end, inclusive="both")].copy()

    raw = _load_ohlcv_csv(config.data_path).reset_index(drop=True)
    price = raw.loc[
        raw["timestamp"].between(period_start, period_end, inclusive="both"),
        ["timestamp", "_close"],
    ]
    if price.empty:
        raise ValueError("No price rows exist in the requested period")
    stride = max(1, len(price) // int(max_background_points))
    price_plot = price.iloc[::stride]
    _, chan_feed = _prepare_price_frames(config.data_path, config.chan)
    bsp["_discovery_timestamp"] = bsp["timestamp"]
    if "snapshot_first_seen" in bsp.columns:
        positions = pd.to_numeric(bsp["snapshot_first_seen"], errors="coerce") - 1
        valid = positions.notna() & positions.between(0, len(chan_feed) - 1)
        if valid.any():
            bsp.loc[valid, "_discovery_timestamp"] = pd.to_datetime(
                chan_feed.iloc[positions.loc[valid].astype(int).to_numpy()]["timestamp"].to_numpy()
            )
    bsp["_is_delayed"] = bsp["_discovery_timestamp"] > bsp["timestamp"]

    type_buttons = widgets.ToggleButtons(
        options=[(f"Type {value}", str(value).lower()) for value in example_types],
        description="BSP type:", button_style="info",
    )
    direction_widget = widgets.Dropdown(
        options=[("All", "all"), ("Buy", "buy"), ("Sell", "sell")],
        value="all", description="Direction:",
    )
    delayed_widget = widgets.Checkbox(value=True, description="Include delayed")
    slider_range_widget = widgets.ToggleButtons(
        options=[
            ("Middle 90%", "90"), ("Middle 95%", "95"),
            ("Middle 98%", "98"), ("Middle 99%", "99"), ("Full", "full"),
        ],
        value="90", description="Slider range:", button_style="",
        style={"description_width": "100px"},
    )
    update_button = widgets.Button(description="Update Plot", button_style="success", icon="refresh")
    reset_button = widgets.Button(description="Reset Thresholds", icon="undo")
    status = widgets.HTML()
    controls_box = widgets.VBox()
    plot_output = widgets.Output()

    # The most meaningful Chan thresholds for each point family. A blank/NaN
    # field means that bound is disabled.
    indicator_thresholds = (("macd_value", "MACD"), ("rsi", "RSI"))
    profiles = {
        "1": (("divergence_rate", "Divergence"), ("bi_amp_rate", "Bi amp rate"),
              ("bi_klu_cnt", "Bi bars"), ("zs_cnt", "ZS count"),
              *indicator_thresholds),
        "1p": (("divergence_rate", "Divergence"), ("bi_amp_rate", "Bi amp rate"),
               ("bi_klu_cnt", "Bi bars"), *indicator_thresholds),
        "2": (("retrace_rate", "Retrace"), ("break_bi_amp_rate", "Break amp rate"),
              ("bi_amp_rate", "Bi amp rate"), ("bi_klu_cnt", "Bi bars"),
              *indicator_thresholds),
        "2s": (("retrace_rate", "Retrace"), ("break_bi_amp_rate", "Break amp rate"),
               ("bi_amp_rate", "Bi amp rate"), ("bi_klu_cnt", "Bi bars"),
               ("level", "2s level"), *indicator_thresholds),
        "3a": (("zs_height", "ZS height"), ("bi_amp_rate", "Bi amp rate"),
               ("bi_klu_cnt", "Bi bars"), *indicator_thresholds),
        "3b": (("zs_height", "ZS height"), ("bi_amp_rate", "Bi amp rate"),
               ("bi_klu_cnt", "Bi bars"), *indicator_thresholds),
    }
    threshold_widgets: dict[str, object] = {}
    threshold_full_ranges: dict[str, tuple[float, float]] = {}
    secondary_threshold_widgets: dict[str, object] = {}
    secondary_enabled_widgets: dict[str, object] = {}

    def rebuild_threshold_controls(*_):
        threshold_widgets.clear()
        threshold_full_ranges.clear()
        secondary_threshold_widgets.clear()
        secondary_enabled_widgets.clear()
        rows = []
        selected_type = type_buttons.value
        aliases = _BSP_PLOT_THRESHOLD_COLUMNS.get(selected_type, {})
        type_rows = bsp.loc[bsp["bsp_type"] == selected_type]
        for alias, label in profiles.get(type_buttons.value, ()):
            column = aliases.get(alias, alias)
            if column not in type_rows.columns:
                rows.append(widgets.HTML(f"<i>{label}: workbook column {column} is unavailable</i>"))
                continue
            numeric = pd.to_numeric(type_rows[column], errors="coerce").dropna()
            if numeric.empty:
                rows.append(widgets.HTML(f"<i>{label}: no numeric values available</i>"))
                continue
            range_mode = slider_range_widget.value
            if range_mode == "full":
                lower, upper = float(numeric.min()), float(numeric.max())
            else:
                central_fraction = float(range_mode) / 100.0
                tail = (1.0 - central_fraction) / 2.0
                lower, upper = map(float, numeric.quantile([tail, 1.0 - tail]).to_numpy())
            if np.isclose(lower, upper):
                padding = max(abs(lower) * 0.01, 1.0)
                slider_min, slider_max = lower - padding, upper + padding
            else:
                slider_min, slider_max = lower, upper
            step = max((slider_max - slider_min) / 250.0, 1e-8)
            slider = widgets.FloatRangeSlider(
                value=(lower, upper), min=slider_min, max=slider_max, step=step,
                description=f"{label}:", continuous_update=False,
                readout=True, readout_format=".4g",
                layout=widgets.Layout(width="720px"),
                style={"description_width": "140px"},
            )
            threshold_widgets[alias] = slider
            threshold_full_ranges[alias] = (lower, upper)
            slider.observe(update_plot, names="value")
            rows.append(slider)
            if alias in {"macd_value", "rsi"}:
                second_slider = widgets.FloatRangeSlider(
                    value=(lower, upper), min=slider_min, max=slider_max, step=step,
                    description=f"{label} range 2:", continuous_update=False,
                    readout=True, readout_format=".4g", disabled=True,
                    layout=widgets.Layout(width="720px"),
                    style={"description_width": "140px"},
                )
                second_enabled = widgets.Checkbox(
                    value=False, description=f"Use second {label} range",
                    indent=False,
                )

                def toggle_second(change, *, target=second_slider):
                    target.disabled = not bool(change["new"])
                    update_plot()

                second_enabled.observe(toggle_second, names="value")
                second_slider.observe(update_plot, names="value")
                secondary_threshold_widgets[alias] = second_slider
                secondary_enabled_widgets[alias] = second_enabled
                rows.extend((second_enabled, second_slider))
        controls_box.children = tuple(rows)

    def current_rules() -> dict:
        rules = {}
        for alias, slider in threshold_widgets.items():
            minimum, maximum = map(float, slider.value)
            full_minimum, full_maximum = threshold_full_ranges[alias]
            bounds = {}
            tolerance = max(abs(full_maximum - full_minimum) * 1e-9, 1e-12)
            if minimum > full_minimum + tolerance:
                bounds["min"] = minimum
            if maximum < full_maximum - tolerance:
                bounds["max"] = maximum
            second_enabled = secondary_enabled_widgets.get(alias)
            if second_enabled is not None and second_enabled.value:
                second_slider = secondary_threshold_widgets[alias]
                second_minimum, second_maximum = map(float, second_slider.value)
                second_bounds = {}
                if second_minimum > full_minimum + tolerance:
                    second_bounds["min"] = second_minimum
                if second_maximum < full_maximum - tolerance:
                    second_bounds["max"] = second_maximum
                rules[alias] = {"ranges": [bounds, second_bounds]}
            elif bounds:
                rules[alias] = bounds
        return rules

    def update_plot(_=None):
        selected_type = type_buttons.value
        subset = bsp.loc[bsp["bsp_type"] == selected_type].copy()
        original_count = len(subset)
        if not delayed_widget.value:
            subset = subset.loc[~subset["_is_delayed"]]
        if direction_widget.value != "all":
            subset = subset.loc[subset["direction"] == direction_widget.value]
        pre_threshold_count = len(subset)
        rules = current_rules()
        if rules:
            subset, _ = _apply_bsp_plot_type_thresholds(
                subset, {selected_type: rules}
            )
        figure = go.Figure()
        figure.add_trace(go.Scattergl(
            x=price_plot["timestamp"], y=price_plot["_close"], mode="lines",
            name=f"{config.chan.code.upper()} close",
            line={"color": "#62748A", "width": 1},
        ))
        for point_direction, color, symbol in (
            ("buy", "#16803C", "triangle-up"), ("sell", "#C33C3C", "triangle-down")
        ):
            group = subset.loc[subset["direction"] == point_direction]
            if len(group):
                macd_hover = pd.to_numeric(
                    group.get("macd_value", pd.Series(np.nan, index=group.index)),
                    errors="coerce",
                )
                rsi_hover = pd.to_numeric(
                    group.get("rsi", pd.Series(np.nan, index=group.index)),
                    errors="coerce",
                )
                figure.add_trace(go.Scattergl(
                    x=group["timestamp"], y=group["klu_close"], mode="markers",
                    name=f"{point_direction.title()} ({len(group):,})",
                    marker={"color": color, "symbol": symbol, "size": 9},
                    customdata=np.column_stack((
                        group["bsp_type"], group["_discovery_timestamp"],
                        macd_hover, rsi_hover,
                    )),
                    hovertemplate=(
                        "Time=%{x}<br>Price=%{y:.4f}<br>Type=%{customdata[0]}"
                        "<br>Discovered=%{customdata[1]}<br>MACD=%{customdata[2]:.5g}"
                        "<br>RSI=%{customdata[3]:.3f}<extra></extra>"
                    ),
                ))
        figure.update_layout(
            title=f"{config.chan.code.upper()} BSP {selected_type}: {len(subset):,} points",
            xaxis_title="Timestamp", yaxis_title="Price", template="plotly_white",
            height=650, hovermode="closest", legend={"orientation": "h"},
        )
        status.value = (
            f"<b>Type {selected_type}</b>: workbook period count {original_count:,} → "
            f"direction/delay count {pre_threshold_count:,} → threshold count <b>{len(subset):,}</b>"
        )
        with plot_output:
            clear_output(wait=True)
            # Plotly's normal notebook MIME renderer requires nbformat. Some
            # lightweight Jupyter kernels have IPython/widgets but omit it.
            # Fall back to self-contained HTML so the explorer still works.
            try:
                import nbformat
                nbformat_ok = tuple(int(part) for part in nbformat.__version__.split(".")[:2]) >= (4, 2)
            except (ImportError, AttributeError, ValueError):
                nbformat_ok = False
            if nbformat_ok:
                display(figure)
            else:
                display(HTML(figure.to_html(
                    full_html=False,
                    include_plotlyjs=True,
                    config={"responsive": True, "displaylogo": False},
                )))

    def reset_thresholds(_=None):
        for alias, slider in threshold_widgets.items():
            slider.value = threshold_full_ranges[alias]
        for alias, slider in secondary_threshold_widgets.items():
            slider.value = threshold_full_ranges[alias]
            slider.disabled = True
            secondary_enabled_widgets[alias].value = False
        update_plot()

    type_buttons.observe(rebuild_threshold_controls, names="value")
    type_buttons.observe(update_plot, names="value")
    slider_range_widget.observe(rebuild_threshold_controls, names="value")
    slider_range_widget.observe(update_plot, names="value")
    direction_widget.observe(update_plot, names="value")
    delayed_widget.observe(update_plot, names="value")
    update_button.on_click(update_plot)
    reset_button.on_click(reset_thresholds)
    rebuild_threshold_controls()
    dashboard = widgets.VBox([
        widgets.HTML("<h3>Interactive Chan BSP Threshold Explorer</h3>"),
        type_buttons,
        widgets.HBox([direction_widget, delayed_widget]),
        slider_range_widget,
        widgets.HTML(
            "<small>Slider endpoints are open bounds: leaving a handle at an endpoint "
            "does not remove values outside the displayed percentile range.</small>"
        ),
        controls_box,
        widgets.HBox([update_button, reset_button]),
        status,
        plot_output,
    ])
    display(dashboard)
    update_plot()
    return dashboard


def plot_mature_bspoints_from_excel(
    config: RealisticSimulationConfig,
    bsp_workbook_path: str | Path | None = None,
    path: str | Path | None = None,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    bsp_types: tuple[str, ...] = ("1", "1p", "2", "2s", "3a", "3b"),
    direction: str | None = None,
    minimum_mature_rate: float = 1.0,
    max_background_points: int = 75_000,
    annotate_types: bool = False,
    show: bool = True,
) -> Path | None:
    """Plot only BSPs carrying an explicit mature flag or mature-rate value.

    Snapshot persistence is deliberately not interpreted as maturity. The BSP
    workbook must contain ``is_mature``, ``is_mature_point``, or
    ``mature_rate`` as produced by a maturity-aware export.
    """
    import matplotlib.pyplot as plt

    if direction is not None and str(direction).lower() not in {"buy", "sell"}:
        raise ValueError("direction must be 'buy', 'sell', or None")
    if not 0.0 <= float(minimum_mature_rate) <= 1.0:
        raise ValueError("minimum_mature_rate must be between 0 and 1")
    if int(max_background_points) <= 0:
        raise ValueError("max_background_points must be greater than zero")

    workbook = Path(bsp_workbook_path or config.bsp_workbook_path)
    bsp = pd.read_excel(workbook, sheet_name=config.bsp_sheet_name)
    required = {"timestamp", "direction", "bsp_type", "klu_close"}
    missing = required.difference(bsp.columns)
    if missing:
        raise KeyError(f"BSP workbook is missing columns: {sorted(missing)}")

    maturity_columns = [
        column for column in ("is_mature", "is_mature_point", "mature_rate")
        if column in bsp.columns
    ]
    if not maturity_columns:
        raise ValueError(
            f"{workbook} has no explicit maturity column. Regenerate it with "
            "is_mature/is_mature_point or mature_rate; snapshot persistence is "
            "not a valid maturity label."
        )

    bsp["timestamp"] = pd.to_datetime(bsp["timestamp"], errors="raise")
    bsp["direction"] = bsp["direction"].astype(str).str.lower()
    bsp["bsp_type"] = (
        bsp["bsp_type"].astype(str).str.lower().str.replace(r"\.0$", "", regex=True)
    )
    bsp["klu_close"] = pd.to_numeric(bsp["klu_close"], errors="coerce")
    mature = pd.Series(False, index=bsp.index)
    for column in ("is_mature", "is_mature_point"):
        if column in bsp.columns:
            values = bsp[column]
            mature |= values.fillna(False).astype(str).str.strip().str.lower().isin(
                {"1", "1.0", "true", "yes", "y"}
            )
    if "mature_rate" in bsp.columns:
        mature |= pd.to_numeric(bsp["mature_rate"], errors="coerce").ge(
            float(minimum_mature_rate)
        )

    period_start = pd.Timestamp(start) if start is not None else bsp["timestamp"].min()
    period_end = pd.Timestamp(end) if end is not None else bsp["timestamp"].max()
    allowed_types = {str(value).lower() for value in bsp_types}
    point_mask = (
        mature
        & bsp["timestamp"].between(period_start, period_end, inclusive="both")
        & bsp["bsp_type"].isin(allowed_types)
    )
    if direction is not None:
        point_mask &= bsp["direction"].eq(str(direction).lower())
    points = bsp.loc[point_mask].sort_values("timestamp")

    raw = _load_ohlcv_csv(config.data_path)
    price = raw.loc[
        raw["timestamp"].between(period_start, period_end, inclusive="both"),
        ["timestamp", "_close"],
    ]
    if price.empty:
        raise ValueError("No price rows exist in the requested plot period")
    stride = max(1, len(price) // int(max_background_points))

    fig, ax = plt.subplots(figsize=(22, 9))
    ax.plot(
        price.iloc[::stride]["timestamp"], price.iloc[::stride]["_close"],
        color="#62748A", linewidth=0.8,
        label=f"{config.chan.code.upper()} close", zorder=1,
    )
    for point_direction, color, marker in (
        ("buy", "#16803C", "^"), ("sell", "#C33C3C", "v")
    ):
        group = points.loc[points["direction"] == point_direction]
        if group.empty:
            continue
        ax.scatter(
            group["timestamp"], group["klu_close"], color=color, marker=marker,
            s=46, alpha=0.86, linewidths=0.35,
            label=f"Mature {point_direction.title()} ({len(group):,})", zorder=3,
        )
        if annotate_types:
            for row in group.itertuples(index=False):
                ax.annotate(
                    str(row.bsp_type), (row.timestamp, row.klu_close),
                    xytext=(0, 7 if point_direction == "buy" else -11),
                    textcoords="offset points", ha="center", fontsize=8, color=color,
                )

    ax.set_title(
        f"{config.chan.code.upper()} — Mature Chan BSPs ({len(points):,})\n"
        f"{period_start} to {period_end}"
    )
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()

    saved_path: Path | None = None
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=180, bbox_inches="tight")
        saved_path = target.resolve()
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(
        f"[Mature BSP] columns={maturity_columns}; points={len(points):,}; "
        f"saved={saved_path}"
    )
    return saved_path


def plot_mature_bsp_quality_from_excel(
    config: RealisticSimulationConfig,
    maturity_workbook_path: str | Path,
    path: str | Path | None = None,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    bsp_types: tuple[str, ...] = ("1", "1p", "2", "2s", "3a", "3b"),
    direction: str | None = None,
    include_censored: bool = False,
    plot_at: Literal["bsp", "maturity"] = "maturity",
    max_background_points: int = 75_000,
    annotate_types: bool = False,
    show: bool = True,
) -> Path | None:
    """Plot mature BSPs colored by held, covered, invalidated, or censored."""
    import matplotlib.pyplot as plt

    if direction is not None and str(direction).lower() not in {"buy", "sell"}:
        raise ValueError("direction must be 'buy', 'sell', or None")
    if plot_at not in {"bsp", "maturity"}:
        raise ValueError("plot_at must be 'bsp' or 'maturity'")
    quality = pd.read_excel(maturity_workbook_path, sheet_name="Mature BSP Quality")
    required = {
        "bsp_timestamp", "maturity_timestamp", "direction", "bsp_type",
        "klu_close", "quality_status",
    }
    missing = required.difference(quality.columns)
    if missing:
        raise KeyError(f"Mature BSP Quality sheet is missing columns: {sorted(missing)}")
    quality["bsp_timestamp"] = pd.to_datetime(quality["bsp_timestamp"], errors="raise")
    quality["maturity_timestamp"] = pd.to_datetime(quality["maturity_timestamp"], errors="raise")
    quality["direction"] = quality["direction"].astype(str).str.lower()
    quality["bsp_type"] = quality["bsp_type"].astype(str).str.lower().str.replace(r"\.0$", "", regex=True)
    quality["klu_close"] = pd.to_numeric(quality["klu_close"], errors="coerce")
    event_column = "bsp_timestamp" if plot_at == "bsp" else "maturity_timestamp"
    price_column = "klu_close"
    if plot_at == "maturity" and "maturity_market_price" in quality.columns:
        quality["maturity_market_price"] = pd.to_numeric(
            quality["maturity_market_price"], errors="coerce"
        )
        price_column = "maturity_market_price"
    period_start = pd.Timestamp(start) if start is not None else quality[event_column].min()
    period_end = pd.Timestamp(end) if end is not None else quality[event_column].max()
    mask = (
        quality[event_column].between(period_start, period_end, inclusive="both")
        & quality["bsp_type"].isin({str(value).lower() for value in bsp_types})
    )
    if direction is not None:
        mask &= quality["direction"].eq(str(direction).lower())
    if not include_censored:
        mask &= ~quality["quality_status"].eq("censored")
    points = quality.loc[mask].copy()

    raw = _load_ohlcv_csv(config.data_path)
    price = raw.loc[
        raw["timestamp"].between(period_start, period_end, inclusive="both"),
        ["timestamp", "_close"],
    ]
    if price.empty:
        raise ValueError("No price rows exist in the requested plot period")
    stride = max(1, len(price) // int(max_background_points))
    fig, ax = plt.subplots(figsize=(22, 9))
    ax.plot(
        price.iloc[::stride]["timestamp"], price.iloc[::stride]["_close"],
        color="#62748A", linewidth=0.8, label=f"{config.chan.code.upper()} close", zorder=1,
    )
    styles = {
        "held": ("#16803C", "o", "Held"),
        "covered": ("#C33C3C", "X", "Covered"),
        "structurally_invalidated": ("#D97706", "s", "Structurally invalidated"),
        "censored": ("#7C3AED", "D", "Censored"),
    }
    for status_name, (color, marker, label) in styles.items():
        group = points.loc[points["quality_status"] == status_name]
        if group.empty:
            continue
        ax.scatter(
            group[event_column], group[price_column], color=color, marker=marker,
            s=48, alpha=0.86, linewidths=0.35,
            label=f"{label} ({len(group):,})", zorder=3,
        )
        if annotate_types:
            for row in group.itertuples(index=False):
                ax.annotate(
                    str(row.bsp_type), (getattr(row, event_column), getattr(row, price_column)),
                    xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=8, color=color,
                )
    held = int(points["quality_status"].eq("held").sum())
    resolved = int(points["quality_status"].isin({"held", "covered", "structurally_invalidated"}).sum())
    ax.set_title(
        f"{config.chan.code.upper()} — Mature BSP post-confirmation quality\n"
        f"{period_start} to {period_end} | held={held:,}/{resolved:,} resolved"
    )
    ax.set_xlabel("BSP timestamp" if plot_at == "bsp" else "Maturity timestamp")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    saved_path: Path | None = None
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=180, bbox_inches="tight")
        saved_path = target.resolve()
    if show:
        plt.show()
    else:
        plt.close(fig)
    return saved_path


def load_virtual_bsp_backtest_data(
    config: RealisticSimulationConfig,
    virtual_bsp_path: str | Path,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read price CSV and saved virtual BSP workbook for direct replay."""

    candidates = pd.read_excel(
        virtual_bsp_path, sheet_name="Virtual BSP Candidates"
    )
    required = {"bsp_timestamp", "direction", "klu_close"}
    missing = required.difference(candidates.columns)
    if missing:
        raise KeyError(
            f"Virtual BSP workbook is missing columns: {sorted(missing)}"
        )
    timestamp_source = (
        "trigger_timestamp"
        if "trigger_timestamp" in candidates.columns
        else "bsp_timestamp"
    )
    candidates["timestamp"] = pd.to_datetime(
        candidates[timestamp_source], errors="raise"
    )
    candidates["bsp_timestamp"] = pd.to_datetime(
        candidates["bsp_timestamp"], errors="raise"
    )
    candidates["direction"] = candidates["direction"].astype(str).str.lower()
    candidates["bsp_type"] = "virtual_bi"
    if "segment_direction" not in candidates.columns and "seg_direction" in candidates.columns:
        candidates["segment_direction"] = candidates["seg_direction"]
    if "segment_confirmed" not in candidates.columns and "seg_is_sure" in candidates.columns:
        candidates["segment_confirmed"] = candidates["seg_is_sure"]

    raw = _load_ohlcv_csv(config.data_path)
    period_start = pd.Timestamp(start) if start is not None else candidates["timestamp"].min()
    period_end = pd.Timestamp(end) if end is not None else candidates["timestamp"].max()
    if pd.isna(period_start) or pd.isna(period_end):
        raise ValueError("The virtual BSP workbook contains no candidates")
    if period_end < period_start:
        raise ValueError("end must be greater than or equal to start")
    raw = raw.loc[
        raw["timestamp"].between(period_start, period_end, inclusive="both")
    ].copy()
    if raw.empty:
        raise ValueError("No price rows exist in the requested backtest period")
    price = raw.assign(
        open=raw["_open"],
        high=raw["_high"],
        low=raw["_low"],
        close=raw["_close"],
        volume=raw["_vol"],
    )[["timestamp", "open", "high", "low", "close", "volume"]]
    candidates = candidates.loc[
        candidates["timestamp"].between(period_start, period_end, inclusive="both")
    ].copy()
    candidates = candidates.loc[candidates["timestamp"].isin(price["timestamp"])]
    candidates = candidates.sort_values("timestamp").drop_duplicates(
        ["timestamp", "direction", "bsp_type"], keep="first"
    )
    return price.reset_index(drop=True), candidates.reset_index(drop=True)


def default_virtual_bsp_strategy_config() -> SegBspStrategyConfig:
    """Return a simple long-only configuration for virtual-Bi candidates."""

    return SegBspStrategyConfig(
        entry_segment_directions=frozenset({"up", "down"}),
        entry_bsp_types=frozenset({"virtual_bi"}),
        sell_bsp_types=frozenset({"virtual_bi"}),
        required_buy_signals=1,
        buy_lookback_bars=5,
        required_sell_signals=1,
        sell_lookback_bars=5,
        exit_on_down_segment=False,
        allow_unconfirmed_entry=True,
        position_fraction=1.0,
    )


def run_virtual_bsp_simulation_from_excel(
    config: RealisticSimulationConfig,
    virtual_bsp_path: str | Path | None = None,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    strategy_config: SegBspStrategyConfig | None = None,
) -> RealisticSimulationOutput:
    """Backtest saved virtual candidates with next-bar-open execution."""

    workbook_path = virtual_bsp_path or config.bsp_workbook_path
    effective_start = config.chan.start if start is None else start
    effective_end = config.chan.end if end is None else end
    price, candidates = load_virtual_bsp_backtest_data(
        config, workbook_path, start=effective_start, end=effective_end
    )
    result = run_bsp_backtest(
        price,
        candidates,
        SegBspStrategy(strategy_config or config.strategy),
        config.backtest,
        RiskManager(config.risk),
    )
    return RealisticSimulationOutput(
        result=result,
        price_df=price,
        bsp_df=candidates,
        chan_feed_df=pd.DataFrame(),
        config=config,
    )


def plot_immediate_vs_virtual_bspoints_from_files(
    config: RealisticSimulationConfig | None = None,
    path: str | Path | None = None,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    show: bool = True,
    max_background_points: int = 75_000,
    virtual_lookback_bars: int = 5,
) -> Path | None:
    """Compare immediate confirmed BSPs with possible virtual-Bi BSPs.

    Both panels use historical endpoint timestamps and prices.  The upper
    panel excludes delayed confirmed BSPs.  The lower panel shows first-seen
    endpoints of unconfirmed virtual Bis, without implying that those
    provisional candidates will later become confirmed BSPs.
    """

    import matplotlib.pyplot as plt

    if int(max_background_points) <= 0:
        raise ValueError("max_background_points must be greater than zero")
    cfg = config or RealisticSimulationConfig()
    comparison_cfg = replace(
        cfg,
        ignore_delayed_bspoints=False,
        max_delayed_price_gap_pct=None,
        max_discovery_delay_minutes=None,
    )
    price, confirmed, chan_feed = prepare_simulation_data(comparison_cfg)
    price = price.copy()
    price["timestamp"] = pd.to_datetime(price["timestamp"], errors="raise")
    period_start = pd.Timestamp(start) if start is not None else price["timestamp"].min()
    period_end = pd.Timestamp(end) if end is not None else price["timestamp"].max()
    if period_end < period_start:
        raise ValueError("end must be greater than or equal to start")

    price = price.loc[
        price["timestamp"].between(period_start, period_end, inclusive="both"),
        ["timestamp", "close"],
    ].copy()
    if price.empty:
        raise ValueError("No price rows exist in the requested period")

    confirmed = confirmed.copy()
    confirmed["bsp_timestamp"] = pd.to_datetime(
        confirmed["bsp_timestamp"], errors="raise"
    )
    confirmed["trigger_timestamp"] = pd.to_datetime(
        confirmed["trigger_timestamp"], errors="raise"
    )
    confirmed["klu_close"] = pd.to_numeric(confirmed["klu_close"], errors="raise")
    confirmed = confirmed.loc[
        confirmed["bsp_timestamp"].between(
            period_start, period_end, inclusive="both"
        )
        & (confirmed["trigger_timestamp"] <= confirmed["bsp_timestamp"])
    ].copy()

    virtual = generate_virtual_bsp_candidates(
        chan_feed,
        comparison_cfg.chan,
        start=period_start,
        end=period_end,
        virtual_lookback_bars=virtual_lookback_bars,
    )

    stride = max(1, len(price) // int(max_background_points))
    price_plot = price.iloc[::stride]
    colors = {"buy": "#16803C", "sell": "#C33C3C"}
    markers = {"buy": "^", "sell": "v"}
    fig, axes = plt.subplots(2, 1, figsize=(20, 12), sharex=True, sharey=True)
    panels = [
        (axes[0], confirmed, f"Immediate confirmed BSPs: {len(confirmed):,}"),
        (axes[1], virtual, f"Possible virtual-Bi BSPs: {len(virtual):,}"),
    ]
    for ax, events, title in panels:
        ax.plot(
            price_plot["timestamp"],
            price_plot["close"],
            color="#62748A",
            linewidth=0.65,
            alpha=0.80,
            label=f"{comparison_cfg.chan.code.upper()} close",
            zorder=1,
        )
        if not events.empty:
            for direction in ("buy", "sell"):
                group = events.loc[
                    events["direction"].astype(str).str.lower() == direction
                ]
                if group.empty:
                    continue
                ax.scatter(
                    group["bsp_timestamp"],
                    group["klu_close"],
                    s=25,
                    color=colors[direction],
                    marker=markers[direction],
                    alpha=0.80,
                    linewidths=0.35,
                    label=f"{direction.title()} BSP",
                    zorder=3,
                )
        ax.set_title(title)
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.18)
        ax.legend(loc="best", ncol=3)

    axes[1].set_xlabel("Historical endpoint timestamp")
    fig.suptitle(
        f"{comparison_cfg.chan.code.upper()} immediate confirmed vs virtual BSPs: "
        f"{period_start} to {period_end}",
        fontsize=14,
    )
    fig.autofmt_xdate()
    fig.tight_layout()

    saved_path: Path | None = None
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=180, bbox_inches="tight")
        saved_path = target.resolve()
    if show:
        plt.show()
    else:
        plt.close(fig)
    return saved_path


def plot_delayed_bspoints_from_files(
    config: RealisticSimulationConfig | None = None,
    path: str | Path | None = None,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    include_not_delayed: bool = False,
    max_price_gap_pct: float | None = None,
    selection: Literal["all", "largest_time", "largest_price_gap"] = "all",
    top_n: int = 50,
    annotate_largest: int = 0,
) -> Path:
    """Plot delayed BSPs from a saved workbook without running a backtest."""

    cfg = config or RealisticSimulationConfig()
    if cfg.replay_mode != "saved":
        raise ValueError(
            "plot_delayed_bspoints_from_files requires replay_mode='saved' "
            "because it reads BSPs from the configured workbook"
        )
    price_df, bsp_df, chan_feed_df = prepare_simulation_data(cfg)
    plotting_output = RealisticSimulationOutput(
        result=None,  # type: ignore[arg-type]
        price_df=price_df,
        bsp_df=bsp_df,
        chan_feed_df=chan_feed_df,
        config=cfg,
    )
    return plot_delayed_bspoints(
        plotting_output,
        path=path,
        start=start,
        end=end,
        include_not_delayed=include_not_delayed,
        max_price_gap_pct=max_price_gap_pct,
        selection=selection,
        top_n=top_n,
        annotate_largest=annotate_largest,
    )


def plot_simulation_result(
    output: RealisticSimulationOutput,
    path: str | Path | None = None,
    *,
    title: str | None = None,
    show: bool = True,
    max_background_points: int = 75_000,
) -> Path | None:
    """Plot executed trades, portfolio equity, and drawdown for a simulation."""

    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    if output.result is None:
        raise ValueError("output.result is missing; run the simulation before plotting")
    if int(max_background_points) <= 0:
        raise ValueError("max_background_points must be greater than zero")

    required_price = {"timestamp", "close"}
    missing_price = required_price.difference(output.price_df.columns)
    if missing_price:
        raise KeyError(f"price_df is missing columns: {sorted(missing_price)}")
    required_equity = {"timestamp", "equity"}
    missing_equity = required_equity.difference(output.result.equity.columns)
    if missing_equity:
        raise KeyError(f"equity is missing columns: {sorted(missing_equity)}")

    price = output.price_df[["timestamp", "close"]].copy().reset_index(drop=True)
    price["timestamp"] = pd.to_datetime(price["timestamp"], errors="raise")
    price_stride = max(1, len(price) // int(max_background_points))
    price_plot = price.iloc[::price_stride]

    fills = output.result.trades.copy()
    if not fills.empty:
        required_fills = {"side", "exec_idx", "px"}
        missing_fills = required_fills.difference(fills.columns)
        if missing_fills:
            raise KeyError(f"fills are missing columns: {sorted(missing_fills)}")
        exec_idx = pd.to_numeric(fills["exec_idx"], errors="raise").astype("int64")
        invalid = (exec_idx < 0) | (exec_idx >= len(price))
        if invalid.any():
            raise ValueError(
                f"{int(invalid.sum())} fill execution indexes are outside price_df"
            )
        fills["execution_timestamp"] = price.iloc[exec_idx.to_numpy()][
            "timestamp"
        ].to_numpy()
        buys = fills.loc[fills["side"].astype(str).str.lower() == "buy"]
        sells = fills.loc[fills["side"].astype(str).str.lower() == "sell"]
    else:
        buys = fills
        sells = fills

    equity = output.result.equity[["timestamp", "equity"]].copy()
    equity["timestamp"] = pd.to_datetime(equity["timestamp"], errors="raise")
    equity["running_max"] = equity["equity"].cummax()
    equity["drawdown"] = equity["equity"] / equity["running_max"] - 1.0
    equity_stride = max(1, len(equity) // int(max_background_points))
    equity_plot = equity.iloc[::equity_stride]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(18, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.4, 0.8]},
    )
    price_ax, equity_ax, drawdown_ax = axes

    price_ax.plot(
        price_plot["timestamp"],
        price_plot["close"],
        color="#477DA8",
        linewidth=0.7,
        label=f"{output.config.chan.code.upper()} close",
    )
    if not buys.empty:
        price_ax.scatter(
            buys["execution_timestamp"],
            buys["px"],
            marker="^",
            s=28,
            color="#16803C",
            label="Executed buy",
            zorder=3,
        )
    if not sells.empty:
        price_ax.scatter(
            sells["execution_timestamp"],
            sells["px"],
            marker="v",
            s=28,
            color="#C33C3C",
            label="Executed sell",
            zorder=3,
        )

    if title is None:
        suffix = (
            " — Delayed BSPs Removed"
            if output.config.ignore_delayed_bspoints
            else ""
        )
        title = f"{output.config.chan.code.upper()} Simulation Result{suffix}"
    price_ax.set_title(title)
    price_ax.set_ylabel("Price")
    price_ax.grid(alpha=0.20)
    price_ax.legend(loc="best")

    equity_ax.plot(
        equity_plot["timestamp"],
        equity_plot["equity"],
        color="#166534",
        linewidth=1.1,
    )
    equity_ax.set_ylabel("Portfolio equity")
    equity_ax.grid(alpha=0.20)

    drawdown_ax.fill_between(
        equity_plot["timestamp"],
        equity_plot["drawdown"],
        0,
        color="#B42318",
        alpha=0.45,
    )
    drawdown_ax.set_ylabel("Drawdown")
    drawdown_ax.set_xlabel("Date")
    drawdown_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    drawdown_ax.grid(alpha=0.20)

    fig.tight_layout()
    saved_path: Path | None = None
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=180, bbox_inches="tight")
        saved_path = target.resolve()
    if show:
        plt.show()
    else:
        plt.close(fig)
    return saved_path


__all__ = [
    "ChanSimulationConfig",
    "RealisticSimulationConfig",
    "RealisticSimulationOutput",
    "SimulationTradeAnalysis",
    "analyze_simulation_trades",
    "attach_trigger_timestamps",
    "default_strategy_config",
    "display_simulation_analysis",
    "generate_trigger_timed_bsp",
    "generate_virtual_bsp_candidates",
    "save_virtual_bsp_candidates_to_excel",
    "plot_virtual_bsp_candidates_from_excel",
    "plot_virtual_bspoints_from_excel",
    "plot_bsp_type_examples_from_excel",
    "interactive_bsp_type_explorer_from_excel",
    "plot_mature_bspoints_from_excel",
    "plot_mature_bsp_quality_from_excel",
    "load_virtual_bsp_backtest_data",
    "default_virtual_bsp_strategy_config",
    "run_virtual_bsp_simulation_from_excel",
    "prepare_simulation_data",
    "print_simulation_summary",
    "plot_delayed_bspoints",
    "plot_delayed_bspoints_from_files",
    "plot_bsp_delay_comparison",
    "plot_bsp_delay_comparison_from_files",
    "plot_immediate_vs_delayed_bspoints_from_files",
    "plot_immediate_vs_virtual_bspoints_from_files",
    "plot_simulation_result",
    "run_realistic_simulation",
    "save_simulation_to_excel",
]
