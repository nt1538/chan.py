from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pandas as pd

from CustomBuySellPoint.Signal import ChanSignal, SignalTracker
from CustomBuySellPoint.Strategy import BaseStrategy, StrategyContext
from Trade.RiskManager import RiskManager
from Trade.TradeEngine import ExecutionEngine
from .BacktestChanConfig import BacktestChanConfig
from .parameterEvaluate.eval_strategy import evaluate_backtest


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    metrics: dict[str, Any]
    engine: ExecutionEngine


def _normalize_price(price_df: pd.DataFrame, cfg: BacktestChanConfig) -> pd.DataFrame:
    required = {cfg.timestamp_col, cfg.open_col, cfg.close_col}
    missing = required.difference(price_df.columns)
    if missing:
        raise KeyError(f"price_df missing columns: {sorted(missing)}")
    data = price_df.copy()
    data[cfg.timestamp_col] = pd.to_datetime(data[cfg.timestamp_col])
    return data.sort_values(cfg.timestamp_col).drop_duplicates(cfg.timestamp_col, keep="last").reset_index(drop=True)


def run_bsp_backtest(
    price_df: pd.DataFrame,
    bsp_df: pd.DataFrame,
    strategy: BaseStrategy,
    config: BacktestChanConfig | None = None,
    risk_manager: RiskManager | None = None,
) -> BacktestResult:
    """Replay first-seen BSP snapshots without using future bars."""
    cfg = config or BacktestChanConfig()
    price = _normalize_price(price_df, cfg)
    bsp = bsp_df.copy()
    if "timestamp" not in bsp.columns:
        raise KeyError("bsp_df missing column: timestamp")
    bsp["timestamp"] = pd.to_datetime(bsp["timestamp"])
    by_time = {ts: group.to_dict("records") for ts, group in bsp.sort_values("timestamp").groupby("timestamp", sort=False)}

    engine = ExecutionEngine(cfg.initial_capital, cfg.fee_pct, cfg.slippage_pct)
    tracker = SignalTracker()
    equity_rows: list[dict[str, Any]] = []
    strategy.reset()

    for idx, row in price.iterrows():
        timestamp = row[cfg.timestamp_col]
        fill = engine.maybe_execute_pending(idx)
        close = float(row[cfg.close_col])
        engine.update_market_price(close)
        context = StrategyContext(timestamp, idx, row.to_dict(), engine.position_view(), tracker)
        if fill is not None:
            strategy.on_fill(fill, context)

        queued = False
        if risk_manager is not None and idx + 1 < len(price):
            risk_intent = risk_manager.evaluate(engine.position_view(), close, idx)
            if risk_intent is not None:
                queued = engine.place_order_for_next_bar(
                    risk_intent.side, idx, idx + 1, float(price.iloc[idx + 1][cfg.open_col]),
                    risk_intent.reason, risk_intent.position_fraction, overwrite=False,
                )

        for record in by_time.get(timestamp, ()):
            signal = replace(ChanSignal.from_bsp_row(record), bar_index=idx)
            if not tracker.add(signal):
                continue
            context = StrategyContext(timestamp, idx, row.to_dict(), engine.position_view(), tracker)
            for intent in strategy.on_bsp(signal, context):
                if idx + 1 >= len(price) or queued:
                    break
                queued = engine.place_order_for_next_bar(
                    intent.side, idx, idx + 1, float(price.iloc[idx + 1][cfg.open_col]),
                    intent.reason, intent.position_fraction, overwrite=False,
                )
                if queued:
                    break

        for intent in strategy.on_bar(StrategyContext(timestamp, idx, row.to_dict(), engine.position_view(), tracker)):
            if idx + 1 < len(price) and not queued:
                queued = engine.place_order_for_next_bar(
                    intent.side, idx, idx + 1, float(price.iloc[idx + 1][cfg.open_col]),
                    intent.reason, intent.position_fraction, overwrite=False,
                )
        equity_rows.append({"timestamp": timestamp, "bar_index": idx, "equity": engine.mark_to_market(close), "position": engine.pos})

    if cfg.close_at_end and engine.pos == 1 and len(price):
        idx = len(price) - 1
        engine.place_order_for_next_bar("sell", idx, idx, float(price.iloc[idx][cfg.close_col]), "end_of_backtest")
        engine.maybe_execute_pending(idx)
        equity_rows[-1]["equity"] = engine.cash
        equity_rows[-1]["position"] = 0

    trades = pd.DataFrame(engine.trades)
    equity = pd.DataFrame(equity_rows)
    metrics = evaluate_backtest(equity, trades, cfg.initial_capital, cfg.bars_per_year)
    return BacktestResult(trades=trades, equity=equity, metrics=metrics, engine=engine)
