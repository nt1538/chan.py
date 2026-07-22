from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def evaluate_backtest(equity: pd.DataFrame, trades: pd.DataFrame, initial_capital: float, bars_per_year: int) -> dict[str, Any]:
    if equity.empty:
        return {"total_return": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "closed_trades": 0, "win_rate": 0.0, "profit_factor": 0.0}
    values = equity["equity"].astype(float)
    returns = values.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    drawdown = values / values.cummax() - 1.0
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe = float(returns.mean() / std * math.sqrt(bars_per_year)) if std > 0 else 0.0
    sells = trades.loc[trades.get("side", pd.Series(dtype=str)).eq("sell")].copy() if not trades.empty else trades
    pnl = pd.to_numeric(sells.get("pnl", pd.Series(dtype=float)), errors="coerce").dropna()
    gains = float(pnl[pnl > 0].sum())
    losses = abs(float(pnl[pnl < 0].sum()))
    return {
        "final_equity": float(values.iloc[-1]),
        "total_return": float(values.iloc[-1] / initial_capital - 1.0),
        "max_drawdown": float(drawdown.min()),
        "sharpe": sharpe,
        "closed_trades": int(len(pnl)),
        "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
        "profit_factor": gains / losses if losses > 0 else (float("inf") if gains > 0 else 0.0),
    }
