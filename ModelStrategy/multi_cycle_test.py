from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd

from CustomBuySellPoint.Strategy import BaseStrategy
from Trade.RiskManager import RiskManager
from .BacktestChanConfig import BacktestChanConfig
from .backtest import run_bsp_backtest


def run_period_tests(periods: Iterable[Tuple[str, pd.DataFrame, pd.DataFrame]], strategy_factory, config: BacktestChanConfig | None = None, risk_manager: RiskManager | None = None) -> pd.DataFrame:
    rows = []
    for name, price_df, bsp_df in periods:
        strategy: BaseStrategy = strategy_factory()
        result = run_bsp_backtest(price_df, bsp_df, strategy, config, risk_manager)
        rows.append({"period": name, **result.metrics})
    return pd.DataFrame(rows)
