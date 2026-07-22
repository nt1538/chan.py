from __future__ import annotations

from dataclasses import asdict
from itertools import product
from typing import Iterable, Mapping, Any

import pandas as pd

from CustomBuySellPoint.SegBspStrategy import SegBspStrategy, SegBspStrategyConfig
from ModelStrategy.BacktestChanConfig import BacktestChanConfig
from ModelStrategy.backtest import run_bsp_backtest
from Trade.RiskManager import RiskConfig, RiskManager


def grid_search_seg_bsp(
    price_df: pd.DataFrame,
    bsp_df: pd.DataFrame,
    parameter_grid: Mapping[str, Iterable[Any]],
    backtest_config: BacktestChanConfig | None = None,
    risk_config: RiskConfig | None = None,
    score: str = "sharpe",
) -> pd.DataFrame:
    """Exhaustive in-sample search; callers should validate winners out of sample."""
    names = list(parameter_grid)
    rows = []
    for values in product(*(parameter_grid[name] for name in names)):
        overrides = dict(zip(names, values))
        for key in ("entry_segment_directions", "entry_bsp_types", "sell_bsp_types"):
            if key in overrides:
                overrides[key] = frozenset(overrides[key])
        strategy_config = SegBspStrategyConfig(**overrides)
        result = run_bsp_backtest(
            price_df, bsp_df, SegBspStrategy(strategy_config), backtest_config,
            RiskManager(risk_config) if risk_config else None,
        )
        rows.append({**asdict(strategy_config), **result.metrics})
    result_df = pd.DataFrame(rows)
    return result_df.sort_values(score, ascending=False, na_position="last").reset_index(drop=True) if not result_df.empty else result_df
