import unittest

import pandas as pd

from CustomBuySellPoint.SegBspStrategy import SegBspStrategy, SegBspStrategyConfig
from ModelStrategy.BacktestChanConfig import BacktestChanConfig
from ModelStrategy.backtest import run_bsp_backtest
from Trade.RiskManager import RiskConfig, RiskManager


class RuleStrategyBacktestTest(unittest.TestCase):
    def setUp(self):
        ts = pd.date_range("2025-01-02 09:30", periods=8, freq="5min")
        self.price = pd.DataFrame({"timestamp": ts, "open": [100, 101, 102, 103, 104, 105, 106, 107], "close": [100, 101, 102, 103, 104, 105, 106, 107]})
        self.ts = ts

    def test_buy_and_three_sell_signal_bars_use_next_open(self):
        bsp = pd.DataFrame([
            {"timestamp": self.ts[0], "klu_close": 100, "direction": "buy", "bsp_type": "1", "segment_direction": "up"},
            {"timestamp": self.ts[2], "klu_close": 102, "direction": "sell", "bsp_type": "1", "segment_direction": "up"},
            {"timestamp": self.ts[3], "klu_close": 103, "direction": "sell", "bsp_type": "2", "segment_direction": "up"},
            {"timestamp": self.ts[4], "klu_close": 104, "direction": "sell", "bsp_type": "3a", "segment_direction": "up"},
        ])
        result = run_bsp_backtest(self.price, bsp, SegBspStrategy(), BacktestChanConfig(fee_pct=0))
        self.assertEqual(result.trades["side"].tolist(), ["buy", "sell"])
        self.assertEqual(result.trades["px"].tolist(), [101.0, 105.0])
        self.assertEqual(result.metrics["closed_trades"], 1)

    def test_down_segment_exits_on_next_bar(self):
        bsp = pd.DataFrame([
            {"timestamp": self.ts[0], "price": 100, "direction": "buy", "bsp_type": "1", "segment_direction": "up"},
            {"timestamp": self.ts[2], "price": 102, "direction": "sell", "bsp_type": "1", "segment_direction": "down"},
        ])
        result = run_bsp_backtest(self.price, bsp, SegBspStrategy(), BacktestChanConfig(fee_pct=0))
        self.assertEqual(result.trades.iloc[1]["px"], 103.0)
        self.assertEqual(result.trades.iloc[1]["reason"], "segment direction down")

    def test_stop_loss(self):
        price = self.price.copy()
        price.loc[2:, ["open", "close"]] = [95, 95]
        bsp = pd.DataFrame([{"timestamp": self.ts[0], "price": 100, "direction": "buy", "bsp_type": "1", "segment_direction": "up"}])
        result = run_bsp_backtest(price, bsp, SegBspStrategy(), BacktestChanConfig(fee_pct=0), RiskManager(RiskConfig(stop_loss_pct=0.03)))
        self.assertEqual(result.trades.iloc[1]["reason"], "stop_loss")


if __name__ == "__main__":
    unittest.main()
