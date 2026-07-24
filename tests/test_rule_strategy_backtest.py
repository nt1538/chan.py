import unittest

import pandas as pd

from CustomBuySellPoint.SegBspStrategy import SegBspStrategy, SegBspStrategyConfig
from ModelStrategy.BacktestChanConfig import BacktestChanConfig
from ModelStrategy.backtest import run_bsp_backtest
from Trade.RiskManager import RiskConfig, RiskManager
from CustomBuySellPoint import TypeAwareSegBspStrategy


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

    def test_type_aware_exit_uses_entry_type_window(self):
        ts = pd.date_range("2025-01-02 09:30", periods=10, freq="5min")
        price = pd.DataFrame({"timestamp": ts, "open": range(100, 110), "close": range(100, 110)})
        sell_rows = [
            {"timestamp": ts[2], "price": 102, "direction": "sell", "bsp_type": "2", "segment_direction": "up"},
            {"timestamp": ts[7], "price": 107, "direction": "sell", "bsp_type": "2", "segment_direction": "up"},
        ]

        type_1_bsp = pd.DataFrame([
            {"timestamp": ts[0], "price": 100, "direction": "buy", "bsp_type": "1", "segment_direction": "up"},
            *sell_rows,
        ])
        type_1 = run_bsp_backtest(price, type_1_bsp, TypeAwareSegBspStrategy(), BacktestChanConfig(fee_pct=0))
        self.assertEqual(type_1.trades.iloc[-1]["reason"], "end_of_backtest")
        self.assertAlmostEqual(type_1.trades.iloc[0]["qty"], 100000 / 101)

        type_2_bsp = pd.DataFrame([
            {"timestamp": ts[0], "price": 100, "direction": "buy", "bsp_type": "2", "segment_direction": "up"},
            *sell_rows,
        ])
        type_2 = run_bsp_backtest(price, type_2_bsp, TypeAwareSegBspStrategy(), BacktestChanConfig(fee_pct=0))
        self.assertIn("in 8 bars", type_2.trades.iloc[-1]["reason"])
        self.assertEqual(type_2.trades.iloc[-1]["px"], 108.0)

    def test_required_buy_signals_uses_distinct_bars(self):
        bsp = pd.DataFrame([
            {"timestamp": self.ts[0], "price": 100, "direction": "buy", "bsp_type": "1", "segment_direction": "up"},
            {"timestamp": self.ts[1], "price": 101, "direction": "buy", "bsp_type": "2", "segment_direction": "up"},
        ])
        strategy = SegBspStrategy(SegBspStrategyConfig(required_buy_signals=2, buy_lookback_bars=3))
        result = run_bsp_backtest(self.price, bsp, strategy, BacktestChanConfig(fee_pct=0))
        self.assertEqual(result.trades.iloc[0]["side"], "buy")
        self.assertEqual(result.trades.iloc[0]["px"], 102.0)
        self.assertIn("2 buy-signal bars", result.trades.iloc[0]["reason"])

    def test_impossible_signal_window_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "distinct bars"):
            SegBspStrategyConfig(required_sell_signals=15, sell_lookback_bars=5)

    def test_per_type_buy_and_sell_requirements_are_independent(self):
        ts = pd.date_range("2025-01-02 09:30", periods=10, freq="5min")
        price = pd.DataFrame({"timestamp": ts, "open": range(100, 110), "close": range(100, 110)})
        bsp = pd.DataFrame([
            {"timestamp": ts[0], "price": 100, "direction": "buy", "bsp_type": "1", "segment_direction": "up"},
            {"timestamp": ts[1], "price": 101, "direction": "buy", "bsp_type": "2", "segment_direction": "up"},
            {"timestamp": ts[2], "price": 102, "direction": "buy", "bsp_type": "1", "segment_direction": "up"},
            {"timestamp": ts[4], "price": 104, "direction": "sell", "bsp_type": "2", "segment_direction": "up"},
            {"timestamp": ts[5], "price": 105, "direction": "sell", "bsp_type": "1", "segment_direction": "up"},
            {"timestamp": ts[6], "price": 106, "direction": "sell", "bsp_type": "2", "segment_direction": "up"},
        ])
        strategy = SegBspStrategy(SegBspStrategyConfig(
            required_buy_signals_by_type={"1": 2, "2": 3},
            required_sell_signals_by_type={"1": 2, "2": 2},
            buy_lookback_bars=5,
            sell_lookback_bars=5,
            exit_on_down_segment=False,
        ))
        result = run_bsp_backtest(price, bsp, strategy, BacktestChanConfig(fee_pct=0))
        self.assertEqual(result.trades.iloc[0]["px"], 103.0)
        self.assertIn("2/2 type-1 buy-signal bars", result.trades.iloc[0]["reason"])
        self.assertEqual(result.trades.iloc[1]["px"], 107.0)
        self.assertIn("2/2 type-2 sell-signal bars", result.trades.iloc[1]["reason"])

    def test_down_segment_entry_uses_its_own_exit_policy(self):
        ts = pd.date_range("2025-01-02 09:30", periods=9, freq="5min")
        price = pd.DataFrame({"timestamp": ts, "open": range(100, 109), "close": range(100, 109)})
        bsp = pd.DataFrame([
            {"timestamp": ts[0], "price": 100, "direction": "buy", "bsp_type": "1", "segment_direction": "down"},
            {"timestamp": ts[1], "price": 101, "direction": "buy", "bsp_type": "1", "segment_direction": "down"},
            {"timestamp": ts[3], "price": 103, "direction": "sell", "bsp_type": "2", "segment_direction": "down"},
            {"timestamp": ts[5], "price": 105, "direction": "sell", "bsp_type": "2", "segment_direction": "down"},
        ])
        strategy = SegBspStrategy(SegBspStrategyConfig(
            entry_segment_directions=frozenset({"up", "down"}),
            required_buy_signals_by_segment={"down": {"1": 2}, "up": {"1": 1}},
            buy_lookback_bars_by_segment={"down": 5, "up": 3},
            required_sell_signals_by_entry_segment={"down": {"2": 2}, "up": {"2": 3}},
            sell_lookback_bars_by_entry_segment={"down": 6, "up": 4},
            exit_segment_directions_by_entry_segment={"up": frozenset({"down"}), "down": frozenset()},
        ))
        result = run_bsp_backtest(price, bsp, strategy, BacktestChanConfig(fee_pct=0))
        self.assertEqual(result.trades.iloc[0]["px"], 102.0)
        self.assertIn("down segment", result.trades.iloc[0]["reason"])
        self.assertEqual(result.trades.iloc[1]["px"], 106.0)
        self.assertIn("current down segment", result.trades.iloc[1]["reason"])

    def test_exit_policy_uses_current_segment_not_entry_segment(self):
        ts = pd.date_range("2025-01-02 09:30", periods=9, freq="5min")
        price = pd.DataFrame({"timestamp": ts, "open": range(100, 109), "close": range(100, 109)})
        bsp = pd.DataFrame([
            {"timestamp": ts[0], "price": 100, "direction": "buy", "bsp_type": "1", "segment_direction": "up"},
            {"timestamp": ts[2], "price": 102, "direction": "sell", "bsp_type": "1", "segment_direction": "up"},
            {"timestamp": ts[4], "price": 104, "direction": "sell", "bsp_type": "1", "segment_direction": "down"},
            {"timestamp": ts[5], "price": 105, "direction": "sell", "bsp_type": "1", "segment_direction": "down"},
        ])
        strategy = SegBspStrategy(SegBspStrategyConfig(
            required_sell_signals_by_segment={"up": {"1": 5}, "down": {"1": 2}},
            sell_lookback_bars_by_segment={"up": 8, "down": 4},
            exit_segment_directions_by_entry_segment={"up": frozenset()},
        ))
        result = run_bsp_backtest(price, bsp, strategy, BacktestChanConfig(fee_pct=0))
        self.assertEqual(result.trades.iloc[1]["px"], 106.0)
        self.assertIn("current down segment", result.trades.iloc[1]["reason"])


if __name__ == "__main__":
    unittest.main()
