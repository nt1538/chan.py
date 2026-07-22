import unittest

import numpy as np
import pandas as pd

import pipelineCurrent
from Bandit import LinUCBBandit
from DataPipeline import load_5m_index
from FeaturePipeline import compute_daily_kline_features
from Pipeline import run_daily_bandit_then_5m_xgb


class PipelineModuleTest(unittest.TestCase):
    def test_legacy_runner_points_to_modular_runner(self):
        self.assertIs(pipelineCurrent.run_daily_bandit_then_5m_xgb, run_daily_bandit_then_5m_xgb)

    def test_ohlcv_features_and_index(self):
        times = pd.date_range("2025-01-01", periods=45, freq="D")
        normalized = pd.DataFrame({"timestamp": times, "_open": np.arange(45) + 100, "_high": np.arange(45) + 102,
                                   "_low": np.arange(45) + 99, "_close": np.arange(45) + 101, "_vol": 1000})
        featured = compute_daily_kline_features(normalized)
        self.assertTrue({"ret1", "atr_14", "slope40"}.issubset(featured.columns))
        indexed = load_5m_index(normalized, str(times[0]), str(times[-1]))
        self.assertEqual(len(indexed[0]), 45)

    def test_bandit_state_round_trip(self):
        bandit = LinUCBBandit(3, 2)
        bandit.update(1, np.array([1.0, 0.5]), 0.2)
        restored = LinUCBBandit.from_state_dict(bandit.state_dict())
        np.testing.assert_allclose(restored.A[1], bandit.A[1])
        np.testing.assert_allclose(restored.b[1], bandit.b[1])


if __name__ == "__main__":
    unittest.main()
