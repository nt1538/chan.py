import numpy as np
import pandas as pd

from kline_2day_forecaster.config import ForecastConfig
from kline_2day_forecaster.features import add_technical_features, normalize_ohlcv
from kline_2day_forecaster.labels import (add_same_time_return_label,
                                           add_same_time_direction_label,
                                           add_trading_day_extreme_labels,
                                           add_two_day_extreme_labels)
from kline_2day_forecaster.pipeline import _select_model_rows


def test_labels_exclude_current_bar_and_require_full_horizon():
    df = pd.DataFrame({"close": [10, 10, 10, 10], "high": [100, 11, 12, 13], "low": [1, 9, 8, 7]})
    got = add_two_day_extreme_labels(df, horizon_bars=2)
    assert got.loc[0, "target_max_gain_2d"] == 0.2
    assert got.loc[0, "target_max_loss_2d"] == -0.2
    assert got["target_max_gain_2d"].tail(2).isna().all()


def test_max_gain_is_signed_and_not_clipped_to_zero():
    df = pd.DataFrame({
        "close": [10, 8, 7],
        "high": [10, 9, 8],
        "low": [10, 7, 6],
    })
    got = add_two_day_extreme_labels(df, horizon_bars=2)
    assert np.isclose(got.loc[0, "target_max_gain_2d"], -0.1)


def test_features_are_causal_under_future_mutation():
    n = 220
    raw = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=n, freq="5min"), "Open": np.arange(n) + 10,
                        "High": np.arange(n) + 11, "Low": np.arange(n) + 9, "Close": np.arange(n) + 10.5, "Volume": np.arange(n) + 100})
    first = add_technical_features(normalize_ohlcv(raw))
    raw.loc[200:, "Close"] *= 10
    second = add_technical_features(normalize_ohlcv(raw))
    pd.testing.assert_frame_equal(first.iloc[:200], second.iloc[:200])


def test_trading_day_labels_use_actual_variable_bar_counts():
    timestamps = pd.to_datetime([
        "2024-01-02 09:30", "2024-01-02 15:55",
        "2024-01-03 09:30", "2024-01-03 12:00", "2024-01-03 15:55",
        "2024-01-04 09:30", "2024-01-04 15:55",
    ])
    frame = pd.DataFrame({"timestamp": timestamps, "close": 10.0,
                          "high": [10, 11, 12, 13, 14, 15, 16],
                          "low": [10, 9, 8, 7, 6, 5, 4]})
    got = add_trading_day_extreme_labels(frame, horizon_days=2)
    # Jan 2 09:30 ends at the same time on the second future session, Jan 4.
    assert np.isclose(got.loc[0, "target_max_gain_2d"], 0.5)
    assert np.isclose(got.loc[0, "target_max_loss_2d"], -0.5)
    assert got.loc[0, "target_horizon_end_timestamp"] == timestamps[5]
    # Jan 3 onward lacks a second subsequent trading date.
    assert got.loc[2:, "target_max_gain_2d"].isna().all()


def test_same_time_next_session_return_requires_exact_bar():
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-01-05 09:30", "2024-01-05 09:35",
            "2024-01-08 09:30", "2024-01-08 09:40",
        ]),
        "close": [100.0, 200.0, 110.0, 220.0],
    })
    got = add_same_time_return_label(frame, horizon_days=1)
    assert np.isclose(got.loc[0, "target_exact_return"], 0.10)
    assert pd.isna(got.loc[1, "target_exact_return"])
    assert got.loc[0, "target_horizon_end_timestamp"] == frame.loc[2, "timestamp"]


def test_precomputed_bsp_groups_do_not_require_chan():
    n = 40
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-02 09:30", periods=n, freq="5min"),
        "open": np.arange(n) + 100.0, "high": np.arange(n) + 101.0,
        "low": np.arange(n) + 99.0, "close": np.arange(n) + 100.5,
        "volume": 1_000.0,
        "bsp_types": ["1p", "2s", "3b"] + [""] * (n - 3),
    })
    got = add_technical_features(frame, windows=(2,), rsi_periods=(2,),
                                 atr_periods=(2,), macd_periods=((2, 3, 2),))
    assert got.loc[0, "tech_bsp_is_type_1"] == 1.0
    assert got.loc[1, "tech_bsp_is_type_2"] == 1.0
    assert got.loc[2, "tech_bsp_is_type_3"] == 1.0


def test_same_time_direction_target_preserves_missing_endpoints():
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-01-05 09:30", "2024-01-05 09:35",
            "2024-01-08 09:30", "2024-01-08 09:40",
        ]),
        "close": [100.0, 200.0, 110.0, 180.0],
    })
    got = add_same_time_direction_label(frame, horizon_days=1)
    assert got.loc[0, "target_up"] == 1.0
    assert pd.isna(got.loc[1, "target_up"])


def test_regular_session_sampling_is_anchored_at_open():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-02 04:00", "2024-01-02 19:55", freq="5min")
    })
    config = ForecastConfig(
        "unused.csv", regular_session_only=True,
        regular_session_start="09:30", regular_session_end="15:55",
        sample_every_minutes=60,
    )
    got = _select_model_rows(frame, config)
    assert got["timestamp"].dt.strftime("%H:%M").tolist() == [
        "09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30",
    ]
