"""Forward-looking training labels; never include these columns as features."""

from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_COLUMNS = ["target_max_gain_2d", "target_max_loss_2d"]
TARGET_METADATA_COLUMNS = ["target_horizon_end_timestamp"]
EXACT_RETURN_TARGET = "target_exact_return"
DIRECTION_TARGET = "target_up"


def target_columns(target_mode: str) -> list[str]:
    mode = str(target_mode).strip().lower()
    if mode == "extremes":
        return list(TARGET_COLUMNS)
    if mode == "exact_return":
        return [EXACT_RETURN_TARGET]
    if mode == "up_direction":
        return [DIRECTION_TARGET]
    raise ValueError("target_mode must be 'extremes', 'exact_return', or 'up_direction'")


def add_two_day_extreme_labels(frame: pd.DataFrame, horizon_bars: int) -> pd.DataFrame:
    """Label extrema over bars t+1..t+horizon, relative to close at t.

    Both targets are signed returns.  Consequently max gain can be negative
    when every future high is below the current close, and max loss can be
    positive when every future low is above it.  The final horizon rows are
    unlabeled rather than silently receiving a shorter look-ahead.
    """
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be positive")
    df = frame.copy()
    future_high = df["high"].shift(-1).iloc[::-1].rolling(horizon_bars, min_periods=horizon_bars).max().iloc[::-1]
    future_low = df["low"].shift(-1).iloc[::-1].rolling(horizon_bars, min_periods=horizon_bars).min().iloc[::-1]
    df[TARGET_COLUMNS[0]] = future_high / df["close"] - 1.0
    df[TARGET_COLUMNS[1]] = future_low / df["close"] - 1.0
    return df


def add_trading_day_extreme_labels(frame: pd.DataFrame, horizon_days: int = 2) -> pd.DataFrame:
    """Label extrema from t+1 through the same time N trading sessions later.

    For example, with ``horizon_days=2``, a Monday 10:00 row uses all bars after
    that row through Wednesday 10:00 inclusive. Weekends and holidays are
    handled by advancing over distinct trading dates present in the data. If
    the final session has no bar at the exact clock time, its latest bar not
    later than that time is used. Rows lacking the Nth future session remain
    unlabeled.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be positive")
    if "timestamp" not in frame:
        raise ValueError("timestamp is required for trading-day labels")
    df = frame.copy()
    timestamps = pd.to_datetime(df["timestamp"])
    if not timestamps.is_monotonic_increasing:
        raise ValueError("timestamp must be sorted ascending")
    sessions = timestamps.dt.normalize()
    unique_days = pd.Index(sessions.drop_duplicates())
    day_position = pd.Series(np.arange(len(unique_days)), index=unique_days)
    positions = sessions.map(day_position).to_numpy(int)
    timestamp_values = timestamps.to_numpy(dtype="datetime64[ns]")
    time_offsets = timestamp_values - sessions.to_numpy(dtype="datetime64[ns]")

    target_positions = positions + int(horizon_days)
    valid = target_positions < len(unique_days)
    end_indices = np.full(len(df), -1, dtype=np.int64)
    valid_rows = np.flatnonzero(valid)
    if len(valid_rows):
        target_days = unique_days.to_numpy(dtype="datetime64[ns]")[target_positions[valid]]
        target_times = target_days + time_offsets[valid]
        end_indices[valid] = np.searchsorted(
            timestamp_values, target_times, side="right"
        ) - 1
        # Do not accept an endpoint that fell back into the prior session.
        endpoint_sessions = sessions.iloc[end_indices[valid]].to_numpy(dtype="datetime64[ns]")
        valid[valid_rows] &= endpoint_sessions == target_days

    highs = pd.to_numeric(df["high"], errors="coerce").to_numpy(float)
    lows = pd.to_numeric(df["low"], errors="coerce").to_numpy(float)
    future_high = np.full(len(df), np.nan)
    future_low = np.full(len(df), np.nan)
    horizon_end = np.full(len(df), np.datetime64("NaT"), dtype="datetime64[ns]")

    # Endpoints increase monotonically, so two deques provide O(n) range extrema.
    from collections import deque
    max_queue: deque[int] = deque()
    min_queue: deque[int] = deque()
    right = 0
    for left in range(len(df)):
        if not valid[left]:
            continue
        end = int(end_indices[left])
        while right <= end:
            while max_queue and highs[max_queue[-1]] <= highs[right]:
                max_queue.pop()
            while min_queue and lows[min_queue[-1]] >= lows[right]:
                min_queue.pop()
            max_queue.append(right)
            min_queue.append(right)
            right += 1
        while max_queue and max_queue[0] <= left:
            max_queue.popleft()
        while min_queue and min_queue[0] <= left:
            min_queue.popleft()
        if max_queue and min_queue and end > left:
            future_high[left] = highs[max_queue[0]]
            future_low[left] = lows[min_queue[0]]
            horizon_end[left] = timestamp_values[end]

    df[TARGET_COLUMNS[0]] = future_high / df["close"].to_numpy(float) - 1.0
    df[TARGET_COLUMNS[1]] = future_low / df["close"].to_numpy(float) - 1.0
    df[TARGET_METADATA_COLUMNS[0]] = horizon_end
    return df


def add_same_time_return_label(frame: pd.DataFrame, horizon_days: int = 1) -> pd.DataFrame:
    """Label close return at the exact same clock time on a future session."""
    if horizon_days < 1:
        raise ValueError("horizon_days must be positive")
    if "timestamp" not in frame:
        raise ValueError("timestamp is required for same-time return labels")
    df = frame.copy()
    timestamps = pd.to_datetime(df["timestamp"])
    if not timestamps.is_monotonic_increasing:
        raise ValueError("timestamp must be sorted ascending")
    sessions = timestamps.dt.normalize()
    unique_days = pd.Index(sessions.drop_duplicates())
    positions = sessions.map(pd.Series(np.arange(len(unique_days)), index=unique_days)).to_numpy(int)
    timestamp_values = timestamps.to_numpy(dtype="datetime64[ns]")
    offsets = timestamp_values - sessions.to_numpy(dtype="datetime64[ns]")
    target_positions = positions + int(horizon_days)
    valid = target_positions < len(unique_days)
    endpoint = np.full(len(df), -1, dtype=np.int64)
    valid_rows = np.flatnonzero(valid)
    if len(valid_rows):
        target_days = unique_days.to_numpy(dtype="datetime64[ns]")[target_positions[valid]]
        target_times = target_days + offsets[valid]
        candidates = np.searchsorted(timestamp_values, target_times, side="left")
        matched = candidates < len(df)
        matched[matched] &= timestamp_values[candidates[matched]] == target_times[matched]
        valid[valid_rows] &= matched
        endpoint[valid_rows[matched]] = candidates[matched]
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(float)
    future_close = np.full(len(df), np.nan)
    horizon_end = np.full(len(df), np.datetime64("NaT"), dtype="datetime64[ns]")
    rows = np.flatnonzero(valid)
    if len(rows):
        future_close[rows] = close[endpoint[rows]]
        horizon_end[rows] = timestamp_values[endpoint[rows]]
    df[EXACT_RETURN_TARGET] = future_close / close - 1.0
    df[TARGET_METADATA_COLUMNS[0]] = horizon_end
    return df


def add_same_time_direction_label(frame: pd.DataFrame, horizon_days: int = 1) -> pd.DataFrame:
    """Label whether the exact same-time future-session close return is positive."""
    df = add_same_time_return_label(frame, horizon_days)
    valid = df[EXACT_RETURN_TARGET].notna()
    df[DIRECTION_TARGET] = np.nan
    df.loc[valid, DIRECTION_TARGET] = (
        df.loc[valid, EXACT_RETURN_TARGET] > 0.0
    ).astype(float)
    return df
