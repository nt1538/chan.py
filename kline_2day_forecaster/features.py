"""Causal technical features calculated for every input bar.

All rolling computations use the current and earlier bars only.  Input columns
not recognized as standard OHLCV fields are retained (and numeric columns can
therefore be used by the model), satisfying the "every datapoint" contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


CONFIGURABLE_INDICATOR_ARGUMENTS = (
    "stochastic_period", "stochastic_d_period", "williams_period", "cci_period",
    "cci_constant", "mfi_period", "bollinger_period", "bollinger_std_multiplier",
    "dmi_period", "adx_period", "rsl_period", "ppo_periods",
    "keltner_ema_period", "keltner_atr_period", "keltner_atr_multiplier",
    "starc_sma_period", "starc_atr_period", "starc_atr_multiplier",
    "tsi_periods", "ultimate_periods", "ultimate_weights", "obv_z_period",
    "obv_z_min_periods", "ad_line_z_period", "ad_line_z_min_periods",
    "cmf_period", "doji_body_ratio", "pattern_shadow_body_ratio",
    "first_hour_minutes", "relative_volume_lookback_days",
    "relative_volume_min_days",
)


def technical_feature_kwargs(config) -> dict:
    """Translate persisted ``technical_*`` settings to function arguments."""
    source = config if isinstance(config, dict) else vars(config)
    result = {}
    tuple_arguments = {"ppo_periods", "tsi_periods", "ultimate_periods", "ultimate_weights"}
    for argument in CONFIGURABLE_INDICATOR_ARGUMENTS:
        key = f"technical_{argument}"
        if key in source:
            result[argument] = tuple(source[key]) if argument in tuple_arguments else source[key]
    return result


def technical_warmup_bars(config) -> int:
    """Largest causal history requirement among configured indicators."""
    source = config if isinstance(config, dict) else vars(config)
    get = lambda name, default: source.get(name, default)
    values = [
        *get("technical_windows", (200,)), *get("technical_rsi_periods", (14,)),
        *get("technical_atr_periods", (14,)),
        *(slow + signal for _, slow, signal in get("technical_macd_periods", ((12, 26, 9),))),
        get("technical_stochastic_period", 14) + get("technical_stochastic_d_period", 3),
        get("technical_williams_period", 14), get("technical_cci_period", 20),
        get("technical_mfi_period", 14), get("technical_bollinger_period", 20),
        get("technical_dmi_period", 14) + get("technical_adx_period", 14),
        get("technical_rsl_period", 14), *get("technical_ppo_periods", (12, 26)),
        get("technical_keltner_ema_period", 20), get("technical_keltner_atr_period", 14),
        get("technical_starc_sma_period", 5), get("technical_starc_atr_period", 10),
        sum(get("technical_tsi_periods", (25, 13))), *get("technical_ultimate_periods", (7, 14, 28)),
        get("technical_obv_z_period", 78), get("technical_ad_line_z_period", 78),
        get("technical_cmf_period", 20),
        get("technical_relative_volume_lookback_days", 20),
    ]
    return max(map(int, values))


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize common column spellings while preserving every source column."""
    df = frame.copy()
    lookup = {str(c).lower().replace(" ", "_"): c for c in df.columns}
    ts = next((lookup[k] for k in ("timestamp", "datetime", "date", "time") if k in lookup), df.columns[0])
    df["timestamp"] = pd.to_datetime(df[ts], errors="coerce")
    for name, candidates in {
        "open": ("open", "o"), "high": ("high", "h"), "low": ("low", "l"),
        "close": ("close", "adj_close", "adjclose", "c"), "volume": ("volume", "vol", "v"),
    }.items():
        source = next((lookup[x] for x in candidates if x in lookup), None)
        if source is None and name == "volume":
            df[name] = 0.0
        elif source is None:
            raise ValueError(f"Input CSV is missing required {name!r} column")
        else:
            df[name] = pd.to_numeric(df[source], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.div(b.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def add_technical_features(
    frame: pd.DataFrame,
    *,
    windows: tuple[int, ...] | list[int] | None = None,
    rsi_periods: tuple[int, ...] | list[int] | None = None,
    atr_periods: tuple[int, ...] | list[int] | None = None,
    macd_periods: tuple[tuple[int, int, int], ...] | list[tuple[int, int, int]] | None = None,
    stochastic_period: int = 14,
    stochastic_d_period: int = 3,
    williams_period: int = 14,
    cci_period: int = 20,
    cci_constant: float = 0.015,
    mfi_period: int = 14,
    bollinger_period: int = 20,
    bollinger_std_multiplier: float = 2.0,
    dmi_period: int = 14,
    adx_period: int = 14,
    rsl_period: int = 14,
    ppo_periods: tuple[int, int] = (12, 26),
    keltner_ema_period: int = 20,
    keltner_atr_period: int = 14,
    keltner_atr_multiplier: float = 2.0,
    starc_sma_period: int = 5,
    starc_atr_period: int = 10,
    starc_atr_multiplier: float = 2.0,
    tsi_periods: tuple[int, int] = (25, 13),
    ultimate_periods: tuple[int, int, int] = (7, 14, 28),
    ultimate_weights: tuple[float, float, float] = (4.0, 2.0, 1.0),
    obv_z_period: int = 78,
    obv_z_min_periods: int = 20,
    ad_line_z_period: int = 78,
    ad_line_z_min_periods: int = 20,
    cmf_period: int = 20,
    doji_body_ratio: float = 0.10,
    pattern_shadow_body_ratio: float = 2.0,
    first_hour_minutes: int = 60,
    relative_volume_lookback_days: int = 20,
    relative_volume_min_days: int = 5,
    regular_session_start: str = "09:30",
    regular_session_end: str = "15:55",
) -> pd.DataFrame:
    """Add causal multi-horizon technical features.

    Periods are expressed in bars. Legacy RSI-14, ATR-14 and MACD(12,26,9)
    names are retained, while additional configurations receive explicit
    period suffixes.
    """
    default_windows = (
        2, 3, 5, 10, 12, 20, 26, 39, 50, 78, 100, 156, 200, 234, 390, 780,
    )
    windows = tuple(sorted({int(x) for x in (windows or default_windows)}))
    rsi_periods = tuple(sorted({
        int(x) for x in (rsi_periods or (14, 39, 78, 156, 390))
    }))
    atr_periods = tuple(sorted({
        int(x) for x in (atr_periods or (14, 39, 78, 156, 390))
    }))
    macd_periods = tuple(tuple(map(int, values)) for values in (
        macd_periods or ((12, 26, 9), (39, 78, 20), (78, 156, 39), (156, 390, 78))
    ))
    if any(period < 2 for period in (*windows, *rsi_periods, *atr_periods)):
        raise ValueError("Technical feature periods must be at least 2 bars")
    if any(fast < 2 or slow <= fast or signal < 2 for fast, slow, signal in macd_periods):
        raise ValueError("Each MACD tuple must satisfy 2 <= fast < slow and signal >= 2")
    period_values = (
        stochastic_period, stochastic_d_period, williams_period, cci_period,
        mfi_period, bollinger_period, dmi_period, adx_period, rsl_period,
        *ppo_periods, keltner_ema_period, keltner_atr_period,
        starc_sma_period, starc_atr_period, *tsi_periods, *ultimate_periods,
        obv_z_period, obv_z_min_periods, ad_line_z_period,
        ad_line_z_min_periods, cmf_period,
        first_hour_minutes, relative_volume_lookback_days,
        relative_volume_min_days,
    )
    if any(int(period) < 1 for period in period_values):
        raise ValueError("All technical indicator periods must be positive")
    if ppo_periods[0] >= ppo_periods[1]:
        raise ValueError("technical_ppo_periods must satisfy fast < slow")
    if obv_z_min_periods > obv_z_period or ad_line_z_min_periods > ad_line_z_period:
        raise ValueError("Z-score min_periods cannot exceed its rolling period")
    if relative_volume_min_days > relative_volume_lookback_days:
        raise ValueError("relative_volume_min_days cannot exceed its lookback")
    if any(value <= 0 for value in (
        cci_constant, bollinger_std_multiplier, keltner_atr_multiplier,
        starc_atr_multiplier, *ultimate_weights,
        doji_body_ratio, pattern_shadow_body_ratio,
    )):
        raise ValueError("Technical indicator constants and multipliers must be positive")
    df = frame.copy()
    o, h, l, c, v = (df[x].astype(float) for x in ("open", "high", "low", "close", "volume"))
    prev = c.shift(1)
    ret1 = c.pct_change(fill_method=None)
    tr = pd.concat([(h - l).abs(), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    rng = (h - l).replace(0.0, np.nan)
    df["tech_ret_1"] = ret1
    df["tech_gap"] = _safe_div(o, prev) - 1.0
    df["tech_range_pct"] = _safe_div(h - l, prev)
    df["tech_body_pct"] = _safe_div(c - o, o)
    df["tech_upper_shadow_pct"] = _safe_div(h - pd.concat([o, c], axis=1).max(axis=1), o)
    df["tech_lower_shadow_pct"] = _safe_div(pd.concat([o, c], axis=1).min(axis=1) - l, o)
    df["tech_close_position"] = (c - l).div(rng)
    df["tech_volume_log1p"] = np.log1p(v.clip(lower=0.0))

    # If BSP annotations were precomputed into the input CSV, expose their
    # grouped families without rebuilding a Chan model. A bar may contain
    # multiple comma/pipe-separated types in ``bsp_types``.
    bsp_source = next((name for name in ("bsp_types", "bsp_type") if name in df.columns), None)
    if bsp_source is not None:
        bsp_text = df[bsp_source].fillna("").astype(str).str.lower()
        df["tech_bsp_is_type_1"] = bsp_text.str.contains(r"(?:^|[^0-9a-z])1p?(?:$|[^0-9a-z])", regex=True).astype(float)
        df["tech_bsp_is_type_2"] = bsp_text.str.contains(r"(?:^|[^0-9a-z])2(?:p|s)?(?:$|[^0-9a-z])", regex=True).astype(float)
        df["tech_bsp_is_type_3"] = bsp_text.str.contains(r"(?:^|[^0-9a-z])3(?:p|a|b)?(?:$|[^0-9a-z])", regex=True).astype(float)

    for w in windows:
        df[f"tech_return_{w}"] = c.pct_change(w, fill_method=None)
        df[f"tech_sma_distance_{w}"] = _safe_div(c, c.rolling(w, min_periods=w).mean()) - 1.0
        df[f"tech_ema_distance_{w}"] = _safe_div(c, c.ewm(span=w, adjust=False, min_periods=w).mean()) - 1.0
        df[f"tech_volatility_{w}"] = ret1.rolling(w, min_periods=w).std()
        df[f"tech_volume_z_{w}"] = (v - v.rolling(w, min_periods=w).mean()).div(v.rolling(w, min_periods=w).std().replace(0, np.nan))

    # Consolidate the frame before adding indicator families. This avoids
    # pandas fragmentation warnings with the expanded multi-horizon inventory.
    df = df.copy()

    delta = c.diff()
    for period in atr_periods:
        atr = tr.rolling(period, min_periods=period).mean()
        df[f"tech_atr_{period}_pct"] = _safe_div(atr, c)
    for period in rsi_periods:
        gain = delta.clip(lower=0).ewm(
            alpha=1 / period, adjust=False, min_periods=period
        ).mean()
        loss = (-delta.clip(upper=0)).ewm(
            alpha=1 / period, adjust=False, min_periods=period
        ).mean()
        rs = gain.div(loss.replace(0, np.nan))
        df[f"tech_rsi_{period}"] = 100.0 - 100.0 / (1.0 + rs)
    for fast, slow, signal in macd_periods:
        fast_ema = c.ewm(span=fast, adjust=False, min_periods=fast).mean()
        slow_ema = c.ewm(span=slow, adjust=False, min_periods=slow).mean()
        macd = fast_ema - slow_ema
        macd_signal = macd.ewm(
            span=signal, adjust=False, min_periods=signal
        ).mean()
        suffix = "" if (fast, slow, signal) == (12, 26, 9) else f"_{fast}_{slow}_{signal}"
        df[f"tech_macd_pct{suffix}"] = _safe_div(macd, c)
        df[f"tech_macd_signal_pct{suffix}"] = _safe_div(macd_signal, c)
        df[f"tech_macd_hist_pct{suffix}"] = _safe_div(macd - macd_signal, c)
    stochastic_low = l.rolling(stochastic_period).min()
    stochastic_high = h.rolling(stochastic_period).max()
    df["tech_stochastic_k"] = 100 * (c - stochastic_low).div((stochastic_high - stochastic_low).replace(0, np.nan))
    df["tech_stochastic_d"] = df["tech_stochastic_k"].rolling(stochastic_d_period, min_periods=stochastic_d_period).mean()
    williams_low, williams_high = l.rolling(williams_period).min(), h.rolling(williams_period).max()
    df[f"tech_williams_r_{williams_period}"] = -100 * (williams_high - c).div((williams_high - williams_low).replace(0, np.nan))
    typical = (h + l + c) / 3.0
    typical_mean = typical.rolling(cci_period, min_periods=cci_period).mean()
    mean_deviation = typical.rolling(cci_period, min_periods=cci_period).apply(
        lambda values: np.mean(np.abs(values - values.mean())), raw=True
    )
    df[f"tech_cci_{cci_period}"] = (typical - typical_mean).div((cci_constant * mean_deviation).replace(0, np.nan))
    positive_flow = (typical * v).where(typical.diff() > 0, 0.0)
    negative_flow = (typical * v).where(typical.diff() < 0, 0.0)
    money_ratio = positive_flow.rolling(mfi_period, min_periods=mfi_period).sum().div(
        negative_flow.rolling(mfi_period, min_periods=mfi_period).sum().replace(0, np.nan)
    )
    df[f"tech_mfi_{mfi_period}"] = 100.0 - 100.0 / (1.0 + money_ratio)
    boll_mid = c.rolling(bollinger_period, min_periods=bollinger_period).mean()
    boll_std = c.rolling(bollinger_period, min_periods=bollinger_period).std()
    df[f"tech_bollinger_z_{bollinger_period}"] = (c - boll_mid).div(boll_std.replace(0, np.nan))
    df[f"tech_bollinger_bandwidth_{bollinger_period}"] = _safe_div(2.0 * bollinger_std_multiplier * boll_std, boll_mid)
    up_move, down_move = h.diff(), -l.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    dmi_atr = tr.ewm(alpha=1 / dmi_period, adjust=False, min_periods=dmi_period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / dmi_period, adjust=False, min_periods=dmi_period).mean().div(dmi_atr.replace(0, np.nan))
    minus_di = 100 * minus_dm.ewm(alpha=1 / dmi_period, adjust=False, min_periods=dmi_period).mean().div(dmi_atr.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs().div((plus_di + minus_di).replace(0, np.nan))
    df[f"tech_plus_di_{dmi_period}"], df[f"tech_minus_di_{dmi_period}"] = plus_di, minus_di
    adx_name = "tech_adx_14" if (dmi_period, adx_period) == (14, 14) else f"tech_adx_{dmi_period}_{adx_period}"
    df[adx_name] = dx.ewm(alpha=1 / adx_period, adjust=False, min_periods=adx_period).mean()

    # Standalone versions of the remaining useful indicators exposed by the
    # Chan K-line unit. These depend only on OHLCV and never instantiate Chan.
    keltner_ema = c.ewm(span=keltner_ema_period, adjust=False, min_periods=keltner_ema_period).mean()
    keltner_atr = tr.ewm(alpha=1 / keltner_atr_period, adjust=False, min_periods=keltner_atr_period).mean()
    df[f"tech_rsl_{rsl_period}"] = _safe_div(c, c.rolling(rsl_period, min_periods=rsl_period).mean()) * 100.0
    ppo_fast, ppo_slow = ppo_periods
    ppo_slow_ema = c.ewm(span=ppo_slow, adjust=False, min_periods=ppo_slow).mean()
    df[f"tech_ppo_{ppo_fast}_{ppo_slow}"] = _safe_div(
        c.ewm(span=ppo_fast, adjust=False, min_periods=ppo_fast).mean() - ppo_slow_ema,
        ppo_slow_ema,
    ) * 100.0
    keltner_suffix = "20" if (keltner_ema_period, keltner_atr_period) == (20, 14) else f"{keltner_ema_period}_{keltner_atr_period}"
    df[f"tech_keltner_position_{keltner_suffix}"] = (c - keltner_ema).div((keltner_atr_multiplier * keltner_atr).replace(0, np.nan))
    df[f"tech_keltner_width_{keltner_suffix}_pct"] = _safe_div(2.0 * keltner_atr_multiplier * keltner_atr, keltner_ema)
    starc_atr = tr.ewm(alpha=1 / starc_atr_period, adjust=False, min_periods=starc_atr_period).mean()
    starc_sma = c.rolling(starc_sma_period, min_periods=starc_sma_period).mean()
    df[f"tech_starc_position_{starc_sma_period}_{starc_atr_period}"] = (c - starc_sma).div((starc_atr_multiplier * starc_atr).replace(0, np.nan))
    df[f"tech_starc_width_{starc_sma_period}_{starc_atr_period}_pct"] = _safe_div(2.0 * starc_atr_multiplier * starc_atr, starc_sma)

    momentum = c.diff()
    tsi_long, tsi_short = tsi_periods
    double_smoothed = momentum.ewm(span=tsi_long, adjust=False, min_periods=tsi_long).mean().ewm(
        span=tsi_short, adjust=False, min_periods=tsi_short
    ).mean()
    double_abs = momentum.abs().ewm(span=tsi_long, adjust=False, min_periods=tsi_long).mean().ewm(
        span=tsi_short, adjust=False, min_periods=tsi_short
    ).mean()
    df[f"tech_tsi_{tsi_long}_{tsi_short}"] = 100.0 * double_smoothed.div(double_abs.replace(0, np.nan))

    buying_pressure = c - pd.concat([l, prev], axis=1).min(axis=1)
    true_range = pd.concat([h, prev], axis=1).max(axis=1) - pd.concat([l, prev], axis=1).min(axis=1)
    ultimate_averages = [buying_pressure.rolling(p, min_periods=p).sum().div(true_range.rolling(p, min_periods=p).sum().replace(0, np.nan)) for p in ultimate_periods]
    weight_total = float(sum(ultimate_weights))
    df[f"tech_ultimate_oscillator_{'_'.join(map(str, ultimate_periods))}"] = 100.0 * sum(w * avg for w, avg in zip(ultimate_weights, ultimate_averages)) / weight_total

    money_flow_multiplier = ((c - l) - (h - c)).div(rng)
    money_flow_volume = money_flow_multiplier.fillna(0.0) * v
    df["tech_ad_line"] = money_flow_volume.cumsum()
    ad_mean = df["tech_ad_line"].rolling(ad_line_z_period, min_periods=ad_line_z_min_periods).mean()
    ad_std = df["tech_ad_line"].rolling(ad_line_z_period, min_periods=ad_line_z_min_periods).std().replace(0, np.nan)
    df[f"tech_ad_line_z_{ad_line_z_period}"] = (df["tech_ad_line"] - ad_mean).div(ad_std)
    df[f"tech_cmf_{cmf_period}"] = money_flow_volume.rolling(cmf_period, min_periods=cmf_period).sum().div(
        v.rolling(cmf_period, min_periods=cmf_period).sum().replace(0, np.nan)
    )

    # Compact causal candle-pattern flags corresponding to the most common
    # pattern signals in the Chan indicator inventory.
    body = (c - o).abs()
    upper_shadow = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_shadow = pd.concat([o, c], axis=1).min(axis=1) - l
    df["tech_pattern_doji"] = (body <= doji_body_ratio * rng).astype(float)
    df["tech_pattern_hammer"] = ((lower_shadow >= pattern_shadow_body_ratio * body) & (upper_shadow <= body)).astype(float)
    df["tech_pattern_shooting_star"] = ((upper_shadow >= pattern_shadow_body_ratio * body) & (lower_shadow <= body)).astype(float)
    prev_o, prev_c = o.shift(1), c.shift(1)
    df["tech_pattern_bullish_engulfing"] = (
        (prev_c < prev_o) & (c > o) & (o <= prev_c) & (c >= prev_o)
    ).astype(float)
    df["tech_pattern_bearish_engulfing"] = (
        (prev_c > prev_o) & (c < o) & (o >= prev_c) & (c <= prev_o)
    ).astype(float)
    df["tech_obv"] = (np.sign(c.diff()).fillna(0.0) * v).cumsum()
    obv_mean = df["tech_obv"].rolling(obv_z_period, min_periods=obv_z_min_periods).mean()
    obv_std = df["tech_obv"].rolling(obv_z_period, min_periods=obv_z_min_periods).std().replace(0, np.nan)
    df[f"tech_obv_z_{obv_z_period}"] = (df["tech_obv"] - obv_mean) / obv_std
    df["tech_vwap_session"] = (typical * v).groupby(df["timestamp"].dt.date).cumsum().div(v.groupby(df["timestamp"].dt.date).cumsum().replace(0, np.nan))
    df["tech_vwap_distance"] = _safe_div(c, df["tech_vwap_session"]) - 1.0

    minute = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["time_minute_of_day"] = minute
    df["time_day_of_week"] = df["timestamp"].dt.dayofweek
    df["time_sin"] = np.sin(2 * np.pi * minute / 1440.0)
    df["time_cos"] = np.cos(2 * np.pi * minute / 1440.0)
    df = _add_session_context_features(
        df, regular_session_start=regular_session_start,
        regular_session_end=regular_session_end,
        first_hour_minutes=first_hour_minutes,
        relative_volume_lookback_days=relative_volume_lookback_days,
        relative_volume_min_days=relative_volume_min_days,
    )
    return df


def _clock_minutes(value: str, name: str) -> int:
    try:
        parts = str(value).split(":")
        if len(parts) != 2:
            raise ValueError
        hour, minute = map(int, parts)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must use HH:MM format") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"{name} must use a valid HH:MM time")
    return hour * 60 + minute


def _add_session_context_features(
    frame: pd.DataFrame,
    *,
    regular_session_start: str,
    regular_session_end: str,
    first_hour_minutes: int,
    relative_volume_lookback_days: int,
    relative_volume_min_days: int,
) -> pd.DataFrame:
    """Add point-in-time regular-session and prior-session context."""
    df = frame.copy()
    start = _clock_minutes(regular_session_start, "regular_session_start")
    end = _clock_minutes(regular_session_end, "regular_session_end")
    if start > end:
        raise ValueError("regular_session_start must not be after regular_session_end")
    timestamp = pd.to_datetime(df["timestamp"])
    session = timestamp.dt.normalize()
    minute = timestamp.dt.hour * 60 + timestamp.dt.minute
    regular = minute.between(start, end, inclusive="both")
    premarket = minute < start
    c = pd.to_numeric(df["close"], errors="coerce")
    o = pd.to_numeric(df["open"], errors="coerce")
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")
    v = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    regular_rows = pd.DataFrame({
        "session": session[regular].to_numpy(), "open": o[regular].to_numpy(),
        "high": h[regular].to_numpy(), "low": l[regular].to_numpy(),
        "close": c[regular].to_numpy(),
    })
    daily = regular_rows.groupby("session", sort=True).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
    )
    previous_close = session.map(daily["close"].shift(1))
    previous_high = session.map(daily["high"].shift(1))
    previous_low = session.map(daily["low"].shift(1))
    regular_open = session.map(daily["open"])
    df["tech_return_since_regular_open"] = (c / regular_open - 1.0).where(minute >= start)
    df["tech_return_since_previous_close"] = c / previous_close - 1.0
    df["tech_overnight_gap"] = (regular_open / previous_close - 1.0).where(minute >= start)
    df["tech_distance_from_previous_high"] = c / previous_high - 1.0
    df["tech_distance_from_previous_low"] = c / previous_low - 1.0
    df["tech_distance_from_previous_close"] = c / previous_close - 1.0

    premarket_last = pd.Series(c[premarket].to_numpy(), index=session[premarket]).groupby(level=0).last()
    final_premarket_return = session.map(premarket_last) / previous_close - 1.0
    df["tech_premarket_return"] = final_premarket_return.where(~premarket, c / previous_close - 1.0)

    regular_high = h.where(regular).groupby(session).cummax()
    regular_low = l.where(regular).groupby(session).cummin()
    df["tech_current_session_high_distance"] = (c / regular_high - 1.0).where(regular)
    df["tech_current_session_low_distance"] = (c / regular_low - 1.0).where(regular)

    first_hour_end = min(end, start + int(first_hour_minutes) - 1)
    first_hour_mask = regular & minute.le(first_hour_end)
    first_hour_close = pd.Series(
        c[first_hour_mask].to_numpy(), index=session[first_hour_mask]
    ).groupby(level=0).last()
    first_hour_return = session.map(first_hour_close) / regular_open - 1.0
    df["tech_first_hour_return"] = first_hour_return.where(minute > first_hour_end)

    cumulative_volume = v.where(regular).groupby(session).cumsum().where(regular)
    df["tech_regular_session_volume_so_far"] = cumulative_volume
    volume_table = pd.DataFrame({
        "minute": minute[regular].to_numpy(), "cumulative": cumulative_volume[regular].to_numpy(),
    }, index=df.index[regular])
    prior_average = volume_table.groupby("minute", sort=False)["cumulative"].transform(
        lambda values: values.shift(1).rolling(
            int(relative_volume_lookback_days),
            min_periods=int(relative_volume_min_days),
        ).mean()
    )
    relative_volume = pd.Series(np.nan, index=df.index, dtype=float)
    relative_volume.loc[regular] = cumulative_volume.loc[regular].div(prior_average.to_numpy())
    df["tech_relative_volume_at_same_time"] = relative_volume

    regular_return = c.pct_change(fill_method=None).where(
        regular & regular.shift(1, fill_value=False)
    )
    realized = regular_return.loc[regular].groupby(session.loc[regular]).expanding(
        min_periods=2
    ).std().reset_index(level=0, drop=True)
    df["tech_realized_volatility_today"] = np.nan
    df.loc[realized.index, "tech_realized_volatility_today"] = realized

    df["tech_previous_day_return"] = session.map(daily["close"].shift(1) / daily["close"].shift(2) - 1.0)
    df["tech_previous_5_day_return"] = session.map(daily["close"].shift(1) / daily["close"].shift(6) - 1.0)
    return df
