"""Rich point-in-time Chan snapshots built from a bounded rolling window."""

from __future__ import annotations

from enum import Enum
import re
from typing import Any, Dict, List
from time import perf_counter

import numpy as np
import pandas as pd


def _number(value: Any, default: float = 0.0) -> float:
    """Convert scalar/enums/bools to a finite model value."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, Enum):
        value = getattr(value, "value", 0)
    try:
        result = float(value)
        return result if np.isfinite(result) else float(default)
    except Exception:
        return float(default)


def _direction(value: Any) -> float:
    text = str(getattr(value, "name", value)).lower()
    return 1.0 if text.endswith("up") else (-1.0 if text.endswith("down") else 0.0)


def _last(value: Any) -> Any:
    try:
        return value[-1] if len(value) else None
    except Exception:
        return None


def _length(value: Any) -> int:
    try:
        return len(value)
    except Exception:
        return 0


def _call(obj: Any, name: str, default: float = 0.0) -> float:
    try:
        return _number(getattr(obj, name)(), default)
    except Exception:
        return float(default)


def _flatten_indicator(out: Dict[str, float], prefix: str, value: Any) -> None:
    """Flatten numeric indicator records and pattern dictionaries."""
    if value is None:
        return
    if isinstance(value, (int, float, bool, np.number, Enum)):
        out[prefix] = _number(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            _flatten_indicator(out, f"{prefix}_{str(key).lower()}", item)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _flatten_indicator(out, f"{prefix}_{index}", item)
    elif hasattr(value, "__dict__"):
        for key, item in vars(value).items():
            if not str(key).startswith("_") and isinstance(item, (int, float, bool, np.number, Enum)):
                out[f"{prefix}_{str(key).lower()}"] = _number(item)


def _indicator_snapshot(klu: Any) -> Dict[str, float]:
    """Extract every technical indicator currently stored on CKLine_Unit."""
    out: Dict[str, float] = {}
    for attr in (
        "macd", "boll", "demark", "rsi", "rsl", "kdj", "dmi",
        "demand_index", "ad_line", "bb_vals", "kc_vals", "starc_vals",
        "sma", "ema", "atr", "stochastic", "roc", "williams_r", "cci",
        "mfi", "tsi", "uo", "psar", "candlestick_patterns",
        "price_patterns", "volume_patterns", "trend",
    ):
        _flatten_indicator(out, f"chan_tech_{attr}", getattr(klu, attr, None))
    return _stationarize_indicator_snapshot(out, _number(getattr(klu, "close", 0.0)))


def _stationarize_indicator_snapshot(values: Dict[str, float], close: float) -> Dict[str, float]:
    """Convert Chan's absolute price-level indicators into stationary ratios."""
    if not close:
        return values
    result: Dict[str, float] = {}
    distance_tokens = (
        "chan_tech_sma_", "chan_tech_ema_", "chan_tech_trend_",
        "chan_tech_bb_vals_", "chan_tech_kc_vals_", "chan_tech_starc_vals_",
    )
    distance_exact = {
        "chan_tech_macd_fast_ema", "chan_tech_macd_slow_ema",
        "chan_tech_boll_up", "chan_tech_boll_down", "chan_tech_boll_mid",
        "chan_tech_psar",
    }
    percent_exact = {
        "chan_tech_macd_dif", "chan_tech_macd_dea", "chan_tech_macd_macd",
        "chan_tech_boll_theta", "chan_tech_atr", "chan_tech_demand_index",
    }
    for key, value in values.items():
        if key in distance_exact or key.startswith(distance_tokens):
            result[f"{key}_distance"] = value / close - 1.0
        elif key in percent_exact:
            result[f"{key}_pct"] = value / close
        else:
            result[key] = value
    return result


def _stationarize_structure_values(values: Dict[str, float], close: float) -> Dict[str, float]:
    """Normalize absolute prices/amps exported by BSP feature dictionaries."""
    if not close:
        return values
    result: Dict[str, float] = {}
    for key, value in values.items():
        lower = key.lower()
        is_bsp_price = lower.startswith("chan_latest_bsp_klu_") and lower.rsplit("_", 1)[-1] in {"open", "high", "low", "close"}
        is_absolute_amp = lower.endswith("_amp") and not lower.endswith("_amp_rate")
        is_absolute_height = lower.endswith("_zs_height")
        if is_bsp_price:
            result[f"{key}_distance"] = value / close - 1.0
        elif is_absolute_amp or is_absolute_height:
            result[f"{key}_pct"] = value / close
        else:
            result[key] = value
    return result


def _structure_snapshot(data: Any, close: float, window_size: int, new_bsps: List[Dict]) -> Dict[str, float]:
    """Extract current merged-K-line, Bi, segment, ZhongShu and BSP state."""
    bi_list = getattr(data, "bi_list", [])
    seg_list = getattr(data, "seg_list", [])
    segseg_list = getattr(data, "segseg_list", [])
    zs_list = getattr(data, "zs_list", [])
    segzs_list = getattr(data, "segzs_list", [])
    last_klc, bi, seg = _last(data), _last(bi_list), _last(seg_list)
    zs, seg_zs = _last(zs_list), _last(segzs_list)
    try:
        bsps = data.bs_point_lst.getSortedBspList()
    except Exception:
        bsps = []
    try:
        seg_bsps = data.seg_bs_point_lst.getSortedBspList()
    except Exception:
        seg_bsps = []
    out = {
        "chan_window_size": float(window_size),
        "chan_merged_kline_count": float(_length(data)),
        "chan_bi_count": float(_length(bi_list)),
        "chan_seg_count": float(_length(seg_list)),
        "chan_segseg_count": float(_length(segseg_list)),
        "chan_zs_count": float(_length(zs_list)),
        "chan_seg_zs_count": float(_length(segzs_list)),
        "chan_bsp_count": float(_length(bsps)),
        "chan_seg_bsp_count": float(_length(seg_bsps)),
        "chan_new_bsp_count": float(len(new_bsps)),
    }
    if last_klc is not None:
        out.update({
            "chan_last_klc_direction": _direction(getattr(last_klc, "dir", None)),
            "chan_last_klc_fx": _number(getattr(last_klc, "fx", 0)),
            "chan_last_klc_high_distance": close / _number(getattr(last_klc, "high", close), close) - 1.0,
            "chan_last_klc_low_distance": close / _number(getattr(last_klc, "low", close), close) - 1.0,
            "chan_last_klc_unit_count": float(_length(getattr(last_klc, "lst", []))),
        })
    if bi is not None:
        begin, end = _call(bi, "get_begin_val", close), _call(bi, "get_end_val", close)
        out.update({
            "chan_last_bi_direction": _direction(getattr(bi, "dir", None)),
            "chan_last_bi_sure": _number(getattr(bi, "is_sure", False)),
            "chan_last_bi_begin_distance": close / begin - 1.0 if begin else 0.0,
            "chan_last_bi_end_distance": close / end - 1.0 if end else 0.0,
            "chan_last_bi_amplitude_pct": _call(bi, "amp") / begin if begin else 0.0,
            "chan_last_bi_klu_count": _call(bi, "get_klu_cnt"),
            "chan_last_bi_klc_count": _call(bi, "get_klc_cnt"),
            "chan_last_bi_seg_index": _number(getattr(bi, "seg_idx", -1), -1),
            "chan_last_bi_has_bsp": _number(getattr(bi, "bsp", None) is not None),
            "chan_last_bi_virtual_end_count": float(_length(getattr(bi, "sure_end", []))),
        })
        # Compare the current Bi with the previous Bi in the same direction
        # (normally three positions back because Bi directions alternate).
        # These are point-in-time ratios; no later confirmation is consulted.
        previous_same_direction = bi_list[-3] if _length(bi_list) >= 3 else None
        if (
            previous_same_direction is not None
            and getattr(previous_same_direction, "dir", None) == getattr(bi, "dir", None)
        ):
            from Common.CEnum import MACD_ALGO
            previous_end = _call(previous_same_direction, "get_end_val", close)
            current_end = _call(bi, "get_end_val", close)
            is_down = _direction(getattr(bi, "dir", None)) < 0
            price_extended = (
                current_end < previous_end if is_down else current_end > previous_end
            )
            out["chan_last_bi_price_extension"] = _number(price_extended)
            out["chan_last_bi_previous_end_distance"] = (
                close / previous_end - 1.0 if previous_end else 0.0
            )
            for metric_name, algorithm in (
                ("area", MACD_ALGO.AREA),
                ("peak", MACD_ALGO.PEAK),
                ("full_area", MACD_ALGO.FULL_AREA),
                ("slope", MACD_ALGO.SLOPE),
                ("amplitude", MACD_ALGO.AMP),
                ("rsi", MACD_ALGO.RSI),
            ):
                try:
                    previous_metric = float(
                        previous_same_direction.cal_macd_metric(
                            algorithm, is_reverse=False
                        )
                    )
                    current_metric = float(
                        bi.cal_macd_metric(algorithm, is_reverse=True)
                    )
                    ratio = current_metric / (abs(previous_metric) + 1e-7)
                    out[f"chan_last_bi_{metric_name}_ratio"] = _number(ratio)
                    out[f"chan_last_bi_{metric_name}_divergence_strength"] = (
                        max(0.0, 1.0 - ratio) if price_extended else 0.0
                    )
                except Exception:
                    continue
    if seg is not None:
        begin, end = _call(seg, "get_begin_val", close), _call(seg, "get_end_val", close)
        out.update({
            "chan_last_seg_direction": _direction(getattr(seg, "dir", None)),
            "chan_last_seg_sure": _number(getattr(seg, "is_sure", False)),
            "chan_last_seg_begin_distance": close / begin - 1.0 if begin else 0.0,
            "chan_last_seg_end_distance": close / end - 1.0 if end else 0.0,
            "chan_last_seg_amplitude_pct": _call(seg, "cal_amp"),
            "chan_last_seg_slope": _call(seg, "cal_klu_slope"),
            "chan_last_seg_bi_count": _call(seg, "cal_bi_cnt"),
            "chan_last_seg_klu_count": _call(seg, "get_klu_cnt"),
            "chan_last_seg_zs_count": float(_length(getattr(seg, "zs_lst", []))),
            "chan_last_seg_has_bsp": _number(getattr(seg, "bsp", None) is not None),
        })
    for prefix, current in (("chan_last_zs", zs), ("chan_last_seg_zs", seg_zs)):
        if current is not None:
            low, high, mid = (_number(getattr(current, key, close), close) for key in ("low", "high", "mid"))
            out.update({
                f"{prefix}_sure": _number(getattr(current, "is_sure", False)),
                f"{prefix}_low_distance": close / low - 1.0 if low else 0.0,
                f"{prefix}_high_distance": close / high - 1.0 if high else 0.0,
                f"{prefix}_mid_distance": close / mid - 1.0 if mid else 0.0,
                f"{prefix}_width_pct": (high - low) / mid if mid else 0.0,
                f"{prefix}_peak_width_pct": (_number(getattr(current, "peak_high", high)) - _number(getattr(current, "peak_low", low))) / mid if mid else 0.0,
                f"{prefix}_bi_count": float(_length(getattr(current, "bi_lst", []))),
                f"{prefix}_sub_zs_count": float(_length(getattr(current, "sub_zs_lst", []))),
                f"{prefix}_has_bi_in": _number(getattr(current, "bi_in", None) is not None),
                f"{prefix}_has_bi_out": _number(getattr(current, "bi_out", None) is not None),
            })
    if new_bsps:
        latest = new_bsps[-1]
        out["chan_latest_new_bsp_is_buy"] = _number(str(latest.get("direction", "")).lower() == "buy")
        out["chan_latest_new_bsp_is_sell"] = _number(str(latest.get("direction", "")).lower() == "sell")
        # A bar can expose more than one BSP. Cohort flags represent the union
        # so type-vs-complement experiments do not silently discard earlier
        # points when only the latest record supplies the detailed snapshot.
        type_text = " ".join(
            str(point.get("bsp_types", point.get("bsp_type", ""))).lower()
            for point in new_bsps
        )
        latest_types = set(re.findall(r"1p|2p|2s|3p|3a|3b|1|2|3", type_text))
        for kind in ("1", "1p", "2", "2s", "3a", "3b"):
            out[f"chan_latest_new_bsp_type_{kind}"] = _number(kind in latest_types)
        out["chan_latest_new_bsp_is_type_1"] = _number(bool(latest_types & {"1", "1p"}))
        out["chan_latest_new_bsp_is_type_2"] = _number(bool(latest_types & {"2", "2s", "2p"}))
        out["chan_latest_new_bsp_is_type_3"] = _number(bool(latest_types & {"3", "3a", "3b", "3p"}))
        for key, value in latest.items():
            if isinstance(value, (int, float, bool, np.number)):
                out[f"chan_latest_bsp_{key}"] = _number(value)
    current_bsp = _last(bsps)
    if current_bsp is not None:
        out["chan_current_last_bsp_is_buy"] = _number(getattr(current_bsp, "is_buy", False))
        out["chan_current_last_bsp_is_segbsp"] = _number(getattr(current_bsp, "is_segbsp", False))
        out["chan_current_last_bsp_mature_rate"] = _number(getattr(current_bsp, "mature_rate", 0.0))
        out["chan_current_last_bsp_is_mature"] = _number(getattr(current_bsp, "is_mature_point", False))
        types = {str(getattr(kind, "value", kind)).lower() for kind in getattr(current_bsp, "type", [])}
        for kind in ("1", "1p", "2", "2s", "3a", "3b"):
            out[f"chan_current_last_bsp_type_{kind}"] = _number(kind in types)
        out["chan_current_last_bsp_is_type_1"] = _number(bool(types & {"1", "1p"}))
        out["chan_current_last_bsp_is_type_2"] = _number(bool(types & {"2", "2s", "2p"}))
        out["chan_current_last_bsp_is_type_3"] = _number(bool(types & {"3", "3a", "3b", "3p"}))
        try:
            for key, value in current_bsp.features.to_dict().items():
                _flatten_indicator(out, f"chan_current_last_bsp_feature_{str(key).lower()}", value)
        except Exception:
            pass
    return _stationarize_structure_values(out, close)


def _full_chan_config():
    """Enable the technical-indicator inventory already implemented by Chan."""
    from ChanConfig import CChanConfig
    return CChanConfig({
        "trigger_step": True, "print_warning": False,
        "cal_rsi": True, "cal_kdj": True, "cal_dmi": True, "cal_rsl": True,
        "cal_demand_index": True, "cal_adline": True, "cal_bb_vals": True,
        "cal_kc_vals": True, "cal_starc_vals": True,
        "cal_sma": True, "sma_periods": [5, 10, 20, 50, 100, 200],
        "cal_ema": True, "ema_periods": [5, 10, 20, 50, 100, 200],
        "cal_atr": True, "cal_stochastic": True,
        "cal_roc": True, "roc_periods": [5, 10, 20, 50, 100],
        "cal_williams": True, "cal_cci": True, "cal_mfi": True,
        "cal_tsi": True, "cal_uo": True, "cal_psar": True, "cal_demark": True,
        "cal_candlestick_patterns": True, "cal_price_patterns": True,
        "cal_volume_patterns": True, "mean_metrics": [5, 10, 20, 50, 100, 200],
        "trend_metrics": [5, 10, 20, 50, 100, 200],
    })


def add_chan_features(
    frame: pd.DataFrame,
    symbol: str,
    window_bars: int = 500,
    *,
    verbose: bool = False,
    progress_every_rows: int = 5_000,
    level: Any | None = None,
    feature_prefix: str = "chan_",
) -> pd.DataFrame:
    """Rebuild a complete Chan state from the latest ``window_bars`` per row."""
    if window_bars < 3:
        raise ValueError("window_bars must be at least 3")
    if progress_every_rows < 1:
        raise ValueError("progress_every_rows must be positive")
    from Common.CEnum import KL_TYPE
    from Pipeline.DailyBandit5mPipeline import build_klu
    from sliding_window_chan import SlidingWindowChan

    level = KL_TYPE.K_5M if level is None else level
    sliding = SlidingWindowChan(code=symbol, lv_list=[level], config=_full_chan_config(), max_klines=window_bars)
    rows: List[Dict[str, float]] = []
    chunks: List[pd.DataFrame] = []
    bars_since_bsp = window_bars + 1
    last_bsp_direction = 0.0
    started = perf_counter()
    if verbose:
        print(f"[Chan] Starting {len(frame):,} rows; rolling window={window_bars:,}", flush=True)
    for i, row in frame.iterrows():
        klu = build_klu(row.timestamp, row.open, row.high, row.low, row.close, row.volume)
        # Pass the absolute index ourselves, then discard SlidingWindowChan's
        # redundant unbounded history. Its deque still retains exactly the
        # requested rolling window.
        window_chan, new_bsps = sliding._process_single_kline(klu, i)
        sliding.all_klines.clear()
        # CKLine_Unit links neighbors bidirectionally. Break the deque's left
        # boundary so a 500-bar window cannot retain the entire earlier series.
        if sliding.kline_window:
            sliding.kline_window[0].pre = None
        if new_bsps:
            bars_since_bsp = 0
            last_bsp_direction = 1.0 if str(new_bsps[-1].get("direction", "")).lower() == "buy" else -1.0
        else:
            bars_since_bsp += 1
        data = window_chan.kl_datas[level]
        state = _structure_snapshot(data, float(row.close), min(i + 1, window_bars), new_bsps)
        state.update(_indicator_snapshot(klu))
        state["chan_bars_since_bsp"] = float(bars_since_bsp)
        state["chan_last_bsp_direction"] = last_bsp_direction
        rows.append(state)
        # A list containing hundreds of thousands of wide dictionaries can use
        # multiple GB. Convert and release records incrementally.
        if len(rows) >= 2000:
            chunks.append(pd.DataFrame.from_records(rows))
            rows = []
        completed = i + 1
        if verbose and (completed % progress_every_rows == 0 or completed == len(frame)):
            elapsed = perf_counter() - started
            rate = completed / elapsed if elapsed else 0.0
            remaining = (len(frame) - completed) / rate if rate else 0.0
            print(
                f"[Chan] {completed:,}/{len(frame):,} ({100 * completed / len(frame):.1f}%) "
                f"| {rate:,.1f} rows/s | elapsed {elapsed / 60:.1f}m | ETA {remaining / 60:.1f}m",
                flush=True,
            )
    if rows:
        chunks.append(pd.DataFrame.from_records(rows))
    chan_frame = pd.concat(chunks, ignore_index=True, sort=False) if chunks else pd.DataFrame(index=frame.index)
    if verbose:
        print(f"[Chan] Complete: {len(chan_frame):,} rows, {len(chan_frame.columns):,} Chan features", flush=True)
    if feature_prefix != "chan_":
        chan_frame = chan_frame.rename(columns={
            column: f"{feature_prefix}{column[len('chan_'):]}"
            for column in chan_frame.columns
            if str(column).startswith("chan_")
        })
    return pd.concat([frame.reset_index(drop=True), chan_frame], axis=1)
