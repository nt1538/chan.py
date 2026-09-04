"""Notebook-friendly loading and inference helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from .chan_state import add_chan_features
from .features import (add_technical_features, normalize_ohlcv,
                       technical_feature_kwargs, technical_warmup_bars)
from .labels import TARGET_COLUMNS
from .models import predict_lstm


def load_forecaster(model_path: str | Path) -> Dict[str, Any]:
    """Load a training artifact while preserving its exact feature order."""
    artifact = joblib.load(model_path)
    required = {"models", "features", "targets", "config"}
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"Invalid forecaster artifact; missing: {sorted(missing)}")
    return artifact


def _prepare_inference_frame(
    csv_path: str | Path,
    artifact: Dict[str, Any],
    *,
    enable_chan: Optional[bool] = None,
    symbol: Optional[str] = None,
    latest_only: bool = False,
) -> pd.DataFrame:
    """Recreate training features without generating forward-looking labels."""
    cfg = artifact["config"]
    use_chan = bool(cfg.get("enable_chan", True)) if enable_chan is None else bool(enable_chan)
    source = normalize_ohlcv(pd.read_csv(csv_path))
    if latest_only:
        windows = tuple(cfg.get("technical_windows", (200,)))
        rsi_periods = tuple(cfg.get("technical_rsi_periods", (14,)))
        atr_periods = tuple(cfg.get("technical_atr_periods", (14,)))
        macd_periods = tuple(cfg.get("technical_macd_periods", ((12, 26, 9),)))
        technical_history = technical_warmup_bars(cfg)
        history = max(int(cfg.get("chan_window_bars", 500)),
                      int(cfg.get("lstm_sequence_length", 78)), technical_history)
        source = source.tail(history).reset_index(drop=True)
    frame = add_technical_features(
        source,
        windows=tuple(cfg.get("technical_windows", (2, 3, 5, 10, 12, 20, 26, 39, 50, 78, 156, 200))),
        rsi_periods=tuple(cfg.get("technical_rsi_periods", (14,))),
        atr_periods=tuple(cfg.get("technical_atr_periods", (14,))),
        macd_periods=tuple(tuple(values) for values in cfg.get("technical_macd_periods", ((12, 26, 9),))),
        **technical_feature_kwargs(cfg),
        regular_session_start=str(cfg.get("regular_session_start", "09:30")),
        regular_session_end=str(cfg.get("regular_session_end", "15:55")),
    )
    if use_chan:
        frame = add_chan_features(
            frame, str(symbol or cfg.get("symbol", "UNKNOWN")),
            int(cfg.get("chan_window_bars", 500)),
        )
    # Optional source fields may be absent in a live CSV. The saved imputer will
    # handle them, but sklearn still requires the original column names/order.
    missing = [feature for feature in artifact["features"] if feature not in frame.columns]
    if missing:
        frame = pd.concat(
            [frame, pd.DataFrame(np.nan, index=frame.index, columns=missing)], axis=1
        )
    return frame


def predict_from_csv(
    model_path: str | Path,
    csv_path: str | Path,
    *,
    latest_only: bool = True,
    enable_chan: Optional[bool] = None,
    symbol: Optional[str] = None,
    prediction_model: Optional[str] = None,
    ensemble_xgboost_weight: Optional[float] = None,
) -> pd.DataFrame:
    """Predict two-day extremes for the newest bar or every supplied bar.

    Supply at least 200 historical bars so rolling and Chan state can warm up.
    Chan defaults to the same enabled/disabled setting used during training.
    """
    artifact = load_forecaster(model_path)
    frame = _prepare_inference_frame(
        csv_path, artifact, enable_chan=enable_chan, symbol=symbol,
        latest_only=latest_only,
    )
    features = artifact["features"]
    selected = frame.tail(1).copy() if latest_only else frame.copy()
    result = selected[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    has_xgboost = bool(artifact["models"])
    has_lstm = "lstm" in artifact

    def restore(target: str, values):
        encoding = artifact.get("target_encoding", {}).get(target)
        return -np.asarray(values) if encoding == "positive_magnitude" else values

    if has_xgboost:
        for target in artifact["targets"]:
            model = artifact["models"][target]
            values = (
                model.predict_proba(selected[features])[:, 1]
                if artifact["config"].get("target_mode") == "up_direction"
                else model.predict(selected[features])
            )
            result[f"xgboost_{target}"] = restore(
                target, values
            )

    if has_lstm:
        # Compute sequences on the full history even when only the newest row is
        # returned; selecting the last row first would discard temporal context.
        ends, sequence_predictions = predict_lstm(
            artifact["lstm"], frame[features].to_numpy(dtype=np.float32)
        )
        for target_index, target in enumerate(artifact["targets"]):
            if latest_only:
                lstm_values = restore(target, sequence_predictions[-1:, target_index])
            else:
                lstm_values = np.full(len(frame), np.nan)
                lstm_values[ends] = restore(target, sequence_predictions[:, target_index])
            result[f"lstm_{target}"] = lstm_values
            if has_xgboost:
                weight = float(
                    artifact["config"].get("ensemble_xgboost_weight", 0.5)
                    if ensemble_xgboost_weight is None else ensemble_xgboost_weight
                )
                if not 0.0 <= weight <= 1.0:
                    raise ValueError("ensemble_xgboost_weight must be between 0 and 1")
                result[f"ensemble_{target}"] = (
                    weight * result[f"xgboost_{target}"]
                    + (1.0 - weight) * result[f"lstm_{target}"]
                )

    default_model = artifact["config"].get(
        "prediction_model", "ensemble" if has_lstm and has_xgboost else ("lstm" if has_lstm else "xgboost")
    )
    chosen = str(prediction_model or default_model).lower()
    available = ({"xgboost"} if has_xgboost else set()) | ({"lstm"} if has_lstm else set())
    if has_xgboost and has_lstm:
        available.add("ensemble")
    if chosen not in available:
        raise ValueError(f"prediction_model={chosen!r}; available models are {sorted(available)}")
    for target in artifact["targets"]:
        result[f"predicted_{target}"] = result[f"{chosen}_{target}"]
        if target == "target_up":
            threshold = float(artifact["config"].get("direction_probability_threshold", 0.5))
            result["predicted_target_up_class"] = (
                result["predicted_target_up"] >= threshold
            ).astype(int)
    if "target_max_loss_2d" in artifact["targets"]:
        result["predicted_max_loss_magnitude_2d"] = (
            -result["predicted_target_max_loss_2d"]
        ).clip(lower=0.0)
    return result.reset_index(drop=True)
