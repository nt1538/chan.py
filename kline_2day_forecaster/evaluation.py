"""Notebook-friendly diagnostic plots for saved forecasting runs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix,
                             log_loss, r2_score, roc_auc_score, roc_curve)

from .features import normalize_ohlcv
from .labels import TARGET_COLUMNS


def plot_direction_accuracy_vs_market_trend(
    output_dir: str | Path,
    *,
    model: str = "xgboost",
    accuracy_rolling_bars: int = 390,
    trend_lookback_bars: int = 390,
    probability_threshold: float | None = None,
    show: bool = True,
) -> Dict[str, str]:
    """Compare rolling sign accuracy with the contemporaneous market trend.

    Market trend is the sampled TQQQ close return over ``trend_lookback_bars``.
    Prediction direction comes from exact-return sign or classifier probability.
    """
    if accuracy_rolling_bars < 2 or trend_lookback_bars < 1:
        raise ValueError("accuracy_rolling_bars must be >= 2 and trend_lookback_bars positive")
    out = Path(output_dir)
    predictions_path, artifact_path = out / "test_predictions.csv", out / "model.joblib"
    if not predictions_path.exists() or not artifact_path.exists():
        raise FileNotFoundError("Both test_predictions.csv and model.joblib are required")
    artifact = joblib.load(artifact_path)
    predictions = pd.read_csv(predictions_path, parse_dates=["timestamp"])
    cfg = artifact.get("config", {})
    mode = str(cfg.get("target_mode", ""))
    if mode == "exact_return":
        target, prediction_column = "target_exact_return", f"{model}_target_exact_return"
        required = {target, prediction_column}
        missing = required.difference(predictions.columns)
        if missing:
            raise ValueError(f"Missing exact-return columns: {sorted(missing)}")
        actual_up = predictions[target].astype(float).gt(0)
        predicted_up = predictions[prediction_column].astype(float).gt(0)
    elif mode == "up_direction":
        target, prediction_column = "target_up", f"{model}_target_up"
        required = {target, prediction_column}
        missing = required.difference(predictions.columns)
        if missing:
            raise ValueError(f"Missing direction columns: {sorted(missing)}")
        threshold = float(
            cfg.get("direction_probability_threshold", 0.5)
            if probability_threshold is None else probability_threshold
        )
        if not 0 < threshold < 1:
            raise ValueError("probability_threshold must be between 0 and 1")
        actual_up = predictions[target].astype(int).eq(1)
        predicted_up = predictions[prediction_column].astype(float).ge(threshold)
    else:
        raise ValueError("This plot supports target_mode='exact_return' or 'up_direction'")

    source_path = Path(str(cfg.get("input_csv", "")))
    if not source_path.exists():
        raise FileNotFoundError(f"Could not load configured market data: {source_path}")
    market = normalize_ohlcv(pd.read_csv(source_path))[["timestamp", "close"]]
    data = predictions[["timestamp"]].copy()
    data["actual_up"] = actual_up.astype(int)
    data["predicted_up"] = predicted_up.astype(int)
    data["direction_correct"] = data["actual_up"].eq(data["predicted_up"]).astype(float)
    data = data.merge(market, on="timestamp", how="left").dropna(subset=["close"])
    minimum = max(2, accuracy_rolling_bars // 4)
    data["rolling_direction_accuracy"] = data["direction_correct"].rolling(
        accuracy_rolling_bars, min_periods=minimum
    ).mean()
    data["market_trend_return"] = data["close"].pct_change(
        trend_lookback_bars, fill_method=None
    )
    data["market_regime"] = np.where(
        data["market_trend_return"].isna(), "unknown",
        np.where(data["market_trend_return"] >= 0, "rising", "falling"),
    )
    data["rolling_actual_up_rate"] = data["actual_up"].rolling(
        accuracy_rolling_bars, min_periods=minimum
    ).mean()

    known = data[data["market_regime"].ne("unknown")]
    regime_summary = known.groupby("market_regime", sort=False).agg(
        rows=("direction_correct", "size"),
        direction_accuracy=("direction_correct", "mean"),
        actual_up_rate=("actual_up", "mean"),
        average_trend_return=("market_trend_return", "mean"),
    ).reset_index()
    quality_path = out / f"{model}_direction_accuracy_vs_market_trend.csv"
    summary_path = out / f"{model}_direction_accuracy_by_market_regime.csv"
    data.to_csv(quality_path, index=False)
    regime_summary.to_csv(summary_path, index=False)

    fig, axes = plt.subplots(3, 1, figsize=(17, 13), sharex=True, constrained_layout=True)
    fig.suptitle(
        f"{model.upper()} direction accuracy versus actual TQQQ trend",
        fontsize=15,
    )
    axes[0].plot(data["timestamp"], data["close"], color="black", linewidth=1)
    axes[0].set(title="Actual TQQQ close", ylabel="Price")

    trend_percent = data["market_trend_return"] * 100
    axes[1].plot(data["timestamp"], trend_percent, color="tab:blue", linewidth=1,
                 label=f"{trend_lookback_bars}-observation market return")
    axes[1].fill_between(data["timestamp"], 0, trend_percent, where=trend_percent.ge(0),
                         color="tab:green", alpha=0.25, label="Rising regime")
    axes[1].fill_between(data["timestamp"], 0, trend_percent, where=trend_percent.lt(0),
                         color="tab:red", alpha=0.25, label="Falling regime")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set(title="Actual market trend", ylabel="Lookback return (%)")
    axes[1].legend(loc="upper left")

    accuracy_percent = data["rolling_direction_accuracy"] * 100
    axes[2].plot(data["timestamp"], accuracy_percent, color="tab:purple",
                 linewidth=1.2, label="Rolling direction accuracy")
    axes[2].plot(data["timestamp"], data["rolling_actual_up_rate"] * 100,
                 color="tab:orange", alpha=0.7, label="Actual up frequency")
    axes[2].axhline(50, color="gray", linestyle="--", linewidth=0.8,
                    label="50% reference")
    axes[2].set(title=f"Direction accuracy ({accuracy_rolling_bars:,} observations)",
                ylabel="Percent", xlabel="Time", ylim=(0, 100))
    axes[2].legend(loc="best")
    for axis in axes:
        axis.grid(alpha=0.2)

    plot_path = out / f"{model}_direction_accuracy_vs_market_trend.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return {
        "plot_path": str(plot_path.resolve()),
        "rolling_data_csv": str(quality_path.resolve()),
        "regime_summary_csv": str(summary_path.resolve()),
    }


def plot_exact_return_rolling_quality(
    output_dir: str | Path,
    *,
    model: str = "xgboost",
    rolling_bars: int = 390,
    min_periods: int | None = None,
    max_time_points: int = 3_000,
    show: bool = True,
) -> Dict[str, str]:
    """Show when an exact-return model is useful and when it fails.

    Positive ``rolling_mae_skill_vs_zero`` means the model beats predicting
    zero in that window. The timestamped quality CSV includes point errors,
    rolling errors, correlation, directional accuracy, volatility and bias.
    """
    if rolling_bars < 2 or max_time_points < 1:
        raise ValueError("rolling_bars must be >= 2 and max_time_points positive")
    minimum = max(2, rolling_bars // 4) if min_periods is None else int(min_periods)
    if not 2 <= minimum <= rolling_bars:
        raise ValueError("min_periods must be between 2 and rolling_bars")
    out = Path(output_dir)
    predictions_path = out / "test_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing test predictions: {predictions_path}")
    frame = pd.read_csv(predictions_path, parse_dates=["timestamp"])
    target, prediction_column = "target_exact_return", f"{model}_target_exact_return"
    missing = {"timestamp", target, prediction_column}.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing exact-return columns: {sorted(missing)}")
    quality = frame[["timestamp", target, prediction_column]].dropna().copy()
    if len(quality) < minimum:
        raise ValueError(f"Need at least {minimum} non-null predictions")
    actual = quality[target].astype(float)
    predicted = quality[prediction_column].astype(float)
    error = predicted - actual
    quality["prediction_error"] = error
    quality["absolute_error"] = error.abs()
    quality["direction_correct"] = (np.sign(predicted) == np.sign(actual)).astype(float)
    quality["rolling_mae"] = quality["absolute_error"].rolling(
        rolling_bars, min_periods=minimum
    ).mean()
    quality["rolling_rmse"] = np.sqrt(error.pow(2).rolling(
        rolling_bars, min_periods=minimum
    ).mean())
    quality["rolling_zero_baseline_mae"] = actual.abs().rolling(
        rolling_bars, min_periods=minimum
    ).mean()
    quality["rolling_mae_skill_vs_zero"] = 1.0 - quality["rolling_mae"].div(
        quality["rolling_zero_baseline_mae"].replace(0, np.nan)
    )
    quality["rolling_correlation"] = actual.rolling(
        rolling_bars, min_periods=minimum
    ).corr(predicted)
    quality["rolling_direction_accuracy"] = quality["direction_correct"].rolling(
        rolling_bars, min_periods=minimum
    ).mean()
    quality["rolling_actual_volatility"] = actual.rolling(
        rolling_bars, min_periods=minimum
    ).std()
    quality["rolling_prediction_bias"] = error.rolling(
        rolling_bars, min_periods=minimum
    ).mean()

    csv_path = out / f"{model}_exact_return_rolling_quality.csv"
    quality.to_csv(csv_path, index=False)
    step = max(1, int(np.ceil(len(quality) / max_time_points)))
    shown = quality.iloc[::step]
    fig, axes = plt.subplots(4, 1, figsize=(17, 15), sharex=True, constrained_layout=True)
    fig.suptitle(
        f"{model.upper()} exact-return rolling quality ({rolling_bars:,} observations)",
        fontsize=15,
    )

    axes[0].plot(shown["timestamp"], shown[target] * 100, linewidth=0.8,
                 alpha=0.75, label="Actual return")
    axes[0].plot(shown["timestamp"], shown[prediction_column] * 100,
                 linewidth=0.9, label="Predicted return")
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set(ylabel="Return (%)", title="Actual versus predicted")
    axes[0].legend(loc="upper right")

    axes[1].plot(quality["timestamp"], quality["rolling_mae"] * 100,
                 label="Model rolling MAE", color="tab:blue")
    axes[1].plot(quality["timestamp"], quality["rolling_zero_baseline_mae"] * 100,
                 label="Always-zero rolling MAE", color="tab:gray", alpha=0.85)
    axes[1].set(ylabel="MAE (%)", title="Rolling error versus zero-return baseline")
    axes[1].legend(loc="upper right")
    skill_axis = axes[1].twinx()
    skill_axis.plot(quality["timestamp"], quality["rolling_mae_skill_vs_zero"] * 100,
                    color="tab:green", alpha=0.55, label="MAE skill")
    skill_axis.axhline(0, color="tab:red", linestyle="--", linewidth=0.8)
    skill_axis.set_ylabel("Skill vs zero (%)")

    axes[2].plot(quality["timestamp"], quality["rolling_correlation"],
                 label="Rolling correlation", color="tab:purple")
    axes[2].axhline(0, color="tab:purple", linestyle="--", linewidth=0.8)
    axes[2].set(ylabel="Correlation", title="Rolling ranking and direction quality")
    direction_axis = axes[2].twinx()
    direction_axis.plot(
        quality["timestamp"], quality["rolling_direction_accuracy"] * 100,
        label="Directional accuracy", color="tab:orange", alpha=0.75,
    )
    direction_axis.axhline(50, color="tab:orange", linestyle="--", linewidth=0.8)
    direction_axis.set_ylabel("Direction accuracy (%)")

    axes[3].plot(quality["timestamp"], quality["rolling_actual_volatility"] * 100,
                 label="Actual return volatility", color="tab:red")
    axes[3].set(ylabel="Volatility (%)", xlabel="Time",
                title="Market volatility and prediction bias")
    bias_axis = axes[3].twinx()
    bias_axis.plot(quality["timestamp"], quality["rolling_prediction_bias"] * 100,
                   label="Prediction bias", color="tab:brown", alpha=0.75)
    bias_axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
    bias_axis.set_ylabel("Bias (%)")
    for axis in axes:
        axis.grid(alpha=0.2)

    plot_path = out / f"{model}_exact_return_rolling_quality.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return {
        "rolling_quality_plot": str(plot_path.resolve()),
        "rolling_quality_csv": str(csv_path.resolve()),
    }


def plot_direction_results(
    output_dir: str | Path,
    *,
    model: str = "xgboost",
    probability_threshold: float | None = None,
    rolling_bars: int = 390,
    max_time_points: int = 2_000,
    calibration_bins: int = 10,
    top_features: int = 25,
    show: bool = True,
) -> Dict[str, str]:
    """Plot diagnostics for a saved ``target_mode='up_direction'`` run."""
    if rolling_bars < 1 or max_time_points < 1 or calibration_bins < 2:
        raise ValueError("rolling_bars/max_time_points must be positive and calibration_bins >= 2")
    out = Path(output_dir)
    predictions_path, artifact_path = out / "test_predictions.csv", out / "model.joblib"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing test predictions: {predictions_path}")
    frame = pd.read_csv(predictions_path, parse_dates=["timestamp"])
    target, probability_column = "target_up", f"{model}_target_up"
    missing = {"timestamp", target, probability_column}.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing direction-result columns: {sorted(missing)}")
    data = frame[["timestamp", target, probability_column]].dropna().copy()
    if len(data) < 2 or data[target].nunique() < 2:
        raise ValueError("Direction plots require at least two rows and both target classes")

    artifact = joblib.load(artifact_path) if artifact_path.exists() else None
    saved_threshold = (
        artifact.get("config", {}).get("direction_probability_threshold", 0.5)
        if artifact else 0.5
    )
    threshold = float(saved_threshold if probability_threshold is None else probability_threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError("probability_threshold must be between 0 and 1")
    actual = data[target].to_numpy(int)
    probability = data[probability_column].to_numpy(float)
    predicted = (probability >= threshold).astype(int)
    accuracy = float(accuracy_score(actual, predicted))
    auc_value = float(roc_auc_score(actual, probability))
    loss = float(log_loss(actual, probability, labels=[0, 1]))
    brier = float(brier_score_loss(actual, probability))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    fig.suptitle(
        f"{model.upper()} next-session direction | Accuracy={accuracy:.1%}  "
        f"AUC={auc_value:.3f}  Log loss={loss:.3f}  Brier={brier:.3f}", fontsize=14,
    )
    step = max(1, int(np.ceil(len(data) / max_time_points)))
    shown = data.iloc[::step]
    shown_actual = actual[::step]
    axes[0, 0].plot(shown["timestamp"], probability[::step], linewidth=1,
                    label="Predicted P(up)")
    axes[0, 0].scatter(shown["timestamp"], shown_actual, c=shown_actual,
                       cmap="coolwarm", vmin=0, vmax=1, s=8, alpha=0.35,
                       label="Actual class")
    axes[0, 0].axhline(threshold, color="black", linestyle="--", linewidth=1,
                       label=f"Threshold={threshold:.2f}")
    axes[0, 0].set(title="Predicted probability and actual direction",
                   ylabel="Probability / class", ylim=(-0.05, 1.05))
    axes[0, 0].legend(loc="best")

    false_positive, true_positive, _ = roc_curve(actual, probability)
    axes[0, 1].plot(false_positive, true_positive, linewidth=2,
                    label=f"ROC (AUC={auc_value:.3f})")
    axes[0, 1].plot([0, 1], [0, 1], "--", color="gray", label="Random")
    axes[0, 1].set(title="Receiver operating characteristic",
                   xlabel="False-positive rate", ylabel="True-positive rate",
                   xlim=(0, 1), ylim=(0, 1))
    axes[0, 1].legend(loc="lower right")

    observed, forecast = calibration_curve(
        actual, probability, n_bins=int(calibration_bins), strategy="quantile"
    )
    axes[1, 0].plot(forecast, observed, marker="o", label="Model")
    axes[1, 0].plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    axes[1, 0].set(title="Probability calibration", xlabel="Mean predicted probability",
                   ylabel="Observed up frequency", xlim=(0, 1), ylim=(0, 1))
    axes[1, 0].legend(loc="best")

    rolling_accuracy = pd.Series(
        (predicted == actual).astype(float), index=data.index
    ).rolling(rolling_bars, min_periods=max(1, rolling_bars // 4)).mean() * 100
    axes[1, 1].plot(data["timestamp"], rolling_accuracy, linewidth=1,
                    label="Rolling accuracy")
    majority_baseline = max(float(actual.mean()), 1.0 - float(actual.mean())) * 100
    axes[1, 1].axhline(50, color="gray", linestyle="--", linewidth=1,
                       label="Random=50%")
    axes[1, 1].axhline(majority_baseline, color="tab:red", linestyle=":", linewidth=1,
                       label=f"Majority={majority_baseline:.1f}%")
    axes[1, 1].set(title=f"Rolling accuracy ({rolling_bars:,} bars)",
                   xlabel="Time", ylabel="Accuracy (%)", ylim=(0, 100))
    axes[1, 1].legend(loc="best")
    for axis in axes.flat:
        axis.grid(alpha=0.2)

    diagnostics_path = out / f"{model}_direction_diagnostics.png"
    fig.savefig(diagnostics_path, dpi=160, bbox_inches="tight")
    generated = {"direction_diagnostics": str(diagnostics_path.resolve())}
    if show:
        plt.show()
    else:
        plt.close(fig)

    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    matrix_fig, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center",
                      color="white" if matrix[row, column] > matrix.max() / 2 else "black",
                      fontsize=14)
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Predicted down", "Predicted up"],
             yticklabels=["Actual down", "Actual up"], title=f"Confusion matrix (threshold={threshold:.2f})",
             xlabel="Prediction", ylabel="Actual")
    matrix_fig.colorbar(image, ax=axis, shrink=0.8)
    matrix_path = out / f"{model}_direction_confusion_matrix.png"
    matrix_fig.savefig(matrix_path, dpi=160, bbox_inches="tight")
    generated["direction_confusion_matrix"] = str(matrix_path.resolve())
    if show:
        plt.show()
    else:
        plt.close(matrix_fig)

    if model == "xgboost" and top_features > 0 and artifact is not None:
        pipeline = artifact.get("models", {}).get(target)
        estimator = getattr(pipeline, "named_steps", {}).get("model")
        importance = getattr(estimator, "feature_importances_", None)
        if importance is not None:
            count = min(int(top_features), len(importance))
            order = np.argsort(importance)[-count:]
            names = np.asarray(artifact["features"], dtype=object)[order]
            values = np.asarray(importance)[order]
            importance_fig, axis = plt.subplots(
                figsize=(11, max(6, count * 0.32)), constrained_layout=True
            )
            axis.barh(names, values)
            axis.set(title="XGBoost feature importance: next-session direction",
                     xlabel="Importance")
            axis.grid(axis="x", alpha=0.2)
            importance_path = out / "xgboost_direction_feature_importance.png"
            importance_fig.savefig(importance_path, dpi=160, bbox_inches="tight")
            generated["direction_feature_importance"] = str(importance_path.resolve())
            if show:
                plt.show()
            else:
                plt.close(importance_fig)
    return generated


def plot_exact_return_results(
    output_dir: str | Path,
    *,
    model: str = "xgboost",
    rolling_bars: int = 390,
    max_time_points: int = 2_000,
    top_features: int = 25,
    show: bool = True,
) -> Dict[str, str]:
    """Plot diagnostics for a same-time exact-return forecasting run.

    Reads ``test_predictions.csv`` and, when available, ``model.joblib`` from
    ``output_dir``. All metrics use the complete test set; downsampling applies
    only to the time-series display. Returns paths of generated PNG files.
    """
    if rolling_bars < 1 or max_time_points < 1:
        raise ValueError("rolling_bars and max_time_points must be positive")
    out = Path(output_dir)
    predictions_path = out / "test_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing test predictions: {predictions_path}")
    frame = pd.read_csv(predictions_path, parse_dates=["timestamp"])
    target = "target_exact_return"
    prediction_column = f"{model}_{target}"
    required = {"timestamp", target, prediction_column}
    missing = required.difference(frame.columns)
    if missing:
        available = sorted(c for c in frame if c.endswith(target) and c != target)
        raise ValueError(f"Missing columns {sorted(missing)}. Available predictions: {available}")
    data = frame[["timestamp", target, prediction_column]].dropna().copy()
    if len(data) < 2:
        raise ValueError("At least two non-null test predictions are required")
    actual = data[target].to_numpy(float)
    predicted = data[prediction_column].to_numpy(float)
    residual = predicted - actual
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    r2 = float(r2_score(actual, predicted))
    correlation = float(np.corrcoef(actual, predicted)[0, 1])
    directional = (np.sign(actual) == np.sign(predicted)).astype(float)
    directional_accuracy = float(directional.mean())

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    fig.suptitle(
        f"{model.upper()} exact next-session return | "
        f"MAE={mae:.5f}  RMSE={rmse:.5f}  R²={r2:.3f}  "
        f"Direction={directional_accuracy:.1%}", fontsize=14,
    )
    step = max(1, int(np.ceil(len(data) / max_time_points)))
    shown = data.iloc[::step]
    axes[0, 0].plot(shown["timestamp"], actual[::step] * 100, label="Actual", linewidth=1)
    axes[0, 0].plot(shown["timestamp"], predicted[::step] * 100, label="Predicted", linewidth=1)
    axes[0, 0].axhline(0, color="black", linewidth=0.7)
    axes[0, 0].set(title="Test-period returns", ylabel="Return (%)")
    axes[0, 0].legend()

    axes[0, 1].scatter(actual * 100, predicted * 100, s=9, alpha=0.25)
    low = float(min(actual.min(), predicted.min()) * 100)
    high = float(max(actual.max(), predicted.max()) * 100)
    axes[0, 1].plot([low, high], [low, high], "--", linewidth=1, label="Perfect")
    axes[0, 1].set(title=f"Actual vs predicted (correlation={correlation:.3f})",
                   xlabel="Actual return (%)", ylabel="Predicted return (%)")
    axes[0, 1].legend()

    axes[1, 0].hist(residual * 100, bins=60, alpha=0.85)
    axes[1, 0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set(title=f"Prediction errors (mean={residual.mean() * 100:.3f}%)",
                   xlabel="Predicted minus actual (%)", ylabel="K-lines")

    min_periods = max(1, rolling_bars // 4)
    rolling_mae = pd.Series(np.abs(residual), index=data.index).rolling(
        rolling_bars, min_periods=min_periods
    ).mean() * 100
    rolling_direction = pd.Series(directional, index=data.index).rolling(
        rolling_bars, min_periods=min_periods
    ).mean() * 100
    axes[1, 1].plot(data["timestamp"], rolling_mae, color="tab:blue", label="MAE")
    axes[1, 1].set(title=f"Rolling quality ({rolling_bars:,} bars)",
                   xlabel="Time", ylabel="MAE (%)", ylim=(0, None))
    direction_axis = axes[1, 1].twinx()
    direction_axis.plot(data["timestamp"], rolling_direction, color="tab:orange",
                        alpha=0.8, label="Directional accuracy")
    direction_axis.axhline(50, color="tab:orange", linestyle="--", linewidth=0.8)
    direction_axis.set_ylabel("Directional accuracy (%)")
    lines = axes[1, 1].lines[:1] + direction_axis.lines[:1]
    axes[1, 1].legend(lines, [line.get_label() for line in lines], loc="upper right")
    for axis in axes.flat:
        axis.grid(alpha=0.2)

    diagnostics_path = out / f"{model}_exact_return_diagnostics.png"
    fig.savefig(diagnostics_path, dpi=160, bbox_inches="tight")
    generated = {"exact_return_diagnostics": str(diagnostics_path.resolve())}
    if show:
        plt.show()
    else:
        plt.close(fig)

    artifact_path = out / "model.joblib"
    if model == "xgboost" and top_features > 0 and artifact_path.exists():
        artifact = joblib.load(artifact_path)
        pipeline = artifact.get("models", {}).get(target)
        estimator = getattr(pipeline, "named_steps", {}).get("model")
        importance = getattr(estimator, "feature_importances_", None)
        if importance is not None:
            count = min(int(top_features), len(importance))
            order = np.argsort(importance)[-count:]
            names = np.asarray(artifact["features"], dtype=object)[order]
            values = np.asarray(importance)[order]
            importance_fig, axis = plt.subplots(
                figsize=(11, max(6, count * 0.32)), constrained_layout=True
            )
            axis.barh(names, values)
            axis.set(title="XGBoost feature importance: exact next-session return",
                     xlabel="Importance")
            axis.grid(axis="x", alpha=0.2)
            importance_path = out / "xgboost_exact_return_feature_importance.png"
            importance_fig.savefig(importance_path, dpi=160, bbox_inches="tight")
            generated["exact_return_feature_importance"] = str(importance_path.resolve())
            if show:
                plt.show()
            else:
                plt.close(importance_fig)
    return generated


def plot_forecast_results(
    output_dir: str | Path,
    *,
    model: str = "xgboost",
    rolling_bars: int = 390,
    max_time_points: int = 2_000,
    top_features: int = 25,
    show: bool = True,
) -> Dict[str, str]:
    """Plot test diagnostics and XGBoost importance from a training directory.

    Returns the generated PNG paths. Losses are displayed as positive downside
    magnitudes, which makes their charts easier to compare with gains.
    """
    out = Path(output_dir)
    predictions_path = out / "test_predictions.csv"
    artifact_path = out / "model.joblib"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing test predictions: {predictions_path}")

    frame = pd.read_csv(predictions_path, parse_dates=["timestamp"])
    generated: Dict[str, str] = {}
    labels = {
        "target_max_gain_2d": "Signed return to maximum high in next 2 trading days",
        "target_max_loss_2d": "Signed return to minimum low in next 2 trading days",
    }

    for target in TARGET_COLUMNS:
        prediction_column = f"{model}_{target}"
        if prediction_column not in frame:
            available = sorted(c for c in frame if c.endswith(target) and c != target)
            raise ValueError(
                f"No {model!r} predictions for {target}. Available: {available}"
            )
        data = frame[["timestamp", target, prediction_column]].dropna().copy()
        actual = data[target].to_numpy(float)
        predicted = data[prediction_column].to_numpy(float)
        if target == "target_max_loss_2d":
            actual, predicted = -actual, -predicted
        residual = predicted - actual

        fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
        fig.suptitle(f"{model.upper()}: {labels[target]}", fontsize=15)

        # Downsample only the time-series display; all rows remain in metrics.
        step = max(1, int(np.ceil(len(data) / max(1, max_time_points))))
        shown = data.iloc[::step]
        shown_actual = actual[::step]
        shown_predicted = predicted[::step]
        axes[0, 0].plot(shown["timestamp"], shown_actual, label="Actual", linewidth=1)
        axes[0, 0].plot(shown["timestamp"], shown_predicted, label="Predicted", linewidth=1)
        axes[0, 0].set(title="Test-period predictions", ylabel="Return magnitude")
        axes[0, 0].legend()

        axes[0, 1].scatter(actual, predicted, s=8, alpha=0.25)
        low = float(np.nanmin([actual.min(), predicted.min()]))
        high = float(np.nanmax([actual.max(), predicted.max()]))
        axes[0, 1].plot([low, high], [low, high], "--", linewidth=1, label="Perfect")
        axes[0, 1].set(
            title=f"Actual vs predicted (correlation={np.corrcoef(actual, predicted)[0, 1]:.3f})",
            xlabel="Actual return magnitude", ylabel="Predicted return magnitude",
        )
        axes[0, 1].legend()

        axes[1, 0].hist(residual, bins=60, alpha=0.85)
        axes[1, 0].axvline(0, linestyle="--", linewidth=1)
        axes[1, 0].set(
            title=f"Prediction error (mean={residual.mean():.4f})",
            xlabel="Predicted minus actual", ylabel="Number of K-lines",
        )

        rolling_mae = pd.Series(np.abs(residual)).rolling(
            max(1, int(rolling_bars)), min_periods=max(1, int(rolling_bars) // 4)
        ).mean()
        axes[1, 1].plot(data["timestamp"], rolling_mae, linewidth=1)
        axes[1, 1].set(
            title=f"Rolling MAE ({rolling_bars:,} bars)",
            xlabel="Time", ylabel="Mean absolute error",
        )
        for axis in axes.flat:
            axis.grid(alpha=0.2)

        name = "gain" if target.endswith("gain_2d") else "loss"
        path = out / f"{model}_{name}_diagnostics.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        generated[f"{name}_diagnostics"] = str(path)
        if show:
            plt.show()
        else:
            plt.close(fig)

    if model == "xgboost" and artifact_path.exists() and top_features > 0:
        artifact = joblib.load(artifact_path)
        for target in TARGET_COLUMNS:
            pipeline = artifact.get("models", {}).get(target)
            estimator = getattr(pipeline, "named_steps", {}).get("model")
            importance = getattr(estimator, "feature_importances_", None)
            if importance is None:
                continue
            count = min(int(top_features), len(importance))
            order = np.argsort(importance)[-count:]
            names = np.asarray(artifact["features"], dtype=object)[order]
            values = np.asarray(importance)[order]
            fig, axis = plt.subplots(figsize=(11, max(6, count * 0.32)), constrained_layout=True)
            axis.barh(names, values)
            axis.set(title=f"XGBoost feature importance: {labels[target]}", xlabel="Importance")
            axis.grid(axis="x", alpha=0.2)
            name = "gain" if target.endswith("gain_2d") else "loss"
            path = out / f"xgboost_{name}_feature_importance.png"
            fig.savefig(path, dpi=160, bbox_inches="tight")
            generated[f"{name}_feature_importance"] = str(path)
            if show:
                plt.show()
            else:
                plt.close(fig)

    return generated


def plot_bsp_comparison_results(
    output_dir: str | Path,
    *,
    cohorts: list[str] | tuple[str, ...] | None = None,
    rolling_bars: int = 390,
    max_time_points: int = 2_000,
    top_features: int = 20,
    show: bool = True,
) -> Dict[str, str]:
    """Plot forecast-style diagnostics for every saved BSP comparison cohort.

    The comparison predictions are sparse for BSP cohorts, so ``rolling_bars``
    means rolling cohort observations (BSP signals), not consecutive market
    K-lines. For ``all_5min_technical`` it still corresponds to five-minute
    bars. Loss targets are displayed as positive downside magnitudes, matching
    :func:`plot_forecast_results`.
    """
    out = Path(output_dir)
    prediction_dir = out / "test_predictions"
    importance_path = out / "feature_importance_long.csv"
    if not prediction_dir.exists():
        raise FileNotFoundError(f"Missing comparison predictions: {prediction_dir}")

    available = sorted(path.stem for path in prediction_dir.glob("*.csv"))
    if not available:
        raise FileNotFoundError(f"No cohort prediction CSV files in: {prediction_dir}")
    selected = available if cohorts is None else [str(value) for value in cohorts]
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise ValueError(f"Unknown cohorts {unknown}. Available: {available}")
    if int(rolling_bars) < 1 or int(max_time_points) < 1:
        raise ValueError("rolling_bars and max_time_points must be positive")

    importance = (
        pd.read_csv(importance_path) if importance_path.exists() else pd.DataFrame()
    )
    plot_dir = out / "plots" / "forecast_diagnostics"
    plot_dir.mkdir(parents=True, exist_ok=True)
    generated: Dict[str, str] = {}
    labels = {
        "target_max_gain_2d": "Signed return to maximum future high",
        "target_max_loss_2d": "Signed return to minimum future low",
    }

    for cohort in selected:
        frame = pd.read_csv(
            prediction_dir / f"{cohort}.csv", parse_dates=["timestamp"]
        )
        direction = (
            "buy" if cohort.endswith("_buy")
            else "sell" if cohort.endswith("_sell") else "all bars"
        )
        for target in TARGET_COLUMNS:
            prediction_column = f"predicted_{target}"
            required = {"timestamp", target, prediction_column}
            missing = required.difference(frame.columns)
            if missing:
                raise KeyError(f"{cohort}.csv is missing: {sorted(missing)}")
            data = frame[["timestamp", target, prediction_column]].dropna().copy()
            if data.empty:
                continue
            actual = data[target].to_numpy(float)
            predicted = data[prediction_column].to_numpy(float)
            display_name = "signed return"
            if target == "target_max_loss_2d":
                actual, predicted = -actual, -predicted
                display_name = "downside magnitude"
            residual = predicted - actual
            correlation = (
                float(np.corrcoef(actual, predicted)[0, 1])
                if len(actual) > 1 and np.std(actual) > 0 and np.std(predicted) > 0
                else float("nan")
            )

            fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
            fig.suptitle(
                f"XGBoost comparison — {cohort} ({direction})\n{labels[target]}",
                fontsize=15,
            )
            step = max(1, int(np.ceil(len(data) / int(max_time_points))))
            shown = data.iloc[::step]
            axes[0, 0].plot(
                shown["timestamp"], actual[::step], label="Actual", linewidth=1
            )
            axes[0, 0].plot(
                shown["timestamp"], predicted[::step], label="Predicted", linewidth=1
            )
            axes[0, 0].set(
                title="Test-period predictions", ylabel=display_name.capitalize()
            )
            axes[0, 0].legend()

            axes[0, 1].scatter(actual, predicted, s=9, alpha=0.28)
            low = float(np.nanmin([actual.min(), predicted.min()]))
            high = float(np.nanmax([actual.max(), predicted.max()]))
            axes[0, 1].plot([low, high], [low, high], "--", linewidth=1, label="Perfect")
            axes[0, 1].set(
                title=f"Actual vs predicted (correlation={correlation:.3f})",
                xlabel=f"Actual {display_name}", ylabel=f"Predicted {display_name}",
            )
            axes[0, 1].legend()

            axes[1, 0].hist(residual, bins=min(60, max(15, len(residual) // 5)), alpha=0.85)
            axes[1, 0].axvline(0, linestyle="--", linewidth=1)
            axes[1, 0].set(
                title=f"Prediction error (mean={residual.mean():.4f})",
                xlabel="Predicted minus actual", ylabel="Observations",
            )

            rolling_window = min(int(rolling_bars), len(residual))
            rolling_mae = pd.Series(np.abs(residual)).rolling(
                rolling_window, min_periods=max(1, rolling_window // 4)
            ).mean()
            axes[1, 1].plot(data["timestamp"], rolling_mae, linewidth=1)
            observation_label = "bars" if cohort == "all_5min_technical" else "signals"
            axes[1, 1].set(
                title=f"Rolling MAE ({rolling_window:,} {observation_label})",
                xlabel="Time", ylabel="Mean absolute error",
            )
            for axis in axes.flat:
                axis.grid(alpha=0.2)

            short_target = "gain" if target.endswith("gain_2d") else "loss"
            path = plot_dir / f"{cohort}_{short_target}_diagnostics.png"
            fig.savefig(path, dpi=160, bbox_inches="tight")
            generated[f"{cohort}_{short_target}_diagnostics"] = str(path.resolve())
            if show:
                plt.show()
            else:
                plt.close(fig)

            if importance.empty or int(top_features) <= 0:
                continue
            rows = importance.loc[
                importance["cohort"].eq(cohort) & importance["target"].eq(target)
            ].copy()
            if rows.empty:
                continue
            value_column = (
                "normalized_gain" if "normalized_gain" in rows else "raw_gain"
            )
            rows = rows.nlargest(min(int(top_features), len(rows)), value_column)
            rows = rows.sort_values(value_column)
            fig, axis = plt.subplots(
                figsize=(11, max(6, len(rows) * 0.32)), constrained_layout=True
            )
            axis.barh(rows["feature"], rows[value_column], color="#2563EB")
            axis.set(
                title=f"XGBoost feature importance — {cohort}: {short_target}",
                xlabel="Normalized XGBoost gain",
            )
            axis.grid(axis="x", alpha=0.2)
            path = plot_dir / f"{cohort}_{short_target}_feature_importance.png"
            fig.savefig(path, dpi=160, bbox_inches="tight")
            generated[f"{cohort}_{short_target}_feature_importance"] = str(path.resolve())
            if show:
                plt.show()
            else:
                plt.close(fig)

    return generated


def plot_paired_bsp_model_results(
    output_dir: str | Path,
    *,
    cohorts: list[str] | tuple[str, ...] | None = None,
    paired_subdir: str = "paired_bsp_predictions",
    rolling_signals: int = 100,
    max_time_points: int = 2_000,
    show: bool = True,
) -> Dict[str, str]:
    """Plot all-bar versus BSP-model predictions on identical BSP test rows."""
    out = Path(output_dir)
    paired_dir = out / paired_subdir
    if not paired_dir.exists():
        raise FileNotFoundError(
            f"Missing paired predictions directory: {paired_dir}. Run "
            "create_paired_bsp_model_predictions() first."
        )
    suffix = "_paired_predictions.csv"
    available = sorted(
        path.name[:-len(suffix)] for path in paired_dir.glob(f"*{suffix}")
    )
    selected = available if cohorts is None else [str(value) for value in cohorts]
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise ValueError(f"Unknown paired cohorts {unknown}. Available: {available}")
    if not selected:
        raise ValueError("No paired BSP prediction files were found")
    if int(rolling_signals) < 1 or int(max_time_points) < 1:
        raise ValueError("rolling_signals and max_time_points must be positive")

    plot_dir = out / "plots" / "paired_model_comparison"
    plot_dir.mkdir(parents=True, exist_ok=True)
    generated: Dict[str, str] = {}

    for cohort in selected:
        frame = pd.read_csv(
            paired_dir / f"{cohort}{suffix}", parse_dates=["timestamp"]
        ).sort_values("timestamp")
        for target in TARGET_COLUMNS:
            all_column = f"all_bars_predicted_{target}"
            bsp_column = f"bsp_specific_predicted_{target}"
            required = {"timestamp", target, all_column, bsp_column}
            missing = required.difference(frame.columns)
            if missing:
                raise KeyError(f"{cohort} paired CSV is missing: {sorted(missing)}")
            data = frame[["timestamp", target, all_column, bsp_column]].dropna().copy()
            if data.empty:
                continue
            actual = data[target].to_numpy(float)
            all_prediction = data[all_column].to_numpy(float)
            bsp_prediction = data[bsp_column].to_numpy(float)
            display_label = "Signed return"
            if target == "target_max_loss_2d":
                actual = -actual
                all_prediction = -all_prediction
                bsp_prediction = -bsp_prediction
                display_label = "Downside magnitude"

            all_error = all_prediction - actual
            bsp_error = bsp_prediction - actual
            all_mae = float(np.mean(np.abs(all_error)))
            bsp_mae = float(np.mean(np.abs(bsp_error)))
            all_r2 = float(r2_score(actual, all_prediction))
            bsp_r2 = float(r2_score(actual, bsp_prediction))
            short_target = "gain" if target.endswith("gain_2d") else "loss"

            fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
            fig.suptitle(
                f"Paired model comparison — {cohort}: {short_target}\n"
                f"same {len(data):,} BSP test rows",
                fontsize=15,
            )

            step = max(1, int(np.ceil(len(data) / int(max_time_points))))
            shown = data.iloc[::step]
            axes[0, 0].plot(
                shown["timestamp"], actual[::step], label="Actual", linewidth=1.1,
                color="#334155",
            )
            axes[0, 0].plot(
                shown["timestamp"], all_prediction[::step],
                label="All-bars model", linewidth=1, alpha=0.85,
            )
            axes[0, 0].plot(
                shown["timestamp"], bsp_prediction[::step],
                label="BSP-specific model", linewidth=1, alpha=0.85,
            )
            axes[0, 0].set(title="Predictions on identical rows", ylabel=display_label)
            axes[0, 0].legend()

            axes[0, 1].scatter(
                actual, all_prediction, s=10, alpha=0.25,
                label=f"All bars (R²={all_r2:.3f})",
            )
            axes[0, 1].scatter(
                actual, bsp_prediction, s=10, alpha=0.25,
                label=f"BSP-specific (R²={bsp_r2:.3f})",
            )
            low = float(np.nanmin([actual.min(), all_prediction.min(), bsp_prediction.min()]))
            high = float(np.nanmax([actual.max(), all_prediction.max(), bsp_prediction.max()]))
            axes[0, 1].plot([low, high], [low, high], "--", color="#334155", linewidth=1)
            axes[0, 1].set(
                title="Actual versus predicted",
                xlabel=f"Actual {display_label.lower()}",
                ylabel=f"Predicted {display_label.lower()}",
            )
            axes[0, 1].legend()

            axes[1, 0].hist(
                all_error, bins=50, alpha=0.55,
                label=f"All bars (MAE={all_mae:.4f})",
            )
            axes[1, 0].hist(
                bsp_error, bins=50, alpha=0.55,
                label=f"BSP-specific (MAE={bsp_mae:.4f})",
            )
            axes[1, 0].axvline(0, linestyle="--", color="#334155", linewidth=1)
            axes[1, 0].set(
                title="Paired prediction-error distributions",
                xlabel="Predicted minus actual", ylabel="Signals",
            )
            axes[1, 0].legend()

            window = min(int(rolling_signals), len(data))
            minimum = max(1, window // 4)
            all_rolling = pd.Series(np.abs(all_error)).rolling(
                window, min_periods=minimum
            ).mean()
            bsp_rolling = pd.Series(np.abs(bsp_error)).rolling(
                window, min_periods=minimum
            ).mean()
            axes[1, 1].plot(
                data["timestamp"], all_rolling,
                label="All-bars model", linewidth=1,
            )
            axes[1, 1].plot(
                data["timestamp"], bsp_rolling,
                label="BSP-specific model", linewidth=1,
            )
            axes[1, 1].set(
                title=f"Rolling MAE ({window:,} BSP signals)",
                xlabel="Time", ylabel="Mean absolute error",
            )
            axes[1, 1].legend()
            for axis in axes.flat:
                axis.grid(alpha=0.2)

            path = plot_dir / f"{cohort}_{short_target}_paired_comparison.png"
            fig.savefig(path, dpi=160, bbox_inches="tight")
            generated[f"{cohort}_{short_target}"] = str(path.resolve())
            if show:
                plt.show()
            else:
                plt.close(fig)

    return generated


def plot_daily_forecast_review(
    output_dir: str | Path,
    *,
    model: str = "xgboost",
    start: str | None = None,
    end: str | None = None,
    dates: list[str] | tuple[str, ...] | None = None,
    max_days: int | None = None,
    show: bool = False,
    save_individual_pngs: bool = False,
) -> Dict[str, object]:
    """Create one review page per test day and combine all pages into a PDF.

    Each page contains the current day's five-minute candles, daily averages of
    the two-day prediction/target columns, and the next trading day's realized
    high/low returns measured from the current day's final close.
    """
    out = Path(output_dir)
    artifact = joblib.load(out / "model.joblib")
    predictions = pd.read_csv(out / "test_predictions.csv", parse_dates=["timestamp"])
    input_csv = Path(artifact["config"]["input_csv"])
    if not input_csv.exists():
        candidate = Path.cwd() / input_csv
        input_csv = candidate if candidate.exists() else input_csv
    prices = normalize_ohlcv(pd.read_csv(input_csv))

    prediction_columns = {
        target: f"{model}_{target}" for target in TARGET_COLUMNS
    }
    missing = [column for column in prediction_columns.values() if column not in predictions]
    if missing:
        raise ValueError(f"Missing {model!r} prediction columns: {missing}")

    predictions["trading_date"] = predictions["timestamp"].dt.normalize()
    prices["trading_date"] = prices["timestamp"].dt.normalize()
    selected_dates = pd.Index(predictions["trading_date"].drop_duplicates()).sort_values()
    if dates is not None:
        requested = pd.to_datetime(list(dates)).normalize()
        selected_dates = selected_dates[selected_dates.isin(requested)]
    if start is not None:
        selected_dates = selected_dates[selected_dates >= pd.Timestamp(start).normalize()]
    if end is not None:
        selected_dates = selected_dates[selected_dates <= pd.Timestamp(end).normalize()]
    if max_days is not None:
        selected_dates = selected_dates[:max(0, int(max_days))]
    if not len(selected_dates):
        raise ValueError("No test dates matched the requested daily plot range")

    all_price_dates = pd.Index(prices["trading_date"].drop_duplicates()).sort_values()
    png_dir = out / "daily_forecast_review"
    if save_individual_pngs:
        png_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out / f"{model}_daily_forecast_review.pdf"
    png_paths: list[str] = []
    plotted_days = 0

    with PdfPages(pdf_path) as pdf:
        for day in selected_dates:
            current = prices.loc[prices["trading_date"] == day].copy()
            daily_predictions = predictions.loc[predictions["trading_date"] == day]
            later_dates = all_price_dates[all_price_dates > day]
            if current.empty or daily_predictions.empty or not len(later_dates):
                continue
            next_day = later_dates[0]
            following = prices.loc[prices["trading_date"] == next_day]
            reference_close = float(current["close"].iloc[-1])

            predicted_gain = float(daily_predictions[prediction_columns["target_max_gain_2d"]].mean())
            predicted_loss = float(-daily_predictions[prediction_columns["target_max_loss_2d"]].mean())
            actual_gain = float(daily_predictions["target_max_gain_2d"].mean())
            actual_loss = float(-daily_predictions["target_max_loss_2d"].mean())
            next_gain = float(following["high"].max() / reference_close - 1.0)
            next_loss = float(max(0.0, 1.0 - following["low"].min() / reference_close))

            fig, (price_axis, summary_axis) = plt.subplots(
                2, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [2.1, 1]},
                constrained_layout=True,
            )
            fig.suptitle(
                f"Daily forecast review: {day.date()} | next trading day: {next_day.date()}",
                fontsize=15,
            )

            # Draw lightweight candlesticks without requiring mplfinance.
            x = np.arange(len(current))
            candle_width = 0.65
            for i, row in enumerate(current.itertuples(index=False)):
                rising = row.close >= row.open
                color = "tab:green" if rising else "tab:red"
                price_axis.vlines(i, row.low, row.high, color=color, linewidth=0.8)
                bottom = min(row.open, row.close)
                height = max(abs(row.close - row.open), reference_close * 1e-5)
                price_axis.add_patch(Rectangle(
                    (i - candle_width / 2, bottom), candle_width, height,
                    facecolor=color, edgecolor=color, alpha=0.8,
                ))
            tick_count = min(8, len(current))
            tick_positions = np.unique(np.linspace(0, len(current) - 1, tick_count).astype(int))
            tick_labels = current["timestamp"].iloc[tick_positions].dt.strftime("%H:%M")
            price_axis.set_xticks(tick_positions, tick_labels)
            price_axis.set(
                title=f"Five-minute K-lines — final close ${reference_close:.2f}",
                xlabel="Time", ylabel="Price",
            )
            price_axis.grid(alpha=0.2)

            categories = ["Maximum gain", "Maximum loss magnitude"]
            positions = np.arange(2)
            width = 0.24
            predicted_values = np.array([predicted_gain, predicted_loss]) * 100
            target_values = np.array([actual_gain, actual_loss]) * 100
            next_values = np.array([next_gain, next_loss]) * 100
            bars = [
                summary_axis.bar(positions - width, predicted_values, width, label="Average prediction (2-day)"),
                summary_axis.bar(positions, target_values, width, label="Average actual target (2-day)"),
                summary_axis.bar(positions + width, next_values, width, label="Next-day realized move"),
            ]
            for group in bars:
                summary_axis.bar_label(group, fmt="%.2f%%", padding=3)
            summary_axis.set_xticks(positions, categories)
            summary_axis.set(
                title="Daily forecast averages and following-day realized range",
                ylabel="Return magnitude (%)",
            )
            summary_axis.legend()
            summary_axis.grid(axis="y", alpha=0.2)

            pdf.savefig(fig, dpi=160, bbox_inches="tight")
            plotted_days += 1
            if save_individual_pngs:
                png_path = png_dir / f"{model}_{day.strftime('%Y-%m-%d')}.png"
                fig.savefig(png_path, dpi=160, bbox_inches="tight")
                png_paths.append(str(png_path))
            if show:
                plt.show()
            else:
                plt.close(fig)

    return {
        "pdf_path": str(pdf_path),
        "daily_png_paths": png_paths,
        "requested_days": int(len(selected_dates)),
        "plotted_days": plotted_days,
    }
