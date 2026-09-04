"""Plots for saved down/neutral/up return-classification runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


CLASS_NAMES = ("down", "neutral", "up")
CLASS_COLORS = ("tab:red", "0.55", "tab:green")


def _load_three_class_run(output_dir: str | Path) -> tuple[Path, pd.DataFrame, dict]:
    out = Path(output_dir)
    prediction_path = out / "three_class_test_predictions.csv"
    model_path = out / "three_class_model.joblib"
    if not prediction_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            "The output directory must contain three_class_test_predictions.csv "
            "and three_class_model.joblib"
        )
    frame = pd.read_csv(prediction_path, parse_dates=["timestamp"]).sort_values("timestamp")
    required = {
        "timestamp", "target_exact_return", "target_return_class",
        "predicted_return_class", "probability_down", "probability_neutral",
        "probability_up",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    return out, frame.reset_index(drop=True), joblib.load(model_path)


def _feature_importance(artifact: dict, limit: int) -> pd.DataFrame:
    features = list(artifact.get("features", []))
    pipeline = artifact.get("model")
    estimator = pipeline.named_steps.get("model") if hasattr(pipeline, "named_steps") else pipeline
    values = getattr(estimator, "feature_importances_", None)
    if values is None or len(values) != len(features):
        return pd.DataFrame(columns=["feature", "importance"])
    return (pd.DataFrame({"feature": features, "importance": values})
            .nlargest(limit, "importance").sort_values("importance"))


def plot_three_class_results(
    output_dir: str | Path,
    *,
    rolling_bars: int = 100,
    max_time_points: int = 2_000,
    top_features: int = 25,
    normalize_confusion: bool = True,
    show: bool = True,
) -> Dict[str, Any]:
    """Create a six-panel overview of a saved three-class test result."""
    if rolling_bars < 2 or max_time_points < 2 or top_features < 1:
        raise ValueError("rolling_bars/max_time_points must be >= 2 and top_features positive")
    out, frame, artifact = _load_three_class_run(output_dir)
    actual = frame["target_return_class"].astype(int).to_numpy()
    predicted = frame["predicted_return_class"].astype(int).to_numpy()
    correct = (actual == predicted).astype(float)
    frame["rolling_accuracy"] = pd.Series(correct).rolling(rolling_bars, min_periods=max(2, rolling_bars // 4)).mean()
    frame["rolling_balanced_accuracy"] = np.nan
    minimum = max(2, rolling_bars // 4)
    for end in range(minimum, len(frame) + 1):
        start = max(0, end - rolling_bars)
        recalls = []
        for value in range(3):
            mask = actual[start:end] == value
            if mask.any():
                recalls.append(float((predicted[start:end][mask] == value).mean()))
        frame.loc[end - 1, "rolling_balanced_accuracy"] = np.mean(recalls) if recalls else np.nan

    display = frame.iloc[-max_time_points:].copy()
    matrix = confusion_matrix(actual, predicted, labels=[0, 1, 2]).astype(float)
    if normalize_confusion:
        denominators = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, denominators, out=np.zeros_like(matrix), where=denominators != 0)

    fig, axes = plt.subplots(3, 2, figsize=(16, 14), constrained_layout=True)
    fig.suptitle("Three-class next-session return model — test results", fontsize=16)

    ax = axes[0, 0]
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if normalize_confusion else None)
    for row in range(3):
        for column in range(3):
            label = f"{matrix[row, column]:.1%}" if normalize_confusion else f"{matrix[row, column]:.0f}"
            ax.text(column, row, label, ha="center", va="center")
    ax.set_xticks(range(3), CLASS_NAMES); ax.set_yticks(range(3), CLASS_NAMES)
    ax.set(xlabel="Predicted class", ylabel="Actual class", title="Confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046)

    ax = axes[0, 1]
    positions = np.arange(3); width = 0.36
    actual_counts = np.bincount(actual, minlength=3) / len(actual)
    predicted_counts = np.bincount(predicted, minlength=3) / len(predicted)
    ax.bar(positions - width / 2, actual_counts, width, label="Actual")
    ax.bar(positions + width / 2, predicted_counts, width, label="Predicted")
    ax.set_xticks(positions, CLASS_NAMES); ax.set(title="Class distribution", ylabel="Share")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}"); ax.legend(); ax.grid(axis="y", alpha=.2)

    ax = axes[1, 0]
    ax.plot(display["timestamp"], display["rolling_accuracy"], label="Accuracy", linewidth=1.2)
    ax.plot(display["timestamp"], display["rolling_balanced_accuracy"], label="Balanced accuracy", linewidth=1.2)
    ax.axhline(1 / 3, color="black", linestyle="--", alpha=.5, label="1/3 reference")
    ax.set(title=f"Rolling quality ({rolling_bars} sampled bars)", ylabel="Score")
    ax.legend(); ax.grid(alpha=.2)

    ax = axes[1, 1]
    for name, color in zip(CLASS_NAMES, CLASS_COLORS):
        ax.plot(display["timestamp"], display[f"probability_{name}"], color=color, alpha=.8, linewidth=.9, label=name)
    ax.set(title="Predicted class probabilities", ylabel="Probability", ylim=(0, 1))
    ax.legend(ncol=3); ax.grid(alpha=.2)

    ax = axes[2, 0]
    ax.plot(display["timestamp"], display["target_exact_return"] * 100, color="black", linewidth=.8)
    thresholds = artifact.get("thresholds", {})
    if "down" in thresholds: ax.axhline(100 * thresholds["down"], color="tab:red", linestyle="--")
    if "up" in thresholds: ax.axhline(100 * thresholds["up"], color="tab:green", linestyle="--")
    wrong = display[display["target_return_class"] != display["predicted_return_class"]]
    ax.scatter(wrong["timestamp"], wrong["target_exact_return"] * 100, s=10, color="tab:orange", label="Wrong class", zorder=3)
    ax.set(title="Actual return and classification errors", ylabel="Next-session return (%)")
    ax.legend(); ax.grid(alpha=.2)

    ax = axes[2, 1]
    importance = _feature_importance(artifact, top_features)
    if importance.empty:
        ax.text(.5, .5, "Feature importance unavailable", ha="center", va="center")
    else:
        ax.barh(importance["feature"], importance["importance"], color="tab:blue")
    ax.set(title=f"Top {min(top_features, len(importance))} feature importances", xlabel="XGBoost importance")

    png_path = out / "three_class_results.png"
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    rolling_path = out / "three_class_rolling_quality.csv"
    frame[["timestamp", "target_exact_return", "target_class_name", "predicted_class_name",
           "rolling_accuracy", "rolling_balanced_accuracy"]].to_csv(rolling_path, index=False)
    if show: plt.show()
    else: plt.close(fig)
    return {"figure": fig, "plot_path": str(png_path), "rolling_quality_path": str(rolling_path)}


def plot_three_class_tail_quality(
    output_dir: str | Path,
    *,
    rolling_bars: int = 100,
    show: bool = True,
) -> Dict[str, Any]:
    """Plot tail detection separately from correct up/down tail direction."""
    if rolling_bars < 2:
        raise ValueError("rolling_bars must be >= 2")
    out, frame, _ = _load_three_class_run(output_dir)
    actual = frame["target_return_class"].astype(int)
    predicted = frame["predicted_return_class"].astype(int)
    actual_tail, predicted_tail = actual.ne(1), predicted.ne(1)
    detected = (actual_tail & predicted_tail).astype(float)
    direction_correct = (actual.eq(predicted) & actual_tail & predicted_tail).astype(float)
    min_periods = max(2, rolling_bars // 4)
    actual_count = actual_tail.astype(float).rolling(rolling_bars, min_periods=min_periods).sum()
    predicted_count = predicted_tail.astype(float).rolling(rolling_bars, min_periods=min_periods).sum()
    true_positive = detected.rolling(rolling_bars, min_periods=min_periods).sum()
    direction_hits = direction_correct.rolling(rolling_bars, min_periods=min_periods).sum()
    recall = true_positive / actual_count.replace(0, np.nan)
    precision = true_positive / predicted_count.replace(0, np.nan)
    direction_accuracy = direction_hits / true_positive.replace(0, np.nan)

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True, constrained_layout=True)
    axes[0].plot(frame["timestamp"], actual_tail.rolling(rolling_bars, min_periods=min_periods).mean(), label="Actual tail rate")
    axes[0].plot(frame["timestamp"], predicted_tail.rolling(rolling_bars, min_periods=min_periods).mean(), label="Predicted tail rate")
    axes[0].set(title=f"Rolling tail frequency ({rolling_bars} sampled bars)", ylabel="Rate")
    axes[0].legend(); axes[0].grid(alpha=.2)
    axes[1].plot(frame["timestamp"], recall, label="Tail recall")
    axes[1].plot(frame["timestamp"], precision, label="Tail precision")
    axes[1].plot(frame["timestamp"], direction_accuracy, label="Correct direction among detected tails")
    axes[1].set(title="Rolling tail prediction quality", ylabel="Score", xlabel="Time", ylim=(0, 1))
    axes[1].legend(); axes[1].grid(alpha=.2)
    png_path = out / "three_class_tail_quality.png"
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    if show: plt.show()
    else: plt.close(fig)
    return {"figure": fig, "plot_path": str(png_path)}
