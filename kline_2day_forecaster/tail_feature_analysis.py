"""Training-safe feature profiling for unusually large gains and losses."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ForecastConfig
from .features import technical_warmup_bars
from .pipeline import _feature_columns, _select_model_rows, build_dataset
from .three_class_forecaster import _chronological_splits


def _config_from_artifact(saved: dict, output_dir: Path) -> ForecastConfig:
    allowed = {item.name for item in fields(ForecastConfig)}
    values = {key: value for key, value in saved.items() if key in allowed}
    values["output_dir"] = str(output_dir)
    values["target_mode"] = "exact_return"
    if "input_csv" not in values:
        raise ValueError("The saved artifact does not contain input_csv")
    return ForecastConfig(**values)


def _effect_table(
    part: pd.DataFrame,
    feature_columns: list[str],
    down_threshold: float,
    up_threshold: float,
    split: str,
) -> pd.DataFrame:
    returns = pd.to_numeric(part["target_exact_return"], errors="coerce")
    groups = {
        "loser": returns <= down_threshold,
        "normal": (returns > down_threshold) & (returns < up_threshold),
        "gainer": returns >= up_threshold,
    }
    rows = []
    for feature in feature_columns:
        values = pd.to_numeric(part[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        normal = values[groups["normal"]]
        scale = float(normal.std(ddof=0))
        normal_mean = float(normal.mean())
        if not np.isfinite(scale) or scale <= 1e-12 or not np.isfinite(normal_mean):
            continue
        row: Dict[str, Any] = {"split": split, "feature": feature}
        for name, mask in groups.items():
            sample = values[mask]
            mean, median = float(sample.mean()), float(sample.median())
            row[f"{name}_rows"] = int(sample.notna().sum())
            row[f"{name}_mean"] = mean
            row[f"{name}_median"] = median
            row[f"{name}_missing_rate"] = float(sample.isna().mean())
            row[f"{name}_effect"] = (mean - normal_mean) / scale if np.isfinite(mean) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_tail_feature_profiles(
    output_dir: str | Path,
    *,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    top_features: int = 30,
    distribution_features: int = 6,
    show: bool = True,
) -> Dict[str, Any]:
    """Find features shared by the largest next-session gainers and losers.

    Tail cutoffs and feature ranking are learned exclusively from the training
    split. The same cutoffs and ranking are then applied unchanged to validation
    and test data, which makes stability across periods visible without leakage.
    """
    if not 0 < lower_quantile < upper_quantile < 1:
        raise ValueError("Require 0 < lower_quantile < upper_quantile < 1")
    if top_features < 1 or distribution_features < 1:
        raise ValueError("top_features and distribution_features must be positive")
    out = Path(output_dir)
    candidates = (out / "three_class_model.joblib", out / "model.joblib")
    model_path = next((path for path in candidates if path.exists()), None)
    if model_path is None:
        raise FileNotFoundError("No three_class_model.joblib or model.joblib was found")
    artifact = joblib.load(model_path)
    config = _config_from_artifact(artifact.get("config", {}), out)

    data = build_dataset(config)
    warmup = max(config.min_history_bars, technical_warmup_bars(config))
    labeled = data.dropna(subset=["target_exact_return"]).iloc[warmup:].copy()
    labeled = _select_model_rows(labeled, config)
    splits = _chronological_splits(labeled, config)
    saved_features = list(artifact.get("features", []))
    features = [name for name in saved_features if name in labeled.columns]
    if not features:
        features = _feature_columns(
            labeled,
            use_standard_technical_features=config.use_standard_technical_features,
            use_chan_bsp_features=config.use_chan_bsp_features,
        )
    train_returns = pd.to_numeric(splits["train"]["target_exact_return"], errors="coerce")
    down_threshold = float(train_returns.quantile(lower_quantile))
    up_threshold = float(train_returns.quantile(upper_quantile))

    profiles = pd.concat([
        _effect_table(part, features, down_threshold, up_threshold, name)
        for name, part in splits.items()
    ], ignore_index=True)
    train_profile = profiles[profiles["split"] == "train"].copy()
    train_profile["importance_score"] = train_profile[["loser_effect", "gainer_effect"]].abs().max(axis=1)
    train_profile["common_tail_effect"] = (
        train_profile["loser_effect"] + train_profile["gainer_effect"]
    ) / 2
    train_profile["directional_effect"] = (
        train_profile["gainer_effect"] - train_profile["loser_effect"]
    )
    ranking = train_profile.sort_values("importance_score", ascending=False).reset_index(drop=True)

    ordered = ranking["feature"].head(top_features).tolist()
    wide = profiles[profiles["feature"].isin(ordered)].pivot(
        index="feature", columns="split", values=["loser_effect", "gainer_effect"]
    )
    columns = [(effect, split) for split in ("train", "validation", "test")
               for effect in ("loser_effect", "gainer_effect")]
    wide = wide.reindex(index=ordered, columns=pd.MultiIndex.from_tuples(columns))
    wide.columns = [f"{split}_{'loser' if effect == 'loser_effect' else 'gainer'}"
                    for effect, split in columns]
    wide = wide.reset_index()

    profile_path = out / "tail_feature_profiles_all_splits.csv"
    ranking_path = out / "tail_feature_ranking_train.csv"
    stability_path = out / "tail_feature_stability.csv"
    profiles.to_csv(profile_path, index=False)
    ranking.to_csv(ranking_path, index=False)
    wide.to_csv(stability_path, index=False)

    matrix = wide.set_index("feature").to_numpy(float)
    bound = max(0.25, float(np.nanpercentile(np.abs(matrix), 95))) if matrix.size else 1.0
    fig, ax = plt.subplots(figsize=(13, max(6, .38 * len(wide) + 2)), constrained_layout=True)
    image = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-bound, vmax=bound)
    ax.set_yticks(range(len(wide)), wide["feature"])
    ax.set_xticks(range(len(wide.columns) - 1), wide.columns[1:], rotation=35, ha="right")
    ax.set_title("Large-gainer / large-loser feature effects across time splits")
    fig.colorbar(image, ax=ax, label="Standard deviations from normal-group mean")
    heatmap_path = out / "tail_feature_effect_heatmap.png"
    fig.savefig(heatmap_path, dpi=160, bbox_inches="tight")
    if show: plt.show()
    else: plt.close(fig)

    selected = ordered[:distribution_features]
    rows = int(np.ceil(len(selected) / 2))
    dist_fig, axes = plt.subplots(rows, 2, figsize=(14, 4 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    train = splits["train"]
    train_return = pd.to_numeric(train["target_exact_return"], errors="coerce")
    masks = (train_return <= down_threshold,
             (train_return > down_threshold) & (train_return < up_threshold),
             train_return >= up_threshold)
    for ax, feature in zip(axes, selected):
        values = pd.to_numeric(train[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite = values.dropna()
        if finite.empty:
            continue
        low, high = finite.quantile([.01, .99])
        arrays = [values[mask].clip(low, high).dropna() for mask in masks]
        ax.boxplot(arrays, tick_labels=["loser", "normal", "gainer"], showfliers=False)
        ax.set_title(feature); ax.grid(axis="y", alpha=.2)
    for ax in axes[len(selected):]:
        ax.set_visible(False)
    dist_fig.suptitle("Training distributions for the strongest tail features", fontsize=15)
    distribution_path = out / "tail_feature_distributions.png"
    dist_fig.savefig(distribution_path, dpi=160, bbox_inches="tight")
    if show: plt.show()
    else: plt.close(dist_fig)

    counts = {
        name: {
            "loser": int((part["target_exact_return"] <= down_threshold).sum()),
            "normal": int(((part["target_exact_return"] > down_threshold) &
                           (part["target_exact_return"] < up_threshold)).sum()),
            "gainer": int((part["target_exact_return"] >= up_threshold).sum()),
        } for name, part in splits.items()
    }
    return {
        "thresholds": {"down": down_threshold, "up": up_threshold},
        "counts": counts,
        "top_features": ranking.head(top_features),
        "ranking_path": str(ranking_path),
        "all_profiles_path": str(profile_path),
        "stability_path": str(stability_path),
        "heatmap_path": str(heatmap_path),
        "distribution_plot_path": str(distribution_path),
    }
