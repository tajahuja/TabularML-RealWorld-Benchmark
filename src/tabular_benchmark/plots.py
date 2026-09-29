"""Generate publication-style plots from the checked-in experiment tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import ROOT

MODEL_LABELS = {
    "dummy_mean": "Mean dummy", "ridge": "Ridge", "random_forest": "Random forest",
    "hist_gradient_boosting": "Hist. gradient boosting", "mlp": "MLP",
}


def _style() -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "figure.titlesize": 12, "axes.spines.top": False,
        "axes.spines.right": False, "figure.dpi": 120,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def generate_figures(results_dir: Path | None = None) -> list[Path]:
    results_dir = results_dir or ROOT / "results"
    figure_dir = results_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    _style()
    raw = pd.read_csv(results_dir / "raw_results.csv")
    summary = pd.read_csv(results_dir / "summary.csv")
    shifts = pd.read_csv(results_dir / "feature_shift.csv")
    degradation = pd.read_csv(results_dir / "random_temporal_degradation.csv")
    paths: list[Path] = []

    # Random-vs-time performance; error bars are standard deviation across three seeds.
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=False)
    for ax, (dataset, feature_set) in zip(axes.flat, [
        ("bike_sharing", "minimal"), ("bike_sharing", "engineered"),
        ("metro_traffic", "minimal"), ("metro_traffic", "engineered"),
    ]):
        subset = summary[(summary.dataset == dataset) & (summary.feature_set == feature_set)]
        models = [m for m in MODEL_LABELS if m in set(subset.model)]
        x = np.arange(len(models))
        for offset, strategy, label, color in [
            (-0.10, "random", "Random holdout", "#2A6F97"),
            (0.10, "temporal", "Chronological holdout", "#C44536"),
        ]:
            rows = subset.set_index(["model", "split_strategy"])
            means = [rows.loc[(m, strategy), "test_mae_mean"] for m in models]
            stds = [rows.loc[(m, strategy), "test_mae_std"] for m in models]
            ax.errorbar(x + offset, means, yerr=stds, fmt="o", capsize=3, color=color, label=label)
        ax.set_xticks(x, [MODEL_LABELS[m] for m in models], rotation=25, ha="right")
        ax.set_ylabel("Test MAE (lower is better)")
        ax.set_title(f"{dataset.replace('_', ' ').title()} · {feature_set}")
        ax.grid(axis="y", alpha=0.22)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Random and chronological holdout error")
    fig.tight_layout()
    path = figure_dir / "random_vs_temporal_mae.png"
    fig.savefig(path); plt.close(fig); paths.append(path)

    # Feature engineering difference on the final chronological holdout.
    temporal = summary[summary.split_strategy == "temporal"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, dataset in zip(axes, ["bike_sharing", "metro_traffic"]):
        subset = temporal[temporal.dataset == dataset]
        pivot = subset.pivot(index="model", columns="feature_set", values="test_mae_mean")
        models = [m for m in MODEL_LABELS if m in pivot.index]
        x = np.arange(len(models))
        ax.plot(x, pivot.loc[models, "minimal"], "o-", color="#6C757D", label="Minimal")
        ax.plot(x, pivot.loc[models, "engineered"], "o-", color="#2A9D8F", label="Engineered")
        ax.set_xticks(x, [MODEL_LABELS[m] for m in models], rotation=25, ha="right")
        ax.set_title(dataset.replace("_", " ").title())
        ax.set_ylabel("Chronological test MAE")
        ax.grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False)
    fig.suptitle("Effect of the predeclared engineered feature set")
    fig.tight_layout()
    path = figure_dir / "feature_engineering_effect.png"
    fig.savefig(path); plt.close(fig); paths.append(path)

    # Relative MAE degradation, where positive means higher temporal error.
    mae = degradation[(degradation.metric == "mae") & (degradation.feature_set == "engineered")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=False)
    for ax, dataset in zip(axes, ["bike_sharing", "metro_traffic"]):
        pivot = mae[mae.dataset == dataset].pivot(index="model", columns="feature_set", values="relative_degradation_pct")
        # The degradation table contains both feature settings; select engineering explicitly above.
        vals = mae[mae.dataset == dataset].set_index("model")["relative_degradation_pct"]
        models = [m for m in MODEL_LABELS if m in vals.index]
        y = np.arange(len(models))
        colors = ["#C44536" if vals[m] > 0 else "#2A9D8F" for m in models]
        ax.barh(y, [vals[m] for m in models], color=colors, alpha=0.88)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y, [MODEL_LABELS[m] for m in models])
        ax.set_xlabel("MAE increase: (temporal − random) / random (%)")
        ax.set_title(dataset.replace("_", " ").title())
        ax.grid(axis="x", alpha=0.22)
    fig.suptitle("Relative MAE degradation on the engineered features")
    fig.tight_layout()
    path = figure_dir / "mae_degradation.png"
    fig.savefig(path); plt.close(fig); paths.append(path)

    # Largest measured covariate PSI values; PSI bins/categories are train-derived.
    engineered = shifts[(shifts.feature_set == "engineered") & (shifts.comparison == "train_to_test")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, dataset in zip(axes, ["bike_sharing", "metro_traffic"]):
        sub = engineered[engineered.dataset == dataset].nlargest(10, "psi").sort_values("psi")
        ax.barh(sub.feature, sub.psi, color="#457B9D")
        ax.set_xlabel("PSI (descriptive; higher means larger change)")
        ax.set_title(dataset.replace("_", " ").title())
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Largest train-to-test feature distribution changes")
    fig.tight_layout()
    path = figure_dir / "feature_shift_psi.png"
    fig.savefig(path); plt.close(fig); paths.append(path)

    # Missingness deltas are reported as percentage points, separate from PSI.
    missing = engineered[engineered.comparison == "train_to_test"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, dataset in zip(axes, ["bike_sharing", "metro_traffic"]):
        sub = missing[missing.dataset == dataset].copy()
        sub["abs_delta"] = sub["missingness_change_percentage_points"].abs()
        sub = sub.nlargest(10, "abs_delta").sort_values("missingness_change_percentage_points")
        colors = ["#C44536" if value > 0 else "#2A9D8F" for value in sub.missingness_change_percentage_points]
        ax.barh(sub.feature, sub.missingness_change_percentage_points, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Missingness change (percentage points)")
        ax.set_title(dataset.replace("_", " ").title())
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Train-to-test change in feature missingness")
    fig.tight_layout()
    path = figure_dir / "missingness_shift.png"
    fig.savefig(path); plt.close(fig); paths.append(path)

    # The MAE figure is enough for the central question; include training-time evidence separately.
    temporal_raw = raw[raw.split_strategy == "temporal"]
    fit = temporal_raw.groupby(["dataset", "model"], as_index=False).fit_seconds.mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, dataset in zip(axes, ["bike_sharing", "metro_traffic"]):
        sub = fit[fit.dataset == dataset].set_index("model")
        models = [m for m in MODEL_LABELS if m in sub.index]
        ax.bar([MODEL_LABELS[m] for m in models], [sub.loc[m, "fit_seconds"] for m in models], color="#6D597A")
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylabel("Fit wall time (seconds)")
        ax.set_title(dataset.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Mean fit time across three model seeds (one local run)")
    fig.tight_layout()
    path = figure_dir / "training_time.png"
    fig.savefig(path); plt.close(fig); paths.append(path)
    return paths
