"""Run the fixed random, chronological, and rolling-origin experiment matrix."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .data import RAW_DIR, ROOT, build_feature_table, download_datasets, load_dataset
from .metrics import performance_degradation, regression_metrics
from .models import MODEL_NAMES, make_pipeline
from .shift import compare_feature_distributions, compare_target_distributions
from .splits import expanding_window_indices, train_validation_test_indices

SEEDS = (11, 29, 47)
FEATURE_SETS = ("minimal", "engineered")
DATASETS = ("bike_sharing", "metro_traffic")


def _measure_pipeline(
    dataset: str,
    feature_set: str,
    model_name: str,
    split_strategy: str,
    seed: int,
    fold: str,
    frame: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    test_idx: np.ndarray | None,
) -> dict[str, Any]:
    feature_columns = numeric + categorical
    x_train = frame.iloc[train_idx][feature_columns]
    y_train = frame.iloc[train_idx]["target"].to_numpy(float)
    x_val = frame.iloc[validation_idx][feature_columns]
    y_val = frame.iloc[validation_idx]["target"].to_numpy(float)
    model = make_pipeline(model_name, numeric, categorical, seed)
    start = perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = perf_counter() - start
    start = perf_counter()
    val_pred = model.predict(x_val)
    val_seconds = perf_counter() - start
    val_metrics = regression_metrics(y_val, val_pred)
    record: dict[str, Any] = {
        "dataset": dataset, "feature_set": feature_set, "model": model_name,
        "split_strategy": split_strategy, "seed": seed, "fold": fold,
        "train_rows": len(train_idx), "validation_rows": len(validation_idx),
        "test_rows": 0 if test_idx is None else len(test_idx),
        "fit_seconds": fit_seconds,
        "validation_inference_ms_per_row": 1000 * val_seconds / max(len(x_val), 1),
        "validation_mae": val_metrics["mae"],
        "validation_rmse": val_metrics["rmse"],
        "validation_r2": val_metrics["r2"],
    }
    if test_idx is not None:
        x_test = frame.iloc[test_idx][feature_columns]
        y_test = frame.iloc[test_idx]["target"].to_numpy(float)
        start = perf_counter()
        test_pred = model.predict(x_test)
        test_seconds = perf_counter() - start
        test_metrics = regression_metrics(y_test, test_pred)
        record.update({
            "test_inference_ms_per_row": 1000 * test_seconds / max(len(x_test), 1),
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
        })
    return record


def _shift_records(dataset: str, feature_set: str, frame: pd.DataFrame, numeric: list[str], categorical: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train, validation, test = train_validation_test_indices(frame, "temporal")
    columns = numeric + categorical
    reference = frame.iloc[train][columns]
    val_frame, test_frame = frame.iloc[validation], frame.iloc[test]
    feature_records = []
    for part_name, part in [("train_to_validation", val_frame), ("train_to_test", test_frame)]:
        feature_records.extend(compare_feature_distributions(
            reference, part[columns], numeric, categorical, dataset, feature_set, part_name
        ))
    target = {
        "dataset": dataset, "feature_set": feature_set,
        "train_to_validation": compare_target_distributions(
            frame.iloc[train]["target"], val_frame["target"], dataset
        ),
        "train_to_test": compare_target_distributions(
            frame.iloc[train]["target"], test_frame["target"], dataset
        ),
    }
    return feature_records, target


def _dataset_profile_markdown(profiles: list[dict[str, Any]]) -> str:
    lines = [
        "# Dataset profile", "",
        "Profiles are computed from downloaded raw source files before forecast-example construction.",
        "The modeling table groups repeated timestamps where the target agrees and requires a prior-hour row.", "",
        "| Dataset | Raw rows | Raw columns | Target | Temporal coverage | Duplicate rows | Repeated-time rows |",
        "|---|---:|---:|---|---|---:|---:|",
    ]
    for p in profiles:
        lines.append(
            f"| {p['dataset']} | {p['source_rows']} | {p['source_columns']} | {p['target']} | "
            f"{p['temporal_start']} to {p['temporal_end']} | {p['duplicate_rows']} | {p['rows_with_repeated_timestamp']} |"
        )
    for p in profiles:
        lines += ["", f"## {p['dataset']}", "", f"Normalized unique-time rows: {p['normalized_unique_timestamp_rows']}.", "",
                  f"Target distribution: mean {p['target_distribution']['mean']:.2f}, "
                  f"median {p['target_distribution']['median']:.2f}, range "
                  f"{p['target_distribution']['min']:.2f} to {p['target_distribution']['max']:.2f}.", "",
                  "Missing values by raw column:", ""]
        lines.extend([f"- `{key}`: {value}" for key, value in p["missing_values_by_column"].items() if value])
        if not any(p["missing_values_by_column"].values()):
            lines.append("- None")
        lines += ["", "Exclusions and leakage controls:", ""]
        lines.extend([f"- {note}" for note in p["leakage_or_exclusion_notes"]])
    return "\n".join(lines) + "\n"


def run_all(raw_dir: Path = RAW_DIR, output_dir: Path | None = None) -> dict[str, Path]:
    """Execute the preregistered CPU-sized matrix and write raw results."""
    output_dir = output_dir or ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    required_files = [raw_dir / "bike_sharing_hour.csv", raw_dir / "Metro_Interstate_Traffic_Volume.csv.gz"]
    if not all(path.exists() for path in required_files):
        download_datasets(raw_dir)
    datasets = [load_dataset(name, raw_dir) for name in DATASETS]
    profile_records = [dataset.profile for dataset in datasets]
    (output_dir / "dataset_profile.json").write_text(json.dumps(profile_records, indent=2) + "\n")
    (ROOT / "docs" / "dataset_profile.md").write_text(_dataset_profile_markdown(profile_records))

    result_rows: list[dict[str, Any]] = []
    rolling_rows: list[dict[str, Any]] = []
    shift_rows: list[dict[str, Any]] = []
    target_shift_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for feature_set in FEATURE_SETS:
            print(f"Preparing {dataset.name} / {feature_set} ...", flush=True)
            frame, numeric, categorical = build_feature_table(dataset, feature_set)
            if len(frame) < 100:
                raise ValueError(f"Too few forecast examples for {dataset.name}: {len(frame)}")
            feature_metadata = {
                "dataset": dataset.name, "feature_set": feature_set,
                "feature_rows": len(frame), "numeric_feature_count": len(numeric),
                "categorical_feature_count": len(categorical),
                "numeric_features": numeric, "categorical_features": categorical,
                "forecast_examples_start": frame["timestamp"].min().isoformat(),
                "forecast_examples_end": frame["timestamp"].max().isoformat(),
            }
            (output_dir / f"features_{dataset.name}_{feature_set}.json").write_text(
                json.dumps(feature_metadata, indent=2) + "\n"
            )
            feature_shift, target_shift = _shift_records(dataset.name, feature_set, frame, numeric, categorical)
            shift_rows.extend(feature_shift)
            target_shift_rows.append(target_shift)

            for strategy in ["random", "temporal"]:
                print(f"  {strategy} holdout", flush=True)
                for seed in SEEDS:
                    train_idx, validation_idx, test_idx = train_validation_test_indices(frame, strategy, seed)
                    for model_name in MODEL_NAMES:
                        result_rows.append(_measure_pipeline(
                            dataset.name, feature_set, model_name, strategy, seed, "holdout",
                            frame, numeric, categorical, train_idx, validation_idx, test_idx,
                        ))
            print("  rolling-origin folds", flush=True)
            for fold_number, (train_idx, validation_idx) in enumerate(expanding_window_indices(frame), start=1):
                for model_name in MODEL_NAMES:
                    rolling_rows.append(_measure_pipeline(
                        dataset.name, feature_set, model_name, "rolling_origin", SEEDS[fold_number - 1],
                        f"fold_{fold_number}", frame, numeric, categorical,
                        train_idx, validation_idx, None,
                    ))

    results = pd.DataFrame(result_rows)
    rolling = pd.DataFrame(rolling_rows)
    shifts = pd.DataFrame(shift_rows)
    target_shifts = pd.DataFrame(target_shift_rows)
    results.to_csv(output_dir / "raw_results.csv", index=False)
    results.to_json(output_dir / "raw_results.json", orient="records", indent=2, double_precision=8)
    rolling.to_csv(output_dir / "rolling_origin_results.csv", index=False)
    shifts.to_csv(output_dir / "feature_shift.csv", index=False)
    target_shifts.to_json(output_dir / "target_shift.json", orient="records", indent=2, double_precision=8)

    test_rows = results[results["split_strategy"].isin(["random", "temporal"])]
    metric_summary = test_rows.groupby(
        ["dataset", "feature_set", "model", "split_strategy"], as_index=False
    ).agg(
        test_mae_mean=("test_mae", "mean"), test_mae_std=("test_mae", "std"),
        test_rmse_mean=("test_rmse", "mean"), test_rmse_std=("test_rmse", "std"),
        test_r2_mean=("test_r2", "mean"), test_r2_std=("test_r2", "std"),
        fit_seconds_mean=("fit_seconds", "mean"),
        inference_ms_per_row_mean=("test_inference_ms_per_row", "mean"),
    )
    metric_summary.to_csv(output_dir / "summary.csv", index=False)
    comparison_rows = []
    pivot = metric_summary.pivot(
        index=["dataset", "feature_set", "model"], columns="split_strategy",
        values=["test_mae_mean", "test_rmse_mean", "test_r2_mean"],
    )
    for index, values in pivot.iterrows():
        dataset_name, feature_set, model_name = index
        if ("test_mae_mean", "random") not in values or ("test_mae_mean", "temporal") not in values:
            continue
        for metric, key in [("mae", "test_mae_mean"), ("rmse", "test_rmse_mean"), ("r2", "test_r2_mean")]:
            d = performance_degradation(metric, values[(key, "random")], values[(key, "temporal")])
            comparison_rows.append({"dataset": dataset_name, "feature_set": feature_set, "model": model_name, **d})
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "random_temporal_degradation.csv", index=False)
    return {
        "profile": output_dir / "dataset_profile.json",
        "raw_csv": output_dir / "raw_results.csv",
        "raw_json": output_dir / "raw_results.json",
        "summary": output_dir / "summary.csv",
        "rolling": output_dir / "rolling_origin_results.csv",
        "shift": output_dir / "feature_shift.csv",
        "target_shift": output_dir / "target_shift.json",
        "degradation": output_dir / "random_temporal_degradation.csv",
    }
