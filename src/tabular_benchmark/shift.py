"""Descriptive shift measurements that do not assume independent time samples."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population stability index with bins defined only from reference values."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        center = float(np.median(expected))
        delta = max(abs(center) * 1e-6, 1e-6)
        edges = np.array([-np.inf, center - delta, center + delta, np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    exp_count = np.histogram(expected, bins=edges)[0].astype(float)
    act_count = np.histogram(actual, bins=edges)[0].astype(float)
    exp_share = (exp_count + 0.5) / (exp_count.sum() + 0.5 * len(exp_count))
    act_share = (act_count + 0.5) / (act_count.sum() + 0.5 * len(act_count))
    return float(np.sum((act_share - exp_share) * np.log(act_share / exp_share)))


def _categorical_psi(expected: Iterable[object], actual: Iterable[object]) -> float:
    expected_s = pd.Series(expected, dtype="object").fillna("<MISSING>").astype(str)
    actual_s = pd.Series(actual, dtype="object").fillna("<MISSING>").astype(str)
    categories = sorted(set(expected_s) | set(actual_s))
    if not categories:
        return float("nan")
    exp_count = expected_s.value_counts().reindex(categories, fill_value=0).to_numpy(float)
    act_count = actual_s.value_counts().reindex(categories, fill_value=0).to_numpy(float)
    exp_share = (exp_count + 0.5) / (exp_count.sum() + 0.5 * len(categories))
    act_share = (act_count + 0.5) / (act_count.sum() + 0.5 * len(categories))
    return float(np.sum((act_share - exp_share) * np.log(act_share / exp_share)))


def compare_feature_distributions(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    dataset: str,
    feature_set: str,
    comparison_name: str,
) -> list[dict[str, object]]:
    """Compute PSI, numeric KS effect size, and missingness changes per feature."""
    records: list[dict[str, object]] = []
    for column in numeric_columns + categorical_columns:
        ref = reference[column]
        cmp = comparison[column]
        missing_delta = float(100 * (cmp.isna().mean() - ref.isna().mean()))
        if column in numeric_columns:
            ref_values = pd.to_numeric(ref, errors="coerce").to_numpy(float)
            cmp_values = pd.to_numeric(cmp, errors="coerce").to_numpy(float)
            valid_ref, valid_cmp = ref_values[np.isfinite(ref_values)], cmp_values[np.isfinite(cmp_values)]
            ks_stat = float(ks_2samp(valid_ref, valid_cmp).statistic) if len(valid_ref) and len(valid_cmp) else float("nan")
            psi = _psi(ref_values, cmp_values)
            kind = "numeric"
        else:
            ks_stat = float("nan")
            psi = _categorical_psi(ref, cmp)
            kind = "categorical"
        records.append({
            "dataset": dataset, "feature_set": feature_set,
            "comparison": comparison_name, "feature": column,
            "feature_type": kind, "psi": psi,
            "ks_statistic_descriptive_only": ks_stat,
            "missingness_change_percentage_points": missing_delta,
            "reference_rows": len(reference), "comparison_rows": len(comparison),
        })
    return records


def compare_target_distributions(reference: pd.Series, comparison: pd.Series, dataset: str) -> dict[str, float | str]:
    """Summarize outcome shift without treating adjacent hours as independent tests."""
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(float)
    cmp = pd.to_numeric(comparison, errors="coerce").dropna().to_numpy(float)
    return {
        "dataset": dataset,
        "reference_mean": float(np.mean(ref)),
        "comparison_mean": float(np.mean(cmp)),
        "mean_change": float(np.mean(cmp) - np.mean(ref)),
        "reference_median": float(np.median(ref)),
        "comparison_median": float(np.median(cmp)),
        "median_change": float(np.median(cmp) - np.median(ref)),
        "reference_q90": float(np.quantile(ref, 0.90)),
        "comparison_q90": float(np.quantile(cmp, 0.90)),
        "q90_change": float(np.quantile(cmp, 0.90) - np.quantile(ref, 0.90)),
        "psi": _psi(ref, cmp),
        "ks_statistic_descriptive_only": float(ks_2samp(ref, cmp).statistic),
    }
