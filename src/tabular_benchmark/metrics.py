"""Regression metrics and direction-aware random-to-temporal comparisons."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return the preregistered regression metrics on identical observation sets."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def performance_degradation(metric: str, random_value: float, temporal_value: float) -> dict[str, float | str]:
    """Orient differences so positive means worse temporal performance."""
    if metric in {"mae", "rmse"}:
        delta = temporal_value - random_value
        relative = 100 * delta / max(abs(random_value), 1e-12)
        interpretation = "positive values mean temporal error increased"
    elif metric == "r2":
        delta = random_value - temporal_value
        relative = float("nan")
        interpretation = "positive values mean temporal R-squared decreased; delta is in score points"
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    return {
        "metric": metric,
        "degradation": float(delta),
        "relative_degradation_pct": float(relative),
        "interpretation": interpretation,
    }
