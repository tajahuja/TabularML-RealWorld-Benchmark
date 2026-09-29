import numpy as np

from tabular_benchmark.metrics import performance_degradation, regression_metrics


def test_regression_metrics_are_computed_on_given_targets():
    metrics = regression_metrics(np.array([1.0, 3.0]), np.array([2.0, 2.0]))
    assert metrics["mae"] == 1.0
    assert metrics["rmse"] == 1.0
    assert metrics["r2"] == 0.0


def test_degradation_direction_is_consistent_for_error_and_r2():
    assert performance_degradation("mae", 10, 12)["degradation"] == 2
    assert performance_degradation("rmse", 10, 12)["relative_degradation_pct"] == 20
    assert np.isclose(performance_degradation("r2", 0.8, 0.6)["degradation"], 0.2)
    assert performance_degradation("r2", 0.6, 0.8)["degradation"] < 0
