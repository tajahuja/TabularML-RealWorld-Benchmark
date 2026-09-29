import pandas as pd

from tabular_benchmark.shift import compare_feature_distributions, compare_target_distributions


def test_psi_is_near_zero_for_identical_distributions():
    frame = pd.DataFrame({"x": range(100), "category": ["a", "b"] * 50})
    rows = compare_feature_distributions(
        frame, frame.copy(), ["x"], ["category"], "fixture", "minimal", "same"
    )
    assert max(row["psi"] for row in rows) < 1e-12
    assert rows[0]["ks_statistic_descriptive_only"] == 0


def test_shift_metrics_detect_location_and_category_changes():
    reference = pd.DataFrame({"x": [0.0] * 100, "c": ["a"] * 100})
    changed = pd.DataFrame({"x": [10.0] * 100, "c": ["b"] * 100})
    rows = compare_feature_distributions(
        reference, changed, ["x"], ["c"], "fixture", "engineered", "train_to_test"
    )
    assert all(row["psi"] > 0 for row in rows)
    target = compare_target_distributions(reference.x, changed.x, "fixture")
    assert target["mean_change"] == 10.0


def test_missingness_change_is_reported_separately():
    reference = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    changed = pd.DataFrame({"x": [1.0, None, 3.0, None]})
    row = compare_feature_distributions(reference, changed, ["x"], [], "fixture", "minimal", "test")[0]
    assert row["missingness_change_percentage_points"] == 50.0
