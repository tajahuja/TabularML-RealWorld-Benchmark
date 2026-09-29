import pandas as pd

from tabular_benchmark.data import Dataset, build_feature_table, load_dataset


def test_bike_source_profile_flags_target_components(tmp_path):
    rows = []
    for hour in range(36):
        rows.append({
            "instant": hour + 1, "dteday": f"2011-01-{1 + hour // 24:02d}", "hr": hour % 24,
            "season": 1, "yr": 0, "mnth": 1, "holiday": 0, "weekday": 6,
            "workingday": 0, "weathersit": 1, "temp": 0.2, "atemp": 0.2,
            "hum": 0.5, "windspeed": 0.1, "casual": 1, "registered": 2, "cnt": 3,
        })
    pd.DataFrame(rows).to_csv(tmp_path / "bike_sharing_hour.csv", index=False)
    dataset = load_dataset("bike_sharing", tmp_path)
    assert len(dataset.observations) == 36
    assert any("sum exactly to cnt" in item for item in dataset.profile["leakage_or_exclusion_notes"])
    frame, numeric, categorical = build_feature_table(dataset, "minimal")
    assert len(frame) == 35
    assert "casual" not in numeric + categorical
    assert "registered" not in numeric + categorical


def test_feature_weather_and_lag_values_are_from_prior_hour():
    times = pd.date_range("2020-01-01", periods=200, freq="h")
    observations = pd.DataFrame({
        "timestamp": times, "target": range(200), "temp": range(200), "hum": [0.5] * 200,
        "weathersit": [1] * 200,
    })
    dataset = Dataset("fixture", observations, {}, "target")
    frame, _, _ = build_feature_table(dataset, "engineered")
    row = frame.loc[frame.timestamp.eq(times[180])].iloc[0]
    assert row.weather_temp == 179
    assert row.target_lag_1h == 179
    assert row.target_lag_24h == 156
    assert row.target_lag_168h == 12
    assert row.target_trailing_mean_24h == sum(range(156, 180)) / 24


def test_feature_table_skips_nonconsecutive_forecast_targets():
    times = pd.date_range("2020-01-01", periods=30, freq="h").delete(10)
    observations = pd.DataFrame({"timestamp": times, "target": range(len(times)), "temp": 1.0})
    dataset = Dataset("fixture", observations, {}, "target")
    frame, _, _ = build_feature_table(dataset, "minimal")
    assert times[10] not in set(frame.timestamp)
    assert len(frame) == len(times) - 2
