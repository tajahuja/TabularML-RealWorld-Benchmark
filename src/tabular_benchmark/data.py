"""Dataset download, parsing, profiling, and causal one-step feature construction."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
DATASET_URLS = {
    "bike_sharing": "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
    "metro_traffic": "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip",
}


@dataclass(frozen=True)
class Dataset:
    """A normalized, unique-time dataset and its raw-source profile."""

    name: str
    observations: pd.DataFrame
    profile: dict[str, Any]
    target_name: str


def download_datasets(raw_dir: Path = RAW_DIR, timeout: int = 45) -> list[Path]:
    """Download the two small UCI archives and retain source data under data/raw."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for dataset_name, url in DATASET_URLS.items():
        request = Request(url, headers={"User-Agent": "TabularML-RealWorld-Benchmark/0.1"})
        with urlopen(request, timeout=timeout) as response:
            archive_bytes = response.read()
        with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
            for member in archive.namelist():
                base = Path(member).name
                if dataset_name == "bike_sharing" and base in {"hour.csv", "Readme.txt"}:
                    target = raw_dir / f"bike_sharing_{base}"
                elif dataset_name == "metro_traffic" and base == "Metro_Interstate_Traffic_Volume.csv.gz":
                    target = raw_dir / base
                else:
                    continue
                target.write_bytes(archive.read(member))
                saved.append(target)
    return saved


def _mode_or_missing(values: pd.Series) -> Any:
    modes = values.dropna().mode()
    return modes.iloc[0] if not modes.empty else np.nan


def _profile(
    name: str,
    raw: pd.DataFrame,
    timestamp: pd.Series,
    target: str,
    numeric: list[str],
    categorical: list[str],
    leakage: list[str],
) -> dict[str, Any]:
    valid_time = timestamp.dropna().sort_values()
    duplicates = int(raw.duplicated().sum())
    repeated_times = int(timestamp.duplicated(keep=False).sum())
    target_values = pd.to_numeric(raw[target], errors="coerce").dropna()
    gaps = valid_time.diff().dropna()
    return {
        "dataset": name,
        "source_rows": int(len(raw)),
        "source_columns": int(len(raw.columns)),
        "columns": list(raw.columns),
        "target": target,
        "task": "regression",
        "numerical_features": numeric,
        "categorical_features": categorical,
        "missing_values_by_column": {k: int(v) for k, v in raw.isna().sum().items()},
        "cardinality_by_categorical_feature": {
            k: int(raw[k].nunique(dropna=True)) for k in categorical
        },
        "duplicate_rows": duplicates,
        "rows_with_repeated_timestamp": repeated_times,
        "unique_timestamps": int(valid_time.nunique()),
        "temporal_start": valid_time.min().isoformat() if len(valid_time) else None,
        "temporal_end": valid_time.max().isoformat() if len(valid_time) else None,
        "median_observation_gap_hours": float(gaps.median().total_seconds() / 3600)
        if len(gaps)
        else None,
        "target_distribution": {
            "count": int(target_values.count()),
            "mean": float(target_values.mean()),
            "std": float(target_values.std(ddof=1)),
            "min": float(target_values.min()),
            "q25": float(target_values.quantile(0.25)),
            "median": float(target_values.median()),
            "q75": float(target_values.quantile(0.75)),
            "max": float(target_values.max()),
        },
        "leakage_or_exclusion_notes": leakage,
    }


def load_dataset(name: str, raw_dir: Path = RAW_DIR) -> Dataset:
    """Load an official UCI CSV and normalize it to one row per timestamp."""
    if name == "bike_sharing":
        path = raw_dir / "bike_sharing_hour.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run python -m tabular_benchmark.data --download")
        raw = pd.read_csv(path)
        timestamp = pd.to_datetime(raw["dteday"].astype(str) + " " + raw["hr"].astype(str).str.zfill(2) + ":00:00")
        numeric_weather = ["temp", "atemp", "hum", "windspeed"]
        categorical_weather = ["weathersit"]
        target = "cnt"
        leakage = [
            "instant is an ordered row identifier and is excluded.",
            "casual and registered sum exactly to cnt; both target components are excluded.",
            "Current-hour weather is not used; predictors use the preceding observed hour.",
        ]
        raw_profile_categorical = ["season", "holiday", "weekday", "workingday", "weathersit"]
        observations = pd.DataFrame({
            "timestamp": timestamp, "target": raw[target].astype(float),
            "holiday": raw["holiday"].astype(str),
        })
        for col in numeric_weather + categorical_weather:
            observations[col] = raw[col].to_numpy()
        observations = observations.groupby("timestamp", as_index=False).agg(
            {"target": "first", **{c: "median" for c in numeric_weather},
             **{c: _mode_or_missing for c in categorical_weather + ["holiday"]}}
        )
        profile = _profile(
            name, raw, timestamp, target, numeric_weather,
            raw_profile_categorical, leakage,
        )
    elif name == "metro_traffic":
        path = raw_dir / "Metro_Interstate_Traffic_Volume.csv.gz"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run python -m tabular_benchmark.data --download")
        raw = pd.read_csv(path, compression="gzip")
        timestamp = pd.to_datetime(raw["date_time"], errors="coerce")
        numeric_weather = ["temp", "rain_1h", "snow_1h", "clouds_all"]
        categorical_weather = ["weather_main"]
        target = "traffic_volume"
        leakage = [
            "Repeated timestamps are grouped; the traffic target is constant within each timestamp.",
            "weather_description is omitted because it duplicates weather_main at higher cardinality.",
            "Current-hour weather is not used; predictors use the preceding observed hour.",
            "Blank holiday values are retained as a distinct category after parsing.",
        ]
        raw_profile_categorical = ["holiday", "weather_main", "weather_description"]
        observations = pd.DataFrame({"timestamp": timestamp, "target": raw[target].astype(float)})
        for col in numeric_weather + categorical_weather + ["holiday"]:
            observations[col] = raw[col].to_numpy()
        grouped = observations.groupby("timestamp", as_index=False)
        targets_per_time = grouped["target"].nunique()["target"]
        if (targets_per_time > 1).any():
            raise ValueError("Traffic targets disagree for at least one repeated timestamp.")
        agg: dict[str, Any] = {"target": "first", **{c: "median" for c in numeric_weather}}
        agg.update({c: _mode_or_missing for c in categorical_weather + ["holiday"]})
        observations = grouped.agg(agg)
        profile = _profile(
            name, raw, timestamp, target, numeric_weather,
            raw_profile_categorical, leakage,
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")

    observations = observations.dropna(subset=["timestamp", "target"])
    observations = observations.sort_values("timestamp").drop_duplicates("timestamp", keep="first")
    observations = observations.reset_index(drop=True)
    profile["normalized_unique_timestamp_rows"] = int(len(observations))
    return Dataset(name, observations, profile, target)


def _calendar_features(timestamp: pd.Timestamp) -> dict[str, Any]:
    dow = int(timestamp.dayofweek)
    return {
        "hour": str(timestamp.hour),
        "weekday": str(dow),
        "month": str(timestamp.month),
        "year": str(timestamp.year),
        "is_weekend": str(int(dow >= 5)),
        "day_of_year": int(timestamp.dayofyear),
    }


def build_feature_table(dataset: Dataset, feature_set: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build strict one-hour-ahead examples using known calendar and past observations."""
    if feature_set not in {"minimal", "engineered"}:
        raise ValueError("feature_set must be 'minimal' or 'engineered'")
    obs = dataset.observations.set_index("timestamp").sort_index()
    timestamps = pd.DatetimeIndex(obs.index)
    previous_times = timestamps - pd.Timedelta(hours=1)
    previous = obs.reindex(previous_times)
    has_previous_hour = previous["target"].notna().to_numpy()
    timestamps = timestamps[has_previous_hour]
    current = obs.reindex(timestamps)
    previous = obs.reindex(timestamps - pd.Timedelta(hours=1))
    frame = pd.DataFrame({"timestamp": timestamps, "target": current["target"].to_numpy(float)})
    calendar = pd.DatetimeIndex(timestamps)
    dow = calendar.dayofweek
    frame["cal_hour"] = calendar.hour.astype(str)
    frame["cal_weekday"] = dow.astype(str)
    frame["cal_month"] = calendar.month.astype(str)
    frame["cal_year"] = calendar.year.astype(str)
    frame["cal_is_weekend"] = (dow >= 5).astype(int).astype(str)
    if "holiday" in current.columns:
        frame["cal_holiday"] = current["holiday"].map(
            lambda value: str(value) if pd.notna(value) else "NoHolidayOrUnknown"
        ).to_numpy()
    else:
        frame["cal_holiday"] = "NotProvided"
    for col in ["temp", "atemp", "hum", "windspeed", "rain_1h", "snow_1h", "clouds_all"]:
        if col in previous.columns:
            frame[f"weather_{col}"] = pd.to_numeric(previous[col], errors="coerce").to_numpy()
    for col in ["weathersit", "weather_main"]:
        if col in previous.columns:
            frame[f"weather_{col}"] = previous[col].map(
                lambda value: str(value) if pd.notna(value) else "Unknown"
            ).to_numpy()
    if feature_set == "engineered":
        hour = calendar.hour.to_numpy()
        day_of_week = calendar.dayofweek.to_numpy()
        day_of_year = calendar.dayofyear.to_numpy()
        frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        frame["weekday_sin"] = np.sin(2 * np.pi * day_of_week / 7)
        frame["weekday_cos"] = np.cos(2 * np.pi * day_of_week / 7)
        frame["annual_sin"] = np.sin(2 * np.pi * (day_of_year - 1) / 365.25)
        frame["annual_cos"] = np.cos(2 * np.pi * (day_of_year - 1) / 365.25)
        target_series = pd.to_numeric(obs["target"], errors="coerce")
        for lag in [1, 24, 168]:
            lag_times = timestamps - pd.Timedelta(hours=lag)
            frame[f"target_lag_{lag}h"] = target_series.reindex(lag_times).to_numpy(float)
        full_hourly_index = pd.date_range(timestamps.min(), timestamps.max(), freq="h")
        hourly_targets = target_series.reindex(full_hourly_index)
        for window in [24, 168]:
            trailing = hourly_targets.shift(1).rolling(window, min_periods=int(window * 0.75)).mean()
            frame[f"target_trailing_mean_{window}h"] = trailing.reindex(timestamps).to_numpy(float)
        if "weather_temp" in frame.columns and "weather_hum" in frame.columns:
            frame["temp_humidity_interaction"] = frame["weather_temp"] * frame["weather_hum"]
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    feature_columns = [c for c in frame.columns if c not in {"timestamp", "target"}]
    numeric = [c for c in feature_columns if pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in feature_columns if c not in numeric]
    return frame, numeric, categorical


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download official UCI source archives")
    args = parser.parse_args()
    if args.download:
        for downloaded in download_datasets():
            print(downloaded.relative_to(ROOT))
