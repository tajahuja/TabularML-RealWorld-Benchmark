# Dataset profile

Profiles are computed from downloaded raw source files before forecast-example construction.
The modeling table groups repeated timestamps where the target agrees and requires a prior-hour row.

| Dataset | Raw rows | Raw columns | Target | Temporal coverage | Duplicate rows | Repeated-time rows |
|---|---:|---:|---|---|---:|---:|
| bike_sharing | 17379 | 17 | cnt | 2011-01-01T00:00:00 to 2012-12-31T23:00:00 | 0 | 0 |
| metro_traffic | 48204 | 9 | traffic_volume | 2012-10-02T09:00:00 to 2018-09-30T23:00:00 | 17 | 13074 |

## bike_sharing

Normalized unique-time rows: 17379.

Target distribution: mean 189.46, median 142.00, range 1.00 to 977.00.

Missing values by raw column:

- None

Exclusions and leakage controls:

- instant is an ordered row identifier and is excluded.
- casual and registered sum exactly to cnt; both target components are excluded.
- Current-hour weather is not used; predictors use the preceding observed hour.

## metro_traffic

Normalized unique-time rows: 40575.

Target distribution: mean 3259.82, median 3380.00, range 0.00 to 7280.00.

Missing values by raw column:

- `holiday`: 48143

Exclusions and leakage controls:

- Repeated timestamps are grouped; the traffic target is constant within each timestamp.
- weather_description is omitted because it duplicates weather_main at higher cardinality.
- Current-hour weather is not used; predictors use the preceding observed hour.
- Blank holiday values are retained as a distinct category after parsing.
