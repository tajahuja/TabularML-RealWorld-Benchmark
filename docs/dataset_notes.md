# Dataset selection and notes

## Bike Sharing Dataset

The hourly UCI Bike Sharing data provide two years of hourly rental counts (17,379 rows, 17 columns; January 2011 through December 2012), with weather and calendar covariates and a direct regression target, `cnt`. Its repeating daily and annual cycles make time-based evaluation meaningful. The original `casual` and `registered` counts sum to `cnt`; using them would directly reveal the target, so both are excluded. The ordered `instant` column is excluded as an identifier. The raw file has no missing values or duplicate timestamps. UCI identifies the dataset as CC BY 4.0; see the [dataset page](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) and DOI 10.24432/C5W894.

## Metro Interstate Traffic Volume

The UCI data contain 48,204 records of hourly westbound I-94 traffic volume with weather and holiday fields, spanning October 2012 to September 2018. They offer a second application and a longer time horizon. The source contains 17 exact duplicate rows and repeated timestamps; 40,575 unique timestamps remain after grouping. The target agrees within each timestamp. `holiday` is blank in 48,143 raw rows and is retained as an explicit blank/unknown category. The lower-cardinality `weather_main` is used instead of `weather_description`. UCI identifies the dataset as CC BY 4.0; see the [dataset page](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) and DOI 10.24432/C5X60B.

## Selection limits

These datasets were selected because they are public, licensed for attribution-based use, have a numeric demand/volume target, and support a defensible hourly ordering with mixed calendar/weather predictors. They are two transportation datasets, so their results should not be generalized to tabular problems broadly. The Bike Sharing dataset is relatively short and old; Metro's repeated timestamp records and nearly all-missing holiday field need the stated normalization and interpretation. Raw source profiles are in [`dataset_profile.md`](dataset_profile.md) and machine-readable form in `results/dataset_profile.json`.
