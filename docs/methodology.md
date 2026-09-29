# Methodology

## Research question

How does temporal distribution shift affect the relative performance and generalization of common tabular machine learning models, and how much can feature engineering and model selection mitigate that degradation?

## Prediction task and information set

Each example predicts demand or traffic in hour *t*. At prediction time, predictors include weather from the preceding observed hour and calendar fields known for *t*. The engineered feature set additionally uses target observations at lags 1, 24, and 168 hours; prior-only trailing means over 24 and 168 hours; cyclical hour, weekday, and annual encodings; and a temperature-by-humidity interaction. Missing lags remain missing and are imputed within each training pipeline. Target history is valid for this rolling one-step task, where the preceding outcome is observed before the next prediction. This is not a multi-step forecast and does not simulate recursively forecasting an entire future block without new observations.

The minimal set includes known calendar fields and prior-hour weather. Both settings use median numeric imputation, constant categorical imputation, and one-hot encoding, all fitted within each training split. Ridge and MLP numeric inputs are standardized. The original category values and feature list are saved in `results/features_*.json`.

## Data preparation

The hourly Bike Sharing table is already unique by timestamp. For Metro Interstate Traffic, exact duplicate source rows and repeated timestamps occur. The target is checked to be constant within each timestamp; repeated records are then reduced to one timestamp using the first target, median numeric weather, and modal categorical weather/holiday. This creates a unique hourly timeline while retaining the source-level duplication statistics in the profile. Examples are created only when the preceding hour is present. Gaps therefore do not cause a later observation to be treated as the immediately preceding hour.

Potential leakage controls include exclusion of Bike Sharing's `instant` identifier and `casual`/`registered` target components, use of prior-hour rather than current-hour weather, and strictly prior-only target lags/rolling means. `weather_description` is omitted from Metro because it duplicates the lower-cardinality `weather_main` category. Calendar fields for the target hour are available in advance.

## Splits

- **Random holdout:** 70% training, 15% validation, 15% test, sampled with each of three seeds (11, 29, 47). This is a conventional comparator and is not a deployment estimate for a time-ordered task.
- **Chronological holdout:** earliest 70% train, following 15% validation, final 15% test. There is no row shuffling.
- **Expanding-window validation:** train on the first 50%, 60%, and 70%; validate respectively on the next 10% windows. The windows are ordered in time and training expands between folds.

The random split and chronological split test different information regimes. In particular, random splits interleave neighboring hours and can place observations from later dates in training than some test observations. With lagged outcomes, random validation also benefits from the dense availability of observed historical targets. We therefore treat it as a diagnostic of conventional random evaluation, not as a leakage-free estimate of future deployment performance. Chronological holdout and expanding-window results are the primary temporal evidence.

## Models and fixed settings

The five CPU baselines are a training-mean dummy, Ridge (`alpha=10`), Random Forest (120 trees, leaf size 3), histogram gradient boosting (120 iterations, 15 leaf nodes), and an MLP (64, 32 hidden units; at most 200 iterations, early stopping). Random states are fixed per repetition. No hyperparameter search is performed. XGBoost, LightGBM, CatBoost, and TabM are not part of the measured comparison; this study does not claim a broad leaderboard or a reproduction of Yandex's TabReD or TabM work.

## Outcomes and shift measures

MAE and RMSE are reported in target units; R² is dimensionless. Fit wall time and prediction milliseconds per row are recorded from the local run, and should be treated as machine-specific descriptive measurements. Three-seed means and standard deviations summarize seed variation. The same seed repetitions are not independent temporal samples, so no confidence intervals or significance claims are made.

PSI is computed using bins/categories defined from the training reference. Numeric KS statistics are retained only as descriptive effect sizes: their usual p-values would rely on independence assumptions that hourly series do not satisfy. Missingness change is reported in percentage points. Target distribution summaries are reported separately. PSI and KS do not explain the cause of shift or establish its statistical significance.

## Reproduction

See the top-level README for install, test, download, experiment, and figure commands. `configs/experiments.yaml` records the planned matrix and model settings. `results/raw_results.csv` preserves each holdout fit; `results/rolling_origin_results.csv` preserves each ordered validation fold.
