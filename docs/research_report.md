# Temporal Generalization in Tabular Regression: A Two-Dataset Benchmark

## 1. Abstract

This study compares five common tabular regression baselines on hourly Bike Sharing and Metro Interstate Traffic data. It evaluates random and chronological 70/15/15 holdouts across three seeds, and three expanding-window validations, under minimal and engineered feature sets. On Bike Sharing, all models had higher error on the final chronological test period than under random holdout; engineered features reduced chronological MAE substantially. On Metro traffic, chronological MAE was slightly lower for Random Forest, Ridge, and histogram gradient boosting than for their random-holdout estimates, while the MLP had 108.5% higher error. The engineered features helped most model families, but increased Metro MLP error. These results show that validation strategy and dataset context alter measured performance; they do not establish a universal ranking or causal effect of time shift. The MLP reached its iteration limit in some runs, and only two transportation datasets and three temporal windows were studied.

## 2. Research question

How does temporal distribution shift affect the relative performance and generalization of common tabular machine learning models, and how much can careful feature engineering and model selection mitigate that degradation?

Secondary questions ask how random and time-based evaluation differ, whether simple linear baselines remain competitive, how the engineered feature set affects scores, and whether model rankings persist across later windows.

## 3. Motivation

Many tabular benchmarks use random train/test partitions. For ordered demand data, that can mix neighboring observations and future periods across partitions. A held-out time block is closer to the question of how a model trained on historical records performs later. This small study makes the evaluation strategy explicit and inspects feature/target distribution changes alongside predictive metrics.

## 4. Related work

Yandex Research's TabReD analyzes design pitfalls in tabular deep-learning benchmarks and motivates dataset-aware, carefully specified comparisons. TabM studies an efficient ensemble-style tabular neural architecture. These works motivate the area; this repository has an independently defined scope and is not a reproduction of either work. The comparison uses scikit-learn's MLPRegressor, not TabM. Gradient-boosted trees and tree ensembles provide non-neural baselines, while Ridge and a mean predictor provide linear and naive references.

References:

1. Gorishniy, Y., Rubachev, I., and Babenko, A. *TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks*. arXiv:2406.19380, 2024. [arXiv](https://arxiv.org/abs/2406.19380), [official code](https://github.com/yandex-research/tabred).
2. Gorishniy, Y., Rubachev, I., and Babenko, A. *TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling*. arXiv:2410.24210, 2024. [arXiv](https://arxiv.org/abs/2410.24210), [official code](https://github.com/yandex-research/tabm).
3. Fanaee-T, H. (2013). *Bike Sharing Dataset*. UCI Machine Learning Repository. [doi:10.24432/C5W894](https://doi.org/10.24432/C5W894).
4. Hogue, J. (2019). *Metro Interstate Traffic Volume*. UCI Machine Learning Repository. [doi:10.24432/C5X60B](https://doi.org/10.24432/C5X60B).

## 5. Dataset selection

Both selected UCI datasets have real hourly targets, mixed weather/calendar covariates, clear timestamps, and CC BY 4.0 terms on their UCI pages. Bike Sharing contains 17,379 hourly rows over 2011–2012 and has seasonality and known target-component leakage risks. Metro has 48,204 source rows over 2012–2018 and complements the shorter bike series with a longer traffic history and sparse holiday field. After target-checked grouping, Metro has 40,575 unique timestamps. These transportation datasets do not represent the full variety of tabular ML tasks. Full source-column profiles are in `results/dataset_profile.json` and `docs/dataset_profile.md`.

## 6. Experimental setup

The supervised task is one-hour-ahead rolling regression. For hour *t*, predictors use calendar fields for *t* and weather measured at *t−1*. The engineered feature set adds hour, weekday, and annual sine/cosine encodings; target lags of 1, 24, and 168 hours; prior-only trailing means over 24 and 168 hours; and temperature × humidity. The target history assumes each immediately prior outcome is available by the next prediction. It does not model batch forecasting several hours or days ahead without newly observed labels.

Numeric missing values are median-imputed; categoricals use a constant missing token and one-hot encoding. Imputers, encoders, and scalers are fitted within each model pipeline and split. Ridge and MLP use standard-scaled numeric features. Numeric/category feature names and sample counts are saved in `results/features_*.json`.

## 7. Validation strategy

Random and chronological holdouts each use a 70% / 15% / 15% train/validation/test allocation. Random splitting is repeated with seeds 11, 29, and 47. The chronological split uses the earliest 70% for training, next 15% for validation, and latest 15% for final test, without shuffling. Expanding-window validation uses first 50%, 60%, and 70% for training and the following 10% for validation in each fold.

Random splitting is retained as a conventional comparator. It interleaves adjacent hours and later timestamps and has dense access to historical target values; it is not a forecast-deployment estimate. Temporal holdout and rolling windows are the main evidence for future-period behavior. The final test outcomes are reported, but the validation subset was not used for tuning because no search was conducted.

## 8. Models

Five CPU models were run with fixed settings: mean dummy; Ridge with alpha 10; Random Forest with 120 trees, minimum leaf size 3, and 0.8 max-features; histogram gradient boosting with 120 iterations, learning rate 0.08, 15 leaf nodes, and L2 regularization 1; and an MLP with hidden sizes (64, 32), Adam, early stopping, and a 200-iteration cap. No hyperparameter tuning was performed. XGBoost, LightGBM, CatBoost, and TabM were not executed.

The full matrix has 120 random/chronological holdout fits (2 datasets × 2 feature sets × 5 models × 2 split types × 3 seeds) plus 60 expanding-window fits. Each fit's metrics and timings are in `results/raw_results.csv` and `results/rolling_origin_results.csv`.

## 9. Feature engineering

Minimal predictors are calendar information and previous-hour weather. Engineered predictors add cyclical time, past target history and rolling means, and a temperature-humidity interaction. Calendar fields for prediction hour are known at prediction time; weather at prediction hour is excluded. Target lag values are strictly prior observations. Bike `instant`, `casual`, and `registered` are excluded; the latter two sum to the target. Metro's repeated timestamps are reduced only after verifying that the target is constant within each timestamp. Its high-cardinality `weather_description` field is omitted in favor of `weather_main`.

## 10. Distribution shift analysis

PSI compares train-reference bins/categories with validation/test distributions. For numerical features, the KS statistic is included as an effect-size summary only. Hourly dependence means independent-sample KS significance tests would be inappropriate, so the repository reports no p-values. Missingness changes are in percentage points. Target means, medians, upper quantiles, PSI, and descriptive KS statistics are in `results/target_shift.json`.

Bike Sharing's training target mean was 161.1 rentals; the chronological test mean was 235.3, a +74.2 difference, while the target PSI was 0.171. Metro's training target mean was 3,244.8 vehicles; the test mean was 3,358.6, a +113.8 difference, with target PSI 0.0165. These summaries demonstrate different shifts in the two target distributions, but do not attribute the model error differences to a particular covariate or cause.

## 11. Results

Chronological test MAE means across three seeds (target units):

| Dataset | Features | Dummy | Ridge | Random Forest | Hist. gradient boosting | MLP |
|---|---|---:|---:|---:|---:|---:|
| Bike Sharing | Minimal | 168.23 | 99.68 | 65.63 | 65.20 | 49.35 |
| Bike Sharing | Engineered | 168.23 | 47.73 | 35.88 | 33.95 | 30.74 |
| Metro traffic | Minimal | 1,741.62 | 558.79 | 236.62 | 278.84 | 338.16 |
| Metro traffic | Engineered | 1,741.62 | 270.94 | 137.79 | 154.68 | 372.37 |

The complete summary also reports RMSE, R², seed standard deviations, fit time, and inference time in `results/summary.csv`.

On Bike Sharing's engineered set, the random-to-temporal MAE increases were 18.0% for the dummy, 27.1% Ridge, 61.4% Random Forest, 47.9% histogram gradient boosting, and 54.0% MLP. On Metro, engineered Random Forest, Ridge, and histogram gradient boosting had temporal MAE 5.2%, 6.4%, and 7.6% below their random-holdout MAE; the dummy was +0.3% and MLP +108.5%. Thus temporal validation changed the measured gap by dataset and model. We do not interpret lower temporal error for three Metro models as proof that temporal shift is beneficial; the holdout period may differ in difficulty, and the random split estimates another information regime.

Relative to minimal chronological MAE, the engineered features reduced error for Ridge, Random Forest, and histogram gradient boosting in both datasets, and for MLP on Bike Sharing (37.7% lower); they increased Metro MLP error by 10.1%. MLP had the lowest Bike MAE, while Random Forest had the lowest Metro MAE. These measured wins do not establish a generally superior model.

In engineered expanding-window validation, mean MAE across three folds was 29.18 (MLP; fold SD 7.03) for Bike and 156.17 (Random Forest; fold SD 16.70) for Metro. These are three sequential windows rather than independent replicates.

## 12. Error analysis

The error pattern differs across datasets. Bike Sharing's later period has a substantially higher mean and median target than its training period, and all models lose random-holdout accuracy under chronological evaluation. Metro's target mean shifts less proportionally; the three tree/linear methods have similar or lower chronological error, but MLP error increases sharply. Feature engineering improves all tested Metro models except MLP. This suggests interactions between feature design, model family, and dataset; it does not identify causal mechanisms. No group-level residual audit (such as errors by hour or weather regime) was included, so subgroup failure modes remain to be analyzed.

## 13. Discussion

Random holdout materially understates final-period MAE for every Bike Sharing model. It does not do so uniformly for Metro. Model choice matters: Metro MLP is much less stable across the two strategies than the tree baselines, and MLP's incomplete convergence is an additional concern. Temporal holdout and rolling windows are more relevant for deployment, but only future periods sampled from these particular series are represented. Seed spread captures model randomness, not uncertainty over future time periods.

## 14. Limitations

See [`limitations.md`](limitations.md). In brief: two related domains, a one-step rolling task, only three temporal windows, no search or external test set, a small baseline set, and MLP fits that sometimes reached the iteration cap. Performance-time values are environment-specific. PSI/KS are descriptive and no significance claims are made. No broad or causal conclusion is justified.

## 15. Reproducibility

The run configuration is in `configs/experiments.yaml`; download sources, attribution, and commands are in `data/README.md`. To reproduce from a clone: install `python -m pip install -e ".[test]"`, run `python -m pytest`, download with `python -m tabular_benchmark.data --download`, run `python experiments/run_experiments.py`, then `python experiments/generate_figures.py`. The full raw output is checked in, excluding raw source datasets.

## 16. Future work

1. Add non-transport datasets from distinct domains with temporal structure.
2. Add a fully specified multi-step forecast protocol without realized intermediate targets.
3. Analyze residuals by hour, season, weather, and target range, and diagnose the MLP convergence issue.
4. Tune models only inside chronological training/validation windows and report a nested evaluation.
5. Include CatBoost/LightGBM/XGBoost where installation and split-safe tuning can be controlled; implement TabM only as a separate, faithful experiment.
6. Evaluate multiple historical cutoff dates and uncertainty using block-aware methods.

## 17. Researcher review

- **Question:** Clear, measurable, and aligned with a practical evaluation concern.
- **Design:** The chronological and expanding-window splits are appropriate for one-step rolling regression. The random split is useful as a comparator but should not be presented as equally realistic.
- **Leakage:** Main direct risks are handled: target components and row ID are excluded; target histories and rolling features use earlier observations; current-hour weather is withheld. Random-split comparisons still have an unrealistic interleaved information regime and are interpreted cautiously.
- **Baselines:** Reasonable CPU baselines spanning naive, linear, tree, boosting, and neural families. A tuned tree boosting baseline and a TabM experiment would broaden the comparison.
- **Claims:** Results support dataset-specific differences, not a universal model winner. Engineered features usually helped but not every model/data pairing.
- **Limitations:** Explicit, including dataset scope, model settings, no tuning, temporal dependence, and MLP convergence.
- **Next researcher requests:** Add more domains, multi-horizon evaluation, rolling-origin cutoff replications, residual subgroup audits, nested temporal tuning, and a verified TabM baseline.
- **Evidence of ML understanding:** Task-time information set, leakage exclusions, ordered validation, multiple seeds, feature ablation, shift analysis, directional metric interpretation, and limits on inferential claims.
- **Engineering versus research:** Modular package, configs, tests, and CI demonstrate reproducibility engineering. The research contribution in this repository is the controlled comparison and critical interpretation; it is not a new algorithm.
- **Before an interview:** Be ready to explain why the random split is a diagnostic rather than the deployment estimate, why one-step target lags are available, what Metro timestamp grouping changes, and why the Metro results do not imply a universal temporal penalty. Fix or qualify MLP convergence and add a multi-step setting before making stronger deployment claims.
