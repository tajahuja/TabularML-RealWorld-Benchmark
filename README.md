# TabularML-RealWorld-Benchmark

**A reproducible study of temporal generalization in tabular regression.** This repository asks when random validation misrepresents future performance, whether a compact engineered feature set helps, and how common model families respond to later time periods.

> **Measured result, with scope:** On Bike Sharing, engineered-feature chronological MAE ranged from 30.74 (MLP) to 47.73 (Ridge), versus 19.96 to 37.55 under random holdout. Every model had higher MAE on the final chronological period. On Metro traffic, chronological MAE ranged from 137.79 (Random Forest) to 372.37 (MLP); three models had slightly lower chronological error than random error, while the MLP was substantially worse. Two datasets do not establish a general model ranking.

## Study at a glance

| Dataset | Raw observations | Target | Time span | Why it is included |
|---|---:|---|---|---|
| [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) | 17,379 hourly rows | Bike rentals (`cnt`) | 2011–2012 | Seasonal hourly demand; mixed weather and calendar features |
| [UCI Metro Interstate Traffic](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) | 48,204 source rows; 40,575 unique timestamps after grouping | Traffic volume | 2012–2018 | Longer time span, weather and holiday fields, repeated source timestamps |

Both dataset pages identify their data as CC BY 4.0. Raw files are downloaded from UCI and kept out of Git. Dataset attribution and details are in [`data/README.md`](data/README.md) and [`docs/dataset_notes.md`](docs/dataset_notes.md).

## Research question

> How does temporal distribution shift affect the relative performance and generalization of common tabular machine learning models, and how much can careful feature engineering and model selection mitigate that degradation?

## Experimental design

Each task predicts the target in hour *t* using calendar information known for *t* and weather observed in the preceding hour. The engineered setting adds cyclical hour/weekday/annual encodings, prior target lags and trailing means, and a temperature–humidity interaction. The lagged targets support **rolling one-step-ahead** prediction, where previous outcomes have been observed; this is not a multi-step forecast.

Five fixed CPU baselines are evaluated: mean dummy, Ridge, Random Forest, histogram gradient boosting, and a small MLP. The 120 holdout fits cross two datasets × two feature sets × five models × two split strategies × three seeds. Another 60 fits cover expanding-window validation (three ordered windows per dataset, feature set, and model). MAE, RMSE, R², fit time, and inference time are recorded. Raw per-fit records and summaries are in [`results/`](results/).

| Split | Training / validation / test | Use |
|---|---|---|
| Random | 70% / 15% / 15%, three seeds | Conventional, deliberately optimistic comparator for ordered data |
| Chronological | Earliest 70% / next 15% / latest 15% | Primary future-period holdout |
| Expanding window | Train first 50%, 60%, 70%; validate on next 10% | Check sensitivity across three later windows |

The random split interleaves time and benefits from dense historical target lags, so it is not a deployment estimate. Temporal and rolling results should carry more weight. More detail, including the leakage controls and the different information regimes, is in [`docs/methodology.md`](docs/methodology.md).

## Results

Chronological test MAE (mean across three seeds; lower is better):

| Dataset | Feature set | Mean dummy | Ridge | Random Forest | Hist. gradient boosting | MLP |
|---|---|---:|---:|---:|---:|---:|
| Bike Sharing | Minimal | 168.23 | 99.68 | 65.63 | 65.20 | 49.35 |
| Bike Sharing | Engineered | 168.23 | 47.73 | 35.88 | 33.95 | **30.74** |
| Metro traffic | Minimal | 1,741.62 | 558.79 | **236.62** | 278.84 | 338.16 |
| Metro traffic | Engineered | 1,741.62 | 270.94 | **137.79** | 154.68 | 372.37 |

Key observations from these runs:

- The engineered features reduced chronological MAE for Ridge, Random Forest, and histogram gradient boosting on both datasets. They reduced MLP error on Bike Sharing but increased it by 10.1% on Metro.
- Bike Sharing's engineered random-holdout MAE was lower than its chronological MAE for all five models (increases of about 18%–61%). This matches a meaningful target shift: the final test period's mean rental count was 235.3, against 161.1 in training.
- Metro does not show universal temporal degradation: engineered Random Forest, Ridge, and histogram gradient boosting had slightly lower chronological than random-holdout MAE. MLP error was 108.5% higher; its fits also reached the 200-iteration cap in some runs.
- In engineered expanding-window validation, mean MAE was 29.18 (MLP) on Bike Sharing and 156.17 (Random Forest) on Metro. Fold standard deviations were 7.03 and 16.70, respectively; only three adjacent time windows are represented.

These observations are descriptive for these datasets and periods. Seed standard deviations do not capture uncertainty across independent future periods; no inferential confidence intervals or significance claims are made. Full tables and shift caveats are in [`docs/research_report.md`](docs/research_report.md) and [`docs/limitations.md`](docs/limitations.md).

## Figures

All six figures are generated from the checked-in result tables.

1. [Model performance: random vs chronological](results/figures/random_vs_temporal_mae.png)
2. [Feature engineering effect](results/figures/feature_engineering_effect.png)
3. [Relative MAE change](results/figures/mae_degradation.png)
4. [Feature distribution shift (PSI)](results/figures/feature_shift_psi.png)
5. [Missingness change](results/figures/missingness_shift.png)
6. [Training time](results/figures/training_time.png)

## Reproduce

Python 3.11+ is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[test]"
python -m pytest
python -m tabular_benchmark.data --download
python experiments/run_experiments.py
python experiments/generate_figures.py
```

The full experiment is designed for a CPU workstation or laptop, takes several minutes on the execution environment used for this run, and uses fixed settings in [`configs/experiments.yaml`](configs/experiments.yaml). Runtime depends on hardware. The output includes a raw CSV/JSON record for each holdout fit, rolling validation records, dataset profiles, feature shift, target shift, and aggregate tables. No target labels or generated results are required from external services.

## Research context and boundaries

TabReD motivates careful, time-aware evaluation of tabular models; TabM motivates parameter-efficient deep tabular models. This repository defines its own dataset, task, splits, and baselines and **does not reproduce either paper**. TabM is not implemented here. The measured MLP is a conventional scikit-learn baseline, not a TabM substitute. Further limits and unsupported claims are listed in [`docs/limitations.md`](docs/limitations.md).

## Project layout

```text
configs/                 Dataset and experiment definitions
data/                    Data attribution and download instructions
docs/                    Methodology, profiles, report, limitations, review
experiments/             Experiment and figure entry points
results/                 Raw/aggregated results and six generated figures
src/tabular_benchmark/   Data, splits, models, metrics, shift, plots, runner
tests/                   Unit tests for preprocessing and evaluation logic
```

## License and citations

Project code is MIT licensed. Dataset terms and citations are described in [`data/README.md`](data/README.md). Related work and source links appear in the research report.
