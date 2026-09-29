# Limitations and claim boundaries

- Only two transportation datasets are studied. The findings are dataset-specific and do not establish a general ranking across tabular tasks or domains.
- Each task is one-hour-ahead rolling regression with observed prior targets. It does not estimate a fixed multi-day forecast horizon where intermediate targets are unavailable.
- Random holdout is deliberately retained as a conventional comparator, but it interleaves time and benefits from dense target history; it is not the deployment estimate. Temporal holdout and expanding windows should carry more weight.
- Metro has repeated timestamps and 17 exact duplicate source rows. Grouping is deterministic and target-checked, but it changes the raw sample structure. `holiday` is blank for 48,143 of 48,204 rows.
- Models are fixed, modest scikit-learn baselines. There is no hyperparameter optimization, external test set, XGBoost/LightGBM/CatBoost comparison, or TabM implementation. The MLP hit its 200-iteration cap in some fits; its scores need a convergence-qualified reading.
- Fit and prediction times depend on this execution environment and are not hardware-independent benchmarks. No memory measurements were collected.
- The chronological split and expanding validation windows are a small number of cut points. Seed standard deviations do not measure uncertainty across independent future periods. No confidence intervals or significance claims are made.
- PSI depends on binning and category smoothing. KS statistics are descriptive and no iid p-values are used. Neither measure establishes causality or identifies which shift caused error changes.
- The measured random-versus-temporal difference combines time ordering, period-specific target difficulty, available history, and the sampled split. It should not be interpreted as a causal effect of “time” alone.

Do not claim novelty, a universal best model, that engineered features always help, that all methods degrade under time shift, or that this project reproduces TabReD/TabM.
