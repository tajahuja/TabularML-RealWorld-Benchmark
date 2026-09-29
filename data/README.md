# Data access and attribution

The raw datasets are downloaded at run time from the official [UCI Machine Learning Repository](https://archive.ics.uci.edu/). They are intentionally excluded from Git; `data/raw/` is ignored so a clone stays small and each researcher obtains the source files directly.

Both UCI dataset pages identify the data as **CC BY 4.0**. Attribute the sources as follows when reusing the data:

- Fanaee-T, H. (2013). *Bike Sharing Dataset*. UCI Machine Learning Repository. [doi:10.24432/C5W894](https://doi.org/10.24432/C5W894).
- Hogue, J. (2019). *Metro Interstate Traffic Volume*. UCI Machine Learning Repository. [doi:10.24432/C5X60B](https://doi.org/10.24432/C5X60B).

The project code is MIT licensed; that license does not replace the datasets' attribution and share-alike requirements. Review the source dataset pages before redistribution.

Download both archives:

```bash
python -m tabular_benchmark.data --download
```

Then run the full benchmark:

```bash
python experiments/run_experiments.py
python experiments/generate_figures.py
```

The loader checks target agreement before grouping repeated Metro timestamps. It excludes Bike Sharing's `instant`, `casual`, and `registered` fields to avoid using a row identifier or target components as predictors. See [`docs/dataset_notes.md`](../docs/dataset_notes.md) for selection and preprocessing details.
