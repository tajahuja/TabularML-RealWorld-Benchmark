import numpy as np
import pandas as pd

from tabular_benchmark.models import MODEL_NAMES, make_pipeline


def test_each_baseline_fits_a_mixed_feature_fixture():
    x = pd.DataFrame({"number": np.arange(80, dtype=float), "kind": ["a", "b"] * 40})
    y = np.arange(80, dtype=float) * 0.5
    for name in MODEL_NAMES:
        model = make_pipeline(name, ["number"], ["kind"], seed=11)
        model.fit(x.iloc[:60], y[:60])
        predictions = model.predict(x.iloc[60:])
        assert predictions.shape == (20,)
        assert np.isfinite(predictions).all()
