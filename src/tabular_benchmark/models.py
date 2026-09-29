"""Preprocessing pipelines and fixed, laptop-sized baseline estimators."""

from __future__ import annotations

from typing import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor


MODEL_NAMES = ["dummy_mean", "ridge", "random_forest", "hist_gradient_boosting", "mlp"]


def make_preprocessor(numeric_columns: Sequence[str], categorical_columns: Sequence[str], scale: bool) -> ColumnTransformer:
    numeric_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, list(numeric_columns)),
        ("categorical", categorical_pipe, list(categorical_columns)),
    ], remainder="drop", sparse_threshold=0)


def make_pipeline(
    model_name: str,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    seed: int,
) -> Pipeline:
    """Construct one baseline using predeclared settings and train-only transforms."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model_name}")
    scale = model_name in {"ridge", "mlp"}
    preprocess = make_preprocessor(numeric_columns, categorical_columns, scale)
    estimators = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=10.0),
        "random_forest": RandomForestRegressor(
            n_estimators=120, min_samples_leaf=3, max_features=0.8,
            random_state=seed, n_jobs=1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=120, learning_rate=0.08, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=seed,
        ),
        "mlp": MLPRegressor(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            alpha=0.001, batch_size="auto", learning_rate_init=0.001,
            max_iter=200, early_stopping=True, n_iter_no_change=15,
            random_state=seed,
        ),
    }
    return Pipeline([("preprocess", preprocess), ("model", estimators[model_name])])
