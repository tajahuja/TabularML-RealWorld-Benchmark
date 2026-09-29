import numpy as np
import pandas as pd

from tabular_benchmark.splits import expanding_window_indices, train_validation_test_indices


def test_temporal_split_is_ordered_disjoint_and_chronological():
    frame = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=100, freq="h")})
    train, validation, test = train_validation_test_indices(frame, "temporal")
    assert len(train) + len(validation) + len(test) == len(frame)
    assert train.max() < validation.min() < validation.max() < test.min()
    assert not set(train) & set(validation)
    assert not set(validation) & set(test)


def test_random_split_is_reproducible_and_disjoint():
    frame = pd.DataFrame({"x": range(101)})
    first = train_validation_test_indices(frame, "random", seed=29)
    second = train_validation_test_indices(frame, "random", seed=29)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert len(set(first[0]) | set(first[1]) | set(first[2])) == len(frame)


def test_expanding_windows_only_validate_on_future_rows():
    frame = pd.DataFrame({"x": range(100)})
    windows = expanding_window_indices(frame)
    assert len(windows) == 3
    prior_train = 0
    for train, validation in windows:
        assert train.max() < validation.min()
        assert len(train) > prior_train
        prior_train = len(train)
