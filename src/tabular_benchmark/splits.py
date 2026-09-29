"""Leakage-conscious random, chronological, and expanding-window splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def train_validation_test_indices(
    frame: pd.DataFrame, strategy: str, seed: int = 11,
    train_fraction: float = 0.70, validation_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return disjoint 70/15/15 index arrays; time strategy preserves order."""
    n = len(frame)
    if n < 20:
        raise ValueError("At least 20 observations are required for three-way splits.")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between zero and one.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Training and validation fractions must leave test observations.")
    indices = np.arange(n)
    n_train = int(np.floor(n * train_fraction))
    n_validation = int(np.floor(n * validation_fraction))
    if strategy == "temporal":
        return indices[:n_train], indices[n_train:n_train+n_validation], indices[n_train+n_validation:]
    if strategy != "random":
        raise ValueError("strategy must be 'random' or 'temporal'")
    train, remaining = train_test_split(
        indices, train_size=train_fraction, random_state=seed, shuffle=True
    )
    validation_share = validation_fraction / (1 - train_fraction)
    validation, test = train_test_split(
        remaining, train_size=validation_share, random_state=seed + 1, shuffle=True
    )
    return np.sort(train), np.sort(validation), np.sort(test)


def expanding_window_indices(
    frame: pd.DataFrame,
    windows: tuple[tuple[float, float], ...] = ((0.50, 0.60), (0.60, 0.70), (0.70, 0.80)),
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create expanding training windows followed by chronological validation blocks."""
    n = len(frame)
    results = []
    previous_train_stop = 0
    for train_end, validation_end in windows:
        train_stop = int(np.floor(n * train_end))
        validation_stop = int(np.floor(n * validation_end))
        if train_stop <= previous_train_stop or validation_stop <= train_stop:
            raise ValueError("Expanding windows must increase and contain observations.")
        results.append((np.arange(train_stop), np.arange(train_stop, validation_stop)))
        previous_train_stop = train_stop
    return results
