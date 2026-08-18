from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def split_trajectory_ids(
    trajectory_ids: Iterable[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Dict[str, List[str]]:

    total = train_ratio + val_ratio + test_ratio

    if not np.isclose(total, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total}"
        )

    ids = list(trajectory_ids)

    if len(ids) < 3:
        raise ValueError(
            "At least 3 trajectories are required for "
            "train/validation/test splitting."
        )

    rng = np.random.default_rng(seed)

    shuffled = ids.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)

    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))

    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train:n_train + n_val]
    test_ids = shuffled[n_train + n_val:]

    if not test_ids:
        raise ValueError("Test split is empty.")

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }


def filter_windows_by_split(windows, split_ids):
    return [
        window
        for window in windows
        if window.trajectory_id in set(split_ids)
    ]