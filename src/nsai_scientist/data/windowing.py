from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class TrajectoryWindow:
    dataset: str
    trajectory_id: str
    start_index: int
    end_index: int
    data: pd.DataFrame


def create_windows(
    df: pd.DataFrame,
    dataset: str,
    trajectory_id: str,
    window_length: int,
    stride: int,
) -> List[TrajectoryWindow]:

    if window_length <= 0:
        raise ValueError("window_length must be positive")

    if stride <= 0:
        raise ValueError("stride must be positive")

    if len(df) < window_length:
        return []

    windows = []

    for start in range(0, len(df) - window_length + 1, stride):
        end = start + window_length

        window_df = df.iloc[start:end].copy()

        windows.append(
            TrajectoryWindow(
                dataset=dataset,
                trajectory_id=trajectory_id,
                start_index=start,
                end_index=end - 1,
                data=window_df,
            )
        )

    return windows


def create_dataset_windows(
    trajectories: Dict[str, pd.DataFrame],
    dataset: str,
    window_length: int,
    stride: int,
) -> List[TrajectoryWindow]:

    all_windows = []

    for trajectory_id, df in trajectories.items():
        windows = create_windows(
            df=df,
            dataset=dataset,
            trajectory_id=trajectory_id,
            window_length=window_length,
            stride=stride,
        )

        all_windows.extend(windows)

    return all_windows


def windows_to_array(
    windows: List[TrajectoryWindow],
    feature_columns: List[str],
) -> np.ndarray:

    if not windows:
        raise ValueError("No windows available")

    arrays = []

    for window in windows:
        values = window.data[feature_columns].to_numpy(dtype=np.float32)
        arrays.append(values)

    return np.stack(arrays)