from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_time_statistics(
    df: pd.DataFrame,
) -> dict:
    """Calculate sampling and temporal statistics."""

    results = {}

    for trajectory_id, group in df.groupby(
        "trajectory_id"
    ):

        time = (
            group["time"]
            .to_numpy()
        )

        dt = np.diff(time)

        results[str(trajectory_id)] = {
            "n_points": int(
                len(group)
            ),
            "time_start": float(
                time.min()
            ),
            "time_end": float(
                time.max()
            ),
            "dt_mean": float(
                np.mean(dt)
            ),
            "dt_std": float(
                np.std(dt)
            ),
            "dt_min": float(
                np.min(dt)
            ),
            "dt_max": float(
                np.max(dt)
            ),
            "monotonic": bool(
                np.all(
                    np.diff(time) > 0
                )
            ),
        }

    return results


def autocorrelation(
    series: pd.Series,
    max_lag: int = 100,
) -> pd.Series:
    """Calculate normalized autocorrelation."""

    values = (
        series
        .dropna()
        .to_numpy()
    )

    if len(values) == 0:
        return pd.Series(
            dtype=float
        )

    values = (
        values
        - values.mean()
    )

    denominator = np.dot(
        values,
        values,
    )

    if denominator == 0:
        return pd.Series(
            [1.0],
            index=[0],
            name=series.name,
        )

    max_lag = min(
        max_lag,
        len(values) - 1,
    )

    acf = []

    for lag in range(
        max_lag + 1
    ):

        numerator = np.dot(
            values[:-lag]
            if lag
            else values,
            values[lag:]
            if lag
            else values,
        )

        acf.append(
            numerator / denominator
        )

    return pd.Series(
        acf,
        index=range(
            max_lag + 1
        ),
        name=series.name,
    )