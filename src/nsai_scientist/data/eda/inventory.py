from __future__ import annotations

import pandas as pd


def create_inventory(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Create the P2 dataset inventory."""

    records = []

    for name, df in datasets.items():

        trajectory_sizes = (
            df.groupby(
                "trajectory_id"
            ).size()
        )

        records.append(
            {
                "dataset": name,
                "rows": len(df),
                "columns": len(df.columns),
                "trajectories":
                    df["trajectory_id"].nunique(),
                "time_start":
                    df["time"].min(),
                "time_end":
                    df["time"].max(),
                "time_points_per_trajectory_min":
                    trajectory_sizes.min(),
                "time_points_per_trajectory_max":
                    trajectory_sizes.max(),
                "time_points_per_trajectory_median":
                    trajectory_sizes.median(),
                "missing_values":
                    int(
                        df.isna()
                        .sum()
                        .sum()
                    ),
            }
        )

    return pd.DataFrame(records)