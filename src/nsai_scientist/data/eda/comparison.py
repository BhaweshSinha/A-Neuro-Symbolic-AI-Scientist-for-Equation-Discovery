from __future__ import annotations

import pandas as pd


def compare_datasets(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Create a high-level comparison of all systems."""

    records = []

    for dataset, df in datasets.items():

        numeric = df.select_dtypes(
            include="number"
        )

        state_columns = [
            column
            for column in numeric.columns
            if column != "time"
        ]

        records.append(
            {
                "dataset": dataset,
                "rows": len(df),
                "trajectories":
                    df["trajectory_id"].nunique(),
                "state_variables":
                    len(state_columns),
                "time_start":
                    df["time"].min(),
                "time_end":
                    df["time"].max(),
                "missing_values":
                    int(
                        df.isna()
                        .sum()
                        .sum()
                    ),
            }
        )

    return pd.DataFrame(records)