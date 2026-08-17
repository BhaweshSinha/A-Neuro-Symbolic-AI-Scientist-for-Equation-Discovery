from __future__ import annotations

import pandas as pd


EXCLUDED_COLUMNS = [
    "trajectory_id",
]


def correlation_matrix(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate Pearson correlations."""

    numeric = df.select_dtypes(
        include="number"
    ).drop(
        columns=EXCLUDED_COLUMNS,
        errors="ignore",
    )

    return numeric.corr(
        method="pearson"
    )


def correlation_flags(
    matrix: pd.DataFrame,
    threshold: float = 0.90,
) -> pd.DataFrame:
    """Find strongly correlated variable pairs."""

    records = []

    columns = list(
        matrix.columns
    )

    for i, left in enumerate(
        columns
    ):

        for right in columns[
            i + 1:
        ]:

            value = matrix.loc[
                left,
                right,
            ]

            if (
                pd.notna(value)
                and abs(value)
                >= threshold
            ):

                records.append(
                    {
                        "variable_1":
                            left,
                        "variable_2":
                            right,
                        "correlation":
                            value,
                        "absolute_correlation":
                            abs(value),
                    }
                )

    return pd.DataFrame(
        records
    )