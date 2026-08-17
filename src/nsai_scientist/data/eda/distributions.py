from __future__ import annotations

import pandas as pd


def distribution_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate distribution statistics."""

    numeric = df.select_dtypes(
        include="number"
    )

    result = numeric.agg(
        [
            "min",
            "max",
            "mean",
            "std",
            "median",
        ]
    ).T

    result["skew"] = (
        numeric.skew()
    )

    result["kurtosis"] = (
        numeric.kurtosis()
    )

    return result