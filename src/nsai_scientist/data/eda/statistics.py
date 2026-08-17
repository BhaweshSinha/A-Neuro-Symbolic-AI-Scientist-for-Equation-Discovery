from __future__ import annotations

import pandas as pd


def numeric_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate statistical summary for numerical observations."""

    numeric = df.select_dtypes(
        include="number"
    )

    summary = numeric.describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    ).T

    summary["missing"] = (
        numeric.isna().sum()
    )

    summary["unique"] = (
        numeric.nunique()
    )

    return summary