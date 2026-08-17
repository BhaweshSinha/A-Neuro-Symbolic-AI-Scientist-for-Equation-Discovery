from __future__ import annotations

import pandas as pd


def validate_dataset(
    df: pd.DataFrame,
    dataset: str,
) -> dict:
    """Validate one P2 dataset."""

    errors = []

    if "trajectory_id" not in df.columns:
        errors.append(
            "Missing trajectory_id"
        )

    if "time" not in df.columns:
        errors.append(
            "Missing time column"
        )

    if "trajectory_id" in df.columns:

        if df["trajectory_id"].isna().any():
            errors.append(
                "trajectory_id contains missing values"
            )

    if "time" in df.columns:

        if df["time"].isna().any():
            errors.append(
                "time contains missing values"
            )

    if (
        "trajectory_id" in df.columns
        and "time" in df.columns
    ):

        for trajectory_id, group in (
            df.groupby("trajectory_id")
        ):

            if not group["time"].is_monotonic_increasing:
                errors.append(
                    f"{trajectory_id}: "
                    "time is not monotonic"
                )

    numeric = df.select_dtypes(
        include="number"
    )

    if numeric.isna().any().any():
        errors.append(
            "Numeric columns contain NaN"
        )

    if numeric.isin(
        [float("inf"), float("-inf")]
    ).any().any():
        errors.append(
            "Numeric columns contain Inf"
        )

    return {
        "dataset": dataset,
        "status": (
            "PASS"
            if not errors
            else "FAIL"
        ),
        "errors": errors,
    }


def validate_all(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Validate all datasets."""

    results = [
        validate_dataset(
            df,
            dataset,
        )
        for dataset, df
        in datasets.items()
    ]

    return pd.DataFrame(
        [
            {
                "dataset": result["dataset"],
                "status": result["status"],
                "error_count": len(
                    result["errors"]
                ),
                "errors": "; ".join(
                    result["errors"]
                ),
            }
            for result in results
        ]
    )