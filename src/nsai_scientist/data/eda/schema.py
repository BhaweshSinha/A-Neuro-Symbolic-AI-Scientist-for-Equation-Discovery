from __future__ import annotations

import numpy as np
import pandas as pd


EXPECTED_STATE_COLUMNS = {
    "two_body": [
        "x1",
        "y1",
        "vx1",
        "vy1",
        "x2",
        "y2",
        "vx2",
        "vy2",
    ],
    "lorenz": [
        "x",
        "y",
        "z",
    ],
    "double_pendulum": [
        "theta1",
        "omega1",
        "theta2",
        "omega2",
    ],
    "lotka_volterra": [
        "prey",
        "predator",
    ],
    "sis": [
        "S",
        "I",
    ],
    "sirs": [
        "S",
        "I",
        "R",
    ],
}


def validate_basic_frame(
    df: pd.DataFrame,
    dataset: str,
) -> dict:
    """Validate the structure of a loaded P1 dataset."""

    if dataset not in EXPECTED_STATE_COLUMNS:
        raise ValueError(
            f"Unknown dataset: {dataset}"
        )

    required = [
        "trajectory_id",
        "time",
        *EXPECTED_STATE_COLUMNS[dataset],
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset}: missing required columns: "
            f"{missing}"
        )

    if df["trajectory_id"].isna().any():
        raise ValueError(
            f"{dataset}: trajectory_id contains NaN"
        )

    if df["time"].isna().any():
        raise ValueError(
            f"{dataset}: time contains NaN"
        )

    numeric_columns = [
        "time",
        *EXPECTED_STATE_COLUMNS[dataset],
    ]

    for column in numeric_columns:

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            raise ValueError(
                f"{dataset}: {column} "
                "must be numeric"
            )

    numeric_values = (
        df[numeric_columns]
        .to_numpy()
    )

    if not np.isfinite(
        numeric_values
    ).all():
        raise ValueError(
            f"{dataset}: NaN or Inf detected "
            "in numerical observations"
        )

    trajectory_sizes = (
        df.groupby(
            "trajectory_id"
        ).size()
    )

    monotonic = all(
        group["time"].is_monotonic_increasing
        for _, group
        in df.groupby(
            "trajectory_id"
        )
    )

    return {
        "dataset": dataset,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "trajectory_count": int(
            df["trajectory_id"].nunique()
        ),
        "trajectory_sizes": {
            str(k): int(v)
            for k, v
            in trajectory_sizes.items()
        },
        "time_monotonic": monotonic,
        "missing_values": int(
            df.isna().sum().sum()
        ),
        "numeric_state_columns":
            EXPECTED_STATE_COLUMNS[dataset],
    }