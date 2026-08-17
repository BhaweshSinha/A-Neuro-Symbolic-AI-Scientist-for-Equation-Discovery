from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


PHASE_PAIRS = {
    "two_body": [
        ("x1", "y1"),
        ("x2", "y2"),
    ],
    "lorenz": [
        ("x", "y"),
        ("y", "z"),
        ("x", "z"),
    ],
    "double_pendulum": [
        ("theta1", "theta2"),
        ("omega1", "omega2"),
    ],
    "lotka_volterra": [
        ("prey", "predator"),
    ],
    "sis": [
        ("S", "I"),
    ],
    "sirs": [
        ("S", "I"),
        ("I", "R"),
        ("R", "S"),
    ],
}


def plot_phase_portraits(
    df: pd.DataFrame,
    dataset: str,
    output_dir,
    max_trajectories: int = 10,
):
    """Generate phase-space plots."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairs = PHASE_PAIRS.get(
        dataset,
        [],
    )

    trajectory_ids = (
        df["trajectory_id"]
        .drop_duplicates()
        .tolist()[
            :max_trajectories
        ]
    )

    for x_var, y_var in pairs:

        if (
            x_var not in df.columns
            or y_var not in df.columns
        ):
            continue

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        for trajectory_id in (
            trajectory_ids
        ):

            subset = df[
                df["trajectory_id"]
                == trajectory_id
            ]

            ax.plot(
                subset[x_var],
                subset[y_var],
                alpha=0.7,
                linewidth=1,
            )

        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)

        ax.set_title(
            f"{dataset}: "
            f"{y_var} vs {x_var}"
        )

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        fig.savefig(
            output_dir
            / (
                f"{dataset}_"
                f"{x_var}_"
                f"{y_var}.png"
            ),
            dpi=150,
        )

        plt.close(fig)