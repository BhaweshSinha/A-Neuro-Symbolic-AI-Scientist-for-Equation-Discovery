from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_time_series(
    df: pd.DataFrame,
    dataset: str,
    output_dir,
    max_trajectories: int = 3,
):
    """Plot state variables against time."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    state_columns = [
        column
        for column in df.select_dtypes(
            include="number"
        ).columns
        if column != "time"
    ]

    trajectories = (
        df["trajectory_id"]
        .drop_duplicates()
        .tolist()
        [:max_trajectories]
    )

    for variable in state_columns:

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        for trajectory_id in trajectories:

            subset = df[
                df["trajectory_id"]
                == trajectory_id
            ]

            ax.plot(
                subset["time"],
                subset[variable],
                linewidth=1,
                label=trajectory_id,
            )

        ax.set_xlabel("time")
        ax.set_ylabel(variable)

        ax.set_title(
            f"{dataset} - {variable} vs time"
        )

        ax.legend(
            fontsize=8
        )

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        fig.savefig(
            output_dir
            / f"{dataset}_{variable}_time_series.png",
            dpi=150,
        )

        plt.close(fig)


def plot_distributions(
    df: pd.DataFrame,
    dataset: str,
    output_dir,
):
    """Create histograms for state variables."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    state_columns = [
        column
        for column in df.select_dtypes(
            include="number"
        ).columns
        if column != "time"
    ]

    for variable in state_columns:

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.hist(
            df[variable].dropna(),
            bins=50,
        )

        ax.set_xlabel(variable)
        ax.set_ylabel("Frequency")

        ax.set_title(
            f"{dataset} - distribution of {variable}"
        )

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        fig.savefig(
            output_dir
            / f"{dataset}_{variable}_distribution.png",
            dpi=150,
        )

        plt.close(fig)


def plot_correlation_matrix(
    matrix: pd.DataFrame,
    dataset: str,
    output_dir,
):
    """Plot a correlation matrix."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
    )

    fig.colorbar(
        image,
        ax=ax,
    )

    ax.set_xticks(
        range(len(matrix.columns))
    )

    ax.set_xticklabels(
        matrix.columns,
        rotation=90,
    )

    ax.set_yticks(
        range(len(matrix.index))
    )

    ax.set_yticklabels(
        matrix.index
    )

    ax.set_title(
        f"{dataset} - correlation matrix"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / f"{dataset}_correlation_matrix.png",
        dpi=150,
    )

    plt.close(fig)