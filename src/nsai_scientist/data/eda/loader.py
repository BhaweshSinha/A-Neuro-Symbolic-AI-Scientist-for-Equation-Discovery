from __future__ import annotations

from pathlib import Path

import pandas as pd


DATASETS = [
    "two_body",
    "lorenz",
    "double_pendulum",
    "lotka_volterra",
    "sis",
    "sirs",
]


def find_dataset_files(
    raw_root: Path,
    dataset: str,
) -> list[Path]:
    """Find all trajectory CSV files for one dataset."""

    dataset_root = raw_root / dataset

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_root}"
        )

    files = sorted(
        dataset_root.glob("trajectory_*.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No trajectory CSV files found in {dataset_root}"
        )

    return files


def load_dataset(
    raw_root: Path,
    dataset: str,
) -> pd.DataFrame:
    """
    Load all trajectory CSV files belonging to one system.

    The P1 CSV files do not contain a trajectory_id column.
    The trajectory ID is therefore derived from the filename,
    e.g. trajectory_000.csv -> trajectory_000.
    """

    files = find_dataset_files(
        raw_root,
        dataset,
    )

    frames = []

    for file in files:

        frame = pd.read_csv(file)

        # Derive trajectory ID from filename.
        trajectory_id = file.stem

        frame["trajectory_id"] = trajectory_id

        # Preserve source information for traceability.
        frame["_source_file"] = file.name
        frame["_dataset"] = dataset

        frames.append(frame)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    return combined


def load_all(
    raw_root: Path,
) -> dict[str, pd.DataFrame]:
    """Load all six P1 datasets."""

    return {
        dataset: load_dataset(
            raw_root,
            dataset,
        )
        for dataset in DATASETS
    }