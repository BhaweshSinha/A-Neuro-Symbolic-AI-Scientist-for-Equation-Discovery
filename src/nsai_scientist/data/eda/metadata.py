from __future__ import annotations

import json
from pathlib import Path


def load_trajectory_metadata(
    raw_root: Path,
    dataset: str,
    trajectory_id: str,
) -> dict:
    """Load metadata belonging to one trajectory."""

    metadata_path = (
        raw_root
        / dataset
        / f"{trajectory_id}.json"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    return metadata


def load_dataset_metadata(
    raw_root: Path,
    dataset: str,
) -> dict[str, dict]:
    """Load metadata for every trajectory in a dataset."""

    dataset_root = raw_root / dataset

    metadata_files = sorted(
        dataset_root.glob("trajectory_*.json")
    )

    if not metadata_files:
        raise FileNotFoundError(
            f"No metadata files found in {dataset_root}"
        )

    result = {}

    for path in metadata_files:

        trajectory_id = path.stem

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            result[trajectory_id] = json.load(
                file
            )

    return result


def load_all_metadata(
    raw_root: Path,
    datasets: list[str],
) -> dict[str, dict[str, dict]]:
    """Load metadata for all systems."""

    return {
        dataset: load_dataset_metadata(
            raw_root,
            dataset,
        )
        for dataset in datasets
    }