from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


DATASETS = [
    "two_body",
    "lorenz",
    "double_pendulum",
    "lotka_volterra",
    "sis",
    "sirs",
]


def discover_trajectory_files(dataset_root: str | Path) -> List[Path]:
    root = Path(dataset_root)

    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    files = sorted(root.glob("trajectory_*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No trajectory_*.csv files found in {root}"
        )

    return files


def derive_trajectory_id(path: str | Path) -> str:
    return Path(path).stem


def load_trajectory(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Empty trajectory file: {path}")

    if "t" not in df.columns:
        if "time" in df.columns:
            df = df.rename(columns={"time": "t"})
        elif "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "t"})
        else:
            raise ValueError(
                f"No time column found in {path}. "
                f"Expected one of: t, time, timestamp."
            )

    df = df.copy()

    df["trajectory_id"] = derive_trajectory_id(path)

    df["t"] = pd.to_numeric(df["t"], errors="coerce")

    if df["t"].isna().any():
        raise ValueError(f"Invalid time values in {path}")

    df = df.sort_values("t").reset_index(drop=True)

    return df


def load_dataset(dataset_root: str | Path) -> Dict[str, pd.DataFrame]:
    trajectories = {}

    for path in discover_trajectory_files(dataset_root):
        trajectory_id = derive_trajectory_id(path)
        trajectories[trajectory_id] = load_trajectory(path)

    return trajectories