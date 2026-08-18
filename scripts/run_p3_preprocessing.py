from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from nsai_scientist.data.loaders import load_dataset
from nsai_scientist.data.windowing import (
    create_dataset_windows,
)
from nsai_scientist.data.splitting import (
    split_trajectory_ids,
    filter_windows_by_split,
)
from nsai_scientist.data.scalers import (
    InvertibleScaler,
)
from nsai_scientist.data.graph_builder import (
    build_graph,
)
from nsai_scientist.data.validation import (
    validate_trajectory,
    validate_window,
    validate_graph,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS = [
    "two_body",
    "lorenz",
    "double_pendulum",
    "lotka_volterra",
    "sis",
    "sirs",
]


def load_config(dataset: str):
    path = (
        PROJECT_ROOT
        / "configs"
        / "data"
        / f"{dataset}.yaml"
    )

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
            default=str,
        )


def process_dataset(dataset: str):

    print("=" * 70)
    print(f"P3 PROCESSING: {dataset}")
    print("=" * 70)

    config = load_config(dataset)

    window_length = int(
        config["window"]["length"]
    )

    stride = int(
        config["window"]["stride"]
    )

    split_cfg = config["split"]

    raw_root = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "p1"
        / dataset
    )

    interim_root = (
        PROJECT_ROOT
        / "data"
        / "interim"
        / dataset
    )

    processed_root = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / dataset
    )

    interim_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"[1/8] Loading trajectories from:")
    print(f"      {raw_root}")

    trajectories = load_dataset(raw_root)

    print(
        f"      Loaded {len(trajectories)} trajectories"
    )

    # --------------------------------------------------
    # Trajectory validation
    # --------------------------------------------------

    trajectory_validation = {}

    for trajectory_id, df in trajectories.items():
        trajectory_validation[trajectory_id] = (
            validate_trajectory(df)
        )

    invalid = [
        key
        for key, value
        in trajectory_validation.items()
        if not value["valid"]
    ]

    if invalid:
        raise RuntimeError(
            f"Invalid trajectories in {dataset}: {invalid}"
        )

    print("[2/8] Trajectory validation: PASS")

    # --------------------------------------------------
    # Split by trajectory
    # --------------------------------------------------

    split = split_trajectory_ids(
        trajectories.keys(),
        train_ratio=float(
            split_cfg["train"]
        ),
        val_ratio=float(
            split_cfg["val"]
        ),
        test_ratio=float(
            split_cfg["test"]
        ),
        seed=int(
            split_cfg["seed"]
        ),
    )

    print("[3/8] Trajectory split:")
    print(
        f"      train={len(split['train'])}"
    )
    print(
        f"      val={len(split['val'])}"
    )
    print(
        f"      test={len(split['test'])}"
    )

    # --------------------------------------------------
    # Window generation
    # --------------------------------------------------

    windows = create_dataset_windows(
        trajectories=trajectories,
        dataset=dataset,
        window_length=window_length,
        stride=stride,
    )

    print(
        f"[4/8] Generated {len(windows)} windows"
    )

    if not windows:
        raise RuntimeError(
            f"No windows generated for {dataset}"
        )

    window_validation = [
        validate_window(
            window,
            window_length,
        )
        for window in windows
    ]

    if not all(
        item["valid"]
        for item in window_validation
    ):
        raise RuntimeError(
            f"Window validation failed for {dataset}"
        )

    # --------------------------------------------------
    # Save intermediate window metadata
    # --------------------------------------------------

    window_records = []

    for window in windows:
        window_records.append(
            {
                "dataset": dataset,
                "trajectory_id": window.trajectory_id,
                "start_index": window.start_index,
                "end_index": window.end_index,
                "length": len(window.data),
                "start_time": float(
                    window.data["t"].iloc[0]
                ),
                "end_time": float(
                    window.data["t"].iloc[-1]
                ),
            }
        )

    pd.DataFrame(
        window_records
    ).to_csv(
        interim_root
        / "window_index.csv",
        index=False,
    )

    # --------------------------------------------------
    # Split windows
    # --------------------------------------------------

    train_windows = filter_windows_by_split(
        windows,
        split["train"],
    )

    val_windows = filter_windows_by_split(
        windows,
        split["val"],
    )

    test_windows = filter_windows_by_split(
        windows,
        split["test"],
    )

    print(
        f"      windows: train={len(train_windows)}, "
        f"val={len(val_windows)}, "
        f"test={len(test_windows)}"
    )

    # --------------------------------------------------
    # Determine state columns
    # --------------------------------------------------

    excluded = {
        "t",
        "trajectory_id",
    }

    parameter_columns = {
        "m1",
        "m2",
        "G",
        "l1",
        "l2",
        "g",
        "sigma",
        "rho",
        "beta",
        "alpha",
        "delta",
        "gamma",
        "xi",
        "N",
    }

    feature_columns = [
        column
        for column in trajectories[
            next(iter(trajectories))
        ].columns
        if column not in excluded
        and column not in parameter_columns
    ]

    if not feature_columns:
        raise RuntimeError(
            f"No state columns found for {dataset}"
        )

    print(
        f"      State columns: {feature_columns}"
    )

    # --------------------------------------------------
    # Build arrays
    # --------------------------------------------------

    def windows_to_state_array(window_list):
        return np.stack(
            [
                window.data[
                    feature_columns
                ].to_numpy(
                    dtype=np.float32
                )
                for window in window_list
            ]
        )

    X_train_raw = windows_to_state_array(
        train_windows
    )

    X_val_raw = windows_to_state_array(
        val_windows
    )

    X_test_raw = windows_to_state_array(
        test_windows
    )

    # --------------------------------------------------
    # Scaling — TRAIN ONLY
    # --------------------------------------------------

    scaler = InvertibleScaler(
        config["scaler"]["type"]
    )

    scaler.fit(
        X_train_raw
    )

    X_train = scaler.transform(
        X_train_raw
    )

    X_val = scaler.transform(
        X_val_raw
    )

    X_test = scaler.transform(
        X_test_raw
    )

    scaler.save(
        processed_root
        / "scaler.joblib"
    )

    print(
        "[5/8] Scaling: PASS "
        "(fitted on training split only)"
    )

    # --------------------------------------------------
    # Save intermediate arrays
    # --------------------------------------------------

    np.savez_compressed(
        interim_root
        / "windowed_raw.npz",
        train=X_train_raw,
        val=X_val_raw,
        test=X_test_raw,
    )

    np.savez_compressed(
        processed_root
        / "scaled_windows.npz",
        train=X_train,
        val=X_val,
        test=X_test,
    )

    # --------------------------------------------------
    # Build graph per window
    # --------------------------------------------------

    def build_graph_set(window_list):

        node_features = []
        coordinates = []
        edge_indices = []
        edge_features = []

        for window in window_list:

            graph = build_graph(
                dataset,
                window.data,
            )

            validation = validate_graph(
                graph
            )

            if not validation["valid"]:
                raise RuntimeError(
                    f"Invalid graph for {dataset}"
                )

            node_features.append(
                graph["node_features"]
            )

            coordinates.append(
                graph["coordinates"]
            )

            edge_indices.append(
                graph["edge_index"]
            )

            edge_features.append(
                graph["edge_features"]
            )

        return (
            node_features,
            coordinates,
            edge_indices,
            edge_features,
        )

    train_graph = build_graph_set(
        train_windows
    )

    val_graph = build_graph_set(
        val_windows
    )

    test_graph = build_graph_set(
        test_windows
    )

    print("[6/8] Graph construction: PASS")

    # --------------------------------------------------
    # Save graph tensors
    # --------------------------------------------------

    np.savez_compressed(
        processed_root
        / "train_graphs.npz",
        node_features=np.array(
            train_graph[0],
            dtype=object,
        ),
        coordinates=np.array(
            train_graph[1],
            dtype=object,
        ),
        edge_index=np.array(
            train_graph[2],
            dtype=object,
        ),
        edge_features=np.array(
            train_graph[3],
            dtype=object,
        ),
    )

    np.savez_compressed(
        processed_root
        / "val_graphs.npz",
        node_features=np.array(
            val_graph[0],
            dtype=object,
        ),
        coordinates=np.array(
            val_graph[1],
            dtype=object,
        ),
        edge_index=np.array(
            val_graph[2],
            dtype=object,
        ),
        edge_features=np.array(
            val_graph[3],
            dtype=object,
        ),
    )

    np.savez_compressed(
        processed_root
        / "test_graphs.npz",
        node_features=np.array(
            test_graph[0],
            dtype=object,
        ),
        coordinates=np.array(
            test_graph[1],
            dtype=object,
        ),
        edge_index=np.array(
            test_graph[2],
            dtype=object,
        ),
        edge_features=np.array(
            test_graph[3],
            dtype=object,
        ),
    )

    # --------------------------------------------------
    # Save split metadata
    # --------------------------------------------------

    save_json(
        processed_root
        / "split.json",
        split,
    )

    save_json(
        processed_root
        / "metadata.json",
        {
            "dataset": dataset,
            "window_length": window_length,
            "stride": stride,
            "feature_columns": feature_columns,
            "num_trajectories": len(
                trajectories
            ),
            "num_windows": len(windows),
            "train_windows": len(
                train_windows
            ),
            "val_windows": len(
                val_windows
            ),
            "test_windows": len(
                test_windows
            ),
            "graph_convention": config[
                "graph"
            ],
        },
    )

    print("[7/8] Output artifacts: PASS")

    # --------------------------------------------------
    # Dataset summary
    # --------------------------------------------------

    summary = {
        "dataset": dataset,
        "trajectories": len(
            trajectories
        ),
        "windows": len(windows),
        "train_windows": len(
            train_windows
        ),
        "val_windows": len(
            val_windows
        ),
        "test_windows": len(
            test_windows
        ),
        "window_length": window_length,
        "stride": stride,
        "state_features": feature_columns,
        "status": "PASS",
    }

    save_json(
        processed_root
        / "p3_validation.json",
        summary,
    )

    print("[8/8] P3 dataset processing: COMPLETE")

    return summary


def main():

    print()
    print("=" * 70)
    print("P3 — PREPROCESSING & FEATURE/GRAPH CONSTRUCTION")
    print("=" * 70)
    print()

    summaries = []

    for dataset in DATASETS:
        summaries.append(
            process_dataset(dataset)
        )

    summary_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "p3_summary.csv"
    )

    pd.DataFrame(
        summaries
    ).to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("P3 PIPELINE COMPLETE")
    print("=" * 70)
    print()
    print(
        f"Summary written to: {summary_path}"
    )


if __name__ == "__main__":
    main()