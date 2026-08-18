from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _edge_index_undirected():
    return np.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=np.int64,
    )


def _edge_index_directed_cycle():
    return np.array(
        [
            [0, 1],
            [1, 2],
            [2, 0],
        ],
        dtype=np.int64,
    )


def build_two_body_graph(df: pd.DataFrame) -> Dict[str, np.ndarray]:

    required = [
        "x1", "y1", "vx1", "vy1",
        "x2", "y2", "vx2", "vy2",
    ]

    _require_columns(df, required)

    node_features = np.array(
        [
            [df["x1"].iloc[-1], df["y1"].iloc[-1],
             df["vx1"].iloc[-1], df["vy1"].iloc[-1]],

            [df["x2"].iloc[-1], df["y2"].iloc[-1],
             df["vx2"].iloc[-1], df["vy2"].iloc[-1]],
        ],
        dtype=np.float32,
    )

    coordinates = np.array(
        [
            [df["x1"].iloc[-1], df["y1"].iloc[-1]],
            [df["x2"].iloc[-1], df["y2"].iloc[-1]],
        ],
        dtype=np.float32,
    )

    relative = coordinates[1] - coordinates[0]
    distance = np.linalg.norm(relative)

    edge_features = np.array(
        [[distance], [distance]],
        dtype=np.float32,
    )

    return {
        "node_features": node_features,
        "coordinates": coordinates,
        "edge_index": _edge_index_undirected(),
        "edge_features": edge_features,
    }


def build_double_pendulum_graph(
    df: pd.DataFrame,
) -> Dict[str, np.ndarray]:

    required = [
        "theta1",
        "omega1",
        "theta2",
        "omega2",
    ]

    _require_columns(df, required)

    theta1 = float(df["theta1"].iloc[-1])
    omega1 = float(df["omega1"].iloc[-1])
    theta2 = float(df["theta2"].iloc[-1])
    omega2 = float(df["omega2"].iloc[-1])

    m1 = _optional_parameter(df, "m1", 1.0)
    m2 = _optional_parameter(df, "m2", 1.0)
    l1 = _optional_parameter(df, "l1", 1.0)
    l2 = _optional_parameter(df, "l2", 1.0)

    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)

    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)

    vx1 = l1 * np.cos(theta1) * omega1
    vy1 = l1 * np.sin(theta1) * omega1

    vx2 = vx1 + l2 * np.cos(theta2) * omega2
    vy2 = vy1 + l2 * np.sin(theta2) * omega2

    node_features = np.array(
        [
            [x1, y1, vx1, vy1, theta1, omega1],
            [x2, y2, vx2, vy2, theta2, omega2],
        ],
        dtype=np.float32,
    )

    coordinates = np.array(
        [
            [x1, y1],
            [x2, y2],
        ],
        dtype=np.float32,
    )

    distance = np.linalg.norm(coordinates[1] - coordinates[0])

    edge_features = np.array(
        [[distance], [distance]],
        dtype=np.float32,
    )

    return {
        "node_features": node_features,
        "coordinates": coordinates,
        "edge_index": _edge_index_undirected(),
        "edge_features": edge_features,
    }


def build_lotka_volterra_graph(
    df: pd.DataFrame,
) -> Dict[str, np.ndarray]:

    _require_columns(df, ["prey", "predator"])

    prey = max(float(df["prey"].iloc[-1]), 1e-8)
    predator = max(float(df["predator"].iloc[-1]), 1e-8)

    node_features = np.array(
        [
            [prey],
            [predator],
        ],
        dtype=np.float32,
    )

    coordinates = np.array(
        [
            [np.log(prey)],
            [np.log(predator)],
        ],
        dtype=np.float32,
    )

    distance = abs(float(coordinates[1, 0] - coordinates[0, 0]))

    edge_features = np.array(
        [[distance], [distance]],
        dtype=np.float32,
    )

    return {
        "node_features": node_features,
        "coordinates": coordinates,
        "edge_index": _edge_index_undirected(),
        "edge_features": edge_features,
    }


def build_sis_graph(
    df: pd.DataFrame,
) -> Dict[str, np.ndarray]:

    _require_columns(df, ["S", "I"])

    S = float(df["S"].iloc[-1])
    I = float(df["I"].iloc[-1])

    N = _optional_parameter(
        df,
        "N",
        max(S + I, 1.0),
    )

    node_features = np.array(
        [
            [S / N],
            [I / N],
        ],
        dtype=np.float32,
    )

    coordinates = np.array(
        [
            [0.0],
            [1.0],
        ],
        dtype=np.float32,
    )

    edge_features = np.array(
        [[1.0], [1.0]],
        dtype=np.float32,
    )

    return {
        "node_features": node_features,
        "coordinates": coordinates,
        "edge_index": _edge_index_undirected(),
        "edge_features": edge_features,
    }


def build_sirs_graph(
    df: pd.DataFrame,
) -> Dict[str, np.ndarray]:

    _require_columns(df, ["S", "I", "R"])

    S = float(df["S"].iloc[-1])
    I = float(df["I"].iloc[-1])
    R = float(df["R"].iloc[-1])

    N = _optional_parameter(
        df,
        "N",
        max(S + I + R, 1.0),
    )

    node_features = np.array(
        [
            [S / N],
            [I / N],
            [R / N],
        ],
        dtype=np.float32,
    )

    coordinates = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
        ],
        dtype=np.float32,
    )

    edge_features = np.array(
        [[1.0], [1.0], [1.0]],
        dtype=np.float32,
    )

    return {
        "node_features": node_features,
        "coordinates": coordinates,
        "edge_index": _edge_index_directed_cycle(),
        "edge_features": edge_features,
    }


def build_lorenz_graph(
    df: pd.DataFrame,
) -> Dict[str, np.ndarray]:

    _require_columns(df, ["x", "y", "z"])

    x = float(df["x"].iloc[-1])
    y = float(df["y"].iloc[-1])
    z = float(df["z"].iloc[-1])

    node_features = np.array(
        [
            [x],
            [y],
            [z],
        ],
        dtype=np.float32,
    )

    coordinates = np.array(
        [
            [x],
            [y],
            [z],
        ],
        dtype=np.float32,
    )

    edges = []

    for i in range(3):
        for j in range(3):
            if i != j:
                edges.append([i, j])

    edge_index = np.array(
        edges,
        dtype=np.int64,
    )

    edge_features = []

    for source, target in edge_index:
        distance = abs(
            coordinates[target, 0]
            - coordinates[source, 0]
        )
        edge_features.append([distance])

    return {
        "node_features": node_features,
        "coordinates": coordinates,
        "edge_index": edge_index,
        "edge_features": np.asarray(
            edge_features,
            dtype=np.float32,
        ),
    }


def build_graph(
    dataset: str,
    window: pd.DataFrame,
) -> Dict[str, np.ndarray]:

    builders = {
        "two_body": build_two_body_graph,
        "lorenz": build_lorenz_graph,
        "double_pendulum": build_double_pendulum_graph,
        "lotka_volterra": build_lotka_volterra_graph,
        "sis": build_sis_graph,
        "sirs": build_sirs_graph,
    }

    if dataset not in builders:
        raise ValueError(
            f"Unsupported dataset: {dataset}"
        )

    return builders[dataset](window)


def _require_columns(
    df: pd.DataFrame,
    columns: List[str],
):
    missing = [
        column for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def _optional_parameter(
    df: pd.DataFrame,
    column: str,
    default: float,
) -> float:

    if column not in df.columns:
        return float(default)

    value = pd.to_numeric(
        df[column],
        errors="coerce",
    ).dropna()

    if value.empty:
        return float(default)

    return float(value.iloc[0])