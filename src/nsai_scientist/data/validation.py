from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def validate_trajectory(
    df: pd.DataFrame,
) -> Dict[str, object]:

    result = {
        "rows": len(df),
        "has_time": "t" in df.columns,
        "time_monotonic": False,
        "missing_numeric_values": 0,
        "infinite_numeric_values": 0,
        "valid": False,
    }

    if "t" not in df.columns:
        return result

    time_values = pd.to_numeric(
        df["t"],
        errors="coerce",
    )

    result["time_monotonic"] = bool(
        time_values.notna().all()
        and time_values.is_monotonic_increasing
    )

    numeric = df.select_dtypes(include=[np.number])

    result["missing_numeric_values"] = int(
        numeric.isna().sum().sum()
    )

    result["infinite_numeric_values"] = int(
        np.isinf(numeric.to_numpy()).sum()
    )

    result["valid"] = bool(
        result["has_time"]
        and result["time_monotonic"]
        and result["missing_numeric_values"] == 0
        and result["infinite_numeric_values"] == 0
    )

    return result


def validate_window(
    window,
    expected_length: int,
) -> Dict[str, object]:

    data = window.data

    numeric = data.select_dtypes(include=[np.number])

    return {
        "trajectory_id": window.trajectory_id,
        "start_index": window.start_index,
        "end_index": window.end_index,
        "length": len(data),
        "correct_length": len(data) == expected_length,
        "nan_count": int(numeric.isna().sum().sum()),
        "inf_count": int(
            np.isinf(numeric.to_numpy()).sum()
        ),
        "valid": bool(
            len(data) == expected_length
            and numeric.isna().sum().sum() == 0
            and np.isinf(numeric.to_numpy()).sum() == 0
        ),
    }


def validate_graph(
    graph: Dict[str, np.ndarray],
) -> Dict[str, object]:

    node_features = graph["node_features"]
    coordinates = graph["coordinates"]
    edge_index = graph["edge_index"]
    edge_features = graph["edge_features"]

    finite = all(
        np.isfinite(value).all()
        for value in [
            node_features,
            coordinates,
            edge_features,
        ]
    )

    return {
        "node_count": int(node_features.shape[0]),
        "node_feature_dim": int(node_features.shape[1]),
        "coordinate_shape": list(coordinates.shape),
        "edge_count": int(edge_index.shape[0]),
        "edge_feature_dim": int(edge_features.shape[1]),
        "finite": bool(finite),
        "valid": bool(
            finite
            and edge_index.ndim == 2
            and edge_index.shape[1] == 2
        ),
    }