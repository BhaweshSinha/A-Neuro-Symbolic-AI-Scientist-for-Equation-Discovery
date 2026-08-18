import pandas as pd

from nsai_scientist.data.graph_builder import (
    build_graph,
)


def test_two_body_graph():

    df = pd.DataFrame(
        {
            "x1": [0.0],
            "y1": [0.0],
            "vx1": [1.0],
            "vy1": [0.0],
            "x2": [1.0],
            "y2": [0.0],
            "vx2": [-1.0],
            "vy2": [0.0],
        }
    )

    graph = build_graph(
        "two_body",
        df,
    )

    assert graph[
        "node_features"
    ].shape[0] == 2

    assert graph[
        "edge_index"
    ].shape[0] == 2


def test_sis_graph():

    df = pd.DataFrame(
        {
            "S": [90.0],
            "I": [10.0],
            "N": [100.0],
        }
    )

    graph = build_graph(
        "sis",
        df,
    )

    assert graph[
        "node_features"
    ].shape[0] == 2

    assert graph[
        "edge_index"
    ].shape[0] == 2


def test_sirs_graph():

    df = pd.DataFrame(
        {
            "S": [80.0],
            "I": [10.0],
            "R": [10.0],
            "N": [100.0],
        }
    )

    graph = build_graph(
        "sirs",
        df,
    )

    assert graph[
        "node_features"
    ].shape[0] == 3

    assert graph[
        "edge_index"
    ].shape[0] == 3