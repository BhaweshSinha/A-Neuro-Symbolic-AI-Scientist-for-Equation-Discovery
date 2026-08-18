import pandas as pd

from nsai_scientist.data.loaders import (
    derive_trajectory_id,
    load_trajectory,
)


def test_trajectory_id():

    assert (
        derive_trajectory_id(
            "trajectory_003.csv"
        )
        == "trajectory_003"
    )


def test_loader(tmp_path):

    path = (
        tmp_path
        / "trajectory_000.csv"
    )

    pd.DataFrame(
        {
            "t": [0.0, 0.1, 0.2],
            "x": [1.0, 2.0, 3.0],
        }
    ).to_csv(
        path,
        index=False,
    )

    df = load_trajectory(path)

    assert "trajectory_id" in df.columns
    assert len(df) == 3