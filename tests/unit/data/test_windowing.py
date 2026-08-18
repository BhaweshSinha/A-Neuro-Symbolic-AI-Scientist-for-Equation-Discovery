import pandas as pd

from nsai_scientist.data.windowing import (
    create_windows,
)


def test_window_count():

    df = pd.DataFrame(
        {
            "t": range(10),
            "x": range(10),
        }
    )

    windows = create_windows(
        df,
        dataset="test",
        trajectory_id="trajectory_000",
        window_length=4,
        stride=2,
    )

    assert len(windows) == 4
    assert all(
        len(window.data) == 4
        for window in windows
    )