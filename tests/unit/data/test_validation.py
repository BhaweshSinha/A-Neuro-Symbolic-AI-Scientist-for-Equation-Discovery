import pandas as pd

from nsai_scientist.data.validation import (
    validate_trajectory,
)


def test_valid_trajectory():

    df = pd.DataFrame(
        {
            "t": [0.0, 0.1, 0.2],
            "x": [1.0, 2.0, 3.0],
        }
    )

    result = validate_trajectory(df)

    assert result["valid"] is True