from nsai_scientist.data.splitting import (
    split_trajectory_ids,
)


def test_split_is_complete():

    ids = [
        f"trajectory_{i:03d}"
        for i in range(10)
    ]

    split = split_trajectory_ids(
        ids,
        0.8,
        0.1,
        0.1,
        seed=42,
    )

    combined = (
        split["train"]
        + split["val"]
        + split["test"]
    )

    assert len(combined) == 10
    assert len(set(combined)) == 10

    assert len(split["train"]) == 8
    assert len(split["val"]) == 1
    assert len(split["test"]) == 1