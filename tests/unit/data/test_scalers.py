import numpy as np

from nsai_scientist.data.scalers import (
    InvertibleScaler,
)


def test_standard_scaler_round_trip():

    rng = np.random.default_rng(42)

    original = rng.normal(
        size=(4, 10, 3)
    ).astype(np.float32)

    scaler = InvertibleScaler(
        "standard"
    )

    scaler.fit(original)

    transformed = scaler.transform(
        original
    )

    restored = scaler.inverse_transform(
        transformed
    )

    np.testing.assert_allclose(
        original,
        restored,
        rtol=1e-5,
        atol=1e-5,
    )