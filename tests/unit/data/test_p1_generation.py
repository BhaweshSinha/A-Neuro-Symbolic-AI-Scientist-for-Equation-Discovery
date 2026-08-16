from __future__ import annotations

import numpy as np
import pytest

from nsai_scientist.data.generation.systems import (
    SYSTEMS,
    get_system,
)


@pytest.mark.parametrize(
    "system_name",
    list(SYSTEMS.keys()),
)
def test_system_definition(system_name):
    system = get_system(system_name)

    assert system.name == system_name
    assert len(system.state_names) > 0
    assert len(system.ground_truth_equations) > 0


@pytest.mark.parametrize(
    "system_name",
    list(SYSTEMS.keys()),
)
def test_initial_state_shape(system_name):
    system = get_system(system_name)

    rng = np.random.default_rng(42)

    parameters = system.sample_parameters(
        rng
    )

    state = system.initial_state(
        rng,
        parameters,
    )

    assert state.shape == (
        len(system.state_names),
    )

    assert np.all(
        np.isfinite(state)
    )


@pytest.mark.parametrize(
    "system_name",
    list(SYSTEMS.keys()),
)
def test_derivatives_shape(system_name):
    system = get_system(system_name)

    rng = np.random.default_rng(42)

    parameters = system.sample_parameters(
        rng
    )

    state = system.initial_state(
        rng,
        parameters,
    )

    derivatives = system.derivatives(
        0.0,
        state,
        parameters,
    )

    assert derivatives.shape == state.shape

    assert np.all(
        np.isfinite(derivatives)
    )