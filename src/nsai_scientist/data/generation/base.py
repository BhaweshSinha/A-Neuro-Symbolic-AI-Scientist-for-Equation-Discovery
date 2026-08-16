from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SystemDefinition(ABC):
    """
    Common interface for every P1 dynamical system.
    """

    name: str
    state_names: list[str]
    default_parameters: dict[str, float]
    ground_truth_equations: list[str]

    @abstractmethod
    def derivatives(
        self,
        t: float,
        state: np.ndarray,
        parameters: dict[str, float],
    ) -> np.ndarray:
        """Return dy/dt."""
        raise NotImplementedError

    @abstractmethod
    def initial_state(
        self,
        rng: np.random.Generator,
        parameters: dict[str, float],
    ) -> np.ndarray:
        """Generate a valid initial condition."""
        raise NotImplementedError

    def sample_parameters(
        self,
        rng: np.random.Generator,
    ) -> dict[str, float]:
        """
        Sample parameters for one trajectory.

        By default, use the system's default parameters.
        """
        return dict(self.default_parameters)

    def metadata(self) -> dict[str, Any]:
        return {
            "system": self.name,
            "state_names": self.state_names,
            "default_parameters": self.default_parameters,
            "ground_truth_equations": self.ground_truth_equations,
        }