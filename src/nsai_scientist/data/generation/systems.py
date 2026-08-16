from __future__ import annotations

import numpy as np

from .base import SystemDefinition


class TwoBodySystem(SystemDefinition):
    def __init__(self) -> None:
        super().__init__(
            name="two_body",
            state_names=[
                "x1",
                "y1",
                "vx1",
                "vy1",
                "x2",
                "y2",
                "vx2",
                "vy2",
            ],
            default_parameters={
                "G": 1.0,
                "m1": 1.0,
                "m2": 1.0,
            },
            ground_truth_equations=[
                "a1 = G*m2*(r2-r1)/|r2-r1|^3",
                "a2 = -G*m1*(r2-r1)/|r2-r1|^3",
            ],
        )

    def derivatives(self, t, state, parameters):
        del t

        x1, y1, vx1, vy1, x2, y2, vx2, vy2 = state

        G = parameters["G"]
        m1 = parameters["m1"]
        m2 = parameters["m2"]

        dx = x2 - x1
        dy = y2 - y1

        r2 = dx * dx + dy * dy
        r3 = r2 ** 1.5

        if r3 < 1e-12:
            raise ValueError("Two bodies are too close.")

        ax1 = G * m2 * dx / r3
        ay1 = G * m2 * dy / r3

        ax2 = -G * m1 * dx / r3
        ay2 = -G * m1 * dy / r3

        return np.array(
            [
                vx1,
                vy1,
                ax1,
                ay1,
                vx2,
                vy2,
                ax2,
                ay2,
            ],
            dtype=float,
        )

    def initial_state(self, rng, parameters):
        del rng, parameters

        return np.array(
            [
                -1.0,
                0.0,
                0.0,
                -0.5,
                1.0,
                0.0,
                0.0,
                0.5,
            ],
            dtype=float,
        )


class LorenzSystem(SystemDefinition):
    def __init__(self) -> None:
        super().__init__(
            name="lorenz",
            state_names=["x", "y", "z"],
            default_parameters={
                "sigma": 10.0,
                "rho": 28.0,
                "beta": 8.0 / 3.0,
            },
            ground_truth_equations=[
                "dx/dt = sigma*(y-x)",
                "dy/dt = x*(rho-z)-y",
                "dz/dt = x*y-beta*z",
            ],
        )

    def derivatives(self, t, state, parameters):
        del t

        x, y, z = state

        sigma = parameters["sigma"]
        rho = parameters["rho"]
        beta = parameters["beta"]

        return np.array(
            [
                sigma * (y - x),
                x * (rho - z) - y,
                x * y - beta * z,
            ],
            dtype=float,
        )

    def initial_state(self, rng, parameters):
        del parameters

        return np.array(
            [
                1.0 + rng.normal(0.0, 0.05),
                1.0 + rng.normal(0.0, 0.05),
                1.0 + rng.normal(0.0, 0.05),
            ],
            dtype=float,
        )


class DoublePendulumSystem(SystemDefinition):
    def __init__(self) -> None:
        super().__init__(
            name="double_pendulum",
            state_names=[
                "theta1",
                "omega1",
                "theta2",
                "omega2",
            ],
            default_parameters={
                "m1": 1.0,
                "m2": 1.0,
                "L1": 1.0,
                "L2": 1.0,
                "g": 9.81,
            },
            ground_truth_equations=[
                "dtheta1/dt = omega1",
                "dtheta2/dt = omega2",
                "domega1/dt = coupled nonlinear pendulum dynamics",
                "domega2/dt = coupled nonlinear pendulum dynamics",
            ],
        )

    def derivatives(self, t, state, parameters):
        del t

        theta1, omega1, theta2, omega2 = state

        m1 = parameters["m1"]
        m2 = parameters["m2"]
        L1 = parameters["L1"]
        L2 = parameters["L2"]
        g = parameters["g"]

        delta = theta1 - theta2

        denominator1 = (
            L1
            * (
                2 * m1
                + m2
                - m2 * np.cos(2 * theta1 - 2 * theta2)
            )
        )

        numerator1 = (
            -g * (2 * m1 + m2) * np.sin(theta1)
            - m2 * g * np.sin(theta1 - 2 * theta2)
            - 2
            * np.sin(delta)
            * m2
            * (
                omega2**2 * L2
                + omega1**2 * L1 * np.cos(delta)
            )
        )

        domega1 = numerator1 / denominator1

        denominator2 = (
            L2
            * (
                2 * m1
                + m2
                - m2 * np.cos(2 * theta1 - 2 * theta2)
            )
        )

        numerator2 = (
            2
            * np.sin(delta)
            * (
                omega1**2 * L1 * (m1 + m2)
                + g * (m1 + m2) * np.cos(theta1)
                + omega2**2 * L2 * m2 * np.cos(delta)
            )
        )

        domega2 = numerator2 / denominator2

        return np.array(
            [
                omega1,
                domega1,
                omega2,
                domega2,
            ],
            dtype=float,
        )

    def initial_state(self, rng, parameters):
        del parameters

        return np.array(
            [
                np.pi / 2 + rng.normal(0.0, 0.03),
                0.0,
                np.pi + rng.normal(0.0, 0.03),
                0.0,
            ],
            dtype=float,
        )


class LotkaVolterraSystem(SystemDefinition):
    def __init__(self) -> None:
        super().__init__(
            name="lotka_volterra",
            state_names=["prey", "predator"],
            default_parameters={
                "alpha": 1.5,
                "beta": 1.0,
                "delta": 0.75,
                "gamma": 1.0,
            },
            ground_truth_equations=[
                "dprey/dt = alpha*prey - beta*prey*predator",
                "dpredator/dt = delta*prey*predator - gamma*predator",
            ],
        )

    def derivatives(self, t, state, parameters):
        del t

        prey, predator = state

        alpha = parameters["alpha"]
        beta = parameters["beta"]
        delta = parameters["delta"]
        gamma = parameters["gamma"]

        return np.array(
            [
                alpha * prey - beta * prey * predator,
                delta * prey * predator - gamma * predator,
            ],
            dtype=float,
        )

    def initial_state(self, rng, parameters):
        del parameters

        return np.array(
            [
                2.0 + rng.normal(0.0, 0.05),
                1.0 + rng.normal(0.0, 0.05),
            ],
            dtype=float,
        )


class SISSystem(SystemDefinition):
    def __init__(self) -> None:
        super().__init__(
            name="sis",
            state_names=["S", "I"],
            default_parameters={
                "beta": 0.8,
                "gamma": 0.3,
                "N": 1.0,
            },
            ground_truth_equations=[
                "dS/dt = -beta*S*I/N + gamma*I",
                "dI/dt = beta*S*I/N - gamma*I",
            ],
        )

    def derivatives(self, t, state, parameters):
        del t

        S, I = state

        beta = parameters["beta"]
        gamma = parameters["gamma"]
        N = parameters["N"]

        infection = beta * S * I / N

        return np.array(
            [
                -infection + gamma * I,
                infection - gamma * I,
            ],
            dtype=float,
        )

    def initial_state(self, rng, parameters):
        del rng

        N = parameters["N"]

        I0 = 0.05 * N
        S0 = N - I0

        return np.array([S0, I0], dtype=float)


class SIRSSystem(SystemDefinition):
    def __init__(self) -> None:
        super().__init__(
            name="sirs",
            state_names=["S", "I", "R"],
            default_parameters={
                "beta": 0.8,
                "gamma": 0.3,
                "xi": 0.1,
                "N": 1.0,
            },
            ground_truth_equations=[
                "dS/dt = -beta*S*I/N + xi*R",
                "dI/dt = beta*S*I/N - gamma*I",
                "dR/dt = gamma*I - xi*R",
            ],
        )

    def derivatives(self, t, state, parameters):
        del t

        S, I, R = state

        beta = parameters["beta"]
        gamma = parameters["gamma"]
        xi = parameters["xi"]
        N = parameters["N"]

        infection = beta * S * I / N

        return np.array(
            [
                -infection + xi * R,
                infection - gamma * I,
                gamma * I - xi * R,
            ],
            dtype=float,
        )

    def initial_state(self, rng, parameters):
        del rng

        N = parameters["N"]

        I0 = 0.05 * N
        R0 = 0.0
        S0 = N - I0 - R0

        return np.array(
            [
                S0,
                I0,
                R0,
            ],
            dtype=float,
        )


SYSTEMS = {
    "two_body": TwoBodySystem,
    "lorenz": LorenzSystem,
    "double_pendulum": DoublePendulumSystem,
    "lotka_volterra": LotkaVolterraSystem,
    "sis": SISSystem,
    "sirs": SIRSSystem,
}


def get_system(name: str) -> SystemDefinition:
    if name not in SYSTEMS:
        available = ", ".join(sorted(SYSTEMS))
        raise ValueError(
            f"Unknown system '{name}'. Available: {available}"
        )

    return SYSTEMS[name]()