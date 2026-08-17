from __future__ import annotations

import numpy as np
import pandas as pd


def analyze_two_body(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze basic two-body dynamical quantities."""

    required = [
        "x1", "y1",
        "vx1", "vy1",
        "x2", "y2",
        "vx2", "vy2",
    ]

    if not all(
        column in df.columns
        for column in required
    ):
        return pd.DataFrame()

    result = pd.DataFrame(
        index=df.index
    )

    result["r1"] = np.sqrt(
        df["x1"] ** 2
        + df["y1"] ** 2
    )

    result["r2"] = np.sqrt(
        df["x2"] ** 2
        + df["y2"] ** 2
    )

    result["speed1"] = np.sqrt(
        df["vx1"] ** 2
        + df["vy1"] ** 2
    )

    result["speed2"] = np.sqrt(
        df["vx2"] ** 2
        + df["vy2"] ** 2
    )

    result["separation"] = np.sqrt(
        (
            df["x1"]
            - df["x2"]
        ) ** 2
        + (
            df["y1"]
            - df["y2"]
        ) ** 2
    )

    return result


def analyze_lorenz(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze Lorenz-system state magnitude."""

    required = ["x", "y", "z"]

    if not all(
        column in df.columns
        for column in required
    ):
        return pd.DataFrame()

    result = pd.DataFrame(
        index=df.index
    )

    result["state_norm"] = np.sqrt(
        df["x"] ** 2
        + df["y"] ** 2
        + df["z"] ** 2
    )

    result["xy_radius"] = np.sqrt(
        df["x"] ** 2
        + df["y"] ** 2
    )

    return result


def analyze_double_pendulum(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze double-pendulum state quantities."""

    required = [
        "theta1",
        "omega1",
        "theta2",
        "omega2",
    ]

    if not all(
        column in df.columns
        for column in required
    ):
        return pd.DataFrame()

    result = pd.DataFrame(
        index=df.index
    )

    result["angular_speed_total"] = np.sqrt(
        df["omega1"] ** 2
        + df["omega2"] ** 2
    )

    result["angle_difference"] = (
        df["theta1"]
        - df["theta2"]
    )

    return result


def analyze_lotka_volterra(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze predator-prey state quantities."""

    required = [
        "prey",
        "predator",
    ]

    if not all(
        column in df.columns
        for column in required
    ):
        return pd.DataFrame()

    result = pd.DataFrame(
        index=df.index
    )

    result["total_population"] = (
        df["prey"]
        + df["predator"]
    )

    result["prey_predator_ratio"] = (
        df["prey"]
        / df["predator"].replace(
            0,
            np.nan,
        )
    )

    return result


def analyze_compartmental(
    df: pd.DataFrame,
    dataset: str,
) -> pd.DataFrame:
    """Analyze SIS/SIRS compartment totals."""

    if dataset == "sis":

        required = [
            "S",
            "I",
        ]

        if not all(
            column in df.columns
            for column in required
        ):
            return pd.DataFrame()

        result = pd.DataFrame(
            index=df.index
        )

        result["population_total"] = (
            df["S"] + df["I"]
        )

        return result

    if dataset == "sirs":

        required = [
            "S",
            "I",
            "R",
        ]

        if not all(
            column in df.columns
            for column in required
        ):
            return pd.DataFrame()

        result = pd.DataFrame(
            index=df.index
        )

        result["population_total"] = (
            df["S"]
            + df["I"]
            + df["R"]
        )

        return result

    return pd.DataFrame()


def run_system_analysis(
    df: pd.DataFrame,
    dataset: str,
) -> pd.DataFrame:
    """Run the appropriate system-specific analysis."""

    if dataset == "two_body":
        return analyze_two_body(df)

    if dataset == "lorenz":
        return analyze_lorenz(df)

    if dataset == "double_pendulum":
        return analyze_double_pendulum(df)

    if dataset == "lotka_volterra":
        return analyze_lotka_volterra(df)

    if dataset in {
        "sis",
        "sirs",
    }:
        return analyze_compartmental(
            df,
            dataset,
        )

    return pd.DataFrame()