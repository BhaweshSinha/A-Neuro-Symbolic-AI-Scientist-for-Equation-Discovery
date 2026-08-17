from __future__ import annotations

import numpy as np
import pandas as pd


def analyze_two_body(df: pd.DataFrame) -> dict:

    r1_sq = df["x1"] ** 2 + df["y1"] ** 2
    r2_sq = df["x2"] ** 2 + df["y2"] ** 2

    dx = df["x1"] - df["x2"]
    dy = df["y1"] - df["y2"]

    distance = np.sqrt(
        dx**2 + dy**2
    )

    kinetic = (
        0.5
        * df["m1"]
        * (
            df["vx1"] ** 2
            + df["vy1"] ** 2
        )
        +
        0.5
        * df["m2"]
        * (
            df["vx2"] ** 2
            + df["vy2"] ** 2
        )
    )

    potential = (
        -df["G"]
        * df["m1"]
        * df["m2"]
        / distance
    )

    energy = kinetic + potential

    angular_momentum = (
        df["m1"]
        * (
            df["x1"] * df["vy1"]
            - df["y1"] * df["vx1"]
        )
        +
        df["m2"]
        * (
            df["x2"] * df["vy2"]
            - df["y2"] * df["vx2"]
        )
    )

    return {
        "mean_distance": float(
            distance.mean()
        ),
        "min_distance": float(
            distance.min()
        ),
        "max_distance": float(
            distance.max()
        ),
        "energy_mean": float(
            energy.mean()
        ),
        "energy_std": float(
            energy.std()
        ),
        "angular_momentum_mean": float(
            angular_momentum.mean()
        ),
        "angular_momentum_std": float(
            angular_momentum.std()
        ),
    }


def analyze_sis(
    df: pd.DataFrame,
) -> dict:

    residual = (
        df["S"]
        + df["I"]
        - df["N"]
    )

    return {
        "population_residual_mean":
            float(residual.mean()),
        "population_residual_max_abs":
            float(np.abs(residual).max()),
        "min_S":
            float(df["S"].min()),
        "min_I":
            float(df["I"].min()),
    }


def analyze_sirs(
    df: pd.DataFrame,
) -> dict:

    residual = (
        df["S"]
        + df["I"]
        + df["R"]
        - df["N"]
    )

    return {
        "population_residual_mean":
            float(residual.mean()),
        "population_residual_max_abs":
            float(np.abs(residual).max()),
        "min_S":
            float(df["S"].min()),
        "min_I":
            float(df["I"].min()),
        "min_R":
            float(df["R"].min()),
    }


def analyze_lotka_volterra(
    df: pd.DataFrame,
) -> dict:

    return {
        "prey_min":
            float(df["prey"].min()),
        "predator_min":
            float(df["predator"].min()),
        "prey_positive":
            bool((df["prey"] > 0).all()),
        "predator_positive":
            bool((df["predator"] > 0).all()),
    }


def analyze_generic(
    df: pd.DataFrame,
) -> dict:

    return {
        "rows": int(len(df)),
        "trajectories":
            int(df["trajectory_id"].nunique()),
    }


def analyze_system(
    dataset: str,
    df: pd.DataFrame,
) -> dict:

    if dataset == "two_body":
        return analyze_two_body(df)

    if dataset == "sis":
        return analyze_sis(df)

    if dataset == "sirs":
        return analyze_sirs(df)

    if dataset == "lotka_volterra":
        return analyze_lotka_volterra(df)

    return analyze_generic(df)