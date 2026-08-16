from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.integrate import solve_ivp

from .systems import get_system


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "data" / "p1.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_time_grid(duration: float, dt: float) -> np.ndarray:
    steps = int(round(duration / dt))

    return np.linspace(
        0.0,
        duration,
        steps + 1,
    )


def integrate_trajectory(
    system,
    initial_state: np.ndarray,
    parameters: dict[str, float],
    time_grid: np.ndarray,
    solver_config: dict,
) -> np.ndarray:

    solution = solve_ivp(
        fun=lambda t, state: system.derivatives(
            t,
            state,
            parameters,
        ),
        t_span=(
            float(time_grid[0]),
            float(time_grid[-1]),
        ),
        y0=initial_state,
        t_eval=time_grid,
        method=solver_config["method"],
        rtol=float(solver_config["rtol"]),
        atol=float(solver_config["atol"]),
    )

    if not solution.success:
        raise RuntimeError(
            f"ODE integration failed for {system.name}: "
            f"{solution.message}"
        )

    trajectory = solution.y.T

    if not np.all(np.isfinite(trajectory)):
        raise RuntimeError(
            f"Non-finite values generated for {system.name}."
        )

    return trajectory


def save_csv(
    path: Path,
    time_grid: np.ndarray,
    trajectory: np.ndarray,
    state_names: list[str],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            ["time", *state_names]
        )

        for time, state in zip(
            time_grid,
            trajectory,
        ):
            writer.writerow(
                [
                    f"{time:.12g}",
                    *[
                        f"{value:.12g}"
                        for value in state
                    ],
                ]
            )


def save_json(
    path: Path,
    data: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
        )


def generate_system(
    system_name: str,
    config: dict,
    master_seed: int,
) -> None:

    system = get_system(system_name)

    rng = np.random.default_rng(
        master_seed
    )

    generation_config = config["generation"]

    duration = float(
        generation_config["duration"]
    )

    dt = float(
        generation_config["dt"]
    )

    time_grid = build_time_grid(
        duration,
        dt,
    )

    solver_config = generation_config["solver"]

    n_trajectories = int(
        generation_config[
            "trajectories_per_system"
        ]
    )

    raw_root = (
        PROJECT_ROOT
        / config["output"]["raw_dir"]
        / system_name
    )

    ground_truth_root = (
        PROJECT_ROOT
        / config["output"]["ground_truth_dir"]
    )

    raw_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "system": system.name,
        "state_names": system.state_names,
        "default_parameters": system.default_parameters,
        "ground_truth_equations": system.ground_truth_equations,
        "generation": {
            "trajectories": n_trajectories,
            "duration": duration,
            "dt": dt,
            "solver": solver_config,
            "seed": master_seed,
        },
    }

    save_json(
        ground_truth_root
        / f"{system_name}.json",
        metadata,
    )

    for trajectory_id in range(
        n_trajectories
    ):

        trajectory_seed = (
            master_seed
            + trajectory_id
            + (
                list(config["systems"]).index(
                    system_name
                )
                * 10000
            )
        )

        trajectory_rng = np.random.default_rng(
            trajectory_seed
        )

        parameters = system.sample_parameters(
            trajectory_rng
        )

        initial_state = system.initial_state(
            trajectory_rng,
            parameters,
        )

        trajectory = integrate_trajectory(
            system=system,
            initial_state=initial_state,
            parameters=parameters,
            time_grid=time_grid,
            solver_config=solver_config,
        )

        output_path = (
            raw_root
            / f"trajectory_{trajectory_id:03d}.csv"
        )

        save_csv(
            output_path,
            time_grid,
            trajectory,
            system.state_names,
        )

        trajectory_metadata = {
            "system": system_name,
            "trajectory_id": trajectory_id,
            "seed": trajectory_seed,
            "parameters": parameters,
            "initial_state": initial_state.tolist(),
            "n_samples": len(time_grid),
            "duration": duration,
            "dt": dt,
            "solver": solver_config,
        }

        save_json(
            raw_root
            / f"trajectory_{trajectory_id:03d}.json",
            trajectory_metadata,
        )

        print(
            f"[{system_name}] "
            f"trajectory {trajectory_id + 1}/"
            f"{n_trajectories} generated"
        )


def main() -> None:

    config = load_config()

    seed = int(
        config["seed"]
    )

    systems = config["systems"]

    print("=" * 60)
    print("P1 — Scientific Data Generation")
    print("=" * 60)

    for index, system_name in enumerate(
        systems
    ):

        print()
        print(
            f"[{index + 1}/{len(systems)}] "
            f"Generating {system_name}"
        )

        generate_system(
            system_name=system_name,
            config=config,
            master_seed=seed,
        )

    print()
    print("=" * 60)
    print("P1 DATA GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()