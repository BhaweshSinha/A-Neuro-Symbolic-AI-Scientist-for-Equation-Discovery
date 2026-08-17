from __future__ import annotations

from pathlib import Path

import pandas as pd

from nsai_scientist.data.eda.comparison import (
    compare_datasets,
)
from nsai_scientist.data.eda.correlations import (
    correlation_flags,
    correlation_matrix,
)
from nsai_scientist.data.eda.inventory import (
    create_inventory,
)
from nsai_scientist.data.eda.loader import (
    load_all,
)
from nsai_scientist.data.eda.plots import (
    plot_correlation_matrix,
    plot_distributions,
    plot_time_series,
)
from nsai_scientist.data.eda.schema import (
    EXPECTED_STATE_COLUMNS,
)
from nsai_scientist.data.eda.system_analysis import (
    run_system_analysis,
)
from nsai_scientist.data.eda.validation import (
    validate_all,
)
from nsai_scientist.data.eda.metadata import (
    load_all_metadata,
)
from nsai_scientist.data.eda.phase_space import (
    plot_phase_portraits,
)
from nsai_scientist.data.eda.statistics import (
    numeric_summary,
)
from nsai_scientist.data.eda.timeseries import (
    calculate_time_statistics,
)


DATASETS = [
    "two_body",
    "lorenz",
    "double_pendulum",
    "lotka_volterra",
    "sis",
    "sirs",
]


def _save_json(
    data,
    path: Path,
):
    import json

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            default=str,
        )


def run_p2(config):
    """Execute complete P2 EDA pipeline."""

    raw_root = Path(
        config["raw_root"]
    )

    output_root = Path(
        config["output_root"]
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # 1. Load datasets
    # --------------------------------------------------

    print(
        "\n[1/10] Loading datasets..."
    )

    datasets = load_all(
        raw_root
    )

    metadata = load_all_metadata(
        raw_root,
        DATASETS,
    )

    # --------------------------------------------------
    # 2. Dataset inventory
    # --------------------------------------------------

    print(
        "[2/10] Creating dataset inventory..."
    )

    inventory = create_inventory(
        datasets
    )

    inventory.to_csv(
        output_root
        / "dataset_inventory.csv",
        index=False,
    )

    # --------------------------------------------------
    # 3. Statistical summary
    # --------------------------------------------------

    print(
        "[3/10] Statistical summaries..."
    )

    statistics_dir = (
        output_root
        / "statistical_summary"
    )

    statistics_dir.mkdir(
        exist_ok=True
    )

    for dataset, df in datasets.items():

        summary = numeric_summary(
            df
        )

        summary.to_csv(
            statistics_dir
            / f"{dataset}_summary.csv"
        )

    # --------------------------------------------------
    # 4. Time-series analysis
    # --------------------------------------------------

    print(
        "[4/10] Time-series analysis..."
    )

    time_dir = (
        output_root
        / "time_series"
    )

    time_dir.mkdir(
        exist_ok=True
    )

    for dataset, df in datasets.items():

        result = calculate_time_statistics(
            df
        )

        _save_json(
            result,
            time_dir
            / f"{dataset}_time_statistics.json",
        )

    # --------------------------------------------------
    # 5. Distribution analysis
    # --------------------------------------------------

    print(
        "[5/10] Distribution analysis..."
    )

    distribution_dir = (
        output_root
        / "distribution"
    )

    distribution_dir.mkdir(
        exist_ok=True
    )

    for dataset, df in datasets.items():

        numeric = df.select_dtypes(
            include="number"
        )

        numeric = numeric.drop(
            columns=[
                "trajectory_id"
            ],
            errors="ignore",
        )

        summary = numeric.describe().T

        summary.to_csv(
            distribution_dir
            / f"{dataset}_distribution_summary.csv"
        )

    # --------------------------------------------------
    # 6. Correlation analysis
    # --------------------------------------------------

    print(
        "[6/10] Correlation analysis..."
    )

    correlation_dir = (
        output_root
        / "correlation"
    )

    correlation_dir.mkdir(
        exist_ok=True
    )

    for dataset, df in datasets.items():

        matrix = correlation_matrix(
            df
        )

        matrix.to_csv(
            correlation_dir
            / f"{dataset}_correlation.csv"
        )

        flags = correlation_flags(
            matrix
        )

        flags.to_csv(
            correlation_dir
            / f"{dataset}_high_correlations.csv",
            index=False,
        )

    # --------------------------------------------------
    # 7. Phase-space analysis
    # --------------------------------------------------

    print(
        "[7/10] Phase-space analysis..."
    )

    phase_dir = (
        output_root
        / "phase_space"
    )

    for dataset, df in datasets.items():

        plot_phase_portraits(
            df,
            dataset,
            phase_dir
        )

    # --------------------------------------------------
    # 8. System-specific analysis
    # --------------------------------------------------

    print(
        "[8/10] System-specific analysis..."
    )

    system_dir = (
        output_root
        / "system_specific"
    )

    system_dir.mkdir(
        exist_ok=True
    )

    for dataset, df in datasets.items():

        result = run_system_analysis(
            df,
            dataset,
        )

        if not result.empty:

            result.describe().T.to_csv(
                system_dir
                / f"{dataset}_system_summary.csv"
            )

    # --------------------------------------------------
    # 9. Cross-system comparison
    # --------------------------------------------------

    print(
        "[9/10] Cross-system comparison..."
    )

    comparison = compare_datasets(
        datasets
    )

    comparison.to_csv(
        output_root
        / "cross_system_comparison.csv",
        index=False,
    )

    # --------------------------------------------------
    # 10. EDA plots
    # --------------------------------------------------

    print(
        "[10/10] Generating EDA plots..."
    )

    plots_root = (
        output_root
        / "plots"
    )

    for dataset, df in datasets.items():

        dataset_plot_dir = (
            plots_root
            / dataset
        )

        plot_time_series(
            df,
            dataset,
            dataset_plot_dir,
        )

        plot_distributions(
            df,
            dataset,
            dataset_plot_dir,
        )

        matrix = correlation_matrix(
            df
        )

        plot_correlation_matrix(
            matrix,
            dataset,
            dataset_plot_dir,
        )

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    print(
        "\n[VALIDATION] Running P2 validation..."
    )

    validation = validate_all(
        datasets
    )

    validation.to_csv(
        output_root
        / "p2_validation.csv",
        index=False,
    )

    # --------------------------------------------------
    # Metadata inventory
    # --------------------------------------------------

    metadata_counts = {
        dataset: len(
            dataset_metadata
        )
        for dataset,
        dataset_metadata
        in metadata.items()
    }

    _save_json(
        metadata_counts,
        output_root
        / "metadata_inventory.json",
    )

    # --------------------------------------------------
    # Final manifest
    # --------------------------------------------------

    manifest = {
        "phase": "P2",
        "phase_name":
            "Data Validation and Exploratory Data Analysis",
        "datasets": DATASETS,
        "dataset_count":
            len(DATASETS),
        "status":
            (
                "PASS"
                if (
                    validation["status"]
                    == "PASS"
                ).all()
                else "REVIEW_REQUIRED"
            ),
    }

    _save_json(
        manifest,
        output_root
        / "p2_manifest.json",
    )

    print(
        "\nP2 pipeline completed."
    )

    print(
        validation.to_string(
            index=False
        )
    )