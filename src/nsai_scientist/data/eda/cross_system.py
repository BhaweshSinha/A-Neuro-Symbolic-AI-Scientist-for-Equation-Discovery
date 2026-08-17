from __future__ import annotations

import pandas as pd


def build_cross_system_summary(
    summaries: dict[str, pd.DataFrame],
) -> pd.DataFrame:

    rows = []

    for dataset, summary in summaries.items():

        numeric = summary.select_dtypes(
            include="number"
        )

        rows.append(
            {
                "dataset": dataset,
                "rows": len(summary),
                "numeric_variables":
                    len(numeric.columns),
            }
        )

    return pd.DataFrame(rows)