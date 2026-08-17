from __future__ import annotations

import matplotlib.pyplot as plt


def save_line_plot(
    df,
    x,
    y_columns,
    title,
    output_path,
):

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for column in y_columns:
        ax.plot(
            df[x],
            df[column],
            label=column,
            linewidth=1,
        )

    ax.set_title(title)
    ax.set_xlabel(x)
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
    )

    plt.close(fig)