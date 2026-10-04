from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PLOT DIRECTORY
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

PLOT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "plots"
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TIME SERIES PLOT
# ============================================================

def plot_satellite_timeseries(
    df,
    column,
    title=None
):
    """
    Plot one satellite metric over time.
    """

    if column not in df.columns:

        print(
            f"Skipping plot. "
            f"Column not found: {column}"
        )

        return

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        df["date"],
        df[column],
        marker="o"
    )

    plt.xlabel("Date")
    plt.ylabel(column)

    if title is None:
        title = f"{column} Time Series"

    plt.title(title)

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    output_file = (
        PLOT_DIR
        / f"{column}_timeseries.png"
    )

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.close()

    print(
        f"Saved plot: {output_file}"
    )


# ============================================================
# COMBINED SATELLITE PLOT
# ============================================================

def plot_satellite_metrics(df):
    """
    Generate separate time-series plots for the main
    satellite metrics.
    """

    metrics = [
        "ndvi_mean",
        "ndwi_mean",
        "vv_mean",
        "vh_mean"
    ]

    for metric in metrics:

        plot_satellite_timeseries(
            df,
            metric
        )


# ============================================================
# SAVE DATA
# ============================================================

def save_dataframe(
    df,
    output_path
):
    """
    Save a dataframe to CSV.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved data: {output_path}"
    )