
"""
AWD Satellite Analysis Pipeline
Prototype version using the current combined satellite dataset.

This pipeline:
1. Loads satellite features extracted from Google Earth Engine.
2. Standardizes column names.
3. Cleans and validates satellite observations.
4. Creates temporal and multi-sensor features.
5. Calculates transparent AWD-related indicators.
6. Saves processed datasets and plots.
7. Reports IoT simulation data separately.

No Random Forest model is trained because the current prototype
does not yet have synchronized, validated ground-truth labels.
"""

from pathlib import Path

import pandas as pd

from src.preprocessing import (
    clean_satellite_data,
    generate_quality_report,
)

from src.feature_engineering import create_features

from src.awd_detection import analyze_awd

from src.results import (
    save_dataframe,
    plot_satellite_metrics,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
IOT_FILE = DATA_DIR / "raw" / "iot" / "iot.csv"

SATELLITE_INPUT = (
    PROCESSED_DIR / "satellite_timeseries.csv"
)

# Use separate output files so the original extracted data
# is not overwritten.
ALIGNED_OUTPUT = (
    PROCESSED_DIR / "satellite_timeseries_aligned.csv"
)

FEATURE_OUTPUT = (
    PROCESSED_DIR / "satellite_features.csv"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD CURRENT SATELLITE DATA
# ============================================================

def load_current_satellite_data():
    """
    Load the combined satellite time-series CSV.

    Accepted source features:
        NDVI / ndvi
        NDWI / ndwi
        VV / vv
        VH / vh

    Standardized feature names:
        ndvi_mean
        ndwi_mean
        vv_mean
        vh_mean
    """

    print("\n" + "=" * 70)
    print("STEP 1: LOADING SATELLITE DATA")
    print("=" * 70)

    if not SATELLITE_INPUT.exists():
        raise FileNotFoundError(
            "Satellite input file was not found:\n"
            f"{SATELLITE_INPUT}\n\n"
            "Run src/extract_satellite_features.py first."
        )

    print(f"Reading: {SATELLITE_INPUT}")

    df = pd.read_csv(SATELLITE_INPUT)

    if df.empty:
        raise ValueError(
            "The satellite CSV exists but contains no records."
        )

    # Normalize source column names, e.g. NDVI -> ndvi.
    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    required_columns = [
        "farm_id",
        "date",
        "ndvi",
        "ndwi",
        "vv",
        "vh",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required satellite columns are missing: "
            f"{missing_columns}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Standardize the names expected by the existing modules.
    df = df.rename(
        columns={
            "ndvi": "ndvi_mean",
            "ndwi": "ndwi_mean",
            "vv": "vv_mean",
            "vh": "vh_mean",
        }
    )

    # Validate dates.
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    invalid_dates = int(df["date"].isna().sum())

    if invalid_dates:
        print(
            f"Warning: dropping {invalid_dates} "
            "rows with invalid dates."
        )

        df = df.dropna(
            subset=["date"]
        ).copy()

    if df.empty:
        raise ValueError(
            "No valid satellite observations remain after "
            "date validation."
        )

    # Standardize farm identifiers without inventing new IDs.
    df["farm_id"] = (
        df["farm_id"]
        .astype("string")
        .str.strip()
    )

    df = df.dropna(
        subset=["farm_id"]
    ).copy()

    df = df[
        df["farm_id"] != ""
    ].copy()

    # Convert satellite measurements to numeric.
    metric_columns = [
        "ndvi_mean",
        "ndwi_mean",
        "vv_mean",
        "vh_mean",
    ]

    for column in metric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Sort by farm and date.
    df = df.sort_values(
        ["farm_id", "date"]
    ).reset_index(drop=True)

    # Do not silently choose between conflicting duplicate records.
    duplicate_mask = df.duplicated(
        subset=["farm_id", "date"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())

        raise ValueError(
            f"Found {duplicate_count} rows belonging to duplicate "
            "farm/date combinations.\n"
            "Resolve duplicate observations before continuing."
        )

    print(f"Satellite observations: {len(df)}")
    print(f"Farms: {df['farm_id'].nunique()}")
    print(
        f"Date range: {df['date'].min().date()} "
        f"to {df['date'].max().date()}"
    )

    print("\nSatellite feature columns:")
    print(metric_columns)

    print("\nMissing satellite values:")
    print(
        df[metric_columns]
        .isna()
        .sum()
        .to_string()
    )

    return df


# ============================================================
# REPORT AVAILABLE IOT DATA
# ============================================================

def report_iot_data():
    """
    Report the IoT dataset without forcing an incorrect join.
    """

    print("\n" + "=" * 70)
    print("IOT DATA STATUS")
    print("=" * 70)

    if not IOT_FILE.exists():
        print("IoT CSV not found.")
        print("Continuing with satellite-only analysis.")
        return

    try:
        iot = pd.read_csv(IOT_FILE)
    except Exception as error:
        print(f"Could not read the IoT CSV: {error}")
        return

    print(f"IoT readings available: {len(iot)}")

    if "data_source" in iot.columns:
        print("\nData sources:")
        print(
            iot["data_source"]
            .value_counts(dropna=False)
            .to_string()
        )

    if "timestamp" in iot.columns and not iot.empty:
        timestamps = pd.to_datetime(
            iot["timestamp"],
            errors="coerce",
            utc=True,
        ).dropna()

        if not timestamps.empty:
            print(
                f"\nIoT date range: "
                f"{timestamps.min().date()} "
                f"to {timestamps.max().date()}"
            )

    print(
        "\nIoT readings are retained separately. "
        "They will not be treated as synchronized "
        "satellite ground truth."
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("AWD SATELLITE ANALYSIS PIPELINE")
    print("=" * 70)

    print("Mode: SATELLITE PROTOTYPE")
    print("ML training: DISABLED until valid labels are available")

    # --------------------------------------------------------
    # STEP 1: Load the current combined satellite CSV.
    # --------------------------------------------------------

    satellite_data = load_current_satellite_data()

    # --------------------------------------------------------
    # STEP 2: Clean and validate the satellite dataset.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 2: DATA CLEANING AND QUALITY REPORT")
    print("=" * 70)

    satellite_data = clean_satellite_data(
        satellite_data
    )

    if satellite_data.empty:
        raise ValueError(
            "Satellite dataset is empty after cleaning."
        )

    generate_quality_report(
        satellite_data
    )

    # --------------------------------------------------------
    # STEP 3: Save the cleaned satellite observations.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 3: SAVING CLEANED SATELLITE DATA")
    print("=" * 70)

    save_dataframe(
        satellite_data,
        ALIGNED_OUTPUT,
    )

    # --------------------------------------------------------
    # STEP 4: Feature engineering.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 4: FEATURE ENGINEERING")
    print("=" * 70)

    feature_data = create_features(
        satellite_data
    )

    print(
        f"Rows after feature engineering: "
        f"{len(feature_data)}"
    )

    print(
        f"Total feature columns: "
        f"{len(feature_data.columns)}"
    )

    # --------------------------------------------------------
    # STEP 5: AWD-related satellite indicators.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 5: AWD SATELLITE INDICATORS")
    print("=" * 70)

    feature_data = analyze_awd(
        feature_data
    )

    print("AWD-related satellite indicators calculated.")

    print(
        "Note: satellite indicators alone do not establish "
        "the actual field water level."
    )

    # --------------------------------------------------------
    # STEP 6: Save feature dataset.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 6: SAVING SATELLITE FEATURES")
    print("=" * 70)

    save_dataframe(
        feature_data,
        FEATURE_OUTPUT,
    )

    # --------------------------------------------------------
    # STEP 7: Generate plots.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 7: GENERATING PLOTS")
    print("=" * 70)

    plot_satellite_metrics(
        feature_data
    )

    # --------------------------------------------------------
    # STEP 8: Report IoT prototype status.
    # --------------------------------------------------------

    report_iot_data()

    # --------------------------------------------------------
    # STEP 9: Final summary.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SATELLITE PROTOTYPE PIPELINE COMPLETED")
    print("=" * 70)

    print("\nGenerated files:")

    print(f"1. {ALIGNED_OUTPUT}")
    print(f"2. {FEATURE_OUTPUT}")

    print(
        "\nPlots directory:\n"
        f"{PROJECT_DIR / 'outputs' / 'plots'}"
    )

    print("\nMachine-learning status:")
    print("Random Forest was not trained.")

    print(
        "Reason: synchronized, validated AWD labels "
        "are not yet available."
    )

    print("\nFuture deployment workflow:")
    print("1. Collect real field water-level measurements.")
    print("2. Obtain satellite observations for the same farms.")
    print("3. Align observations using farm IDs and acquisition dates.")
    print("4. Generate labels using verified field measurements.")
    print("5. Evaluate the model using farm-aware data splits.")
    print("6. Train and evaluate the Random Forest model.")

    print("\nPrototype run finished successfully.")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
