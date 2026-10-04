import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

# Optical observations above this cloud-coverage percentage
# will be considered unreliable for NDVI/NDWI analysis.
#
# This is NOT the AWD threshold.
MAX_CLOUD_PERCENT = 50.0

# Maximum allowed difference between an optical observation
# and the nearest SAR observation.
MAX_SAR_GAP_DAYS = 5


# ============================================================
# BASIC DATA CLEANING
# ============================================================

def clean_satellite_data(df):
    """
    Perform basic cleaning on a satellite dataset.
    """

    data = df.copy()

    # --------------------------------------------------------
    # Check date column
    # --------------------------------------------------------

    if "date" not in data.columns:
        raise ValueError(
            "Satellite dataset must contain a 'date' column."
        )

    # --------------------------------------------------------
    # Convert date
    # --------------------------------------------------------

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    # Remove rows where date could not be parsed
    data = data.dropna(
        subset=["date"]
    )

    # --------------------------------------------------------
    # Convert possible numeric columns
    # --------------------------------------------------------

    for column in data.columns:

        if column == "date":
            continue

        # Try to convert columns to numeric.
        # If a column is not numeric, pandas will produce NaN.
        converted = pd.to_numeric(
            data[column],
            errors="coerce"
        )

        # Only replace the original column if conversion
        # produced at least one valid numeric value.
        if converted.notna().any():
            data[column] = converted

    # --------------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------------

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Sort by date
    # --------------------------------------------------------

    data = (
        data
        .sort_values("date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Handle duplicate dates
    # --------------------------------------------------------

    if data["date"].duplicated().any():

        numeric_columns = data.select_dtypes(
            include=np.number
        ).columns

        aggregation = {
            column: "mean"
            for column in numeric_columns
        }

        data = (
            data
            .groupby(
                "date",
                as_index=False
            )
            .agg(aggregation)
        )

    return data


# ============================================================
# OPTICAL QUALITY CONTROL
# ============================================================

def apply_optical_quality_control(
    df,
    max_cloud_percent=MAX_CLOUD_PERCENT
):
    """
    Apply cloud-quality control to optical observations.

    NDVI and NDWI observations with cloud coverage above
    max_cloud_percent are marked as invalid.

    We keep the observation/date itself because the date may
    still be useful for temporal alignment and SAR matching.

    The satellite measurements themselves are changed to NaN
    when their cloud coverage is too high.
    """

    data = df.copy()

    optical_metrics = [
        "ndvi",
        "ndwi"
    ]

    for metric in optical_metrics:

        cloud_column = (
            f"{metric}_cloud_coverage"
        )

        mean_column = (
            f"{metric}_mean"
        )

        # ----------------------------------------------------
        # If required columns don't exist, skip this metric
        # ----------------------------------------------------

        if cloud_column not in data.columns:
            continue

        if mean_column not in data.columns:
            continue

        # ----------------------------------------------------
        # Make cloud coverage numeric
        # ----------------------------------------------------

        data[cloud_column] = pd.to_numeric(
            data[cloud_column],
            errors="coerce"
        )

        # ----------------------------------------------------
        # Create quality flag
        #
        # We deliberately use object dtype so that the column
        # can contain True, False and None.
        # ----------------------------------------------------

        quality_column = (
            f"{metric}_quality_ok"
        )

        data[quality_column] = (
            data[cloud_column]
            <= max_cloud_percent
        ).astype("object")

        # If cloud coverage is unknown,
        # quality status is also unknown.
        missing_cloud = (
            data[cloud_column].isna()
        )

        data.loc[
            missing_cloud,
            quality_column
        ] = None

        # ----------------------------------------------------
        # Identify cloudy observations
        # ----------------------------------------------------

        invalid_mask = (
            data[cloud_column]
            > max_cloud_percent
        )

        # ----------------------------------------------------
        # Invalidate optical measurements
        #
        # Keep the cloud-coverage and quality columns.
        # Only satellite measurement/statistical columns
        # are set to NaN.
        # ----------------------------------------------------

        metric_columns = [
            column
            for column in data.columns
            if column.startswith(
                f"{metric}_"
            )
            and column != cloud_column
            and column != quality_column
        ]

        for column in metric_columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

            data.loc[
                invalid_mask,
                column
            ] = np.nan

    return data


# ============================================================
# TEMPORAL ALIGNMENT
# ============================================================

def align_satellite_data(
    ndvi,
    ndwi,
    vv,
    vh,
    max_gap_days=MAX_SAR_GAP_DAYS
):
    """
    Align optical and SAR satellite observations.

    NDVI and NDWI are merged using their actual dates.

    VV and VH are merged using their actual SAR dates.

    The nearest SAR observation is then matched to each
    optical observation, provided the difference is within
    max_gap_days.

    The SAR date and time gap are retained.
    """

    # ========================================================
    # 1. Clean individual datasets
    # ========================================================

    ndvi = clean_satellite_data(ndvi)

    ndwi = clean_satellite_data(ndwi)

    vv = clean_satellite_data(vv)

    vh = clean_satellite_data(vh)

    # ========================================================
    # 2. Apply optical quality control
    # ========================================================

    ndvi = apply_optical_quality_control(
        ndvi,
        max_cloud_percent=MAX_CLOUD_PERCENT
    )

    ndwi = apply_optical_quality_control(
        ndwi,
        max_cloud_percent=MAX_CLOUD_PERCENT
    )

    # ========================================================
    # 3. Merge NDVI and NDWI
    # ========================================================

    optical = pd.merge(
        ndvi,
        ndwi,
        on="date",
        how="outer"
    )

    optical = (
        optical
        .sort_values("date")
        .reset_index(drop=True)
    )

    # ========================================================
    # 4. Rename SAR date columns
    # ========================================================

    vv = vv.rename(
        columns={
            "date": "sar_date"
        }
    )

    vh = vh.rename(
        columns={
            "date": "sar_date"
        }
    )

    # ========================================================
    # 5. Merge VV and VH
    # ========================================================

    sar = pd.merge(
        vv,
        vh,
        on="sar_date",
        how="outer"
    )

    sar = (
        sar
        .sort_values("sar_date")
        .reset_index(drop=True)
    )

    # ========================================================
    # 6. Sort before merge_asof
    # ========================================================

    optical = optical.sort_values(
        "date"
    ).reset_index(drop=True)

    sar = sar.sort_values(
        "sar_date"
    ).reset_index(drop=True)

    # ========================================================
    # 7. Match nearest SAR observation
    # ========================================================

    aligned = pd.merge_asof(
        optical,
        sar,
        left_on="date",
        right_on="sar_date",
        direction="nearest",
        tolerance=pd.Timedelta(
            days=max_gap_days
        )
    )

    # ========================================================
    # 8. Calculate SAR time gap
    # ========================================================

    aligned["sar_gap_days"] = (
        aligned["date"]
        - aligned["sar_date"]
    ).abs().dt.total_seconds() / 86400.0

    # ========================================================
    # 9. SAR match validity
    # ========================================================

    aligned["sar_match_valid"] = (
        aligned["sar_date"].notna()
    )

    # ========================================================
    # 10. Sort final dataset
    # ========================================================

    aligned = (
        aligned
        .sort_values("date")
        .reset_index(drop=True)
    )

    return aligned


# ============================================================
# MISSING VALUE REPORT
# ============================================================

def missing_value_report(df):
    """
    Generate a missing-value report.
    """

    report = pd.DataFrame({
        "column": df.columns,

        "missing_count": (
            df.isna()
            .sum()
            .values
        ),

        "missing_percentage": (
            df.isna()
            .mean()
            .values
            * 100
        )
    })

    return report.sort_values(
        "missing_percentage",
        ascending=False
    )


# ============================================================
# QUALITY REPORT
# ============================================================

def generate_quality_report(df):
    """
    Print a satellite data quality report.
    """

    print("\n" + "=" * 70)
    print("SATELLITE DATA QUALITY REPORT")
    print("=" * 70)

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print(
        f"\nTotal observations: {len(df)}"
    )

    print(
        f"Total columns: {len(df.columns)}"
    )

    if "date" in df.columns:

        print(
            f"Date range: "
            f"{df['date'].min().date()} "
            f"to "
            f"{df['date'].max().date()}"
        )

    # --------------------------------------------------------
    # Optical quality
    # --------------------------------------------------------

    for metric in [
        "ndvi",
        "ndwi"
    ]:

        cloud_column = (
            f"{metric}_cloud_coverage"
        )

        quality_column = (
            f"{metric}_quality_ok"
        )

        if cloud_column in df.columns:

            print(
                f"\n{metric.upper()} cloud coverage:"
            )

            print(
                df[cloud_column]
                .describe()
                .to_string()
            )

        if quality_column in df.columns:

            valid_count = (
                df[quality_column]
                .eq(True)
                .sum()
            )

            invalid_count = (
                df[quality_column]
                .eq(False)
                .sum()
            )

            unknown_count = (
                df[quality_column]
                .isna()
                .sum()
            )

            print(
                f"\n{metric.upper()} quality:"
            )

            print(
                f"Valid:   {valid_count}"
            )

            print(
                f"Invalid: {invalid_count}"
            )

            print(
                f"Unknown: {unknown_count}"
            )

    # --------------------------------------------------------
    # SAR matching
    # --------------------------------------------------------

    if "sar_match_valid" in df.columns:

        matched = (
            df["sar_match_valid"]
            .sum()
        )

        print(
            f"\nValid SAR matches: "
            f"{matched}/{len(df)}"
        )

    if "sar_gap_days" in df.columns:

        valid_gaps = (
            df["sar_gap_days"]
            .dropna()
        )

        if len(valid_gaps) > 0:

            print(
                "\nSAR matching gap (days):"
            )

            print(
                valid_gaps
                .describe()
                .to_string()
            )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print(
        "\nMissing values:"
    )

    print(
        missing_value_report(df)
        .to_string(index=False)
    )


# ============================================================
# FUTURE ML DATASET PREPARATION
# ============================================================

def prepare_ml_dataset(
    df,
    target_column="awd_followed"
):
    """
    Prepare a labeled dataset for future ML training.

    This function should only be used after real IoT
    measurements have been integrated.

    The target column should be generated from the actual
    IoT water-level measurements using the AWD threshold.
    """

    if target_column not in df.columns:

        raise ValueError(
            f"Target column '{target_column}' "
            "does not exist.\n"
            "Real IoT data must be integrated before "
            "training the machine-learning model."
        )

    data = df.copy()

    # --------------------------------------------------------
    # Remove rows without labels
    # --------------------------------------------------------

    data = data.dropna(
        subset=[
            target_column
        ]
    )

    # --------------------------------------------------------
    # Select numeric features
    # --------------------------------------------------------

    feature_columns = (
        data
        .select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    # --------------------------------------------------------
    # Remove target from feature list
    # --------------------------------------------------------

    if target_column in feature_columns:

        feature_columns.remove(
            target_column
        )

    # --------------------------------------------------------
    # Create X and y
    # --------------------------------------------------------

    X = data[
        feature_columns
    ]

    y = data[
        target_column
    ]

    return X, y, feature_columns