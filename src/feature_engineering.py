import pandas as pd
import numpy as np


# ============================================================
# TEMPORAL FEATURES
# ============================================================

def add_temporal_features(df):
    """
    Create robust temporal features.

    We deliberately avoid percentage change because indices
    such as NDVI and NDWI can be close to zero, making
    percentage changes unstable.
    """

    data = df.copy()

    data = (
        data
        .sort_values("date")
        .reset_index(drop=True)
    )

    metrics = [
        "ndvi_mean",
        "ndwi_mean",
        "vv_mean",
        "vh_mean"
    ]

    for metric in metrics:

        if metric not in data.columns:
            continue

        # ----------------------------------------------------
        # Absolute temporal change
        # ----------------------------------------------------

        data[
            f"{metric}_delta"
        ] = data[metric].diff()

        # ----------------------------------------------------
        # Absolute magnitude of change
        # ----------------------------------------------------

        data[
            f"{metric}_abs_change"
        ] = data[
            f"{metric}_delta"
        ].abs()

        # ----------------------------------------------------
        # Rolling mean
        # ----------------------------------------------------

        data[
            f"{metric}_rolling_mean_3"
        ] = (
            data[metric]
            .rolling(
                window=3,
                min_periods=2
            )
            .mean()
        )

        # ----------------------------------------------------
        # Rolling standard deviation
        # ----------------------------------------------------

        data[
            f"{metric}_rolling_std_3"
        ] = (
            data[metric]
            .rolling(
                window=3,
                min_periods=2
            )
            .std()
        )

    # --------------------------------------------------------
    # Time between observations
    # --------------------------------------------------------

    data["days_since_previous"] = (
        data["date"]
        .diff()
        .dt.total_seconds()
        / 86400.0
    )

    return data


# ============================================================
# SAR FEATURES
# ============================================================

def add_sar_features(df):
    """
    Create SAR features.

    VV and VH are normally expressed in dB, so VV - VH is
    meaningful as a polarization difference.

    We do NOT create VV/VH arithmetic ratios because ratios
    of dB values are not physically appropriate.
    """

    data = df.copy()

    if (
        "vv_mean" in data.columns
        and "vh_mean" in data.columns
    ):

        data["vv_vh_difference"] = (
            data["vv_mean"]
            - data["vh_mean"]
        )

        data["vv_vh_difference_delta"] = (
            data["vv_vh_difference"].diff()
        )

    return data


# ============================================================
# OPTICAL FEATURES
# ============================================================

def add_optical_features(df):
    """
    Create combined optical features.
    """

    data = df.copy()

    if (
        "ndvi_mean" in data.columns
        and "ndwi_mean" in data.columns
    ):

        data["ndvi_ndwi_difference"] = (
            data["ndvi_mean"]
            - data["ndwi_mean"]
        )

        data["ndvi_ndwi_difference_delta"] = (
            data["ndvi_ndwi_difference"].diff()
        )

    # --------------------------------------------------------
    # Optical quality information
    # --------------------------------------------------------

    if (
        "ndvi_quality_ok" in data.columns
        and "ndwi_quality_ok" in data.columns
    ):

        data["optical_quality_ok"] = (
            data["ndvi_quality_ok"]
            & data["ndwi_quality_ok"]
        )

    return data


# ============================================================
# DATE FEATURES
# ============================================================

def add_date_features(df):
    """
    Add calendar-related features.
    """

    data = df.copy()

    data["year"] = (
        data["date"].dt.year
    )

    data["month"] = (
        data["date"].dt.month
    )

    data["day_of_year"] = (
        data["date"].dt.dayofyear
    )

    data["days_since_start"] = (
        data["date"]
        - data["date"].min()
    ).dt.total_seconds() / 86400.0

    return data


# ============================================================
# COMPLETE FEATURE ENGINEERING PIPELINE
# ============================================================

def create_features(df):
    """
    Run the complete satellite feature engineering pipeline.
    """

    data = df.copy()

    data = add_temporal_features(
        data
    )

    data = add_sar_features(
        data
    )

    data = add_optical_features(
        data
    )

    data = add_date_features(
        data
    )

    return data