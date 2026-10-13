
import pandas as pd
import numpy as np

# Maximum permitted time gap between observations used for
# temporal transitions and rolling statistics.
MAX_TRANSITION_GAP_DAYS = 30

SATELLITE_METRICS = [
    "ndvi_mean",
    "ndwi_mean",
    "vv_mean",
    "vh_mean",
]


# ============================================================
# SHARED TEMPORAL PREPARATION
# ============================================================

def _prepare_temporal_data(df):
    """
    Sort observations and identify valid temporal segments.

    A new segment starts for each farm when:
    - There is no previous observation.
    - The gap exceeds MAX_TRANSITION_GAP_DAYS.

    Temporal calculations must not cross segment boundaries.
    """

    if "date" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'date' column.")

    data = df.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    )

    if data["date"].isna().any():
        raise ValueError(
            "Invalid or missing dates found. "
            "Clean the date column before feature engineering."
        )

    # Support both single-farm datasets and multi-farm datasets.
    if "farm_id" in data.columns:
        if data["farm_id"].isna().any():
            raise ValueError("Missing farm_id values found.")

        group_key = "farm_id"
    else:
        group_key = "_feature_farm_group"
        data[group_key] = "SINGLE_FARM"

    data = (
        data.sort_values([group_key, "date"])
        .reset_index(drop=True)
    )

    # Duplicate observations on the same date can make temporal
    # transitions ambiguous, so reject duplicate farm-date rows.
    if data.duplicated([group_key, "date"]).any():
        raise ValueError(
            "Duplicate farm-date observations found. "
            "Resolve duplicates before feature engineering."
        )

    grouped_dates = data.groupby(
        group_key,
        sort=False,
    )["date"]

    data["days_since_previous"] = (
        grouped_dates.diff().dt.total_seconds() / 86400.0
    )

    # The first observation for a farm has no previous transition.
    data["transition_gap_valid"] = (
        data["days_since_previous"].notna()
        & data["days_since_previous"].le(
            MAX_TRANSITION_GAP_DAYS
        )
    )

    # Start a new segment at each farm's first observation and
    # whenever a gap is longer than the permitted threshold.
    segment_start = (
        data["days_since_previous"].isna()
        | data["days_since_previous"].gt(
            MAX_TRANSITION_GAP_DAYS
        )
    )

    data["_temporal_segment"] = (
        segment_start.groupby(
            data[group_key],
            sort=False,
        ).cumsum()
    )

    data.attrs["_temporal_group_key"] = group_key

    return data


def _segment_groupby(data):
    """Group rows by farm and uninterrupted temporal segment."""

    group_key = data.attrs.get(
        "_temporal_group_key",
        "farm_id" if "farm_id" in data.columns else None,
    )

    if group_key is None:
        raise ValueError(
            "Temporal grouping information is missing."
        )

    return data.groupby(
        [group_key, "_temporal_segment"],
        sort=False,
    )


def _calculate_segmented_delta(data, column):
    """
    Calculate a difference only between consecutive observations
    within the same farm and valid temporal segment.
    """

    return _segment_groupby(data)[column].diff()


# ============================================================
# TEMPORAL FEATURES
# ============================================================

def add_temporal_features(df):
    """
    Create gap-aware satellite temporal features.

    Deltas and rolling statistics never cross gaps longer than
    MAX_TRANSITION_GAP_DAYS.

    Missing satellite measurements remain missing.
    """

    data = _prepare_temporal_data(df)

    segment_groups = _segment_groupby(data)

    for metric in SATELLITE_METRICS:

        if metric not in data.columns:
            continue

        data[metric] = pd.to_numeric(
            data[metric],
            errors="coerce",
        )

        # ----------------------------------------------------
        # Temporal change within a valid segment
        # ----------------------------------------------------

        delta = _calculate_segmented_delta(
            data,
            metric,
        )

        data[f"{metric}_delta"] = delta

        # ----------------------------------------------------
        # Magnitude of temporal change
        # ----------------------------------------------------

        data[f"{metric}_abs_change"] = delta.abs()

        # ----------------------------------------------------
        # Rolling mean within each valid segment
        # ----------------------------------------------------

        data[f"{metric}_rolling_mean_3"] = (
            segment_groups[metric]
            .transform(
                lambda series: series.rolling(
                    window=3,
                    min_periods=2,
                ).mean()
            )
        )

        # ----------------------------------------------------
        # Rolling standard deviation within each segment
        # ----------------------------------------------------

        data[f"{metric}_rolling_std_3"] = (
            segment_groups[metric]
            .transform(
                lambda series: series.rolling(
                    window=3,
                    min_periods=2,
                ).std()
            )
        )

    return data


# ============================================================
# SAR FEATURES
# ============================================================

def add_sar_features(df):
    """
    Create SAR features.

    VV - VH is retained as a polarization difference.
    Changes in that difference are calculated within valid
    temporal segments.
    """

    data = df.copy()

    if (
        "vv_mean" in data.columns
        and "vh_mean" in data.columns
    ):

        data["vv_vh_difference"] = (
            data["vv_mean"] - data["vh_mean"]
        )

        if "_temporal_segment" in data.columns:
            data["vv_vh_difference_delta"] = (
                _calculate_segmented_delta(
                    data,
                    "vv_vh_difference",
                )
            )
        else:
            # Preserve safe behavior if called independently.
            prepared = _prepare_temporal_data(data)

            data["days_since_previous"] = (
                prepared["days_since_previous"]
            )
            data["transition_gap_valid"] = (
                prepared["transition_gap_valid"]
            )
            data["_temporal_segment"] = (
                prepared["_temporal_segment"]
            )

            data["vv_vh_difference_delta"] = (
                _calculate_segmented_delta(
                    data,
                    "vv_vh_difference",
                )
            )

    return data


# ============================================================
# OPTICAL FEATURES
# ============================================================

def add_optical_features(df):
    """
    Create combined optical features.

    Optical differences and their temporal changes preserve
    missing measurements and respect temporal segment boundaries.
    """

    data = df.copy()

    if (
        "ndvi_mean" in data.columns
        and "ndwi_mean" in data.columns
    ):

        data["ndvi_ndwi_difference"] = (
            data["ndvi_mean"] - data["ndwi_mean"]
        )

        if "_temporal_segment" in data.columns:
            data["ndvi_ndwi_difference_delta"] = (
                _calculate_segmented_delta(
                    data,
                    "ndvi_ndwi_difference",
                )
            )
        else:
            prepared = _prepare_temporal_data(data)

            data["days_since_previous"] = (
                prepared["days_since_previous"]
            )
            data["transition_gap_valid"] = (
                prepared["transition_gap_valid"]
            )
            data["_temporal_segment"] = (
                prepared["_temporal_segment"]
            )

            data["ndvi_ndwi_difference_delta"] = (
                _calculate_segmented_delta(
                    data,
                    "ndvi_ndwi_difference",
                )
            )

    # Preserve optical quality information if available.
    if (
        "ndvi_quality_ok" in data.columns
        and "ndwi_quality_ok" in data.columns
    ):

        data["optical_quality_ok"] = (
            data["ndvi_quality_ok"].fillna(False).astype(bool)
            & data["ndwi_quality_ok"].fillna(False).astype(bool)
        )

    return data


# ============================================================
# DATE FEATURES
# ============================================================

def add_date_features(df):
    """Add calendar features without changing temporal gaps."""

    data = df.copy()

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day_of_year"] = data["date"].dt.dayofyear

    if "farm_id" in data.columns:
        first_date = data.groupby("farm_id")["date"].transform(
            "min"
        )
    else:
        first_date = data["date"].min()

    data["days_since_start"] = (
        data["date"] - first_date
    ).dt.total_seconds() / 86400.0

    return data


# ============================================================
# COMPLETE FEATURE ENGINEERING PIPELINE
# ============================================================

def create_features(df):
    """
    Run the complete satellite feature engineering pipeline.

    Existing public function names and feature-column names
    are preserved for compatibility with the main pipeline.
    """

    data = add_temporal_features(df)
    data = add_sar_features(data)
    data = add_optical_features(data)
    data = add_date_features(data)

    # Internal helper columns are not needed in the final output.
    data = data.drop(
        columns=["_temporal_segment", "_feature_farm_group"],
        errors="ignore",
    )

    return data
