
import numpy as np
import pandas as pd


# ============================================================
# AWD CONFIGURATION
# ============================================================

AWD_THRESHOLD_CM = -20.0
MAX_TRANSITION_GAP_DAYS = 30


# ============================================================
# VALIDATION AND TEMPORAL ORDERING
# ============================================================

def prepare_temporal_data(df):
    """
    Validate input and sort observations chronologically by farm.

    Calculate days_since_previous from the actual observation
    dates. Large gaps are retained in the dataset but excluded
    from short-term transition calculations.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    data = df.copy()

    if "date" not in data.columns:
        raise ValueError("Required column 'date' is missing.")

    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    if data["date"].isna().any():
        raise ValueError(
            "Invalid dates found. Clean them before AWD analysis."
        )

    if "farm_id" not in data.columns:
        data["farm_id"] = "UNKNOWN_FARM"

    data["farm_id"] = (
        data["farm_id"].astype("string").str.strip()
    )

    if (
        data["farm_id"].isna().any()
        or (data["farm_id"] == "").any()
    ):
        raise ValueError("Missing farm IDs found.")

    duplicate_mask = data.duplicated(
        subset=["farm_id", "date"],
        keep=False,
    )

    if duplicate_mask.any():
        raise ValueError(
            "Duplicate farm/date observations found. "
            "Resolve duplicates before calculating changes."
        )

    data = data.sort_values(
        ["farm_id", "date"]
    ).reset_index(drop=True)

    # Always calculate the actual gap from dates.
    data["days_since_previous"] = (
        data.groupby("farm_id", sort=False)["date"]
        .diff()
        .dt.total_seconds()
        .div(86400)
    )

    # A transition is suitable for short-term analysis only
    # when the previous observation is within the allowed gap.
    data["transition_gap_valid"] = (
        data["days_since_previous"].le(
            MAX_TRANSITION_GAP_DAYS
        )
    ).fillna(False)

    return data


def grouped_change(data, column):
    """
    Calculate change from the previous observation for each farm.

    Changes are suppressed when the observation gap exceeds
    MAX_TRANSITION_GAP_DAYS. Original measurements are preserved.
    """
    if column not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype=float)

    values = pd.to_numeric(data[column], errors="coerce")

    changes = values.groupby(
        data["farm_id"], sort=False
    ).diff()

    valid_gap = data["transition_gap_valid"]

    return changes.where(valid_gap)


# ============================================================
# OPTICAL DRYING INDICATORS
# ============================================================

def calculate_optical_indicators(df):
    """
    Calculate NDWI and NDVI changes.

    Negative changes can be consistent with drying or vegetation
    decline, but are not confirmed AWD labels.
    """
    data = df.copy()

    if "ndwi_mean" in data.columns:
        data["ndwi_change"] = grouped_change(
            data, "ndwi_mean"
        )

        data["ndwi_drying_signal"] = (
            data["ndwi_change"] < 0
        ).where(data["ndwi_change"].notna())

    if "ndvi_mean" in data.columns:
        data["ndvi_change"] = grouped_change(
            data, "ndvi_mean"
        )

        data["ndvi_decline_signal"] = (
            data["ndvi_change"] < 0
        ).where(data["ndvi_change"].notna())

    return data


# ============================================================
# SENTINEL-1 SAR INDICATORS
# ============================================================

def calculate_sar_indicators(df):
    """
    Calculate Sentinel-1 VV and VH temporal changes.

    SAR changes do not have a universal wet/dry interpretation.
    """
    data = df.copy()

    if "vv_mean" in data.columns:
        data["vv_change"] = grouped_change(
            data, "vv_mean"
        )
        data["vv_abs_change"] = data["vv_change"].abs()

    if "vh_mean" in data.columns:
        data["vh_change"] = grouped_change(
            data, "vh_mean"
        )
        data["vh_abs_change"] = data["vh_change"].abs()

    if "vv_vh_difference" in data.columns:
        data["vv_vh_change"] = grouped_change(
            data, "vv_vh_difference"
        )

    return data


# ============================================================
# OPTICAL TRANSITION SUMMARY
# ============================================================

def calculate_optical_transition(df):
    """
    Summarize available optical signals.

    Unknown signals remain unknown when no optical indicator
    can be calculated for that observation.
    """
    data = df.copy()

    signal_columns = [
        column
        for column in (
            "ndwi_drying_signal",
            "ndvi_decline_signal",
        )
        if column in data.columns
    ]

    if not signal_columns:
        data["optical_change_count"] = np.nan
        data["optical_signal_available"] = False
        data["optical_transition_observed"] = pd.NA
        return data

    signal_values = data[signal_columns]

    data["optical_signal_available"] = (
        signal_values.notna().any(axis=1)
    )

    data["optical_change_count"] = (
        signal_values.fillna(False).astype(int).sum(axis=1)
    )

    data["optical_transition_observed"] = (
        data["optical_change_count"] > 0
    ).where(data["optical_signal_available"])

    return data


# ============================================================
# SAR TRANSITION SUMMARY
# ============================================================

def calculate_sar_transition(df):
    """
    Summarize available absolute SAR changes.

    Magnitude indicates change, not whether an AWD event occurred.
    """
    data = df.copy()

    sar_columns = [
        column
        for column in ("vv_abs_change", "vh_abs_change")
        if column in data.columns
    ]

    if not sar_columns:
        data["sar_change_magnitude"] = np.nan
        data["sar_transition_observed"] = pd.NA
        return data

    data["sar_change_magnitude"] = (
        data[sar_columns].mean(axis=1, skipna=True)
    )

    data["sar_transition_observed"] = (
        data["sar_change_magnitude"] > 0
    ).where(data["sar_change_magnitude"].notna())

    return data


# ============================================================
# MULTI-SENSOR TRANSITION SUMMARY
# ============================================================

def calculate_transition_indicators(df):
    """
    Combine optical and SAR indicators without inventing labels.

    A long observation gap cannot generate a transition from
    the change across that gap. A separately valid sensor signal
    may still be retained if it is genuinely available.
    """
    data = df.copy()

    data = calculate_optical_transition(data)
    data = calculate_sar_transition(data)

    transition_columns = [
        column
        for column in (
            "optical_transition_observed",
            "sar_transition_observed",
        )
        if column in data.columns
    ]

    if not transition_columns:
        data["satellite_transition_observed"] = pd.NA
        return data

    transition_values = data[transition_columns]

    any_sensor_available = (
        transition_values.notna().any(axis=1)
    )

    any_transition = (
        transition_values.fillna(False).astype(bool).any(axis=1)
    )

    data["satellite_transition_observed"] = (
        any_transition.where(any_sensor_available)
    )

    return data


# ============================================================
# COMPLETE AWD INDICATOR PIPELINE
# ============================================================

def analyze_awd(df):
    """
    Run the satellite-only AWD indicator pipeline.

    The -20 cm threshold is metadata only. This function does
    not classify confirmed AWD events.
    """
    data = prepare_temporal_data(df)

    metric_columns = [
        "ndvi_mean",
        "ndwi_mean",
        "vv_mean",
        "vh_mean",
        "vv_vh_difference",
    ]

    for column in metric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column], errors="coerce"
            )

    data = calculate_optical_indicators(data)
    data = calculate_sar_indicators(data)
    data = calculate_transition_indicators(data)

    # Store configuration as metadata, not a satellite label.
    data["iot_awd_threshold_cm"] = AWD_THRESHOLD_CM

    return data


# ============================================================
# OPTIONAL: CLASSIFY FIELD WATER-LEVEL READINGS
# ============================================================

def classify_iot_awd(
    df,
    depth_column="water_depth_cm",
    threshold_cm=AWD_THRESHOLD_CM,
):
    """
    Classify IoT readings only after confirming sensor calibration
    and the signed-depth convention.

    Assumption:
        Negative depth means water is below the soil surface.

    Under that convention:
        depth < threshold -> threshold crossed
        depth >= threshold -> threshold not crossed

    This is not proof that the complete agronomic AWD protocol
    has been satisfied.
    """
    if depth_column not in df.columns:
        raise ValueError(
            f"Required IoT column '{depth_column}' is missing."
        )

    data = df.copy()

    data[depth_column] = pd.to_numeric(
        data[depth_column], errors="coerce"
    )

    data["iot_awd_threshold_cm"] = threshold_cm

    data["iot_threshold_crossed"] = (
        data[depth_column] < threshold_cm
    ).where(data[depth_column].notna())

    data["iot_threshold_status"] = np.select(
        [
            data[depth_column].isna(),
            data[depth_column] < threshold_cm,
        ],
        [
            "UNKNOWN",
            "THRESHOLD_CROSSED",
        ],
        default="THRESHOLD_NOT_CROSSED",
    )

    return data
