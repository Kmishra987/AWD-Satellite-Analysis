import numpy as np
import pandas as pd


# ============================================================
# AWD CONFIGURATION
# ============================================================

AWD_THRESHOLD_CM = -20.0


# ============================================================
# OPTICAL DRYING INDICATOR
# ============================================================

def calculate_optical_indicators(df):
    """
    Calculate optical indicators associated with temporal
    drying/wetting changes.

    These are satellite indicators only.
    They are NOT ground-truth AWD labels.
    """

    data = df.copy()

    # --------------------------------------------------------
    # NDWI
    # --------------------------------------------------------

    if "ndwi_mean" in data.columns:

        data["ndwi_change"] = (
            data["ndwi_mean"].diff()
        )

        # Negative NDWI change is treated as a possible
        # drying-direction signal.
        data["ndwi_drying_signal"] = np.where(
            data["ndwi_change"].notna(),
            data["ndwi_change"] < 0,
            np.nan
        )

    # --------------------------------------------------------
    # NDVI
    # --------------------------------------------------------

    if "ndvi_mean" in data.columns:

        data["ndvi_change"] = (
            data["ndvi_mean"].diff()
        )

        # NDVI decline can indicate vegetation stress,
        # but it is NOT interpreted as AWD by itself.
        data["ndvi_decline_signal"] = np.where(
            data["ndvi_change"].notna(),
            data["ndvi_change"] < 0,
            np.nan
        )

    return data


# ============================================================
# SAR INDICATORS
# ============================================================

def calculate_sar_indicators(df):
    """
    Calculate SAR temporal-change indicators.

    SAR direction is kept as a separate signal rather than
    assigning a universal 'wet' or 'dry' direction because
    SAR response depends on crop, structure and surface
    conditions.
    """

    data = df.copy()

    if "vv_mean" in data.columns:

        data["vv_change"] = (
            data["vv_mean"].diff()
        )

        data["vv_abs_change"] = (
            data["vv_change"].abs()
        )

    if "vh_mean" in data.columns:

        data["vh_change"] = (
            data["vh_mean"].diff()
        )

        data["vh_abs_change"] = (
            data["vh_change"].abs()
        )

    if "vv_vh_difference" in data.columns:

        data["vv_vh_change"] = (
            data["vv_vh_difference"].diff()
        )

    return data


# ============================================================
# MULTI-SENSOR TRANSITION INDICATOR
# ============================================================

def calculate_transition_indicators(df):
    """
    Combine sensor changes into transparent transition
    indicators.

    This does NOT produce an AWD YES/NO label.
    """

    data = df.copy()

    # --------------------------------------------------------
    # Count available sensor changes
    # --------------------------------------------------------

    signal_columns = []

    if "ndwi_drying_signal" in data.columns:
        signal_columns.append(
            "ndwi_drying_signal"
        )

    if "ndvi_decline_signal" in data.columns:
        signal_columns.append(
            "ndvi_decline_signal"
        )

    # --------------------------------------------------------
    # Optical transition count
    # --------------------------------------------------------

    if signal_columns:

        data["optical_change_count"] = (
            data[signal_columns]
            .fillna(False)
            .astype(int)
            .sum(axis=1)
        )

    # --------------------------------------------------------
    # SAR magnitude
    # --------------------------------------------------------

    sar_change_columns = []

    if "vv_abs_change" in data.columns:
        sar_change_columns.append(
            "vv_abs_change"
        )

    if "vh_abs_change" in data.columns:
        sar_change_columns.append(
            "vh_abs_change"
        )

    if sar_change_columns:

        data["sar_change_magnitude"] = (
            data[sar_change_columns]
            .mean(axis=1)
        )

    # --------------------------------------------------------
    # General transition flag
    # --------------------------------------------------------

    if (
        "optical_change_count" in data.columns
        or "sar_change_magnitude" in data.columns
    ):

        data["satellite_transition_observed"] = (
            True
        )

        if "optical_change_count" in data.columns:

            data["satellite_transition_observed"] = (
                data["optical_change_count"] > 0
            )

    else:

        data[
            "satellite_transition_observed"
        ] = False

    return data


# ============================================================
# COMPLETE AWD ANALYSIS
# ============================================================

def analyze_awd(df):
    """
    Run the satellite-only AWD indicator pipeline.

    IMPORTANT:
    No AWD ground-truth label is generated here.

    The actual -20 cm threshold will be applied to real IoT
    water-level measurements later.
    """

    data = df.copy()

    data = calculate_optical_indicators(
        data
    )

    data = calculate_sar_indicators(
        data
    )

    data = calculate_transition_indicators(
        data
    )

    # Store threshold as metadata for future reference.
    data["iot_awd_threshold_cm"] = (
        AWD_THRESHOLD_CM
    )

    return data