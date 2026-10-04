import pandas as pd


# ============================================================
# IOT DATA SCHEMA
# ============================================================

EXPECTED_IOT_COLUMNS = [
    "farm_id",
    "timestamp",
    "water_level_cm"
]


# ============================================================
# LOAD IOT DATA
# ============================================================

def load_iot_data(file_path):
    """
    Load IoT water-level data when it becomes available.

    Expected columns:

        farm_id
        timestamp
        water_level_cm
    """

    df = pd.read_csv(file_path)

    missing_columns = [
        column
        for column in EXPECTED_IOT_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "IoT file is missing required columns: "
            f"{missing_columns}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df["water_level_cm"] = pd.to_numeric(
        df["water_level_cm"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "farm_id",
            "timestamp",
            "water_level_cm"
        ]
    )

    return df.sort_values(
        ["farm_id", "timestamp"]
    ).reset_index(drop=True)


# ============================================================
# CREATE AWD LABEL
# ============================================================

def create_awd_labels(
    iot_df,
    threshold_cm=-20.0
):
    """
    Convert IoT water-level measurements into AWD labels.

    AWD condition:
        water_level_cm <= -20 cm

    Returns:
        1 -> AWD condition reached
        0 -> AWD condition not reached
    """

    data = iot_df.copy()

    data["awd_followed"] = (
        data["water_level_cm"]
        <= threshold_cm
    ).astype(int)

    return data


# ============================================================
# SATELLITE + IOT TEMPORAL MATCHING
# ============================================================

def match_satellite_with_iot(
    satellite_df,
    iot_df,
    tolerance_days=5
):
    """
    Match satellite observations with the nearest IoT
    measurement for each farm.

    This function is intended for the future stage when
    IoT data becomes available.
    """

    satellite = satellite_df.copy()

    iot = iot_df.copy()

    satellite["date"] = pd.to_datetime(
        satellite["date"]
    )

    iot["timestamp"] = pd.to_datetime(
        iot["timestamp"]
    )

    # Satellite date becomes timestamp for matching
    satellite["timestamp"] = satellite["date"]

    satellite = satellite.sort_values(
        "timestamp"
    )

    iot = iot.sort_values(
        "timestamp"
    )

    matched = pd.merge_asof(
        satellite,
        iot,
        on="timestamp",
        by="farm_id",
        direction="nearest",
        tolerance=pd.Timedelta(
            days=tolerance_days
        )
    )

    return matched