
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

IOT_FILE = ROOT / "data" / "raw" / "iot" / "iot.csv"
SATELLITE_FILE = (
    ROOT / "data" / "processed" / "satellite_timeseries.csv"
)
OUTPUT_DIR = ROOT / "data" / "processed"

def main():
    iot = pd.read_csv(IOT_FILE)
    satellite = pd.read_csv(SATELLITE_FILE)

    # Parse timestamps consistently in UTC
    iot["timestamp"] = pd.to_datetime(
        iot["timestamp"], errors="coerce", utc=True
    )
    iot = iot.dropna(subset=["timestamp"])
    iot["date"] = iot["timestamp"].dt.strftime("%Y-%m-%d")

    satellite["date"] = pd.to_datetime(
        satellite["date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    # Convert measurements to numeric without inventing values
    numeric_cols = [
        "water_level_percent",
        "water_depth_cm",
        "sensor_value",
        "awd_status",
        "awd_event_count",
    ]

    for col in numeric_cols:
        if col in iot.columns:
            iot[col] = pd.to_numeric(iot[col], errors="coerce")

    # Sort so the last non-missing event count is well-defined
    iot = iot.sort_values("timestamp")

    # Summarize multiple sensor readings per farm per day
    daily = iot.groupby(
        ["farm_id", "date"], as_index=False
    ).agg(
        iot_reading_count=("timestamp", "count"),
        mean_water_level_percent=("water_level_percent", "mean"),
        mean_water_depth_cm=("water_depth_cm", "mean"),
        mean_sensor_value=("sensor_value", "mean"),
        awd_status_mean=("awd_status", "mean"),
        awd_status_max=("awd_status", "max"),
        last_awd_event_count=(
            "awd_event_count",
            lambda s: s.dropna().iloc[-1]
            if not s.dropna().empty else float("nan")
        ),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    daily_file = OUTPUT_DIR / "iot_daily_summary.csv"
    daily.to_csv(daily_file, index=False)

    # Exact-date merge: do not invent nearby satellite observations
    combined = satellite.merge(
        daily,
        on=["farm_id", "date"],
        how="left",
        validate="one_to_one",
    )

    combined_file = OUTPUT_DIR / "combined_satellite_iot.csv"
    combined.to_csv(combined_file, index=False)

    matched = combined["iot_reading_count"].notna().sum()

    print("\n--- Integration Results ---")
    print("Satellite observations:", len(satellite))
    print("IoT readings:", len(iot))
    print("Daily IoT summaries:", len(daily))
    print("Satellite dates matched with IoT:", matched)
    print("Satellite dates without IoT:", len(combined) - matched)

    print("\nDaily IoT summary:")
    print(daily.to_string(index=False))

    print("\nMatched observations:")
    print(
        combined.loc[
            combined["iot_reading_count"].notna(),
            ["farm_id", "date", "NDVI", "NDWI",
             "VV", "VH", "mean_water_depth_cm",
             "awd_status_mean"]
        ].to_string(index=False)
    )

    print("\nSaved files:")
    print(daily_file)
    print(combined_file)


if __name__ == "__main__":
    main()
