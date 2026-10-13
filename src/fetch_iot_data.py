
import json
import urllib.request
from pathlib import Path

import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "iot" / "iot.csv"

# Your public ThingSpeak channel
CHANNEL_ID = "3515295"
RESULTS_LIMIT = 8000

URL = (
    f"https://api.thingspeak.com/channels/"
    f"{CHANNEL_ID}/feeds.json?results={RESULTS_LIMIT}"
)


def main():
    print("Fetching data from ThingSpeak...")

    try:
        request = urllib.request.Request(
            URL,
            headers={"User-Agent": "AWD-Satellite-Analysis/1.0"}
        )

        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))

    except Exception as error:
        print(f"Could not fetch ThingSpeak data: {error}")
        return

    feeds = payload.get("feeds", [])

    if not feeds:
        print("No readings found. Check your channel and its feed data.")
        return

    # Preserve the original ThingSpeak fields
    df = pd.DataFrame(feeds)

    # Convert ThingSpeak field names into descriptive column names
    field_mapping = {
        "field1": "water_level_percent",
        "field2": "water_depth_cm",
        "field3": "sensor_value",
        "field4": "awd_status",
        "field5": "awd_event_count",
    }

    df = df.rename(columns=field_mapping)

    # Standardize timestamp and identify the farm and data source
    df = df.rename(columns={"created_at": "timestamp"})
    df.insert(0, "farm_id", "FARM001")
    df["data_source"] = "wokwi_simulation"

    # Keep only the columns we need, if present
    columns = [
        "farm_id",
        "timestamp",
        "water_level_percent",
        "water_depth_cm",
        "sensor_value",
        "awd_status",
        "awd_event_count",
        "data_source",
    ]

    df = df[[col for col in columns if col in df.columns]]

    # Do not invent values for missing sensor readings
    for col in [
        "water_level_percent",
        "water_depth_cm",
        "sensor_value",
        "awd_event_count",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Successfully saved {len(df)} readings.")
    print(f"Output file: {OUTPUT_FILE}")
    print("\nLatest readings:")
    print(df.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
