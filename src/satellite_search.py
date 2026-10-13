import csv
from pathlib import Path
from datetime import datetime, timedelta, timezone

import ee


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

# Replace this with the same Project ID used in test_gee.py.
PROJECT_ID = "awd-satellite-analysis"

# Search within a 500-metre radius of each farm coordinate.
BUFFER_METRES = 500

# Search the previous 12 months.
END_DATE = datetime.now(timezone.utc).date()
START_DATE = END_DATE - timedelta(days=365)

# Find the project root and farm registry automatically.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FARM_CSV = PROJECT_ROOT / "data" / "farms" / "farm_registry.csv"


# --------------------------------------------------
# INITIALIZE EARTH ENGINE
# --------------------------------------------------

ee.Initialize(project=PROJECT_ID)

print("\nEarth Engine connected successfully!")
print(f"Search period: {START_DATE} to {END_DATE}")


# --------------------------------------------------
# READ FARM COORDINATES
# --------------------------------------------------

if not FARM_CSV.exists():
    raise FileNotFoundError(
        f"Farm registry not found: {FARM_CSV}"
    )

with open(FARM_CSV, "r", newline="", encoding="utf-8-sig") as file:
    farms = list(csv.DictReader(file))

if not farms:
    raise ValueError(
        "No farms found in farm_registry.csv. "
        "Add at least one farm coordinate."
    )


# --------------------------------------------------
# SEARCH SATELLITE IMAGES
# --------------------------------------------------

for farm in farms:

    farm_id = farm["farm_id"].strip()
    latitude = float(farm["latitude"])
    longitude = float(farm["longitude"])

    print("\n" + "=" * 55)
    print(f"Farm ID: {farm_id}")
    print(f"Latitude: {latitude}")
    print(f"Longitude: {longitude}")

    # Create a circular search area around the farm.
    point = ee.Geometry.Point([longitude, latitude])
    aoi = point.buffer(BUFFER_METRES)

    # --------------------------------------------------
    # SENTINEL-2: OPTICAL SATELLITE IMAGES
    # --------------------------------------------------

    sentinel2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(str(START_DATE), str(END_DATE))
        .filter(
            ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50)
        )
        .sort("system:time_start", False)
    )

    s2_count = sentinel2.size().getInfo()

    print("\nSENTINEL-2 RESULTS")
    print(f"Images matching filters: {s2_count}")

    if s2_count > 0:
        s2_images = sentinel2.limit(5).getInfo()["features"]

        for image in s2_images:
            properties = image["properties"]

            timestamp = properties["system:time_start"]
            acquisition_date = datetime.fromtimestamp(
                timestamp / 1000,
                tz=timezone.utc
            ).strftime("%Y-%m-%d")

            cloud = properties.get(
                "CLOUDY_PIXEL_PERCENTAGE", "Unknown"
            )

            print(
                f"Date: {acquisition_date} | "
                f"Cloud cover: {cloud}%"
            )

    else:
        print("No Sentinel-2 images matched these filters.")

    # --------------------------------------------------
    # SENTINEL-1: RADAR SATELLITE IMAGES
    # --------------------------------------------------

    sentinel1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(str(START_DATE), str(END_DATE))
        .filter(
            ee.Filter.eq("instrumentMode", "IW")
        )
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation", "VV"
            )
        )
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation", "VH"
            )
        )
        .sort("system:time_start", False)
    )

    s1_count = sentinel1.size().getInfo()

    print("\nSENTINEL-1 RESULTS")
    print(f"Images matching filters: {s1_count}")

    if s1_count > 0:
        s1_images = sentinel1.limit(5).getInfo()["features"]

        for image in s1_images:
            properties = image["properties"]

            timestamp = properties["system:time_start"]
            acquisition_date = datetime.fromtimestamp(
                timestamp / 1000,
                tz=timezone.utc
            ).strftime("%Y-%m-%d")

            print(f"Date: {acquisition_date}")

    else:
        print("No Sentinel-1 images matched these filters.")


print("\n" + "=" * 55)
print("Satellite search completed.")