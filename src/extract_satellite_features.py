import csv
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

import ee
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# IMPORTANT:
# Use the same Google Cloud Project ID that you used in test_gee.py
PROJECT_ID = "awd-satellite-analysis"

# Sentinel-2 maximum cloud percentage
MAX_CLOUD_PERCENT = 30

# Sentinel-1 matching window around Sentinel-2 date
S1_MATCH_DAYS = 6

# Search previous 365 days
END_DATE = datetime.now(timezone.utc).date()
START_DATE = END_DATE - timedelta(days=365)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FARM_CSV = (
    PROJECT_ROOT
    / "data"
    / "farms"
    / "farm_registry.csv"
)

BOUNDARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "farms"
    / "farm_boundaries.geojson"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "satellite_timeseries.csv"
)


# ============================================================
# INITIALIZE EARTH ENGINE
# ============================================================

print("\n" + "=" * 70)
print("INITIALIZING GOOGLE EARTH ENGINE")
print("=" * 70)

ee.Initialize(
    project=PROJECT_ID
)

print("Earth Engine connected successfully.")

print(
    f"Search period: {START_DATE} to {END_DATE}"
)


# ============================================================
# LOAD FARM REGISTRY
# ============================================================

if not FARM_CSV.exists():

    raise FileNotFoundError(
        f"\nFarm registry not found:\n{FARM_CSV}"
    )


with open(
    FARM_CSV,
    "r",
    newline="",
    encoding="utf-8-sig"
) as file:

    farms = list(
        csv.DictReader(file)
    )


if not farms:

    raise ValueError(
        "\nNo farms found in farm_registry.csv."
    )


print(
    f"Number of farms in registry: {len(farms)}"
)


# ============================================================
# LOAD FARM BOUNDARIES
# ============================================================

if not BOUNDARY_FILE.exists():

    raise FileNotFoundError(
        f"\nFarm boundary file not found:\n{BOUNDARY_FILE}"
    )


with open(
    BOUNDARY_FILE,
    "r",
    encoding="utf-8"
) as file:

    geojson = json.load(file)


if geojson.get("type") != "FeatureCollection":

    raise ValueError(
        "farm_boundaries.geojson must be a "
        "GeoJSON FeatureCollection."
    )


boundary_features = geojson.get(
    "features",
    []
)


if not boundary_features:

    raise ValueError(
        "\nfarm_boundaries.geojson contains no features."
    )


print(
    f"Farm boundaries found: "
    f"{len(boundary_features)}"
)


# ============================================================
# CREATE FARM BOUNDARY DICTIONARY
# ============================================================

farm_boundaries = {}


for feature in boundary_features:

    properties = feature.get(
        "properties",
        {}
    )

    farm_id = properties.get(
        "farm_id"
    )

    geometry = feature.get(
        "geometry"
    )

    if not farm_id:

        print(
            "WARNING: A boundary has no farm_id. "
            "Skipping it."
        )

        continue

    if not geometry:

        print(
            f"WARNING: {farm_id} has no geometry. "
            "Skipping it."
        )

        continue

    farm_boundaries[
        str(farm_id).strip()
    ] = geometry


print(
    f"Usable farm boundaries: "
    f"{len(farm_boundaries)}"
)


# ============================================================
# SENTINEL-2 CLOUD MASK
# ============================================================

def mask_sentinel2(image):

    scl = image.select(
        "SCL"
    )

    # Remove:
    #
    # 1  = Saturated/defective
    # 3  = Cloud shadow
    # 8  = Medium probability cloud
    # 9  = High probability cloud
    # 10 = Cirrus
    # 11 = Snow/ice

    mask = (
        scl.neq(1)
        .And(scl.neq(3))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )

    return image.updateMask(
        mask
    )


# ============================================================
# STORE ALL RESULTS
# ============================================================

all_results = []


# ============================================================
# PROCESS EACH FARM
# ============================================================

for farm in farms:

    farm_id = farm[
        "farm_id"
    ].strip()


    print("\n")
    print("=" * 70)

    print(
        f"Processing farm: {farm_id}"
    )


    # --------------------------------------------------------
    # FIND FARM BOUNDARY
    # --------------------------------------------------------

    if farm_id not in farm_boundaries:

        print(
            f"WARNING: No boundary found for "
            f"{farm_id}. Skipping farm."
        )

        continue


    geometry_json = farm_boundaries[
        farm_id
    ]


    geometry_type = geometry_json.get(
        "type"
    )


    print(
        f"Boundary type: {geometry_type}"
    )


    # --------------------------------------------------------
    # CONVERT GEOJSON GEOMETRY TO EE GEOMETRY
    # --------------------------------------------------------

    try:

        aoi = ee.Geometry(
            geometry_json
        )

    except Exception as error:

        print(
            f"ERROR converting boundary for "
            f"{farm_id}: {error}"
        )

        continue


    # ========================================================
    # SENTINEL-2 COLLECTION
    # ========================================================

    print(
        "\nSearching Sentinel-2..."
    )


    sentinel2 = (
        ee.ImageCollection(
            "COPERNICUS/S2_SR_HARMONIZED"
        )

        .filterBounds(
            aoi
        )

        .filterDate(
            str(START_DATE),
            str(END_DATE)
        )

        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                MAX_CLOUD_PERCENT
            )
        )

        .map(
            mask_sentinel2
        )

        .sort(
            "system:time_start"
        )
    )


    s2_count = (
        sentinel2
        .size()
        .getInfo()
    )


    print(
        f"Sentinel-2 images found: {s2_count}"
    )


    if s2_count == 0:

        print(
            "No Sentinel-2 images found."
        )

        continue


    # ========================================================
    # SENTINEL-1 COLLECTION
    # ========================================================

    print(
        "Searching Sentinel-1..."
    )


    sentinel1 = (
        ee.ImageCollection(
            "COPERNICUS/S1_GRD"
        )

        .filterBounds(
            aoi
        )

        .filterDate(
            str(START_DATE),
            str(END_DATE)
        )

        .filter(
            ee.Filter.eq(
                "instrumentMode",
                "IW"
            )
        )

        # Require VV
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation",
                "VV"
            )
        )

        # Require VH
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation",
                "VH"
            )
        )
    )


    s1_total = (
        sentinel1
        .size()
        .getInfo()
    )


    print(
        f"Sentinel-1 images found: {s1_total}"
    )


    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    def calculate_features(
        image
    ):

        # ----------------------------------------------------
        # IMAGE DATE
        # ----------------------------------------------------

        image_date = ee.Date(
            image.get(
                "system:time_start"
            )
        )


        # ----------------------------------------------------
        # NDVI
        # ----------------------------------------------------
        #
        # NDVI = (NIR - RED) / (NIR + RED)
        #
        # Sentinel-2:
        # B8 = NIR
        # B4 = Red
        # ----------------------------------------------------

        ndvi = (
            image
            .normalizedDifference(
                [
                    "B8",
                    "B4"
                ]
            )
            .rename(
                "NDVI"
            )
        )


        # ----------------------------------------------------
        # NDWI
        # ----------------------------------------------------
        #
        # NDWI = (GREEN - NIR) / (GREEN + NIR)
        #
        # B3 = Green
        # B8 = NIR
        # ----------------------------------------------------

        ndwi = (
            image
            .normalizedDifference(
                [
                    "B3",
                    "B8"
                ]
            )
            .rename(
                "NDWI"
            )
        )


        # ----------------------------------------------------
        # CALCULATE OPTICAL STATISTICS
        # ----------------------------------------------------

        optical_features = (
            ndvi
            .addBands(
                ndwi
            )
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=aoi,
                scale=10,
                maxPixels=1e9
            )
        )


        # ====================================================
        # FIND NEARBY SENTINEL-1 OBSERVATIONS
        # ====================================================

        s1_nearby = (
            sentinel1
            .filterDate(
                image_date.advance(
                    -S1_MATCH_DAYS,
                    "day"
                ),
                image_date.advance(
                    S1_MATCH_DAYS,
                    "day"
                )
            )
        )


        s1_count = (
            s1_nearby
            .size()
        )


        # ====================================================
        # EMPTY FALLBACK IMAGE
        # ====================================================

        empty_s1 = (
            ee.Image.constant(
                [
                    0,
                    0
                ]
            )
            .rename(
                [
                    "VV",
                    "VH"
                ]
            )
            .updateMask(
                ee.Image.constant(0)
            )
        )


        # ====================================================
        # SENTINEL-1 MEDIAN
        # ====================================================

        s1_image = ee.Image(
            ee.Algorithms.If(
                s1_count.gt(0),

                s1_nearby
                .select(
                    [
                        "VV",
                        "VH"
                    ]
                )
                .median(),

                empty_s1
            )
        )


        # ====================================================
        # RADAR STATISTICS
        # ====================================================

        radar_features = (
            s1_image
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=aoi,
                scale=10,
                maxPixels=1e9
            )
        )


        radar_features = ee.Dictionary(
            radar_features
        )


        # ====================================================
        # RETURN RESULT
        # ====================================================

        return ee.Feature(
            None,
            {

                "farm_id": farm_id,

                "date": image_date.format(
                    "YYYY-MM-dd"
                ),

                "NDVI": optical_features.get(
                    "NDVI"
                ),

                "NDWI": optical_features.get(
                    "NDWI"
                ),

                "VV": radar_features.get(
                    "VV"
                ),

                "VH": radar_features.get(
                    "VH"
                ),

                "sentinel1_count": s1_count,

                "sentinel1_available": s1_count.gt(
                    0
                )
            }
        )


    # ========================================================
    # APPLY FUNCTION
    # ========================================================

    print(
        "\nCalculating NDVI, NDWI, VV and VH..."
    )


    feature_collection = (
        sentinel2
        .map(
            calculate_features
        )
    )


    # ========================================================
    # GET RESULTS
    # ========================================================

    print(
        "Retrieving results from Earth Engine..."
    )


    results = (
        feature_collection
        .getInfo()
        [
            "features"
        ]
    )


    print(
        f"Raw records received: {len(results)}"
    )


    # --------------------------------------------------------
    # ADD RESULTS
    # --------------------------------------------------------

    for feature in results:

        properties = (
            feature["properties"]
        )

        all_results.append(
            properties
        )


# ============================================================
# CHECK WHETHER ANY DATA WAS FOUND
# ============================================================

if not all_results:

    raise RuntimeError(
        "\nNo satellite features were extracted."
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    all_results
)


# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "NDVI",
    "NDWI",
    "VV",
    "VH",
    "sentinel1_count"
]


for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ============================================================
# CONVERT BOOLEAN
# ============================================================

if "sentinel1_available" in df.columns:

    df[
        "sentinel1_available"
    ] = df[
        "sentinel1_available"
    ].astype(bool)


# ============================================================
# REMOVE DUPLICATE FARM + DATE OBSERVATIONS
# ============================================================
#
# Sentinel-2 can return multiple scenes/granules on the same
# date. We combine them into one farm-level observation.
#
# NDVI/NDWI/VV/VH -> mean
# sentinel1_count -> maximum
# sentinel1_available -> maximum
# ============================================================

aggregation = {

    "NDVI": "mean",

    "NDWI": "mean",

    "VV": "mean",

    "VH": "mean",

    "sentinel1_count": "max",

    "sentinel1_available": "max"
}


df = (
    df
    .groupby(
        [
            "farm_id",
            "date"
        ],
        as_index=False
    )
    .agg(
        aggregation
    )
)


# ============================================================
# SORT DATA
# ============================================================

df = df.sort_values(
    [
        "farm_id",
        "date"
    ]
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAVE CLEAN DATASET
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 70)
print(
    "SATELLITE FEATURE EXTRACTION COMPLETED"
)
print("=" * 70)

print(
    f"Final records: {len(df)}"
)

print(
    f"Output file:\n{OUTPUT_CSV}"
)


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\nDataset information:")

print(
    df.info()
)


# ============================================================
# FIRST RECORDS
# ============================================================

print("\nFirst 10 records:")

print(
    df.head(10).to_string(
        index=False
    )
)


# ============================================================
# MISSING VALUES
# ============================================================

print("\nMissing values:")

print(
    df.isnull().sum()
)


# ============================================================
# DATE RANGE
# ============================================================

print("\nDate range:")

print(
    f"First date: {df['date'].min()}"
)

print(
    f"Last date:  {df['date'].max()}"
)


print("\nDone!")