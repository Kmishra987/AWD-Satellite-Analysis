import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GEOJSON_FILE = (
    PROJECT_ROOT
    / "data"
    / "farms"
    / "farm_boundaries.geojson"
)


print("=" * 60)
print("FARM BOUNDARY VALIDATION")
print("=" * 60)

if not GEOJSON_FILE.exists():
    raise FileNotFoundError(
        f"GeoJSON file not found:\n{GEOJSON_FILE}"
    )


with open(
    GEOJSON_FILE,
    "r",
    encoding="utf-8"
) as file:

    data = json.load(file)


print(f"File: {GEOJSON_FILE}")
print(f"GeoJSON type: {data.get('type')}")


# ------------------------------------------------------------
# CHECK FEATURE COLLECTION
# ------------------------------------------------------------

if data.get("type") != "FeatureCollection":

    raise ValueError(
        "The GeoJSON must be a FeatureCollection."
    )


features = data.get("features", [])

print(
    f"Number of features: {len(features)}"
)


if not features:

    print("\nWARNING:")
    print(
        "The GeoJSON currently contains no farm boundaries."
    )

    print(
        "\nExpected structure:"
    )

    print(
        """
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "farm_id": "FARM001"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [...]
      }
    }
  ]
}
"""
    )

    raise SystemExit(
        "\nAdd the farm polygon first."
    )


# ------------------------------------------------------------
# INSPECT FEATURES
# ------------------------------------------------------------

for index, feature in enumerate(features, start=1):

    properties = feature.get(
        "properties",
        {}
    )

    geometry = feature.get(
        "geometry"
    )

    farm_id = properties.get(
        "farm_id"
    )

    print("\n" + "-" * 50)

    print(
        f"Feature {index}"
    )

    print(
        f"farm_id: {farm_id}"
    )

    print(
        f"geometry type: "
        f"{geometry.get('type') if geometry else None}"
    )


# ------------------------------------------------------------
# CHECK FARM ID
# ------------------------------------------------------------

farm_ids = []

for feature in features:

    farm_id = (
        feature
        .get("properties", {})
        .get("farm_id")
    )

    if farm_id:
        farm_ids.append(
            str(farm_id)
        )


print("\n" + "=" * 60)

if "FARM001" in farm_ids:

    print(
        "SUCCESS: FARM001 boundary found."
    )

else:

    print(
        "WARNING: FARM001 boundary was not found."
    )

    print(
        f"Available farm IDs: {farm_ids}"
    )


print("=" * 60)