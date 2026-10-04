from pathlib import Path
import pandas as pd
import geopandas as gpd


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

FARM_DIR = PROJECT_DIR / "data" / "farms"

REGISTRY_FILE = FARM_DIR / "farm_registry.csv"
BOUNDARY_FILE = FARM_DIR / "farm_boundaries.geojson"


# ============================================================
# LOAD FARM REGISTRY
# ============================================================

def load_farm_registry():
    """
    Load the farm registry.

    Expected columns:
        farm_id
        farmer_id
    """

    if not REGISTRY_FILE.exists():
        raise FileNotFoundError(
            f"Farm registry not found:\n{REGISTRY_FILE}"
        )

    df = pd.read_csv(REGISTRY_FILE)

    # Remove accidental spaces from column names
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
    )

    required_columns = {"farm_id", "farmer_id"}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Farm registry is missing columns: {missing}"
        )

    # Clean IDs
    df["farm_id"] = df["farm_id"].astype(str).str.strip()
    df["farmer_id"] = df["farmer_id"].astype(str).str.strip()

    # Check empty IDs
    if (df["farm_id"] == "").any():
        raise ValueError("Farm registry contains empty farm_id values.")

    # Check duplicate farm IDs
    duplicates = df[df["farm_id"].duplicated(keep=False)]

    if not duplicates.empty:
        raise ValueError(
            "Duplicate farm_id values found:\n"
            + str(duplicates)
        )

    return df


# ============================================================
# LOAD FARM BOUNDARIES
# ============================================================

def load_farm_boundaries():
    """
    Load farm boundaries from GeoJSON.

    Every feature must contain a farm_id.
    """

    if not BOUNDARY_FILE.exists():
        raise FileNotFoundError(
            f"Farm boundary file not found:\n{BOUNDARY_FILE}"
        )

    gdf = gpd.read_file(BOUNDARY_FILE)

    # Empty GeoJSON is allowed during initial setup
    if gdf.empty:
        print("WARNING: No farm boundaries are currently available.")
        return gdf

    # Normalize column names
    gdf.columns = [
        str(col).strip().lower()
        for col in gdf.columns
    ]

    if "farm_id" not in gdf.columns:
        raise ValueError(
            "farm_boundaries.geojson must contain a 'farm_id' property."
        )

    # Clean farm IDs
    gdf["farm_id"] = (
        gdf["farm_id"]
        .astype(str)
        .str.strip()
    )

    # Check empty IDs
    if (gdf["farm_id"] == "").any():
        raise ValueError(
            "Farm boundaries contain empty farm_id values."
        )

    # Check duplicate farm IDs
    duplicates = gdf[gdf["farm_id"].duplicated(keep=False)]

    if not duplicates.empty:
        raise ValueError(
            "Duplicate farm_id values found in boundaries:\n"
            + str(duplicates[["farm_id"]])
        )

    # Check geometry
    invalid_geometry = ~gdf.geometry.is_valid

    if invalid_geometry.any():
        print(
            f"WARNING: {invalid_geometry.sum()} "
            "farm geometries are invalid."
        )

    return gdf


# ============================================================
# VALIDATE REGISTRY AND BOUNDARIES
# ============================================================

def validate_farm_data(registry, boundaries):
    """
    Make sure farm registry and farm boundaries agree.
    """

    if boundaries.empty:
        print(
            "Farm validation skipped because "
            "no farm boundaries are available yet."
        )
        return True

    registry_ids = set(registry["farm_id"])
    boundary_ids = set(boundaries["farm_id"])

    missing_boundaries = registry_ids - boundary_ids
    unregistered_boundaries = boundary_ids - registry_ids

    if missing_boundaries:
        print(
            "WARNING: Farms without boundaries:"
        )
        print(sorted(missing_boundaries))

    if unregistered_boundaries:
        print(
            "WARNING: Boundaries without registry entries:"
        )
        print(sorted(unregistered_boundaries))

    if not missing_boundaries and not unregistered_boundaries:
        print("Farm registry and boundaries match successfully.")

    return True


# ============================================================
# CREATE FARM SUMMARY
# ============================================================

def create_farm_summary(registry, boundaries):
    """
    Create a useful farm-level summary.

    This does NOT modify the original data.
    """

    summary = registry.copy()

    if boundaries.empty:
        summary["has_boundary"] = False
        return summary

    boundary_info = boundaries[["farm_id", "geometry"]].copy()

    boundary_info["has_boundary"] = True

    # Calculate centroid only for reporting.
    # We do not use centroid as a replacement for the real polygon.
    boundary_info["centroid"] = boundary_info.geometry.centroid

    boundary_info = boundary_info.drop(columns=["geometry"])

    summary = summary.merge(
        boundary_info,
        on="farm_id",
        how="left"
    )

    summary["has_boundary"] = (
        summary["has_boundary"]
        .fillna(False)
        .astype(bool)
    )

    return summary


# ============================================================
# PRINT FARM INFORMATION
# ============================================================

def print_farm_information(registry, boundaries):
    """
    Print a simple farm data report.
    """

    print("\n" + "=" * 60)
    print("FARM IDENTIFICATION REPORT")
    print("=" * 60)

    print(f"Registered farms: {len(registry)}")

    if boundaries.empty:
        print("Farm boundaries: NOT AVAILABLE")
        print()
        print(
            "The system is ready for farm boundaries, "
            "but no real farm polygons have been supplied yet."
        )
        return

    print(f"Farm boundaries: {len(boundaries)}")

    registry_ids = set(registry["farm_id"])
    boundary_ids = set(boundaries["farm_id"])

    matched = registry_ids & boundary_ids

    print(f"Matched farms: {len(matched)}")
    print(f"Registry only: {len(registry_ids - boundary_ids)}")
    print(f"Boundary only: {len(boundary_ids - registry_ids)}")

    print("\nFarm IDs:")

    for farm_id in sorted(boundary_ids):
        print(f"  - {farm_id}")


# ============================================================
# MAIN FARM LOADING FUNCTION
# ============================================================

def load_farm_data():
    """
    Main function used by the rest of the project.
    """

    registry = load_farm_registry()

    boundaries = load_farm_boundaries()

    validate_farm_data(
        registry,
        boundaries
    )

    print_farm_information(
        registry,
        boundaries
    )

    summary = create_farm_summary(
        registry,
        boundaries
    )

    return registry, boundaries, summary


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    registry, boundaries, summary = load_farm_data()

    print("\nFarm summary:")

    print(summary)