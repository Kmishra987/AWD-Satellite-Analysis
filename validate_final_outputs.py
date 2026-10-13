
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

files = {
    "satellite": ROOT / "data/processed/satellite_timeseries_aligned.csv",
    "features": ROOT / "data/processed/satellite_features.csv",
}

for name, path in files.items():
    print(f"\n{'=' * 60}")
    print(f"CHECKING: {name.upper()}")
    print(f"{'=' * 60}")

    if not path.exists():
        print(f"FAIL: File not found: {path}")
        continue

    df = pd.read_csv(path)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Column names:", df.columns.tolist())

    if df.empty:
        print("FAIL: File is empty.")
        continue

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        print("Invalid dates:", df["date"].isna().sum())

    if "farm_id" in df.columns and "date" in df.columns:
        duplicates = df.duplicated(
            subset=["farm_id", "date"]
        ).sum()
        print("Duplicate farm-date rows:", duplicates)
    else:
        duplicates = None
        print("Farm-date uniqueness: cannot verify")

    numeric = df.select_dtypes(include="number")
    print("Infinite numeric values:",
          int(numeric.isin([float("inf"), float("-inf")]).sum().sum()))

    print("\nMissing values:")
    print(df.isna().sum()[df.isna().sum() > 0])

    if name == "features":
        important = [
            "transition_gap_valid",
            "ndvi_mean_delta",
            "ndwi_mean_delta",
            "vv_mean_delta",
            "vh_mean_delta",
        ]
        print("\nExpected feature columns:")
        for column in important:
            print(f"{column}: {'FOUND' if column in df.columns else 'NOT FOUND'}")

    if duplicates is not None and duplicates > 0:
        print("WARNING: Duplicate farm-date rows found.")
    else:
        print("\nBasic file checks completed.")
