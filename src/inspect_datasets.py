
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

files = {
    "IoT": ROOT / "data" / "raw" / "iot" / "iot.csv",
    "Satellite": ROOT / "data" / "processed" / "satellite_timeseries.csv",
}

for name, path in files.items():
    print(f"\n{'=' * 15} {name} DATA {'=' * 15}")

    if not path.exists():
        print(f"File not found: {path}")
        continue

    df = pd.read_csv(path)
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(
            df["timestamp"], errors="coerce", utc=True
        ).dropna()

        if not timestamps.empty:
            print("First timestamp:", timestamps.min())
            print("Last timestamp:", timestamps.max())

    if "farm_id" in df.columns:
        print("Farm IDs:", df["farm_id"].unique().tolist())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))
