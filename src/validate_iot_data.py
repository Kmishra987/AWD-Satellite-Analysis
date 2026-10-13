
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
csv_path = project_root / "data" / "raw" / "iot" / "iot.csv"

if not csv_path.exists():
    raise FileNotFoundError(f"CSV not found: {csv_path}")

df = pd.read_csv(csv_path)

print("\n--- IoT Dataset Validation ---")
print("Total readings:", len(df))
print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isna().sum())

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce", utc=True
    )
    print("\nInvalid timestamps:", df["timestamp"].isna().sum())
    print("Duplicate timestamps:", df["timestamp"].duplicated().sum())

print("\nData source counts:")
if "data_source" in df.columns:
    print(df["data_source"].value_counts(dropna=False))

print("\nLatest 5 readings:")
print(df.tail(5).to_string(index=False))

print("\nValidation finished.")
