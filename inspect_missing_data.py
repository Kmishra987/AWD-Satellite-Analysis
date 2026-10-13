
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
path = ROOT / "data/processed/satellite_timeseries_aligned.csv"

df = pd.read_csv(path)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values(["farm_id", "date"]).reset_index(drop=True)

measurement_cols = ["ndvi_mean", "ndwi_mean", "vv_mean", "vh_mean"]

df["days_since_previous"] = (
    df.groupby("farm_id")["date"]
    .diff()
    .dt.days
)

missing = df[df[measurement_cols].isna().any(axis=1)]

print("\nMISSING SATELLITE OBSERVATIONS")
print("=" * 70)

if missing.empty:
    print("No missing satellite measurements.")
else:
    print(
        missing[
            ["farm_id", "date", "days_since_previous",
             *measurement_cols, "sentinel1_count",
             "sentinel1_available"]
        ].to_string(index=False)
    )

print("\nMISSING-VALUE COUNTS")
print(df[measurement_cols].isna().sum())

print("\nOBSERVATION INTERVALS")
print(df["days_since_previous"].describe())

print("\nGAPS GREATER THAN 30 DAYS")
print(
    df.loc[
        df["days_since_previous"] > 30,
        ["farm_id", "date", "days_since_previous"]
    ].to_string(index=False)
)
