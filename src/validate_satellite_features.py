
from pathlib import Path

import numpy as np
import pandas as pd

# Project root
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "satellite_features.csv"

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"File not found: {DATA_FILE}\n"
        "Run main.py first to generate satellite_features.csv."
    )

df = pd.read_csv(DATA_FILE)

print("\n========== SATELLITE DATA VALIDATION ==========")

# 1. Dataset overview
print("\n1. DATASET OVERVIEW")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print("Columns:", df.columns.tolist())

# 2. Required columns
print("\n2. REQUIRED COLUMN CHECK")
required = ["farm_id", "date"]
missing_required = [c for c in required if c not in df.columns]

if missing_required:
    print("Missing required columns:", missing_required)
    raise ValueError("Fix the required columns before continuing.")

print("farm_id and date columns are present.")

# 3. Dates and duplicates
print("\n3. DATE AND DUPLICATE CHECK")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

print("Invalid dates:", df["date"].isna().sum())
print("Missing farm IDs:", df["farm_id"].isna().sum())
print("Duplicate farm-date rows:",
      df.duplicated(["farm_id", "date"]).sum())

valid_dates = df.dropna(subset=["date"]).sort_values(
    ["farm_id", "date"]
)

if len(valid_dates):
    print("Date range:",
          valid_dates["date"].min().date(),
          "to",
          valid_dates["date"].max().date())

    valid_dates["gap_days"] = (
        valid_dates.groupby("farm_id")["date"]
        .diff()
        .dt.days
    )

    gaps = valid_dates[valid_dates["gap_days"] > 30]

    print("\nGaps longer than 30 days:")
    if gaps.empty:
        print("None")
    else:
        print(gaps[["farm_id", "date", "gap_days"]].to_string(index=False))

# 4. Missing values
print("\n4. MISSING VALUES")
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if missing.empty:
    print("No missing values.")
else:
    print(missing.to_string())

# 5. Invalid numeric values
print("\n5. INFINITE VALUES")
numeric = df.select_dtypes(include="number")
infinite_counts = pd.Series(
    np.isinf(numeric.to_numpy()).sum(axis=0),
    index=numeric.columns
)
infinite_counts = infinite_counts[infinite_counts > 0]

if infinite_counts.empty:
    print("No infinite numeric values.")
else:
    print(infinite_counts.to_string())

# 6. Farm coverage
print("\n6. FARM COVERAGE")
print("Unique farms:", df["farm_id"].nunique())
print(df["farm_id"].value_counts(dropna=False).to_string())

# 7. Check whether verified target labels exist
print("\n7. TARGET LABEL CHECK")
possible_labels = [
    "awd_label",
    "awd_event",
    "ground_truth",
    "irrigation_event",
    "verified_status",
]

found_labels = [c for c in possible_labels if c in df.columns]

if found_labels:
    print("Possible label columns:", found_labels)
    for column in found_labels:
        print(f"\n{column} distribution:")
        print(df[column].value_counts(dropna=False).to_string())
else:
    print("No recognized verified target-label column found.")
    print("Do not train Random Forest yet.")

# 8. Save validation report
report_path = ROOT / "outputs" / "metrics" / "satellite_validation.txt"
report_path.parent.mkdir(parents=True, exist_ok=True)

with open(report_path, "w", encoding="utf-8") as report:
    report.write("SATELLITE DATA VALIDATION REPORT\n")
    report.write(f"Rows: {len(df)}\n")
    report.write(f"Columns: {len(df.columns)}\n")
    report.write(f"Unique farms: {df['farm_id'].nunique()}\n")
    report.write("\nMissing values:\n")
    report.write(missing.to_string() if not missing.empty else "None")
    report.write("\n\nDuplicate farm-date rows: ")
    report.write(str(df.duplicated(["farm_id", "date"]).sum()))
    report.write("\n\nPossible target labels: ")
    report.write(str(found_labels))

print(f"\nValidation report saved to:\n{report_path}")
print("\n========== VALIDATION COMPLETE ==========")
