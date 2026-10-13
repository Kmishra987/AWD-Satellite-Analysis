
import pandas as pd

from src.awd_detection import (
    MAX_TRANSITION_GAP_DAYS,
    analyze_awd,
)

df = pd.DataFrame({
    "farm_id": [
        "FARM001", "FARM001", "FARM001",
        "FARM001", "FARM002", "FARM002",
    ],
    "date": [
        "2026-06-10",
        "2026-06-20",
        "2026-09-18",
        "2026-09-28",
        "2026-06-10",
        "2026-06-20",
    ],
    "ndvi_mean": [0.70, 0.65, 0.40, 0.35, 0.80, 0.75],
    "ndwi_mean": [0.30, 0.20, 0.05, 0.00, 0.40, 0.30],
    "vv_mean": [-12.0, -13.0, -15.0, -16.0, -11.0, -12.0],
    "vh_mean": [-18.0, -19.0, -21.0, -22.0, -17.0, -18.0],
    "vv_vh_difference": [6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
})

result = analyze_awd(df)

print(result[
    [
        "farm_id",
        "date",
        "days_since_previous",
        "transition_gap_valid",
        "ndwi_change",
        "ndvi_change",
        "vv_change",
        "satellite_transition_observed",
    ]
].to_string(index=False))

# 1. Each farm's first observation has no previous transition.
first_rows = result.groupby("farm_id").head(1)

for column in [
    "ndwi_change",
    "ndvi_change",
    "vv_change",
    "vh_change",
]:
    assert first_rows[column].isna().all(), (
        f"{column} incorrectly crosses farm boundaries."
    )

# 2. The long gap must suppress changes.
long_gap = (
    result["days_since_previous"]
    > MAX_TRANSITION_GAP_DAYS
)

assert long_gap.sum() == 1
assert not result.loc[
    long_gap, "transition_gap_valid"
].any()

for column in [
    "ndwi_change",
    "ndvi_change",
    "vv_change",
    "vh_change",
]:
    assert result.loc[long_gap, column].isna().all(), (
        f"{column} should be missing across the long gap."
    )

# 3. A valid 10-day transition must be retained.
short_gap = (
    result["farm_id"].eq("FARM001")
    & result["date"].eq(pd.Timestamp("2026-09-28"))
)

assert result.loc[
    short_gap, "transition_gap_valid"
].all()

assert result.loc[
    short_gap, "ndwi_change"
].notna().all()

# 4. Satellite transition indicators must not be interpreted
# as confirmed AWD labels.
assert "satellite_transition_observed" in result.columns
assert "iot_awd_threshold_cm" in result.columns

print("\nPASS: Farm boundaries are respected.")
print("PASS: Long-gap changes are suppressed.")
print("PASS: Valid short-gap changes are retained.")
print("PASS: AWD indicators are generated without confirmed labels.")
