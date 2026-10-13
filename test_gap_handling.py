
import pandas as pd

from src.awd_detection import MAX_TRANSITION_GAP_DAYS, analyze_awd

df = pd.DataFrame({
    "farm_id": ["FARM001"] * 4,
    "date": [
        "2026-06-10",
        "2026-06-20",
        "2026-09-18",
        "2026-09-28",
    ],
    "ndvi_mean": [0.70, 0.65, 0.40, 0.35],
    "ndwi_mean": [0.30, 0.20, 0.05, 0.00],
    "vv_mean": [-12.0, -13.0, -15.0, -16.0],
    "vh_mean": [-18.0, -19.0, -21.0, -22.0],
})

result = analyze_awd(df)

columns = [
    "date",
    "days_since_previous",
    "transition_gap_valid",
    "ndwi_change",
    "ndvi_change",
    "vv_change",
    "vh_change",
]

print(result[columns].to_string(index=False))

long_gap = result["days_since_previous"] > MAX_TRANSITION_GAP_DAYS

assert long_gap.sum() == 1, "Expected exactly one gap longer than 30 days."
assert not result.loc[long_gap, "transition_gap_valid"].any(), (
    "The long-gap transition should be marked invalid."
)

for column in ["ndwi_change", "ndvi_change", "vv_change", "vh_change"]:
    assert result.loc[long_gap, column].isna().all(), (
        f"{column} should be missing across the long gap."
    )

short_gap = result["date"].eq(pd.Timestamp("2026-09-28"))

assert result.loc[short_gap, "transition_gap_valid"].all(), (
    "The 10-day transition should be valid."
)
assert result.loc[short_gap, "ndwi_change"].notna().all(), (
    "NDWI change should be retained across the short gap."
)

print("\nPASS: Long-gap changes are suppressed.")
print("PASS: Short-gap changes are retained.")
