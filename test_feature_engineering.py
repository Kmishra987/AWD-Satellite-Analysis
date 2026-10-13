
import pandas as pd

from src.feature_engineering import (
    MAX_TRANSITION_GAP_DAYS,
    create_features,
)

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

result = create_features(df)

print(result[
    [
        "date",
        "days_since_previous",
        "transition_gap_valid",
        "ndvi_mean_delta",
        "ndvi_mean_rolling_mean_3",
        "ndwi_mean_delta",
    ]
].to_string(index=False))

long_gap = (
    result["days_since_previous"]
    > MAX_TRANSITION_GAP_DAYS
)

assert long_gap.sum() == 1
assert not result.loc[
    long_gap, "transition_gap_valid"
].any()

# No delta should be calculated across the long gap.
for column in [
    "ndvi_mean_delta",
    "ndwi_mean_delta",
    "vv_mean_delta",
    "vh_mean_delta",
]:
    assert result.loc[long_gap, column].isna().all(), (
        f"{column} crosses a long gap."
    )

# The first observation after a long gap starts a new segment.
first_after_gap = result["date"].eq(
    pd.Timestamp("2026-09-18")
)

assert result.loc[
    first_after_gap, "ndvi_mean_rolling_mean_3"
].isna().all(), (
    "Rolling mean improperly includes observations from before "
    "the long gap."
)

# A short gap should still allow a valid delta.
short_gap = result["date"].eq(
    pd.Timestamp("2026-09-28")
)

assert result.loc[
    short_gap, "transition_gap_valid"
].all()

assert result.loc[
    short_gap, "ndvi_mean_delta"
].notna().all()

print("\nPASS: Long-gap deltas are suppressed.")
print("PASS: Rolling statistics do not cross long gaps.")
print("PASS: Short-gap temporal features are retained.")
