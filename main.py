from pathlib import Path

from src.load_satellite_data import (
    load_all_satellite_data
)

from src.preprocessing import (
    align_satellite_data,
    clean_satellite_data,
    generate_quality_report
)

from src.feature_engineering import (
    create_features
)

from src.awd_detection import (
    analyze_awd
)

from src.results import (
    save_dataframe,
    plot_satellite_metrics
)


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

PROJECT_DIR = Path(
    __file__
).resolve().parent

PROCESSED_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("AWD SATELLITE ANALYSIS PIPELINE")
    print("=" * 70)

    print(
        "\nMode: SATELLITE-ONLY ANALYSIS"
    )

    print(
        "IoT data: NOT AVAILABLE YET"
    )

    # ========================================================
    # 1. LOAD RAW SATELLITE DATA
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STEP 1: LOADING SATELLITE DATA")
    print("=" * 70)

    datasets = (
        load_all_satellite_data()
    )

    ndvi = datasets["ndvi"]
    ndwi = datasets["ndwi"]
    vv = datasets["vv"]
    vh = datasets["vh"]

    # ========================================================
    # 2. TEMPORAL ALIGNMENT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STEP 2: TEMPORAL ALIGNMENT")
    print("=" * 70)

    satellite_data = align_satellite_data(
        ndvi,
        ndwi,
        vv,
        vh,
        max_gap_days=5
    )

    print(
        f"\nAligned observations: "
        f"{len(satellite_data)}"
    )

    # ========================================================
    # 3. CLEAN DATA
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STEP 3: DATA CLEANING")
    print("=" * 70)

    satellite_data = clean_satellite_data(
        satellite_data
    )

    generate_quality_report(
        satellite_data
    )

    # ========================================================
    # 4. SAVE ALIGNED DATA
    # ========================================================

    aligned_output = (
        PROCESSED_DIR
        / "satellite_timeseries.csv"
    )

    save_dataframe(
        satellite_data,
        aligned_output
    )

    # ========================================================
    # 5. FEATURE ENGINEERING
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STEP 4: FEATURE ENGINEERING")
    print("=" * 70)

    feature_data = create_features(
        satellite_data
    )

    # ========================================================
    # 6. AWD INDICATORS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STEP 5: AWD SATELLITE INDICATORS")
    print("=" * 70)

    feature_data = analyze_awd(
        feature_data
    )

    # ========================================================
    # 7. SAVE FEATURES
    # ========================================================

    feature_output = (
        PROCESSED_DIR
        / "satellite_features.csv"
    )

    save_dataframe(
        feature_data,
        feature_output
    )

    # ========================================================
    # 8. GENERATE PLOTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STEP 6: GENERATING PLOTS")
    print("=" * 70)

    plot_satellite_metrics(
        feature_data
    )

    # ========================================================
    # 9. FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PIPELINE COMPLETED")
    print("=" * 70)

    print(
        "\nGenerated files:"
    )

    print(
        f"1. {aligned_output}"
    )

    print(
        f"2. {feature_output}"
    )

    print(
        "\nPlots are available in:"
    )

    print(
        f"{PROJECT_DIR / 'outputs' / 'plots'}"
    )

    print(
        "\nNo machine-learning model was trained."
    )

    print(
        "Reason: IoT ground-truth data is not "
        "available yet."
    )

    print(
        "\nWhen IoT data becomes available, "
        "the next stage will:"
    )

    print(
        "Satellite features + IoT water level"
    )

    print(
        "        ↓"
    )

    print(
        "AWD label using -20 cm threshold"
    )

    print(
        "        ↓"
    )

    print(
        "Farm-aware train/test split"
    )

    print(
        "        ↓"
    )

    print(
        "Random Forest training"
    )

    print(
        "        ↓"
    )

    print(
        "Satellite-only AWD prediction"
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()