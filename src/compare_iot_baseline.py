
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "iot" / "iot.csv"

MODEL_PATH = (
    ROOT
    / "outputs"
    / "models"
    / "iot_awd_random_forest_prototype.joblib"
)

OUTPUT_PATH = (
    ROOT
    / "outputs"
    / "metrics"
    / "iot_baseline_comparison.json"
)

FEATURES = [
    "water_depth_cm",
    "water_level_percent",
    "sensor_value",
]

TARGET = "awd_status"
THRESHOLD_CM = -20.0
TEST_FRACTION = 0.20


# --------------------------------------------------
# 2. Load and validate data
# --------------------------------------------------

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"IoT dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "timestamp",
        *FEATURES,
        TARGET,
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Parse timestamps and numeric values safely.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # Do not replace missing measurements with zero.
    df = df.dropna(
        subset=["timestamp", *FEATURES, TARGET]
    ).copy()

    if df.empty:
        raise ValueError(
            "No valid rows remain after data validation."
        )

    # The target must contain only binary class labels.
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError(
            "awd_status must contain only 0 and 1."
        )

    df[TARGET] = df[TARGET].astype(int)

    # Stable chronological ordering.
    df = df.sort_values(
        "timestamp",
        kind="stable",
    ).reset_index(drop=True)

    if len(df) < 5:
        raise ValueError(
            "Too few valid rows for a holdout comparison."
        )

    return df


# --------------------------------------------------
# 3. Evaluate one set of predictions
# --------------------------------------------------

def evaluate_predictions(y_true, predictions):
    return {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
        "confusion_matrix_labels_0_1": (
            confusion_matrix(
                y_true,
                predictions,
                labels=[0, 1],
            ).tolist()
        ),
        "classification_report": (
            classification_report(
                y_true,
                predictions,
                labels=[0, 1],
                output_dict=True,
                zero_division=0,
            )
        ),
    }


# --------------------------------------------------
# 4. Compare threshold baseline and Random Forest
# --------------------------------------------------

def main():
    print("=" * 65)
    print("IOT AWD: THRESHOLD VS RANDOM FOREST")
    print("=" * 65)

    # Load the data.
    df = load_data()

    print(f"Valid rows: {len(df)}")
    print("\nTarget class counts:")
    print(df[TARGET].value_counts().sort_index())

    if "data_source" in df.columns:
        print("\nData sources:")
        print(df["data_source"].value_counts(dropna=False))

    # Match the chronological 80/20 split used during training.
    split_index = int(
        len(df) * (1 - TEST_FRACTION)
    )

    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()

    if test.empty:
        raise ValueError("The test set is empty.")

    print(f"\nTraining rows: {len(train)}")
    print(f"Testing rows:  {len(test)}")
    print(
        "Test period:",
        test["timestamp"].iloc[0],
        "to",
        test["timestamp"].iloc[-1],
    )

    y_true = test[TARGET]

    # ----------------------------------------------
    # A. Simple AWD threshold baseline
    # ----------------------------------------------
    #
    # Assumption: negative depth represents the water
    # level below the soil surface.
    #
    # Confirm this convention against sensor calibration
    # before using the rule on physical hardware.

    threshold_predictions = (
        test["water_depth_cm"] <= THRESHOLD_CM
    ).astype(int)

    threshold_results = evaluate_predictions(
        y_true,
        threshold_predictions,
    )

    # ----------------------------------------------
    # B. Existing Random Forest model
    # ----------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained Random Forest model not found: "
            f"{MODEL_PATH}\n"
            "Run src/iot_random_forest_prototype.py first."
        )

    model = joblib.load(MODEL_PATH)

    rf_predictions = model.predict(
        test[FEATURES]
    )

    rf_results = evaluate_predictions(
        y_true,
        rf_predictions,
    )

    # ----------------------------------------------
    # C. Compare predictions directly
    # ----------------------------------------------

    threshold_agreement = float(
        (threshold_predictions == y_true).mean()
    )

    rf_agreement = float(
        (rf_predictions == y_true).mean()
    )

    prediction_agreement = float(
        (
            threshold_predictions.to_numpy()
            == rf_predictions
        ).mean()
    )

    # This check is descriptive, not field validation.
    same_as_threshold = bool(
        (rf_predictions == threshold_predictions.to_numpy()).all()
    )

    results = {
        "experiment": (
            "IoT AWD threshold baseline versus Random Forest"
        ),
        "data_source": "Wokwi simulation",
        "n_valid_rows": int(len(df)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "test_period_start": str(
            test["timestamp"].iloc[0]
        ),
        "test_period_end": str(
            test["timestamp"].iloc[-1]
        ),
        "features": FEATURES,
        "target": TARGET,
        "split": "chronological 80/20 holdout",
        "threshold_cm": THRESHOLD_CM,
        "threshold_sign_convention": (
            "Assumes negative water_depth_cm means "
            "water level below soil surface."
        ),
        "threshold_baseline": threshold_results,
        "random_forest": rf_results,
        "comparison": {
            "threshold_accuracy": threshold_agreement,
            "random_forest_accuracy": rf_agreement,
            "prediction_agreement": prediction_agreement,
            "random_forest_matches_threshold_on_all_test_rows": (
                same_as_threshold
            ),
        },
        "warning": (
            "Prototype-only evaluation using simulated IoT readings. "
            "The AWD target matched the -20 cm threshold on the "
            "inspected dataset. This comparison does not establish "
            "real-field accuracy or generalization to new farms "
            "or sensor hardware."
        ),
    }

    # ----------------------------------------------
    # 5. Save comparison results
    # ----------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    # ----------------------------------------------
    # 6. Display results
    # ----------------------------------------------

    print("\n" + "=" * 65)
    print("HOLDOUT COMPARISON")
    print("=" * 65)

    print("\nThreshold baseline:")
    print(
        f"Accuracy:          "
        f"{threshold_results['accuracy']:.3f}"
    )
    print(
        f"Balanced accuracy: "
        f"{threshold_results['balanced_accuracy']:.3f}"
    )
    print("Confusion matrix:")
    print(
        threshold_results["confusion_matrix_labels_0_1"]
    )

    print("\nRandom Forest:")
    print(
        f"Accuracy:          "
        f"{rf_results['accuracy']:.3f}"
    )
    print(
        f"Balanced accuracy: "
        f"{rf_results['balanced_accuracy']:.3f}"
    )
    print("Confusion matrix:")
    print(
        rf_results["confusion_matrix_labels_0_1"]
    )

    print("\nPrediction comparison:")
    print(
        f"Threshold and RF agree on "
        f"{prediction_agreement * 100:.2f}% of test rows."
    )

    if same_as_threshold:
        print(
            "Finding: Random Forest predictions match "
            "the threshold rule on every test row."
        )
    else:
        print(
            "Finding: Some Random Forest predictions differ "
            "from the threshold rule. Inspect those rows "
            "before drawing conclusions."
        )

    print(f"\nSaved results to:\n{OUTPUT_PATH}")
    print("\nIMPORTANT: These results are not real-field validation.")


if __name__ == "__main__":
    main()
