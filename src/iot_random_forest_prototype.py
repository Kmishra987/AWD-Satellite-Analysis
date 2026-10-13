
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data/raw/iot/iot.csv"
MODEL_DIR = ROOT / "outputs/models"
METRICS_DIR = ROOT / "outputs/metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "water_depth_cm",
    "water_level_percent",
    "sensor_value",
]
TARGET = "awd_status"

# --------------------------------------------------
# 1. Load and validate data
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)

required = FEATURES + [TARGET, "timestamp"]
missing_columns = [c for c in required if c not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df["timestamp"] = pd.to_datetime(
    df["timestamp"], errors="coerce", utc=True
)

for column in FEATURES + [TARGET]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=required).copy()
df = df[df[TARGET].isin([0, 1])].copy()
df = df.sort_values("timestamp").reset_index(drop=True)

if len(df) < 10:
    raise ValueError("Not enough valid observations to run this experiment.")

if df[TARGET].nunique() != 2:
    raise ValueError("Both target classes (0 and 1) are required.")

print("=" * 65)
print("IOT AWD RANDOM FOREST — PROTOTYPE EXPERIMENT")
print("=" * 65)
print("Valid rows:", len(df))
print("\nTarget class counts:")
print(df[TARGET].value_counts().sort_index())

print("\nData source counts:")
if "data_source" in df.columns:
    print(df["data_source"].value_counts(dropna=False))

# --------------------------------------------------
# 2. Chronological holdout
# --------------------------------------------------
split_index = int(len(df) * 0.80)
split_index = max(1, min(split_index, len(df) - 1))

train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

X_train = train[FEATURES]
y_train = train[TARGET].astype(int)

X_test = test[FEATURES]
y_test = test[TARGET].astype(int)

if y_train.nunique() != 2:
    raise ValueError(
        "The chronological training portion does not contain both "
        "classes. Inspect the timestamps and class distribution."
    )

print("\nTraining rows:", len(train))
print("Testing rows:", len(test))
print("Training period:",
      train["timestamp"].min(), "to", train["timestamp"].max())
print("Testing period:",
      test["timestamp"].min(), "to", test["timestamp"].max())

# --------------------------------------------------
# 3. Train model
# --------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# --------------------------------------------------
# 4. Evaluate on later observations
# --------------------------------------------------
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
balanced_accuracy = balanced_accuracy_score(y_test, predictions)

print("\n" + "=" * 65)
print("HOLDOUT EVALUATION")
print("=" * 65)
print(f"Accuracy:          {accuracy:.3f}")
print(f"Balanced accuracy: {balanced_accuracy:.3f}")

print("\nConfusion matrix (rows=true, columns=predicted; labels 0, 1):")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        zero_division=0,
    )
)

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\nFeature importance:")
print(importance.to_string(index=False))

# --------------------------------------------------
# 5. Save model and experiment results
# --------------------------------------------------
model_path = MODEL_DIR / "iot_awd_random_forest_prototype.joblib"
metrics_path = METRICS_DIR / "iot_awd_prototype_metrics.json"
importance_path = METRICS_DIR / "iot_awd_feature_importance.csv"

joblib.dump(model, model_path)
importance.to_csv(importance_path, index=False)

metrics = {
    "experiment": "IoT AWD Random Forest prototype",
    "data_source": "Wokwi simulation; verify source values",
    "n_valid_rows": int(len(df)),
    "n_train": int(len(train)),
    "n_test": int(len(test)),
    "features": FEATURES,
    "target": TARGET,
    "split": "chronological 80/20 holdout",
    "accuracy": float(accuracy),
    "balanced_accuracy": float(balanced_accuracy),
    "test_class_counts": {
        str(int(k)): int(v)
        for k, v in y_test.value_counts().items()
    },
    "confusion_matrix_labels_0_1": confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).tolist(),
    "warning": (
        "Prototype-only evaluation. Repeated readings, small sample size, "
        "single farm and threshold-derived labels may limit generalization."
    ),
}

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

print("\nSaved model:", model_path)
print("Saved metrics:", metrics_path)
print("Saved feature importance:", importance_path)
print("\nExperiment completed.")
print("Do not interpret this result as real-field validation.")
