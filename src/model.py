from pathlib import Path

import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

def create_model():
    """
    Create the Random Forest classifier.

    The model is not trained until actual IoT-derived
    AWD labels are available.
    """

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    return model


# ============================================================
# TRAIN
# ============================================================

def train_model(X_train, y_train):
    """
    Train Random Forest using real labeled data.
    """

    model = create_model()

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# EVALUATE
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):
    """
    Evaluate trained Random Forest.
    """

    predictions = model.predict(
        X_test
    )

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0
        )
    }

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    for name, value in metrics.items():

        print(
            f"{name}: {value:.4f}"
        )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    return metrics, predictions


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(
    model,
    feature_names
):
    """
    Return Random Forest feature importance.
    """

    importance = model.feature_importances_

    result = list(
        zip(
            feature_names,
            importance
        )
    )

    result.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return result


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    output_path
):
    """
    Save trained model to disk.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        output_path
    )

    print(
        f"Model saved to: {output_path}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    model_path
):
    """
    Load trained Random Forest model.
    """

    model_path = Path(
        model_path
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    return joblib.load(
        model_path
    )