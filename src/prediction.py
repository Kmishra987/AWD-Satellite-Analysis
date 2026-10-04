import pandas as pd


# ============================================================
# PREDICTION
# ============================================================

def predict_awd(
    model,
    X
):
    """
    Generate AWD predictions using a trained model.
    """

    predictions = model.predict(X)

    return predictions


# ============================================================
# AWD PROBABILITY
# ============================================================

def predict_awd_probability(
    model,
    X
):
    """
    Generate probability of AWD condition.

    Requires a trained classifier supporting predict_proba().
    """

    probabilities = model.predict_proba(
        X
    )

    # Find probability corresponding to class 1
    if 1 in model.classes_:

        class_index = list(
            model.classes_
        ).index(1)

        return probabilities[
            :, class_index
        ]

    return probabilities[:, 0]


# ============================================================
# CREATE PREDICTION DATAFRAME
# ============================================================

def create_prediction_output(
    dates,
    predictions,
    probabilities=None
):
    """
    Create a clean prediction dataframe.
    """

    result = pd.DataFrame({
        "date": dates,
        "awd_prediction": predictions
    })

    if probabilities is not None:

        result[
            "awd_probability"
        ] = probabilities

    return result