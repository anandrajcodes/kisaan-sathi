
from pathlib import Path

import joblib
import pandas as pd
import shap


# Locate the project folders
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "crop_model.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Crop_recommendation.csv"


def explain_prediction(input_values, top_n=5):
    """
    Explain a crop prediction using SHAP feature contributions.

    input_values must contain the seven model input features.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    saved_data = joblib.load(MODEL_PATH)

    # Support the saved model dictionary used by this project
    model = saved_data["model"]
    features = saved_data["features"]

    # Prepare the input in the same feature order as training
    input_df = pd.DataFrame(
        [{feature: input_values[feature] for feature in features}],
        columns=features,
    )

    # A small background sample keeps explanations practical
    dataset = pd.read_csv(DATA_PATH)
    background = dataset[features].sample(
        n=min(100, len(dataset)),
        random_state=42,
    )

    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional",
    )

    explanation = explainer(input_df, check_additivity=False)

    # SHAP output for a classifier can contain one set of values per class.
    values = explanation.values
    predicted_class = model.predict(input_df)[0]
    class_index = list(model.classes_).index(predicted_class)

    if values.ndim == 3:
        contributions = values[0, :, class_index]
    elif values.ndim == 2:
        contributions = values[0]
    else:
        raise ValueError(
            f"Unexpected SHAP output shape: {values.shape}"
        )

    ranked = sorted(
        zip(features, contributions),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )[:top_n]

    return [
        {
            "feature": feature,
            "input_value": float(input_df.iloc[0][feature]),
            "shap_value": round(float(value), 6),
            "direction": (
                "supports this crop"
                if value > 0
                else "pushes away from this crop"
                if value < 0
                else "little or no effect"
            ),
        }
        for feature, value in ranked
    ]
