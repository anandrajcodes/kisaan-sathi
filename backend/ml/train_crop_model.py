from pathlib import Path

import joblib
import pandas as pd
import matplotlib

# Use a non-interactive backend so saving plots does not require Tkinter.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split


# Project paths
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "Crop_recommendation.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = ROOT / "docs" / "model_evaluation"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

FEATURES = [
    "N", "P", "K", "temperature",
    "humidity", "ph", "rainfall"
]
TARGET = "label"

X = df[FEATURES]
y = df[TARGET]

# Split before training to evaluate on unseen records
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Baseline: always predicts the most frequent crop
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_preds = baseline.predict(X_test)

# Main model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

baseline_accuracy = accuracy_score(y_test, baseline_preds)
model_accuracy = accuracy_score(y_test, predictions)

print("BASELINE ACCURACY:", round(baseline_accuracy, 4))
print("RANDOM FOREST ACCURACY:", round(model_accuracy, 4))

report = classification_report(
    y_test,
    predictions,
    labels=model.classes_,
    zero_division=0,
)
print("\nCLASSIFICATION REPORT:")
print(report)

# Save confusion matrix as a high-resolution PNG
fig, ax = plt.subplots(figsize=(16, 14))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    labels=model.classes_,
    display_labels=model.classes_,
    xticks_rotation=90,
    cmap="Greens",
    values_format="d",
    colorbar=False,
    ax=ax,
)
ax.set_title("Random Forest Crop Classification Confusion Matrix")
ax.set_xlabel("Predicted crop")
ax.set_ylabel("Actual crop")
fig.tight_layout()

confusion_path = REPORT_DIR / "confusion_matrix.png"
fig.savefig(confusion_path, dpi=200, bbox_inches="tight")
plt.close(fig)

# Save text report
report_path = REPORT_DIR / "classification_report.txt"
report_path.write_text(
    "Kissaan AI - Crop Model Evaluation\n"
    f"Dataset records: {len(df)}\n"
    f"Training records: {len(X_train)}\n"
    f"Test records: {len(X_test)}\n"
    f"Crop classes: {y.nunique()}\n"
    f"Baseline accuracy: {baseline_accuracy:.4f}\n"
    f"Random Forest accuracy: {model_accuracy:.4f}\n\n"
    f"Classification report:\n{report}\n",
    encoding="utf-8",
)

# Save model and its expected feature order
model_path = MODEL_DIR / "crop_model.joblib"
joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "classes": model.classes_.tolist()
    },
    model_path
)

print(f"\nModel saved to: {model_path}")
print(f"Confusion matrix saved to: {confusion_path}")
print(f"Classification report saved to: {report_path}")
