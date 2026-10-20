
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "ml" / "models"

TARGET = "glucose_2h_ahead_mg_dl"

data = pd.read_csv(DATA_DIR / "glucose_training_dataset_v2.csv")

v2_artifact = joblib.load(
    MODEL_DIR / "glucose_2h_random_forest_v2.joblib"
)
v3_artifact = joblib.load(
    MODEL_DIR / "glucose_2h_random_forest_v3_weighted.joblib"
)

test_ids = set(v3_artifact["test_patient_ids"])
test = data[data["patient_id"].isin(test_ids)].copy().reset_index(drop=True)

actual = test[TARGET].to_numpy()
X = test[v3_artifact["feature_columns"]]

predictions = {
    "V2": v2_artifact["model"].predict(X),
    "V3": v3_artifact["model"].predict(X)
}

# Evaluate low-glucose classification using a fixed threshold.
actual_low = actual < 70

print("Actual low-glucose readings:", int(actual_low.sum()))

for name, predicted in predictions.items():
    predicted_low = predicted < 70

    true_positive = int((actual_low & predicted_low).sum())
    false_negative = int((actual_low & ~predicted_low).sum())
    false_positive = int((~actual_low & predicted_low).sum())
    true_negative = int((~actual_low & ~predicted_low).sum())

    sensitivity = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative else 0
    )

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive else 0
    )

    print(f"\n=== {name} ===")
    print("True positives:", true_positive)
    print("Missed low readings:", false_negative)
    print("False low predictions:", false_positive)
    print("True negatives:", true_negative)
    print(f"Sensitivity: {sensitivity:.2%}")
    print(f"Precision: {precision:.2%}")
