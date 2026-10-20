
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "ml" / "models"

TARGET = "glucose_2h_ahead_mg_dl"

data = pd.read_csv(DATA_DIR / "glucose_training_dataset_v2.csv")

artifact = joblib.load(
    MODEL_DIR / "glucose_2h_random_forest_v3_weighted.joblib"
)

test_ids = set(artifact["test_patient_ids"])
test = (
    data[data["patient_id"].isin(test_ids)]
    .copy()
    .reset_index(drop=True)
)

actual = test[TARGET].to_numpy()
X = test[artifact["feature_columns"]]
predicted = artifact["model"].predict(X)

actual_low = actual < 70

print("Test readings:", len(actual))
print("Actual low readings:", int(actual_low.sum()))

print("\nExploratory threshold analysis")
print("Threshold | Sensitivity | Precision | False positives | Missed lows")

for threshold in [70, 75, 80, 85, 90, 100, 110, 120]:
    flagged = predicted < threshold

    tp = int((flagged & actual_low).sum())
    fp = int((flagged & ~actual_low).sum())
    fn = int((~flagged & actual_low).sum())

    sensitivity = tp / (tp + fn) if tp + fn else 0
    precision = tp / (tp + fp) if tp + fp else 0

    print(
        f"{threshold:9.0f} | "
        f"{sensitivity:10.2%} | "
        f"{precision:9.2%} | "
        f"{fp:14d} | "
        f"{fn:10d}"
    )
