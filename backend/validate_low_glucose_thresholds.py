
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# Project paths
ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "glucose_training_dataset_v2.csv"
MODEL_PATH = ROOT / "ml" / "models" / "glucose_2h_random_forest_v3_weighted.joblib"

TARGET = "glucose_2h_ahead_mg_dl"
LOW_LIMIT = 70

# Load dataset and existing V3 model information
df = pd.read_csv(DATA_PATH)
artifact = joblib.load(MODEL_PATH)

test_patient_ids = set(artifact["test_patient_ids"])

# Keep the original training patients only.
# The existing test patients remain completely untouched.
train_pool = df[~df["patient_id"].isin(test_patient_ids)].copy()

# Split by patient, not by individual readings.
patients = train_pool[
    ["patient_id", "diabetes_status"]
].drop_duplicates("patient_id")

train_ids, validation_ids = train_test_split(
    patients["patient_id"],
    test_size=0.20,
    random_state=42,
    stratify=patients["diabetes_status"]
)

train_df = train_pool[train_pool["patient_id"].isin(train_ids)].copy()
validation_df = train_pool[
    train_pool["patient_id"].isin(validation_ids)
].copy()

# Use the same input columns as the saved V3 model.
features = artifact["feature_columns"]

X_train = train_df[features]
y_train = train_df[TARGET]

X_val = validation_df[features]
y_val = validation_df[TARGET]

# Rebuild the V3 model with the same settings.
numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = [
    col for col in features if col not in numeric_cols
]

preprocessor = ColumnTransformer([
    (
        "numeric",
        SimpleImputer(strategy="median"),
        numeric_cols
    ),
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical_cols
    )
])

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=100,
            max_depth=18,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    )
])

# Give low and high glucose training examples more weight,
# matching the existing V3 experiment.
weights = np.ones(len(y_train))
weights[y_train.to_numpy() < 70] = 4
weights[y_train.to_numpy() >= 180] = 2

model.fit(X_train, y_train, regressor__sample_weight=weights)

predictions = model.predict(X_val)

actual_low = y_val.to_numpy() < LOW_LIMIT

print("\nVALIDATION DATA")
print("Training patients:", train_ids.nunique())
print("Validation patients:", validation_ids.nunique())
print("Validation readings:", len(y_val))
print("Actual low readings:", int(actual_low.sum()))
print("Validation MAE:", round(mean_absolute_error(y_val, predictions), 3))

print("\nTHRESHOLD ANALYSIS")
print(
    f"{'Threshold':>10} | {'Sensitivity':>11} | "
    f"{'Precision':>9} | {'False positives':>15} | {'Missed lows':>11}"
)

for threshold in [70, 75, 80, 85, 90, 100, 110, 120]:
    flagged = predictions < threshold

    tp = int(np.sum(flagged & actual_low))
    fp = int(np.sum(flagged & ~actual_low))
    fn = int(np.sum(~flagged & actual_low))

    sensitivity = tp / (tp + fn) * 100 if tp + fn else 0
    precision = tp / (tp + fp) * 100 if tp + fp else 0

    print(
        f"{threshold:10} | {sensitivity:10.2f}% | "
        f"{precision:8.2f}% | {fp:15} | {fn:11}"
    )
