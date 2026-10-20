
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "ml" / "models"

TARGET = "glucose_2h_ahead_mg_dl"

# Load V2 data and the existing saved patient split.
artifact = joblib.load(
    MODEL_DIR / "glucose_2h_random_forest_v2.joblib"
)

data = pd.read_csv(
    DATA_DIR / "glucose_training_dataset_v2.csv"
)

train_ids = set(artifact["train_patient_ids"])
test_ids = set(artifact["test_patient_ids"])

train = data[data["patient_id"].isin(train_ids)].copy()
test = data[data["patient_id"].isin(test_ids)].copy()

X_train = train.drop(columns=["patient_id", TARGET])
y_train = train[TARGET]

X_test = test.drop(columns=["patient_id", TARGET])
y_test = test[TARGET].to_numpy()

# Build preprocessing using training data only.
numeric_columns = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

preprocessor = ColumnTransformer([
    (
        "numeric",
        SimpleImputer(strategy="median"),
        numeric_columns
    ),
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical_columns
    )
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

# Give additional weight to rare extreme targets in the TRAINING set.
# These weights use training labels only.
weights = np.ones(len(y_train), dtype=float)
weights[y_train < 70] = 4.0
weights[y_train >= 180] = 2.0

print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
print("Training low-glucose rows:", int((y_train < 70).sum()))
print("Training high-glucose rows:", int((y_train >= 180).sum()))

print("\nTraining weighted Random Forest...")
model.fit(X_train, y_train, regressor__sample_weight=weights)

predicted = model.predict(X_test)

def evaluate_range(label, mask):
    count = int(mask.sum())
    if count == 0:
        print(f"{label}: no test rows")
        return

    actual_values = y_test[mask]
    predicted_values = predicted[mask]

    print(
        f"{label}: n={count}, "
        f"MAE={mean_absolute_error(actual_values, predicted_values):.3f}, "
        f"mean_error={(predicted_values - actual_values).mean():.3f}"
    )

print("\n=== V3 WEIGHTED MODEL RESULTS ===")
print(f"MAE:  {mean_absolute_error(y_test, predicted):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predicted)):.3f}")
print(f"R2:   {r2_score(y_test, predicted):.4f}")

evaluate_range("Low glucose (<70)", y_test < 70)
evaluate_range("Normal glucose (70 to <180)", (y_test >= 70) & (y_test < 180))
evaluate_range("High glucose (>=180)", y_test >= 180)

# Save as a separate experimental model.
output_path = MODEL_DIR / "glucose_2h_random_forest_v3_weighted.joblib"

joblib.dump({
    "model": model,
    "feature_columns": X_train.columns.tolist(),
    "target": TARGET,
    "train_patient_ids": sorted(train_ids),
    "test_patient_ids": sorted(test_ids),
    "random_seed": 42,
    "training_weights": {
        "low_below_70": 4.0,
        "high_at_least_180": 2.0
    }
}, output_path)

# Save test predictions for comparison.
pd.DataFrame({
    "patient_id": test["patient_id"].to_numpy(),
    "actual_glucose": y_test,
    "predicted_glucose": predicted
}).to_csv(DATA_DIR / "v3_weighted_test_predictions.csv", index=False)

print("\nSaved model:", output_path)

