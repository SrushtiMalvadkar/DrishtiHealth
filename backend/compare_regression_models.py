
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "glucose_training_dataset.csv"
MODEL_PATH = BASE_DIR / "ml" / "models" / "glucose_2h_random_forest.joblib"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "model_comparison.csv"

# Load data and original model
df = pd.read_csv(DATA_PATH)
artifact = joblib.load(MODEL_PATH)

original_model = artifact["model"]
features = artifact["feature_columns"]

train_ids = artifact["train_patient_ids"]
test_ids = artifact["test_patient_ids"]

train = df[df["patient_id"].isin(train_ids)].copy()
test = df[df["patient_id"].isin(test_ids)].copy()

X_train = train[features]
y_train = train["glucose_2h_ahead_mg_dl"]

X_test = test[features]
y_test = test["glucose_2h_ahead_mg_dl"]

# Make sure patients do not overlap between training and testing
assert set(train_ids).isdisjoint(set(test_ids))

# Define models to compare
models = {
    "Current Random Forest": clone(original_model),

    "Tuned Random Forest": clone(original_model).set_params(
        regressor=RandomForestRegressor(
            n_estimators=200,
            max_depth=24,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    ),

    "Extra Trees": clone(original_model).set_params(
        regressor=ExtraTreesRegressor(
            n_estimators=200,
            max_depth=24,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    )
}

results = []

print("Training patients:", train["patient_id"].nunique())
print("Testing patients:", test["patient_id"].nunique())
print("Training rows:", len(train))
print("Testing rows:", len(test))

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    actual = y_test.to_numpy()
    errors = predictions - actual

    # Overall metrics
    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    r2 = r2_score(actual, predictions)

    # High-glucose metrics: actual glucose >= 180 mg/dL
    high_mask = actual >= 180
    high_mae = mean_absolute_error(
        actual[high_mask], predictions[high_mask]
    )
    high_mean_error = np.mean(errors[high_mask])

    # Low-glucose metrics: actual glucose < 70 mg/dL
    low_mask = actual < 70

    if low_mask.any():
        low_mae = mean_absolute_error(
            actual[low_mask], predictions[low_mask]
        )
    else:
        low_mae = np.nan

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Low glucose MAE": low_mae,
        "High glucose MAE": high_mae,
        "High glucose mean error": high_mean_error
    })

# Display and save results
comparison = pd.DataFrame(results)

print("\n===== REGRESSION MODEL COMPARISON =====")
print(comparison.round(3).to_string(index=False))

comparison.to_csv(OUTPUT_PATH, index=False)

print(f"\nResults saved to: {OUTPUT_PATH}")
print("\nLower MAE and RMSE are better.")
print("Higher R2 is generally better.")
print("High-glucose mean error closer to zero indicates less average bias.")
