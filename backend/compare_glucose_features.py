
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "glucose_training_dataset.csv"
MODEL_PATH = BASE_DIR / "ml" / "models" / "glucose_2h_random_forest.joblib"

df = pd.read_csv(DATA_PATH)
artifact = joblib.load(MODEL_PATH)

original_model = artifact["model"]
feature_columns = artifact["feature_columns"]

train_ids = artifact["train_patient_ids"]
test_ids = artifact["test_patient_ids"]

train = df[df["patient_id"].isin(train_ids)].copy()
test = df[df["patient_id"].isin(test_ids)].copy()

y_train = train["glucose_2h_ahead_mg_dl"]
y_test = test["glucose_2h_ahead_mg_dl"]

assert set(train["patient_id"]).isdisjoint(set(test["patient_id"]))

reduced_features = [
    col for col in feature_columns
    if col != "baseline_fasting_glucose_mg_dl"
]

feature_sets = {
    "All features": feature_columns,
    "Without baseline fasting glucose": reduced_features
}

results = []

for name, features in feature_sets.items():
    model = clone(original_model)

    # Update the pipeline's column selections to match the features.
    preprocessor = model.named_steps["preprocessor"]

    for transformer_name, transformer, columns in preprocessor.transformers:
        if isinstance(columns, list):
            kept_columns = [col for col in columns if col in features]
            preprocessor.transformers = [
                (
                    n,
                    t,
                    kept_columns if n == transformer_name else c
                )
                for n, t, c in preprocessor.transformers
            ]

    # Refit using only this experiment's selected columns.
    model.fit(train[features], y_train)
    predictions = model.predict(test[features])

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    actual = y_test.to_numpy()
    high_mask = actual >= 180

    high_mae = mean_absolute_error(
        actual[high_mask],
        predictions[high_mask]
    )

    high_mean_error = np.mean(
        predictions[high_mask] - actual[high_mask]
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "High glucose MAE": high_mae,
        "High glucose mean error": high_mean_error
    })

print("\n===== MODEL COMPARISON =====")
print(pd.DataFrame(results).round(3).to_string(index=False))
