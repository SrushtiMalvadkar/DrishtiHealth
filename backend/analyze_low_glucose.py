
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_PATH = (
    BASE_DIR / "ml" / "models" /
    "glucose_2h_random_forest_v2.joblib"
)

TARGET = "glucose_2h_ahead_mg_dl"

# Load the saved V2 model and its held-out test patients.
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
feature_columns = artifact["feature_columns"]
test_ids = set(artifact["test_patient_ids"])

# Load the dataset and patient information.
data = pd.read_csv(DATA_DIR / "glucose_training_dataset_v2.csv")
patients = pd.read_csv(BASE_DIR / "data" / "raw" / "patients.csv")

test = data[data["patient_id"].isin(test_ids)].copy()
test = test.reset_index(drop=True)

X_test = test[feature_columns]
actual = test[TARGET].to_numpy()
predicted = model.predict(X_test)

# Create a table of actual and predicted values.
results = pd.DataFrame({
    "patient_id": test["patient_id"].to_numpy(),
    "actual_glucose": actual,
    "predicted_glucose": predicted
})

results["error"] = results["predicted_glucose"] - results["actual_glucose"]
results["absolute_error"] = results["error"].abs()

results = results.merge(
    patients[["patient_id", "diabetes_status"]],
    on="patient_id",
    how="left",
    validate="many_to_one"
)

# Focus on actual low-glucose readings.
low = results[results["actual_glucose"] < 70].copy()

print("\n=== LOW-GLUCOSE ERROR ANALYSIS ===")
print("Test readings:", len(results))
print("Actual readings below 70:", len(low))

if len(low) > 0:
    print(
        "Low-glucose MAE:",
        round(mean_absolute_error(
            low["actual_glucose"], low["predicted_glucose"]
        ), 3)
    )

    print(
        "Low-glucose RMSE:",
        round(np.sqrt(mean_squared_error(
            low["actual_glucose"], low["predicted_glucose"]
        )), 3)
    )

    # Count low readings whose predictions reach or exceed 70.
    missed_low = low[low["predicted_glucose"] >= 70]

    print(
        "\nActual below 70 but predicted 70 or higher:",
        len(missed_low)
    )

    print(
        "Percentage of low readings predicted at least 70:",
        round(100 * len(missed_low) / len(low), 2), "%"
    )

    print("\nWorst 15 low-glucose prediction errors:")
    print(
        low.sort_values("absolute_error", ascending=False)[
            [
                "patient_id",
                "diabetes_status",
                "actual_glucose",
                "predicted_glucose",
                "error",
                "absolute_error"
            ]
        ].head(15).round(2).to_string(index=False)
    )

    print("\nLow-glucose performance by diabetes status:")
    print(
        low.groupby("diabetes_status").agg(
            readings=("actual_glucose", "size"),
            mean_actual=("actual_glucose", "mean"),
            mean_predicted=("predicted_glucose", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            mean_error=("error", "mean")
        ).round(2).to_string()
    )

# Check potential low-glucose alerts among readings not actually low.
not_low = results[results["actual_glucose"] >= 70]
predicted_low = not_low[not_low["predicted_glucose"] < 70]

print("\n=== PREDICTIONS BELOW 70 WHEN ACTUAL IS 70 OR HIGHER ===")
print("Number of readings:", len(predicted_low))

# Save the complete test predictions for further analysis.
output_path = DATA_DIR / "v2_test_predictions.csv"
results.to_csv(output_path, index=False)

print("\nSaved all test predictions to:", output_path)
