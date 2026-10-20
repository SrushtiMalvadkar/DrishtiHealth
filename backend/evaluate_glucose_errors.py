
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "glucose_training_dataset.csv"
MODEL_PATH = BASE_DIR / "ml" / "models" / "glucose_2h_random_forest.joblib"

# Load the dataset and trained model
df = pd.read_csv(DATA_PATH)
artifact = joblib.load(MODEL_PATH)

# Retrieve the saved model and test patients
model = artifact["model"]
feature_columns = artifact["feature_columns"]
test_patient_ids = artifact["test_patient_ids"]

# Select only the test patients used during evaluation
test_df = df[df["patient_id"].isin(test_patient_ids)].copy()

X_test = test_df[feature_columns]
y_actual = test_df["glucose_2h_ahead_mg_dl"]

# Generate predictions
y_predicted = model.predict(X_test)

# Create an error analysis table
results = test_df[
    ["patient_id", "glucose_2h_ahead_mg_dl"]
].copy()

results["predicted_glucose"] = y_predicted
results["error"] = results["predicted_glucose"] - results[
    "glucose_2h_ahead_mg_dl"
]
results["absolute_error"] = results["error"].abs()

# Overall metrics
mae = mean_absolute_error(y_actual, y_predicted)
rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
r2 = r2_score(y_actual, y_predicted)

print("\n===== OVERALL TEST RESULTS =====")
print(f"Test rows: {len(results)}")
print(f"MAE: {mae:.2f} mg/dL")
print(f"RMSE: {rmse:.2f} mg/dL")
print(f"R2 score: {r2:.4f}")

# Examine the distribution of prediction errors
print("\n===== ABSOLUTE ERROR DISTRIBUTION =====")
print(results["absolute_error"].describe(
    percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
).round(2))

# Categorize actual future glucose levels
results["glucose_category"] = pd.cut(
    results["glucose_2h_ahead_mg_dl"],
    bins=[-np.inf, 70, 180, np.inf],
    labels=["Low (<70)", "Normal range (70-180)", "High (>180)"],
    right=False
)

print("\n===== ERRORS BY ACTUAL GLUCOSE CATEGORY =====")

category_results = results.groupby(
    "glucose_category", observed=False
).agg(
    rows=("absolute_error", "size"),
    MAE=("absolute_error", "mean"),
    RMSE=("error", lambda x: np.sqrt(np.mean(x ** 2))),
    Mean_Error=("error", "mean")
)

print(category_results.round(2))

# Show the largest prediction mistakes
print("\n===== 10 LARGEST PREDICTION ERRORS =====")

largest_errors = results.nlargest(10, "absolute_error")[
    [
        "patient_id",
        "glucose_2h_ahead_mg_dl",
        "predicted_glucose",
        "error",
        "absolute_error"
    ]
]

print(largest_errors.round(2).to_string(index=False))

# Save detailed results for further analysis
output_path = BASE_DIR / "data" / "processed" / "glucose_error_analysis.csv"
results.to_csv(output_path, index=False)

print(f"\nDetailed results saved to:\n{output_path}")
