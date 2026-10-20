
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "glucose_training_dataset.csv"
MODEL_PATH = BASE_DIR / "ml" / "models" / "glucose_2h_random_forest.joblib"
ERROR_PATH = BASE_DIR / "data" / "processed" / "glucose_error_analysis.csv"

df = pd.read_csv(DATA_PATH)
artifact = joblib.load(MODEL_PATH)
errors = pd.read_csv(ERROR_PATH)

model = artifact["model"]
features = artifact["feature_columns"]

test_ids = artifact["test_patient_ids"]
train_ids = artifact["train_patient_ids"]

train = df[df["patient_id"].isin(train_ids)].copy()
test = df[df["patient_id"].isin(test_ids)].copy()

# 1. Compare training and test target distributions
print("\n===== TARGET DISTRIBUTION =====")

for name, data in [("TRAIN", train), ("TEST", test)]:
    target = data["glucose_2h_ahead_mg_dl"]

    print(f"\n{name}")
    print(f"Count: {len(target)}")
    print(f"Mean: {target.mean():.2f}")
    print(f"Median: {target.median():.2f}")
    print(f"90th percentile: {target.quantile(0.90):.2f}")
    print(f"Maximum: {target.max():.2f}")
    print(f"Readings >= 180: {(target >= 180).sum()}")

# 2. Compare actual and predicted glucose in the test set
predictions = model.predict(test[features])

comparison = pd.DataFrame({
    "actual": test["glucose_2h_ahead_mg_dl"].to_numpy(),
    "predicted": predictions
})

comparison["error"] = comparison["predicted"] - comparison["actual"]

print("\n===== HIGH-GLUCOSE PREDICTIONS =====")

high = comparison[comparison["actual"] >= 180]

print(f"High-glucose rows: {len(high)}")
print(f"Actual mean: {high['actual'].mean():.2f}")
print(f"Predicted mean: {high['predicted'].mean():.2f}")
print(f"Mean error: {high['error'].mean():.2f}")
print(f"Actual maximum: {high['actual'].max():.2f}")
print(f"Predicted maximum: {high['predicted'].max():.2f}")

# 3. Inspect the model's most important features
print("\n===== TOP 15 FEATURE IMPORTANCES =====")


print("Pipeline steps:", list(model.named_steps.keys()))

# Find the fitted preprocessing step
preprocessor = model.named_steps["preprocessor"]

# Find the Random Forest without assuming its step name
regressor = next(
    (
        step
        for step in model.named_steps.values()
        if hasattr(step, "feature_importances_")
    ),
    None
)

if regressor is None:
    print("Could not find a model with feature importances.")
else:
    encoded_names = preprocessor.get_feature_names_out()
    importances = regressor.feature_importances_

    importance_table = pd.DataFrame({
        "feature": encoded_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print("\n===== TOP 15 FEATURE IMPORTANCES =====")
    print(importance_table.head(15).to_string(index=False))


encoded_names = preprocessor.get_feature_names_out()
importances = regressor.feature_importances_

importance_table = pd.DataFrame({
    "feature": encoded_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print(importance_table.head(15).to_string(index=False))

# 4. Check missing values in the test inputs
print("\n===== MISSING TEST INPUT VALUES =====")

missing = test[features].isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if missing.empty:
    print("No missing input values.")
else:
    print(missing.to_string())

print("\nInvestigation complete.")
