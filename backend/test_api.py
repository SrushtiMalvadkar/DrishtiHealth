
from pathlib import Path

import pandas as pd
from app import app, FEATURE_COLUMNS

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "glucose_training_dataset_v2.csv"
)

# Load the synthetic dataset
df = pd.read_csv(DATA_PATH)

missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing features: {missing}")

# Find a row containing values for all model features
complete_rows = df.dropna(subset=FEATURE_COLUMNS)

if complete_rows.empty:
    raise ValueError("No complete example row was found.")

row = complete_rows.iloc[0]

# Preserve each feature's original type and value
sample = {}
for feature in FEATURE_COLUMNS:
    value = row[feature]

    # Convert NumPy scalar values to standard Python values
    if hasattr(value, "item"):
        value = value.item()

    sample[feature] = value

print("Testing with a synthetic patient record.")
print("Features supplied:", len(sample))

# Send the sample to the Flask API
with app.test_client() as client:
    response = client.post("/predict", json=sample)

    print("HTTP status:", response.status_code)
    print("Response:")
    print(response.get_json())
