
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "glucose_training_dataset.csv"

df = pd.read_csv(DATA_PATH)

print("\n===== ALL DATASET COLUMNS =====")
for column in df.columns:
    print(column)

print("\n===== POSSIBLE GLUCOSE AND MEAL FEATURES =====")

keywords = [
    "glucose",
    "meal",
    "carb",
    "calorie",
    "food",
    "insulin",
    "medication",
    "change",
    "trend",
    "hour"
]

for column in df.columns:
    if any(word in column.lower() for word in keywords):
        print(f"\nColumn: {column}")
        print(f"Data type: {df[column].dtype}")
        print(f"Missing values: {df[column].isna().sum()}")

        if pd.api.types.is_numeric_dtype(df[column]):
            print(f"Minimum: {df[column].min()}")
            print(f"Median: {df[column].median()}")
            print(f"Maximum: {df[column].max()}")

