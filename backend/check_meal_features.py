
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

data = pd.read_csv(
    BASE_DIR / "data" / "processed" /
    "glucose_training_dataset_v2.csv"
)

column = "hours_since_last_meal"

print("Total rows:", len(data))
print("Missing hours_since_last_meal:", data[column].isna().sum())

print("\nRows with missing meal time:")
print(
    data.loc[
        data[column].isna(),
        ["patient_id", "hour_of_day", "day_of_week",
         "carbs_previous_2h_g", "calories_previous_2h_kcal"]
    ].head(15).to_string(index=False)
)

print("\nMissing meal-time values by patient:")
print(
    data.loc[data[column].isna()]
    .groupby("patient_id")
    .size()
    .sort_values(ascending=False)
    .head(15)
)
