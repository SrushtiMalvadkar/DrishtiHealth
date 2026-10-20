
from pathlib import Path
import pandas as pd

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "processed" / "glucose_training_dataset.csv"
)

df = pd.read_csv(DATA_PATH)

target = "glucose_2h_ahead_mg_dl"

print("=" * 60)
print("DRISHTI HEALTH - FINAL DATASET CHECK")
print("=" * 60)

print("\nDataset shape:", df.shape)
print("Unique patients:", df["patient_id"].nunique())
print("Duplicate rows:", df.duplicated().sum())

print("\nTarget glucose summary:")
print(df[target].describe())

print("\nTarget values outside 40-400 mg/dL:")
outside = ~df[target].between(40, 400)
print(int(outside.sum()))

print("\nMissing values in input columns:")
print(df.drop(columns=[target]).isna().sum().to_string())

print("\nTime feature check:")
print("Hours present:", sorted(df["hour_of_day"].dropna().unique()))
print("Days present:", sorted(df["day_of_week"].dropna().unique()))

print("\nRows per patient:")
print(df.groupby("patient_id").size().describe())

print("\nFINAL CHECK COMPLETE")
