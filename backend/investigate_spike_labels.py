
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

df = pd.read_csv(DATA_DIR / "glucose_predictions.csv")

for col in [
    "current_glucose_mg_dl",
    "glucose_2h_ahead_mg_dl",
    "glucose_change_2h_mg_dl",
    "glucose_spike_in_2h",
    "target_valid_flag",
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[df["target_valid_flag"] == 1].copy()
df = df.dropna(subset=[
    "current_glucose_mg_dl",
    "glucose_2h_ahead_mg_dl",
    "glucose_spike_in_2h",
])

# Recalculate the change using the two recorded glucose values.
df["calculated_change"] = (
    df["glucose_2h_ahead_mg_dl"] - df["current_glucose_mg_dl"]
)

# Check whether the stored change agrees with the recorded values.
df["change_difference"] = (
    df["glucose_change_2h_mg_dl"] - df["calculated_change"]
).abs()

print("=" * 70)
print("GLUCOSE-SPIKE LABEL INVESTIGATION")
print("=" * 70)

print("\n1. Stored versus calculated glucose change")
print("Rows checked:", len(df))
print("Rows with missing stored change:",
      int(df["glucose_change_2h_mg_dl"].isna().sum()))
print("Rows with change mismatch above 0.01 mg/dL:",
      int((df["change_difference"] > 0.01).sum()))

print("\n2. Existing spike label counts")
print(df["glucose_spike_in_2h"].value_counts().sort_index().to_string())

print("\n3. Candidate rule: future glucose > 180 mg/dL")
df["candidate_above_180"] = (
    df["glucose_2h_ahead_mg_dl"] > 180
).astype(int)

print(pd.crosstab(
    df["glucose_spike_in_2h"],
    df["candidate_above_180"],
    rownames=["Stored label"],
    colnames=["Future glucose > 180"],
).to_string())

print("Mismatches:",
      int((df["glucose_spike_in_2h"] != df["candidate_above_180"]).sum()))

print("\n4. Candidate rules based on glucose rise")
for threshold in [20, 30, 40, 50]:
    candidate = (df["calculated_change"] >= threshold).astype(int)
    mismatches = int((df["glucose_spike_in_2h"] != candidate).sum())
    print(f"Rise >= {threshold} mg/dL: {mismatches} mismatches")

print("\n5. Stored label by future glucose range")
df["future_range"] = pd.cut(
    df["glucose_2h_ahead_mg_dl"],
    bins=[-float("inf"), 70, 140, 180, float("inf")],
    labels=["Below 70", "70-140", "140-180", "Above 180"],
    right=False,
)
print(pd.crosstab(
    df["future_range"],
    df["glucose_spike_in_2h"],
    margins=True,
).to_string())

print("\n6. Examples where the stored label differs from future glucose > 180")
mismatch_rows = df[
    df["glucose_spike_in_2h"] != df["candidate_above_180"]
]

columns = [
    "patient_id",
    "prediction_timestamp",
    "current_glucose_mg_dl",
    "glucose_2h_ahead_mg_dl",
    "glucose_change_2h_mg_dl",
    "calculated_change",
    "glucose_spike_in_2h",
]

print(mismatch_rows[columns].head(20).to_string(index=False))

print("\n7. Glucose change summary by stored spike label")
print(
    df.groupby("glucose_spike_in_2h", observed=True)["calculated_change"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(2)
    .to_string()
)

# Test whether the intended rule is future glucose >= 180
candidate_label = (
    df["glucose_2h_ahead_mg_dl"] >= 180
).astype(int)

mismatches = (
    candidate_label != df["glucose_spike_in_2h"]
).sum()

print("\n8. Candidate rule: future glucose >= 180 mg/dL")
print(f"Mismatches: {mismatches}")

print("\nINVESTIGATION COMPLETE - NO FILES MODIFIED")
