
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

def load(name):
    return pd.read_csv(DATA_DIR / name)

patients = load("patients.csv")
sensors = load("sensor_timeseries.csv")
predictions = load("glucose_predictions.csv")
meals = load("meal_logs.csv")
labs = load("lab_observations.csv")
medications = load("medications.csv")

def heading(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)

def numeric_summary(df, columns):
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if not values.empty:
            print(
                f"{column}: count={len(values)}, "
                f"min={values.min():.2f}, "
                f"median={values.median():.2f}, "
                f"max={values.max():.2f}"
            )

heading("1. PATIENT PROFILE")
print("Patients:", len(patients))
for column in [
    "diabetes_status", "diagnosis", "baseline_risk_category",
    "hypertension_status", "dyslipidemia_status"
]:
    if column in patients.columns:
        print(f"\n{column} counts:")
        print(patients[column].value_counts(dropna=False).to_string())

numeric_summary(patients, [
    "age_years", "bmi", "diabetes_duration_years",
    "hba1c_baseline_percent", "baseline_fasting_glucose_mg_dl"
])

heading("2. SENSOR GLUCOSE DISTRIBUTION")
glucose = pd.to_numeric(
    sensors["glucose_mg_dl"], errors="coerce"
).dropna()

print("Available glucose readings:", len(glucose))
print(glucose.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))

print("\nThreshold counts among available hourly readings:")
for label, condition in [
    ("Below 54 mg/dL", glucose < 54),
    ("54 to below 70 mg/dL", (glucose >= 54) & (glucose < 70)),
    ("70 to 180 mg/dL inclusive", glucose.between(70, 180)),
    ("Above 180 mg/dL", glucose > 180),
    ("Above 250 mg/dL", glucose > 250),
]:
    print(f"{label}: {int(condition.sum())} ({condition.mean() * 100:.2f}%)")

heading("3. GLUCOSE PATTERNS BY PATIENT")
sensor_copy = sensors.copy()
sensor_copy["glucose_mg_dl"] = pd.to_numeric(
    sensor_copy["glucose_mg_dl"], errors="coerce"
)

per_patient = sensor_copy.groupby("patient_id")["glucose_mg_dl"].agg(
    readings="count",
    mean_glucose="mean",
    minimum="min",
    maximum="max",
    std_dev="std"
)

print("Patient glucose summary:")
print(per_patient.describe().round(2).to_string())

print("\nPatients with minimum glucose below 54:")
print(int((per_patient["minimum"] < 54).sum()))

print("Patients with maximum glucose above 250:")
print(int((per_patient["maximum"] > 250).sum()))

print("\nFive lowest patient mean glucose values:")
print(per_patient.nsmallest(5, "mean_glucose").round(2).to_string())

print("\nFive highest patient mean glucose values:")
print(per_patient.nlargest(5, "mean_glucose").round(2).to_string())

heading("4. FUTURE TARGETS AND LABELS")
target = pd.to_numeric(
    predictions["glucose_2h_ahead_mg_dl"], errors="coerce"
)
print("Target summary:")
print(target.describe().round(2).to_string())

for column in ["glucose_spike_in_2h", "hypoglycemia_in_2h", "target_valid_flag"]:
    if column in predictions.columns:
        print(f"\n{column} counts:")
        print(predictions[column].value_counts(dropna=False).to_string())

heading("5. LAB TESTS AND UNITS")
print("Test names and units:")
print(
    labs.groupby(["test_name", "unit"], dropna=False)
    .size()
    .rename("records")
    .reset_index()
    .to_string(index=False)
)

labs["result_value"] = pd.to_numeric(
    labs["result_value"], errors="coerce"
)
print("\nLab result ranges by test and unit:")
print(
    labs.groupby(["test_name", "unit"], dropna=False)["result_value"]
    .agg(["count", "min", "median", "max"])
    .round(2)
    .to_string()
)

heading("6. MEAL LOGS")
numeric_summary(meals, [
    "carbohydrates_g", "protein_g", "fat_g",
    "calories_kcal", "fiber_g", "estimated_glycemic_load"
])
print("\nMeal types:")
print(meals["meal_type"].value_counts(dropna=False).to_string())

heading("7. MEDICATION RECORDS")
for column in ["medication_name", "medication_class", "medication_status", "frequency"]:
    print(f"\n{column} counts:")
    print(medications[column].value_counts(dropna=False).to_string())

print("\nDose values:")
numeric_summary(medications, ["dose_value"])

heading("AUDIT COMPLETE")
print("This report identifies patterns for review; it does not validate diagnoses.")
print("No source files were changed.")
