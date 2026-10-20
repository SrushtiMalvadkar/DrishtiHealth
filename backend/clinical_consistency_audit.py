
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

patients = pd.read_csv(DATA_DIR / "patients.csv")
sensors = pd.read_csv(DATA_DIR / "sensor_timeseries.csv")
predictions = pd.read_csv(DATA_DIR / "glucose_predictions.csv")
meals = pd.read_csv(DATA_DIR / "meal_logs.csv")
medications = pd.read_csv(DATA_DIR / "medications.csv")

print("=" * 70)
print("DRISHTI HEALTH - CLINICAL CONSISTENCY AUDIT")
print("=" * 70)

# 1. Compare patient groups.
print("\n1. GLUCOSE BY DIABETES STATUS")

patient_status = patients[
    ["patient_id", "diabetes_status", "hba1c_baseline_percent",
     "baseline_fasting_glucose_mg_dl"]
].copy()

sensor_grouped = sensors.merge(
    patient_status[["patient_id", "diabetes_status"]],
    on="patient_id",
    how="left",
    validate="many_to_one"
)

sensor_grouped["glucose_mg_dl"] = pd.to_numeric(
    sensor_grouped["glucose_mg_dl"], errors="coerce"
)

for status, group in sensor_grouped.groupby("diabetes_status"):
    g = group["glucose_mg_dl"].dropna()
    print(f"\n{status}:")
    print("Available readings:", len(g))
    print("Mean glucose:", round(g.mean(), 2))
    print("Median glucose:", round(g.median(), 2))
    print("Below 70 mg/dL:", int((g < 70).sum()))
    print("Below 54 mg/dL:", int((g < 54).sum()))
    print("Above 180 mg/dL:", int((g > 180).sum()))

print("\nBaseline laboratory values by diagnosis:")
print(
    patient_status.groupby("diabetes_status")[
        ["hba1c_baseline_percent", "baseline_fasting_glucose_mg_dl"]
    ].agg(["count", "min", "median", "max"]).round(2).to_string()
)

# These are review flags, not automatic proof of incorrect diagnosis.
no_diabetes = patient_status[
    patient_status["diabetes_status"] == "no_diabetes"
]

print("\nNo-diabetes patients with HbA1c >= 6.5%:",
      int((no_diabetes["hba1c_baseline_percent"] >= 6.5).sum()))

print("No-diabetes patients with baseline fasting glucose >= 126 mg/dL:",
      int((no_diabetes["baseline_fasting_glucose_mg_dl"] >= 126).sum()))

print("\nType 2 diabetes patients with both baseline values below diagnostic thresholds:")
type2 = patient_status[
    patient_status["diabetes_status"] == "type_2_diabetes"
]
controlled_values = (
    (type2["hba1c_baseline_percent"] < 6.5)
    & (type2["baseline_fasting_glucose_mg_dl"] < 126)
)
print(int(controlled_values.sum()),
      "(not automatically an error; treatment may explain this)")

# 2. Check prediction labels against future glucose.
print("\n2. FUTURE LABEL CONSISTENCY")

target = pd.to_numeric(
    predictions["glucose_2h_ahead_mg_dl"], errors="coerce"
)
valid = target.notna()

checks = {
    "hypoglycemia_in_2h equals target < 70":
        (target[valid] < 70),
    "glucose_spike_in_2h equals target > 180":
        (target[valid] > 180),
}

for label, expected in checks.items():
    # Align expected values with valid prediction rows.
    observed_col = (
        "hypoglycemia_in_2h"
        if label.startswith("hypoglycemia")
        else "glucose_spike_in_2h"
    )
    observed = pd.to_numeric(
        predictions.loc[valid, observed_col], errors="coerce"
    )
    expected = expected.astype(int)
    comparable = observed.notna()
    mismatches = int(
        (observed[comparable].astype(int) != expected[comparable]).sum()
    )
    print(f"{label}: {mismatches} mismatches")

print("Valid targets:", int(valid.sum()))

# 3. Compare meal calories with approximate energy from macronutrients.
print("\n3. MEAL ENERGY CONSISTENCY")

for col in ["carbohydrates_g", "protein_g", "fat_g",
            "calories_kcal", "fiber_g"]:
    meals[col] = pd.to_numeric(meals[col], errors="coerce")

# Standard approximate energy factors: carbohydrate 4, protein 4,
# fat 9 kcal/g. Fiber and food-label rounding can cause differences.
meals["estimated_macro_kcal"] = (
    4 * meals["carbohydrates_g"]
    + 4 * meals["protein_g"]
    + 9 * meals["fat_g"]
)

meals["calorie_difference"] = (
    meals["calories_kcal"] - meals["estimated_macro_kcal"]
)

print("Median absolute calorie difference:",
      round(meals["calorie_difference"].abs().median(), 2), "kcal")
print("Meals differing by more than 20% of recorded calories:",
      int((
          meals["calorie_difference"].abs()
          > 0.20 * meals["calories_kcal"].clip(lower=1)
      ).sum()))

print("\nFive largest calorie differences:")
print(
    meals.assign(
        absolute_difference=meals["calorie_difference"].abs()
    ).nlargest(5, "absolute_difference")[
        ["meal_event_id", "carbohydrates_g", "protein_g", "fat_g",
         "calories_kcal", "estimated_macro_kcal", "calorie_difference"]
    ].round(2).to_string(index=False)
)

# 4. Inspect medication name, dose, and unit combinations.
print("\n4. MEDICATION DOSE AND UNIT REVIEW")
print(
    medications.groupby(
        ["medication_name", "dose_value", "dose_unit", "medication_status"],
        dropna=False
    ).size().rename("records").reset_index()
    .sort_values(["medication_name", "dose_value"])
    .to_string(index=False)
)

print("\nAUDIT COMPLETE - NO FILES MODIFIED")
