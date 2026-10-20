
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

sensor = pd.read_csv(RAW_DIR / "sensor_timeseries.csv")
meals = pd.read_csv(RAW_DIR / "meal_logs.csv")
patients = pd.read_csv(RAW_DIR / "patients.csv")

# Parse timestamps and sort chronologically
sensor["timestamp"] = pd.to_datetime(
    sensor["timestamp"], errors="raise", utc=True
)

meals["meal_timestamp"] = pd.to_datetime(
    meals["meal_timestamp"], errors="raise", utc=True
)

sensor = sensor.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
meals = meals.sort_values(["patient_id", "meal_timestamp"]).reset_index(drop=True)

# Construct the target using the actual reading two hours later.
# This target is never used as an input feature.
sensor["target_timestamp"] = (
    sensor.groupby("patient_id")["timestamp"].shift(-2)
)

future = sensor[["patient_id", "timestamp", "glucose_mg_dl"]].copy()
future = future.rename(columns={
    "timestamp": "target_timestamp",
    "glucose_mg_dl": "glucose_2h_ahead_mg_dl"
})


data = sensor.merge(
    future,
    on=["patient_id", "target_timestamp"],
    how="left",
    validate="many_to_one"
)


# Historical glucose features: shift only within each patient.
grouped_glucose = data.groupby("patient_id")["glucose_mg_dl"]

data["glucose_1h_ago"] = grouped_glucose.shift(1)
data["glucose_2h_ago"] = grouped_glucose.shift(2)
data["glucose_3h_ago"] = grouped_glucose.shift(3)

data["glucose_change_1h"] = (
    data["glucose_mg_dl"] - data["glucose_1h_ago"]
)

# Build meal features using a strict past-only time window.
meal_feature_rows = []

for patient_id, patient_sensors in data.groupby("patient_id", sort=False):
    patient_meals = meals[meals["patient_id"] == patient_id]

    meal_times = patient_meals["meal_timestamp"].astype("int64").to_numpy()
    carbs = patient_meals["carbohydrates_g"].to_numpy(dtype=float)
    calories = patient_meals["calories_kcal"].to_numpy(dtype=float)
    glycemic_load = patient_meals["estimated_glycemic_load"].to_numpy(dtype=float)

    sensor_times = patient_sensors["timestamp"].astype("int64").to_numpy()

    
    for timestamp in sensor_times:
        # Find all meals at or before the current timestamp.
        end = np.searchsorted(meal_times, timestamp, side="right")

        # Find meals within the previous two hours.
        start = np.searchsorted(
            meal_times,
            timestamp - pd.Timedelta(hours=2).value,
            side="right"
        )

        # Features based only on meals in the previous two hours.
        meal_carbs = carbs[start:end]
        meal_calories = calories[start:end]
        meal_load = glycemic_load[start:end]

        # Find the most recent meal, even if it was more than
        # two hours ago.
        if end > 0:
            previous_meal_time = meal_times[end - 1]
            hours_since_meal = (
                timestamp - previous_meal_time
            ) / (60 * 60 * 1e9)
        else:
            # No earlier meal exists in the recorded data.
            hours_since_meal = np.nan

        meal_feature_rows.append({
            "patient_id": patient_id,
            "timestamp": timestamp,
            "carbs_previous_2h_g": meal_carbs.sum(),
            "calories_previous_2h_kcal": meal_calories.sum(),
            "glycemic_load_previous_2h": meal_load.sum(),
            "hours_since_last_meal": hours_since_meal
        })


meal_features = pd.DataFrame(meal_feature_rows)

data["timestamp_ns"] = data["timestamp"].astype("int64")

data = data.merge(
    meal_features,
    left_on=["patient_id", "timestamp_ns"],
    right_on=["patient_id", "timestamp"],
    how="left",
    suffixes=("", "_meal"),
    validate="one_to_one"
)

data = data.drop(columns=["timestamp_ns", "timestamp_meal"])

# Attach patient-level information.
patient_columns = [
    "patient_id", "age_years", "sex", "height_cm", "weight_kg",
    "bmi", "diabetes_status", "diabetes_duration_years",
    "hba1c_baseline_percent", "baseline_fasting_glucose_mg_dl",
    "hypertension_status", "dyslipidemia_status", "smoking_status",
    "activity_level", "usual_sleep_hours"
]

data = data.merge(
    patients[patient_columns],
    on="patient_id",
    how="left",
    validate="many_to_one"
)

# Extract time features from the prediction-time timestamp.
local_time = data["timestamp"].dt.tz_convert("Asia/Kolkata")
data["hour_of_day"] = local_time.dt.hour
data["day_of_week"] = local_time.dt.dayofweek

# Remove identifiers and helper timestamps from model inputs.
# Keep the actual future glucose as the target only.
data = data.drop(columns=[
    "sensor_record_id", "timestamp", "target_timestamp",
    "sensor_source", "glucose_missing_flag", "synthetic_data_flag"
], errors="ignore")

# Do not train on rows without a known future target.
data = data.dropna(subset=["glucose_2h_ahead_mg_dl"])

output_path = PROCESSED_DIR / "glucose_training_dataset_v2.csv"
data.to_csv(output_path, index=False)

print("Saved:", output_path)
print("Dataset shape:", data.shape)
print("Unique patients:", data["patient_id"].nunique())
print("Missing targets:", data["glucose_2h_ahead_mg_dl"].isna().sum())

print("\nNew features:")
for column in [
    "glucose_1h_ago",
    "glucose_2h_ago",
    "glucose_3h_ago",
    "glucose_change_1h",
    "carbs_previous_2h_g",
    "calories_previous_2h_kcal",
    "glycemic_load_previous_2h",
    "hours_since_last_meal"
]:
    print(f"{column}: {data[column].isna().sum()} missing values")
