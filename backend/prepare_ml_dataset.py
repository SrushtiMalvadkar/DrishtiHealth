
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the three datasets needed for the first model.
sensors = pd.read_csv(DATA_DIR / "sensor_timeseries.csv")
predictions = pd.read_csv(DATA_DIR / "glucose_predictions.csv")
patients = pd.read_csv(DATA_DIR / "patients.csv")

print("Loaded sensor, prediction, and patient datasets.")

# Keep only prediction rows with an available, valid target.
predictions["target_valid_flag"] = pd.to_numeric(
    predictions["target_valid_flag"], errors="coerce"
)

predictions["glucose_2h_ahead_mg_dl"] = pd.to_numeric(
    predictions["glucose_2h_ahead_mg_dl"], errors="coerce"
)

predictions = predictions[
    (predictions["target_valid_flag"] == 1)
    & predictions["glucose_2h_ahead_mg_dl"].notna()
].copy()

print("Rows with valid future glucose targets:", len(predictions))

# Attach sensor features using the matching sensor record ID.
sensor_features = [
    "sensor_record_id",
    "glucose_mg_dl",
    "heart_rate_bpm",
    "hrv_ms",
    "steps_last_hour",
    "activity_intensity",
    "sleep_hours_last_night",
    "sleep_state",
    "skin_temperature_c",
    "spo2_percent",
    "sensor_quality",
]

dataset = predictions.merge(
    sensors[sensor_features],
    on="sensor_record_id",
    how="left",
    validate="one_to_one"
)

# Attach patient information.
patient_features = [
    "patient_id",
    "age_years",
    "sex",
    "height_cm",
    "weight_kg",
    "bmi",
    "diabetes_duration_years",
    "hba1c_baseline_percent",
    "baseline_fasting_glucose_mg_dl",
    "hypertension_status",
    "dyslipidemia_status",
    "smoking_status",
    "activity_level",
    "usual_sleep_hours",
]

dataset = dataset.merge(
    patients[patient_features],
    on="patient_id",
    how="left",
    validate="many_to_one"
)

# Extract simple time features from the prediction timestamp.
timestamp = pd.to_datetime(
    dataset["prediction_timestamp"], errors="coerce", utc=True
)

# Convert UTC timestamps to Indian Standard Time.
timestamp = timestamp.dt.tz_convert("Asia/Kolkata")

dataset["hour_of_day"] = timestamp.dt.hour
dataset["day_of_week"] = timestamp.dt.dayofweek

# Define the target: glucose measured two hours later.
target_column = "glucose_2h_ahead_mg_dl"

# Exclude IDs, raw timestamps, and columns derived from future outcomes.
# Keep patient_id temporarily for later patient-level splitting.
keep_columns = [
    "patient_id",
    "glucose_mg_dl",
    "heart_rate_bpm",
    "hrv_ms",
    "steps_last_hour",
    "activity_intensity",
    "sleep_hours_last_night",
    "sleep_state",
    "skin_temperature_c",
    "spo2_percent",
    "sensor_quality",
    "age_years",
    "sex",
    "height_cm",
    "weight_kg",
    "bmi",
    "diabetes_duration_years",
    "hba1c_baseline_percent",
    "baseline_fasting_glucose_mg_dl",
    "hypertension_status",
    "dyslipidemia_status",
    "smoking_status",
    "activity_level",
    "usual_sleep_hours",
    "hour_of_day",
    "day_of_week",
    target_column,
]

dataset = dataset[keep_columns]

# Save the prepared dataset.
output_path = OUTPUT_DIR / "glucose_training_dataset.csv"
dataset.to_csv(output_path, index=False)

print("\nDATASET PREPARATION COMPLETE")
print("Saved to:", output_path)
print("Final rows:", len(dataset))
print("Final columns:", len(dataset.columns))
print("Missing values by column:")
print(dataset.isna().sum().to_string())
print("\nFirst five rows:")
print(dataset.head().to_string(index=False))
