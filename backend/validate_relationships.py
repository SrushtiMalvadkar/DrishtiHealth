
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

def load_csv(name):
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def parse_time(series):
    # Parse timestamps consistently, including their UTC offsets.
    return pd.to_datetime(series, errors="coerce", utc=True)

def report_check(label, condition):
    status = "PASS" if condition else "CHECK"
    print(f"[{status}] {label}")

patients = load_csv("patients.csv")
sensors = load_csv("sensor_timeseries.csv")
predictions = load_csv("glucose_predictions.csv")
meals = load_csv("meal_logs.csv")
labs = load_csv("lab_observations.csv")
medications = load_csv("medications.csv")

print("=" * 70)
print("DRISHTI HEALTH - RELATIONSHIP AND TIME VALIDATION")
print("=" * 70)

# 1. Check patient table uniqueness
report_check(
    "Patient IDs are unique",
    patients["patient_id"].is_unique
)

patient_ids = set(patients["patient_id"].dropna())

# 2. Check patient references in every other dataset
tables = {
    "sensor_timeseries.csv": sensors,
    "glucose_predictions.csv": predictions,
    "meal_logs.csv": meals,
    "lab_observations.csv": labs,
    "medications.csv": medications,
}

for name, df in tables.items():
    referenced = set(df["patient_id"].dropna())
    unknown = referenced - patient_ids
    print(f"\n{name}")
    print(f"  Unique patient IDs: {len(referenced)}")
    print(f"  Unknown patient IDs: {len(unknown)}")
    report_check(
        "All referenced patients exist",
        len(unknown) == 0
    )
    if unknown:
        print(f"  Examples: {list(unknown)[:10]}")

# 3. Check sensor IDs, patient-time uniqueness and intervals
print("\nSENSOR TIMESTAMPS")

sensor_time = parse_time(sensors["timestamp"])
print(f"  Unparseable timestamps: {sensor_time.isna().sum()}")

sensor_check = sensors.assign(_time=sensor_time)
valid_sensor = sensor_check.dropna(subset=["_time"])

report_check(
    "Sensor record IDs are unique",
    sensors["sensor_record_id"].is_unique
)

duplicate_patient_times = valid_sensor.duplicated(
    subset=["patient_id", "_time"]
).sum()

print(f"  Duplicate patient/timestamp pairs: {duplicate_patient_times}")

valid_sensor = valid_sensor.sort_values(["patient_id", "_time"])
valid_sensor["_gap"] = (
    valid_sensor.groupby("patient_id")["_time"].diff()
)

gap_counts = valid_sensor["_gap"].value_counts()
non_hourly = int(
    (valid_sensor["_gap"].notna()
     & (valid_sensor["_gap"] != pd.Timedelta(hours=1))).sum()
)

print(f"  Non-hourly intervals between consecutive records: {non_hourly}")
print("  Most common time intervals:")
print(gap_counts.head(5).to_string() if not gap_counts.empty
      else "  Not enough valid timestamps")

rows_per_patient = sensors.groupby("patient_id").size()
print(f"  Patients represented in sensors: {len(rows_per_patient)}")
print(f"  Minimum sensor rows per patient: {rows_per_patient.min()}")
print(f"  Maximum sensor rows per patient: {rows_per_patient.max()}")

# 4. Match predictions to their sensor records
print("\nPREDICTION-TO-SENSOR MATCHING")

report_check(
    "Prediction record IDs are unique",
    predictions["prediction_record_id"].is_unique
)

report_check(
    "Prediction sensor IDs are unique",
    predictions["sensor_record_id"].is_unique
)

sensor_lookup = sensors[
    ["sensor_record_id", "patient_id", "timestamp", "glucose_mg_dl"]
].copy()

sensor_lookup["_sensor_time"] = parse_time(sensor_lookup["timestamp"])

prediction_check = predictions.copy()
prediction_check["_prediction_time"] = parse_time(
    prediction_check["prediction_timestamp"]
)
prediction_check["_target_time"] = parse_time(
    prediction_check["target_timestamp"]
)

matched = prediction_check.merge(
    sensor_lookup,
    on="sensor_record_id",
    how="left",
    suffixes=("_pred", "_sensor"),
    indicator=True,
    validate="many_to_one"
)

matched_count = int((matched["_merge"] == "both").sum())
print(f"  Predictions matched to sensors: {matched_count}")
print(f"  Predictions without matching sensor: {(matched['_merge'] == 'left_only').sum()}")

matched_rows = matched[matched["_merge"] == "both"].copy()

patient_mismatch = (
    matched_rows["patient_id_pred"] != matched_rows["patient_id_sensor"]
).sum()
print(f"  Patient ID mismatches: {patient_mismatch}")

time_mismatch = (
    matched_rows["_prediction_time"] != matched_rows["_sensor_time"]
).sum()
print(f"  Prediction/sensor timestamp mismatches: {time_mismatch}")

report_check(
    "All predictions match a sensor record",
    matched_count == len(predictions)
)
report_check(
    "Matched patient IDs agree",
    patient_mismatch == 0
)
report_check(
    "Prediction times agree with sensor times",
    time_mismatch == 0
)

# Compare current glucose with the linked sensor reading.
# Both may be missing; compare only when both are present.
both_glucose = matched_rows[
    matched_rows["current_glucose_mg_dl"].notna()
    & matched_rows["glucose_mg_dl"].notna()
]

glucose_mismatch = (
    both_glucose["current_glucose_mg_dl"]
    - both_glucose["glucose_mg_dl"]
).abs() > 0.01

print(f"  Current-glucose mismatches: {int(glucose_mismatch.sum())}")

# 5. Check the two-hour prediction horizon
print("\nTWO-HOUR TARGET VALIDATION")

horizon = matched_rows["_target_time"] - matched_rows["_prediction_time"]
valid_horizon = horizon.notna()

wrong_horizon = int(
    (valid_horizon & (horizon != pd.Timedelta(hours=2))).sum()
)

print(f"  Unparseable prediction times: {prediction_check['_prediction_time'].isna().sum()}")
print(f"  Unparseable target times: {prediction_check['_target_time'].isna().sum()}")
print(f"  Incorrect two-hour horizons among matched rows: {wrong_horizon}")

report_check(
    "All parseable matched targets are two hours ahead",
    wrong_horizon == 0
)

# 6. Check target validity flags against missing target values
print("\nTARGET COMPLETENESS")

flag = pd.to_numeric(
    predictions["target_valid_flag"], errors="coerce"
)
target = predictions["glucose_2h_ahead_mg_dl"]

valid_but_missing = int(((flag == 1) & target.isna()).sum())
invalid_but_present = int(((flag == 0) & target.notna()).sum())

print(f"  Valid flag but missing target: {valid_but_missing}")
print(f"  Invalid flag but target present: {invalid_but_present}")
print(f"  Missing future glucose values: {target.isna().sum()}")

report_check(
    "Valid flags agree with target availability",
    valid_but_missing == 0 and invalid_but_present == 0
)

# 7. Show timestamp coverage for each main time series
print("\nTIME COVERAGE")

for name, df, column in [
    ("Sensors", sensors, "timestamp"),
    ("Predictions", predictions, "prediction_timestamp"),
    ("Meals", meals, "meal_timestamp"),
    ("Labs", labs, "collection_timestamp"),
]:
    times = parse_time(df[column])
    print(f"  {name}: {times.min()} to {times.max()}")

print("\nVALIDATION COMPLETE")
print("Review every CHECK result before proceeding.")
