
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

sensors = pd.read_csv(DATA_DIR / "sensor_timeseries.csv")
predictions = pd.read_csv(DATA_DIR / "glucose_predictions.csv")

# Convert timestamps into a consistent timezone.
sensors["_time"] = pd.to_datetime(
    sensors["timestamp"], errors="coerce", utc=True
)

predictions["_prediction_time"] = pd.to_datetime(
    predictions["prediction_timestamp"], errors="coerce", utc=True
)

predictions["_target_time"] = pd.to_datetime(
    predictions["target_timestamp"], errors="coerce", utc=True
)

# Keep the sensor measurements needed for comparison.
target_sensors = sensors[
    ["patient_id", "_time", "glucose_mg_dl"]
].rename(
    columns={
        "_time": "_target_time",
        "glucose_mg_dl": "actual_future_glucose"
    }
)

# Find the sensor measurement for each patient's target timestamp.
checked = predictions.merge(
    target_sensors,
    on=["patient_id", "_target_time"],
    how="left",
    indicator=True,
    validate="many_to_one"
)

matched = checked[checked["_merge"] == "both"].copy()

# Compare values only when both measurements are available.
comparable = matched[
    matched["glucose_2h_ahead_mg_dl"].notna()
    & matched["actual_future_glucose"].notna()
].copy()

mismatch = (
    comparable["glucose_2h_ahead_mg_dl"]
    - comparable["actual_future_glucose"]
).abs() > 0.01

print("=" * 65)
print("DRISHTI HEALTH - FUTURE GLUCOSE VALIDATION")
print("=" * 65)

print(f"Total prediction rows: {len(predictions)}")
print(f"Rows with a matching future sensor timestamp: {len(matched)}")
print(
    "Rows without a matching future sensor timestamp:",
    int((checked["_merge"] == "left_only").sum())
)

print("\nGLUCOSE VALUE COMPARISON")
print(f"Rows where both values exist: {len(comparable)}")
print(f"Future glucose mismatches: {int(mismatch.sum())}")

print("\nMISSING VALUE CHECK")
print(
    "Missing prediction targets:",
    int(predictions["glucose_2h_ahead_mg_dl"].isna().sum())
)
print(
    "Missing actual future sensor glucose among matched rows:",
    int(matched["actual_future_glucose"].isna().sum())
)

print("\nEXAMPLE MISMATCHES")
if mismatch.any():
    columns = [
        "patient_id",
        "prediction_timestamp",
        "target_timestamp",
        "glucose_2h_ahead_mg_dl",
        "actual_future_glucose"
    ]
    print(comparable.loc[mismatch, columns].head(10).to_string(index=False))
else:
    print("No mismatches found among comparable rows.")

print("\nVALIDATION COMPLETE")
