
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "glucose_training_dataset_v2.csv"
SENSOR_PATH = ROOT / "data" / "raw" / "sensor_timeseries.csv"
MODEL_PATH = ROOT / "ml" / "models" / "glucose_2h_random_forest_v3_weighted.joblib"

TARGET = "glucose_2h_ahead_mg_dl"
THRESHOLDS = [85, 90]
LOW_LIMIT = 70
TIME_TOLERANCE_HOURS = 1


# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

print("Loading datasets...")

df = pd.read_csv(DATA_PATH)
sensor = pd.read_csv(SENSOR_PATH)
artifact = joblib.load(MODEL_PATH)

features = artifact["feature_columns"]

required_columns = [
    "patient_id",
    "diabetes_status",
    TARGET,
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' is missing from processed data."
        )

if "patient_id" not in sensor.columns:
    raise ValueError("patient_id is missing from sensor data.")

if "timestamp" not in sensor.columns:
    raise ValueError("timestamp is missing from sensor data.")

if "glucose_mg_dl" not in sensor.columns:
    raise ValueError("glucose_mg_dl is missing from sensor data.")

missing_features = [c for c in features if c not in df.columns]

if missing_features:
    raise ValueError(
        f"Model features missing from processed data: {missing_features}"
    )


# ---------------------------------------------------------
# 2. RECONSTRUCT TIMESTAMPS
# ---------------------------------------------------------

print("Reconstructing timestamps...")

sensor["timestamp"] = pd.to_datetime(
    sensor["timestamp"],
    errors="raise",
    utc=True,
)

sensor = sensor.sort_values(
    ["patient_id", "timestamp"]
).reset_index(drop=True)

# The target is the glucose reading two hourly records ahead.
sensor["target_timestamp"] = (
    sensor.groupby("patient_id")["timestamp"].shift(-2)
)

future = sensor[
    ["patient_id", "timestamp", "glucose_mg_dl"]
].copy()

future = future.rename(
    columns={
        "timestamp": "target_timestamp",
        "glucose_mg_dl": TARGET,
    }
)

sensor = sensor.merge(
    future,
    on=["patient_id", "target_timestamp"],
    how="left",
    validate="many_to_one",
)

sensor = sensor.sort_values(
    ["patient_id", "timestamp"]
).reset_index(drop=True)

sensor = sensor.dropna(
    subset=[TARGET, "target_timestamp"]
).copy()

# Attach timestamps by chronological row number within each patient.
# This assumes the processed CSV preserved the original row order
# within each patient when it was created.

df["_row_within_patient"] = (
    df.groupby("patient_id").cumcount()
)

timestamp_map = sensor[
    ["patient_id", "timestamp", "target_timestamp"]
].copy()

timestamp_map["_row_within_patient"] = (
    timestamp_map.groupby("patient_id").cumcount()
)

df = df.merge(
    timestamp_map,
    on=["patient_id", "_row_within_patient"],
    how="left",
    validate="one_to_one",
)

if df[["timestamp", "target_timestamp"]].isna().any().any():
    raise ValueError(
        "Timestamp mapping failed. Check that the processed data "
        "preserves the original chronological row order within each patient."
    )

if len(df) != len(sensor):
    raise ValueError(
        f"Row count mismatch: processed={len(df)}, sensor={len(sensor)}"
    )

df = df.drop(columns=["_row_within_patient"])

print("Processed rows:", len(df))
print("Timestamp mapping completed.")


# ---------------------------------------------------------
# 3. RECREATE PATIENT-LEVEL TRAIN/VALIDATION SPLIT
# ---------------------------------------------------------

print("Creating patient-level validation split...")

test_ids = set(artifact["test_patient_ids"])

# Keep the original test patients separate.
train_pool = df[
    ~df["patient_id"].isin(test_ids)
].copy()

patients = train_pool[
    ["patient_id", "diabetes_status"]
].drop_duplicates("patient_id")

train_ids, val_ids = train_test_split(
    patients["patient_id"],
    test_size=0.20,
    random_state=42,
    stratify=patients["diabetes_status"],
)

train_df = train_pool[
    train_pool["patient_id"].isin(train_ids)
].copy()

val_df = train_pool[
    train_pool["patient_id"].isin(val_ids)
].copy()

X_train = train_df[features]
y_train = train_df[TARGET]

X_val = val_df[features]
y_val = val_df[TARGET]


# ---------------------------------------------------------
# 4. TRAIN THE V3-STYLE WEIGHTED MODEL
# ---------------------------------------------------------

print("Training weighted Random Forest model...")

numeric_cols = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_cols = [
    column
    for column in features
    if column not in numeric_cols
]

preprocessor = ColumnTransformer([
    (
        "numeric",
        SimpleImputer(strategy="median"),
        numeric_cols,
    ),
    (
        "categorical",
        Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]),
        categorical_cols,
    ),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=100,
            max_depth=18,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    ),
])

# Weight low and high glucose examples more heavily.
weights = np.ones(len(y_train))

weights[y_train.to_numpy() < LOW_LIMIT] = 4
weights[y_train.to_numpy() >= 180] = 2

model.fit(
    X_train,
    y_train,
    regressor__sample_weight=weights,
)

val_df = val_df.copy()
val_df["actual"] = y_val.to_numpy()
val_df["predicted"] = model.predict(X_val)
print("\nPREDICTION CHECK")
print("Minimum prediction:", round(val_df["predicted"].min(), 2))
print("Maximum prediction:", round(val_df["predicted"].max(), 2))
print("Predictions below 70:", int((val_df["predicted"] < 70).sum()))
print("Predictions below 85:", int((val_df["predicted"] < 85).sum()))
print("Predictions below 90:", int((val_df["predicted"] < 90).sum()))
print("Actual low targets:", int((val_df[TARGET] < 70).sum()))

val_df["timestamp"] = pd.to_datetime(
    val_df["timestamp"],
    utc=True,
)

val_df["target_timestamp"] = pd.to_datetime(
    val_df["target_timestamp"],
    utc=True,
)

val_df = val_df.sort_values(
    ["patient_id", "timestamp"]
).reset_index(drop=True)


# ---------------------------------------------------------
# 5. IDENTIFY ACTUAL LOW-GLUCOSE EPISODES
# ---------------------------------------------------------

print("Identifying actual low-glucose episodes...")

# The actual target value belongs to target_timestamp.
val_df["actual_low"] = val_df["actual"] < LOW_LIMIT

previous_low = (
    val_df.groupby("patient_id")["actual_low"]
    .shift(fill_value=False)
)

target_gap_hours = (
    val_df.groupby("patient_id")["target_timestamp"]
    .diff()
    .dt.total_seconds()
    / 3600
)

val_df["new_actual_episode"] = (
    val_df["actual_low"]
    & (
        (~previous_low)
        | (target_gap_hours > 1.01)
    )
)

val_df["actual_episode_id"] = (
    val_df.groupby("patient_id")["new_actual_episode"].cumsum()
)

actual_episodes = (
    val_df[val_df["actual_low"]]
    .groupby(
        ["patient_id", "actual_episode_id"],
        as_index=False,
    )
    .agg(
        episode_start=("target_timestamp", "min"),
        episode_end=("target_timestamp", "max"),
        readings_in_episode=("actual", "size"),
    )
)

print("\nVALIDATION EVENT-LEVEL EVALUATION")
print("Validation patients:", val_df["patient_id"].nunique())
print("Validation readings:", len(val_df))
print("Actual low readings:", int(val_df["actual_low"].sum()))
print("Actual low episodes:", len(actual_episodes))


# ---------------------------------------------------------
# 6. EVALUATE EACH THRESHOLD
# ---------------------------------------------------------

for threshold in THRESHOLDS:

    print("\n" + "=" * 55)
    print(f"THRESHOLD: {threshold} mg/dL")
    print("=" * 55)

    # A prediction below the threshold creates an alert-positive reading.
    val_df["alert"] = val_df["predicted"] < threshold

    previous_alert = (
        val_df.groupby("patient_id")["alert"]
        .shift(fill_value=False)
    )

    alert_gap_hours = (
        val_df.groupby("patient_id")["target_timestamp"]
        .diff()
        .dt.total_seconds()
        / 3600
    )

    val_df["new_alert_episode"] = (
        val_df["alert"]
        & (
            (~previous_alert)
            | (alert_gap_hours > 1.01)
        )
    )

    val_df["alert_episode_id"] = (
        val_df.groupby("patient_id")["new_alert_episode"].cumsum()
    )

    # Summarize each continuous alert episode.
    alert_episodes = (
        val_df[val_df["alert"]]
        .groupby(
            ["patient_id", "alert_episode_id"],
            as_index=False,
        )
        .agg(
            alert_start=("target_timestamp", "min"),
            alert_end=("target_timestamp", "max"),
            alert_readings=("predicted", "size"),
        )
    )

    total_actual = len(actual_episodes)
    total_alerts = len(alert_episodes)

    matched_actual = set()
    matched_alert = set()

    # Global one-to-one assignment.
    # Each actual episode and each alert episode can be matched once.
    if total_actual > 0 and total_alerts > 0:

        invalid_cost = 1e12

        cost = np.full(
            (total_actual, total_alerts),
            invalid_cost,
            dtype=float,
        )

        tolerance = pd.Timedelta(
            hours=TIME_TOLERANCE_HOURS
        )

        for i, (_, event) in enumerate(
            actual_episodes.iterrows()
        ):
            for j, (_, alert) in enumerate(
                alert_episodes.iterrows()
            ):

                # Episodes from different patients cannot match.
                if event["patient_id"] != alert["patient_id"]:
                    continue

                allowed_start = event["episode_start"] - tolerance
                allowed_end = event["episode_end"] + tolerance

                overlaps = (
                    alert["alert_start"] <= allowed_end
                    and alert["alert_end"] >= allowed_start
                )

                if not overlaps:
                    continue

                # Minimize distance between episode midpoints.
                alert_mid = (
                    alert["alert_start"]
                    + (
                        alert["alert_end"] - alert["alert_start"]
                    ) / 2
                )

                event_mid = (
                    event["episode_start"]
                    + (
                        event["episode_end"] - event["episode_start"]
                    ) / 2
                )

                distance_seconds = abs(
                    (alert_mid - event_mid).total_seconds()
                )

                cost[i, j] = distance_seconds + j * 1e-6

        row_indices, col_indices = linear_sum_assignment(cost)

        for i, j in zip(row_indices, col_indices):

            # Ignore assignments that were not valid time matches.
            if cost[i, j] >= invalid_cost:
                continue

            event = actual_episodes.iloc[i]

            actual_key = (
                event["patient_id"],
                event["actual_episode_id"],
            )

            matched_actual.add(actual_key)
            matched_alert.add(j)

    detected = len(matched_actual)
    matched_alert_count = len(matched_alert)

    missed = total_actual - detected
    unmatched_alerts = total_alerts - matched_alert_count

    sensitivity = (
        detected / total_actual * 100
        if total_actual > 0
        else 0
    )

    precision = (
        matched_alert_count / total_alerts * 100
        if total_alerts > 0
        else 0
    )

    print("Actual low-glucose episodes:", total_actual)
    print("Alert episodes generated:", total_alerts)
    print("Matched alert episodes:", matched_alert_count)
    print("Unmatched alert episodes:", unmatched_alerts)
    print("Actual episodes detected:", detected)
    print("Actual episodes missed:", missed)
    print(f"Event sensitivity: {sensitivity:.2f}%")
    print(f"Event precision: {precision:.2f}%")

print("\nEvaluation complete.")
