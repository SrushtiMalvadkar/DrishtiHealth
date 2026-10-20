
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS


# --------------------------------------------------
# 1. APPLICATION SETUP
# --------------------------------------------------

app = Flask(__name__)
CORS(app)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "glucose_2h_random_forest_v2.joblib"
)

PATIENTS_PATH = PROJECT_ROOT / "data" / "raw" / "patients.csv"
SENSORS_PATH = PROJECT_ROOT / "data" / "raw" / "sensor_timeseries.csv"
PROCESSED_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "glucose_training_dataset_v2.csv"
)


# --------------------------------------------------
# 2. LOAD TRAINED MODEL AND DATA
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not PATIENTS_PATH.exists():
    raise FileNotFoundError(f"Patients CSV not found: {PATIENTS_PATH}")

if not SENSORS_PATH.exists():
    raise FileNotFoundError(f"Sensor CSV not found: {SENSORS_PATH}")

if not PROCESSED_PATH.exists():
    raise FileNotFoundError(
        f"Processed dataset not found: {PROCESSED_PATH}"
    )

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
FEATURE_COLUMNS = artifact["feature_columns"]
TARGET_COLUMN = artifact.get(
    "target",
    "glucose_2h_ahead_mg_dl"
)

patients_df = pd.read_csv(PATIENTS_PATH)
sensors_df = pd.read_csv(SENSORS_PATH)
processed_df = pd.read_csv(PROCESSED_PATH)

# Normalize IDs to strings so CSV IDs match API request IDs.
for dataframe in (patients_df, sensors_df, processed_df):
    if "patient_id" in dataframe.columns:
        dataframe["patient_id"] = (
            dataframe["patient_id"].astype(str).str.strip()
        )

# Parse sensor timestamps and order newest first.
sensors_df["timestamp"] = pd.to_datetime(
    sensors_df["timestamp"],
    errors="coerce"
)

sensors_df = sensors_df.sort_values(
    "timestamp",
    ascending=False,
    na_position="last"
)

# The processed dataset contains historical rows.
# Select the latest row with complete model inputs for each patient.
available_feature_columns = [
    column for column in FEATURE_COLUMNS
    if column in processed_df.columns
]

missing_dataset_columns = [
    column for column in FEATURE_COLUMNS
    if column not in processed_df.columns
]

if missing_dataset_columns:
    raise ValueError(
        "Processed dataset is missing model columns: "
        + ", ".join(missing_dataset_columns)
    )

processed_df["_valid_features"] = processed_df[
    FEATURE_COLUMNS
].notna().all(axis=1)

complete_rows = processed_df[
    processed_df["_valid_features"]
].copy()

latest_features_by_patient = (
    complete_rows.drop_duplicates(
        subset=["patient_id"],
        keep="last"
    )
    .set_index("patient_id")
)

print("Drishti Health backend started.")
print(f"Model loaded: {MODEL_PATH.name}")
print(f"Synthetic patients: {len(patients_df)}")
print(f"Sensor records: {len(sensors_df)}")
print(f"Model features: {len(FEATURE_COLUMNS)}")


# --------------------------------------------------
# 3. HELPER FUNCTIONS
# --------------------------------------------------

def json_safe(value):
    """Convert pandas and NumPy values into JSON-safe values."""

    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, (pd.Timestamp,)):
        if pd.isna(value):
            return None
        return value.isoformat()

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float) and not np.isfinite(value):
        return None

    if pd.isna(value):
        return None

    return value


def dataframe_row_to_dict(row):
    """Convert one pandas row into a JSON-safe dictionary."""

    return {
        str(key): json_safe(value)
        for key, value in row.to_dict().items()
    }


def get_patient_row(patient_id):
    """Return a patient's demographic record, if available."""

    matches = patients_df[
        patients_df["patient_id"] == patient_id
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


def get_latest_sensor(patient_id):
    """Return the newest sensor record for a patient."""

    matches = sensors_df[
        sensors_df["patient_id"] == patient_id
    ]

    if matches.empty:
        return None

    row = matches.iloc[0]
    return dataframe_row_to_dict(row)


def get_latest_processed_features(patient_id):
    """
    Return the latest processed row with all model inputs present.

    The row is historical synthetic data, not a live sensor reading.
    """

    if patient_id not in latest_features_by_patient.index:
        return None

    row = latest_features_by_patient.loc[patient_id]

    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]

    return {
        name: json_safe(row[name])
        for name in FEATURE_COLUMNS
    }


def get_model_input(payload):
    """
    Validate and prepare inputs in the trained model's feature order.

    Missing numeric values become NaN for the model's imputer.
    Categorical values remain strings.
    """

    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    missing_features = [
        name for name in FEATURE_COLUMNS
        if name not in payload
    ]

    if missing_features:
        raise ValueError(
            "Missing required model features: "
            + ", ".join(missing_features)
        )

    input_data = {}

    for name in FEATURE_COLUMNS:
        value = payload[name]

        if value is None or value == "":
            input_data[name] = np.nan
            continue

        if isinstance(value, (dict, list)):
            raise ValueError(
                f"Feature '{name}' must be a number or string."
            )

        if isinstance(value, str):
            value = value.strip()

        if name in model_categorical_features():
            if value is None or value == "":
                input_data[name] = np.nan
            else:
                input_data[name] = str(value)
        else:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Feature '{name}' must be numeric."
                )

            if not np.isfinite(numeric_value):
                input_data[name] = np.nan
            else:
                input_data[name] = numeric_value

    return pd.DataFrame(
        [input_data],
        columns=FEATURE_COLUMNS
    )


def model_categorical_features():
    """
    Return categorical features based on the model's trained pipeline.

    The known categorical fields are used as a safe fallback.
    """

    return {
        "activity_intensity",
        "sleep_state",
        "sensor_quality",
        "sex",
        "diabetes_status",
        "hypertension_status",
        "dyslipidemia_status",
        "smoking_status",
        "activity_level",
    }


def classify_demo_glucose(value):
    """Simple display category for the predicted demo value."""

    if value < 70:
        return "below_demo_range"

    if value >= 180:
        return "above_demo_range"

    return "within_demo_range"


# --------------------------------------------------
# 4. HOME AND HEALTH CHECK
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "application": "Drishti Health",
        "message": "Healthcare Digital Twin API is running.",
        "mode": "synthetic_data_demo",
        "endpoints": [
            "/health",
            "/patients",
            "/patients/<patient_id>",
            "/patients/<patient_id>/features",
            "/predict",
        ],
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "patients_loaded": len(patients_df),
        "sensor_records_loaded": len(sensors_df),
        "model_feature_count": len(FEATURE_COLUMNS),
        "target": TARGET_COLUMN,
        "mode": "synthetic_data_demo",
    })


# --------------------------------------------------
# 5. PATIENT DIRECTORY
# --------------------------------------------------

@app.route("/patients", methods=["GET"])
def list_patients():
    """Return a basic directory of synthetic patients."""

    columns = [
        column for column in [
            "patient_id",
            "age_years",
            "sex",
            "diabetes_status",
            "baseline_risk_category",
        ]
        if column in patients_df.columns
    ]

    records = patients_df[columns].to_dict(orient="records")

    return jsonify({
        "count": len(records),
        "patients": json_safe(records),
        "synthetic_data": True,
    })


# --------------------------------------------------
# 6. PATIENT PROFILE AND LATEST SENSOR RECORD
# --------------------------------------------------

@app.route("/patients/<patient_id>", methods=["GET"])
def patient_details(patient_id):
    """Return a patient's profile and latest available sensor record."""

    patient_id = patient_id.strip()
    row = get_patient_row(patient_id)

    if row is None:
        return jsonify({
            "error": "Patient not found.",
            "patient_id": patient_id,
        }), 404

    patient = dataframe_row_to_dict(row)
    latest_sensor = get_latest_sensor(patient_id)

    return jsonify({
        "patient": patient,
        "latest_sensor": latest_sensor,
        "has_model_features": (
            patient_id in latest_features_by_patient.index
        ),
        "synthetic_data": True,
    })


@app.route("/patients/<patient_id>/history", methods=["GET"])
def patient_history(patient_id):
    """Return up to 24 historical sensor readings in time order."""

    patient_id = patient_id.strip()

    if get_patient_row(patient_id) is None:
        return jsonify({
            "error": "Patient not found.",
            "patient_id": patient_id,
        }), 404

    records = sensors_df[
        sensors_df["patient_id"] == patient_id
    ].copy()

    records = records.dropna(subset=["timestamp"])
    records = records.sort_values("timestamp").tail(24)

    history = []

    for _, row in records.iterrows():
        history.append({
            "timestamp": json_safe(row["timestamp"]),
            "glucose_mg_dl": json_safe(
                row.get("glucose_mg_dl")
            ),
        })

    return jsonify({
        "patient_id": patient_id,
        "count": len(history),
        "history": history,
        "source": "historical_synthetic_sensor_data",
        "synthetic_data": True,
    })



# --------------------------------------------------
# 7. PATIENT MODEL FEATURES
# --------------------------------------------------

@app.route("/patients/<patient_id>/features", methods=["GET"])
def patient_features(patient_id):
    """
    Return model inputs from the latest complete processed historical row.
    """

    patient_id = patient_id.strip()

    if get_patient_row(patient_id) is None:
        return jsonify({
            "error": "Patient not found.",
            "patient_id": patient_id,
        }), 404

    features = get_latest_processed_features(patient_id)

    if features is None:
        return jsonify({
            "error": (
                "No processed row with complete model features "
                "was found for this patient."
            ),
            "patient_id": patient_id,
        }), 404

    return jsonify({
        "patient_id": patient_id,
        "features": features,
        "feature_count": len(features),
        "target": TARGET_COLUMN,
        "source": "historical_synthetic_dataset",
        "synthetic_data": True,
    })


# --------------------------------------------------
# 8. TWO-HOUR GLUCOSE PREDICTION
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    """Predict glucose two hours ahead using the trained model."""

    try:
        payload = request.get_json(silent=True)

        if payload is None:
            return jsonify({
                "error": "Send a valid JSON request body."
            }), 400

        input_df = get_model_input(payload)

        predicted_value = float(model.predict(input_df)[0])

        if not np.isfinite(predicted_value):
            return jsonify({
                "error": "The model returned an invalid prediction."
            }), 500

        return jsonify({
            "predicted_glucose_2h_mg_dl": round(
                predicted_value, 2
            ),
            "target": TARGET_COLUMN,
            "horizon_hours": 2,
            "demo_category": classify_demo_glucose(
                predicted_value
            ),
            "notice": (
                "Synthetic-data research demonstration only. "
                "Not for diagnosis, treatment, or emergency decisions."
            ),
            "synthetic_data": True,
        })

    except ValueError as exc:
        return jsonify({
            "error": str(exc)
        }), 400

    except Exception as exc:
        app.logger.exception("Prediction failed")

        return jsonify({
            "error": "Prediction failed. Check the backend terminal.",
            "details": str(exc),
        }), 500


# --------------------------------------------------
# 9. START SERVER
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
