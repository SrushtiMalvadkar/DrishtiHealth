
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. FILE PATHS AND SETTINGS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "glucose_training_dataset.csv"
)

PATIENTS_PATH = PROJECT_DIR / "data" / "raw" / "patients.csv"

MODEL_DIR = PROJECT_DIR / "ml" / "models"
MODEL_PATH = MODEL_DIR / "glucose_2h_random_forest.joblib"

TARGET = "glucose_2h_ahead_mg_dl"
RANDOM_STATE = 42


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

print("=" * 65)
print("DRISHTI HEALTH - GLUCOSE MODEL TRAINING")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
patient_info = pd.read_csv(PATIENTS_PATH)

required_patient_columns = {"patient_id", "diabetes_status"}
required_dataset_columns = {"patient_id", TARGET}

if not required_patient_columns.issubset(patient_info.columns):
    raise ValueError(
        "patients.csv must contain patient_id and diabetes_status."
    )

if not required_dataset_columns.issubset(df.columns):
    raise ValueError(
        "Training dataset must contain patient_id and target."
    )

if df[TARGET].isna().any():
    raise ValueError("The target contains missing values.")

if df["patient_id"].isna().any():
    raise ValueError("The training dataset contains missing patient IDs.")

if patient_info["patient_id"].duplicated().any():
    raise ValueError("patients.csv contains duplicate patient IDs.")

if patient_info["diabetes_status"].isna().any():
    raise ValueError("Some patients have missing diabetes status.")

# Every training patient must exist in patients.csv.
known_patients = set(patient_info["patient_id"])
dataset_patients = set(df["patient_id"])

unknown_patients = dataset_patients - known_patients

if unknown_patients:
    raise ValueError(
        f"Some training patients are absent from patients.csv: "
        f"{sorted(unknown_patients)[:10]}"
    )

# This script expects the two groups in our synthetic dataset.
expected_statuses = {"type_2_diabetes", "no_diabetes"}
actual_statuses = set(patient_info["diabetes_status"].unique())

if actual_statuses != expected_statuses:
    raise ValueError(
        f"Unexpected diabetes statuses: {actual_statuses}. "
        f"Expected: {expected_statuses}"
    )

print(f"\nDataset shape: {df.shape}")
print(f"Unique patients in training dataset: {df['patient_id'].nunique()}")
print(f"Target: {TARGET}")
print(f"Missing target values: {df[TARGET].isna().sum()}")


# ============================================================
# 3. SEPARATE FEATURES, TARGET AND PATIENT GROUPS
# ============================================================

# patient_id is used only to split and evaluate patient groups.
# It is NOT given to the model as an input feature.
X = df.drop(columns=["patient_id", TARGET])
y = df[TARGET]
groups = df["patient_id"]

categorical_columns = X.select_dtypes(
    include=["object","str", "category", "bool"]
).columns.tolist()

numerical_columns = [
    column
    for column in X.columns
    if column not in categorical_columns
]

print(f"\nNumber of input features: {X.shape[1]}")
print(f"Numerical features: {len(numerical_columns)}")
print(f"Categorical features: {len(categorical_columns)}")


# ============================================================
# 4. STRATIFIED PATIENT-LEVEL TRAIN/TEST SPLIT
# ============================================================

# Create one row per patient for splitting.
# Stratification preserves the diabetes-status proportions.
patient_groups = patient_info[
    ["patient_id", "diabetes_status"]
].drop_duplicates()

train_patient_ids, test_patient_ids = train_test_split(
    patient_groups["patient_id"],
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=patient_groups["diabetes_status"]
)

train_patient_ids = set(train_patient_ids)
test_patient_ids = set(test_patient_ids)

# Assign every row belonging to a patient to the same set.
train_mask = groups.isin(train_patient_ids)
test_mask = groups.isin(test_patient_ids)

X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_test = y.loc[test_mask].copy()

train_groups = groups.loc[train_mask]
test_groups = groups.loc[test_mask]

# Map diabetes status to each test row for group-specific evaluation.
status_map = patient_groups.set_index(
    "patient_id"
)["diabetes_status"]

test_status = test_groups.map(status_map)

shared_patients = train_patient_ids.intersection(test_patient_ids)

print("\n" + "=" * 65)
print("TRAIN/TEST SPLIT")
print("=" * 65)

print(f"Training patients: {len(train_patient_ids)}")
print(f"Testing patients: {len(test_patient_ids)}")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"Patients shared between sets: {len(shared_patients)}")

print("\nAll patients in the source file:")
print(patient_groups["diabetes_status"].value_counts().to_string())

print("\nTraining patient distribution:")
print(
    patient_groups[
        patient_groups["patient_id"].isin(train_patient_ids)
    ]["diabetes_status"].value_counts().to_string()
)

print("\nTesting patient distribution:")
print(
    patient_groups[
        patient_groups["patient_id"].isin(test_patient_ids)
    ]["diabetes_status"].value_counts().to_string()
)

if shared_patients:
    raise RuntimeError("Data leakage detected: patients occur in both sets.")

if len(train_patient_ids) + len(test_patient_ids) != len(patient_groups):
    raise RuntimeError("Some patients were not assigned to a split.")


# ============================================================
# 5. PREPROCESS INPUT FEATURES
# ============================================================

# Learn imputation values and category encodings from training data
# only. The fitted pipeline applies them to test data afterward.
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])


# ============================================================
# 6. DEFINE THE RANDOM FOREST MODEL
# ============================================================

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=100,
            max_depth=18,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )
])


# ============================================================
# 7. TRAIN A SIMPLE BASELINE
# ============================================================

# The baseline always predicts the median glucose target
# from the training set. It gives us a simple comparison.
baseline = DummyRegressor(strategy="median")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)


# ============================================================
# 8. TRAIN RANDOM FOREST
# ============================================================

print("\n" + "=" * 65)
print("TRAINING RANDOM FOREST")
print("=" * 65)

print("Training model. This may take a few minutes...")

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Training complete.")


# ============================================================
# 9. EVALUATION FUNCTIONS
# ============================================================

def calculate_metrics(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted))
    }


def print_metrics(name, actual, predicted):
    metrics = calculate_metrics(actual, predicted)

    print(f"\n{name}")
    print(f"MAE:  {metrics['mae']:.2f} mg/dL")
    print(f"RMSE: {metrics['rmse']:.2f} mg/dL")
    print(f"R²:   {metrics['r2']:.4f}")

    return metrics


# ============================================================
# 10. OVERALL TEST PERFORMANCE
# ============================================================

print("\n" + "=" * 65)
print("OVERALL TEST PERFORMANCE")
print("=" * 65)

baseline_metrics = print_metrics(
    "Median baseline",
    y_test,
    baseline_predictions
)

model_metrics = print_metrics(
    "Random Forest",
    y_test,
    predictions
)


# ============================================================
# 11. PERFORMANCE BY DIABETES STATUS
# ============================================================

print("\n" + "=" * 65)
print("TEST PERFORMANCE BY DIABETES STATUS")
print("=" * 65)

group_metrics = {}

for status in sorted(test_status.unique()):
    group_mask = test_status == status

    actual_group = y_test.loc[group_mask]
    predicted_group = predictions[
        group_mask.to_numpy()
    ]

    print(f"\nDiabetes status: {status}")
    print(f"Test rows: {len(actual_group)}")
    print(f"Test patients: {test_groups.loc[group_mask].nunique()}")

    group_metrics[status] = print_metrics(
        status,
        actual_group,
        predicted_group
    )


# ============================================================
# 12. SAVE THE MODEL AND EVALUATION RESULTS
# ============================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)

saved_artifact = {
    "model": model,
    "feature_columns": X.columns.tolist(),
    "target": TARGET,
    "metrics": {
        "baseline": baseline_metrics,
        "random_forest": model_metrics,
        "by_diabetes_status": group_metrics
    },
    "random_state": RANDOM_STATE,
    "train_patient_ids": sorted(train_patient_ids),
    "test_patient_ids": sorted(test_patient_ids),
    "test_patient_status": {
        str(row.patient_id): row.diabetes_status
        for row in patient_groups[
            patient_groups["patient_id"].isin(test_patient_ids)
        ].itertuples(index=False)
    }
}

joblib.dump(saved_artifact, MODEL_PATH)

print("\n" + "=" * 65)
print("MODEL SAVED")
print("=" * 65)

print(f"Location: {MODEL_PATH}")
print("\nTRAINING AND EVALUATION COMPLETE")
