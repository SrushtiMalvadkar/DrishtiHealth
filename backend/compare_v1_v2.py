
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "ml" / "models" / "glucose_2h_random_forest.joblib"

TARGET = "glucose_2h_ahead_mg_dl"

# Load the original trained model and its saved patient split.
artifact = joblib.load(MODEL_PATH)

print("Saved model artifact keys:")
print(list(artifact.keys()))

train_ids = set(artifact["train_patient_ids"])
test_ids = set(artifact["test_patient_ids"])

# Load both datasets.
v1 = pd.read_csv(DATA_DIR / "glucose_training_dataset.csv")
v2 = pd.read_csv(DATA_DIR / "glucose_training_dataset_v2.csv")

def split_features(dataset, patient_ids):
    subset = dataset[dataset["patient_id"].isin(patient_ids)].copy()

    X = subset.drop(columns=["patient_id", TARGET])
    y = subset[TARGET]

    return X, y

# Original model evaluation on the original dataset.
X1_test, y1_test = split_features(v1, test_ids)

old_model = artifact["model"]
old_columns = artifact["feature_columns"]

X1_test = X1_test[old_columns]
old_predictions = old_model.predict(X1_test)

# Prepare v2 training and test sets using the same patient IDs.
X2_train, y2_train = split_features(v2, train_ids)
X2_test, y2_test = split_features(v2, test_ids)

# Identify numeric and categorical features.
numeric_columns = X2_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X2_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_columns
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_columns
        )
    ]
)

new_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

print("\nTraining the v2 model...")
new_model.fit(X2_train, y2_train)

new_predictions = new_model.predict(X2_test)

def evaluate(name, actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    print(f"\n{name}")
    print(f"Rows evaluated: {len(actual)}")
    print(f"MAE:  {mean_absolute_error(actual, predicted):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(actual, predicted)):.3f}")
    print(f"R²:   {r2_score(actual, predicted):.4f}")

    low = actual < 70
    normal = (actual >= 70) & (actual < 180)
    high = actual >= 180

    for label, mask in [
        ("Low glucose (<70)", low),
        ("Normal range (70 to <180)", normal),
        ("High glucose (>=180)", high)
    ]:
        if mask.any():
            print(
                f"{label}: n={mask.sum()}, "
                f"MAE={mean_absolute_error(actual[mask], predicted[mask]):.3f}, "
                f"Mean error={(predicted[mask] - actual[mask]).mean():.3f}"
            )

evaluate("V1: Original model", y1_test, old_predictions)
evaluate("V2: Model with engineered features", y2_test, new_predictions)

# Save the new model separately so the original remains unchanged.
output_model = BASE_DIR / "ml" / "models" / "glucose_2h_random_forest_v2.joblib"

joblib.dump({
    "model": new_model,
    "feature_columns": X2_train.columns.tolist(),
    "target": TARGET,
    "train_patient_ids": sorted(train_ids),
    "test_patient_ids": sorted(test_ids),
    "random_seed": 42
}, output_model)

print("\nSaved new model:", output_model)
