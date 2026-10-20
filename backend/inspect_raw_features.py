
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

files = [
    "sensor_timeseries.csv",
    "meal_logs.csv",
    "patients.csv"
]

for filename in files:
    path = RAW_DIR / filename

    print("\n" + "=" * 60)
    print(filename.upper())
    print("=" * 60)

    df = pd.read_csv(path)

    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nMissing values:")
    print(df.isna().sum().to_string())

    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
        print("\nTimestamp range:")
        print("Start:", timestamps.min())
        print("End:", timestamps.max())
        print("Unparseable timestamps:", timestamps.isna().sum())
