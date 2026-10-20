
from pathlib import Path
import pandas as pd

# Find the raw data folder relative to this script
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

csv_files = sorted(DATA_DIR.glob("*.csv"))

print("=" * 70)
print("DRISHTI HEALTH - DATASET INSPECTION")
print("=" * 70)
print(f"Folder: {DATA_DIR}")
print(f"CSV files found: {len(csv_files)}")

if not csv_files:
    print("ERROR: No CSV files found. Check the data/raw folder.")
    raise SystemExit(1)

for file_path in csv_files:
    print("\n" + "=" * 70)
    print(f"FILE: {file_path.name}")
    print("=" * 70)

    try:
        df = pd.read_csv(file_path)

        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        print(f"File size: {file_path.stat().st_size:,} bytes")

        print("\nActual column names:")
        for column in df.columns:
            print(f"  - {column!r}")

        print("\nColumn data types:")
        print(df.dtypes.to_string())

        print("\nFirst 5 rows:")
        print(df.head().to_string(index=False))

        print("\nMissing values per column:")
        missing = df.isnull().sum()
        print(missing[missing > 0].to_string() if missing.any()
              else "No missing values detected.")

        print("\nExact duplicate rows:")
        print(int(df.duplicated().sum()))

        print("\nDuplicate column names:")
        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        print(duplicate_columns if duplicate_columns else "None")

        print("\nPotential ID and time columns (name-based hints):")
        candidates = [
            col for col in df.columns
            if any(word in str(col).lower()
                   for word in ["id", "time", "date", "timestamp"])
        ]
        print(candidates if candidates else "None detected by name.")

        # Show distinct counts for likely ID columns
        for col in candidates:
            if "id" in str(col).lower():
                print(
                    f"  {col!r}: {df[col].nunique(dropna=True)} "
                    f"unique non-missing values"
                )

    except Exception as error:
        print(f"ERROR reading this file: {error}")

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)
