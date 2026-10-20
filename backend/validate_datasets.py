
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
files = sorted(DATA_DIR.glob("*.csv"))

print("DRISHTI HEALTH - BASIC DATA VALIDATION")
print("=" * 60)

if not files:
    raise SystemExit("No CSV files found in data/raw.")

issues_found = 0

for path in files:
    print(f"\nDataset: {path.name}")
    issues = []

    try:
        df = pd.read_csv(path)

        # Check whether the file contains any records
        if df.empty:
            issues.append("File has no data rows.")

        # Check for blank or repeated column names
        if any(str(c).strip() == "" for c in df.columns):
            issues.append("One or more column names are blank.")

        if df.columns.duplicated().any():
            issues.append("Column names are duplicated.")

        # Check for completely duplicated records
        duplicate_count = int(df.duplicated().sum())
        if duplicate_count:
            issues.append(
                f"{duplicate_count} exact duplicate rows found."
            )

        # Check for fully empty rows
        empty_rows = int(df.isna().all(axis=1).sum())
        if empty_rows:
            issues.append(f"{empty_rows} completely empty rows found.")

        # Check for completely empty columns
        empty_columns = [
            str(c) for c in df.columns if df[c].isna().all()
        ]
        if empty_columns:
            issues.append(
                f"Completely empty columns: {empty_columns}"
            )

        # Show columns with missing values
        missing = df.isna().sum()
        missing = missing[missing > 0]

        if not missing.empty:
            issues.append(
                "Columns with missing values: "
                + str(missing.to_dict())
            )

        if issues:
            issues_found += len(issues)
            for issue in issues:
                print(f"  CHECK: {issue}")
        else:
            print("  PASS: No basic structural issues detected.")

        print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    except Exception as error:
        issues_found += 1
        print(f"  ERROR: Could not validate file: {error}")

print("\n" + "=" * 60)
print(f"Files checked: {len(files)}")
print(f"Checks needing review: {issues_found}")
print("Basic validation finished.")
