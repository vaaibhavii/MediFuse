"""
Run this FIRST.

It loads every structured CSV declared in config.py and prints:
  - shape
  - column names + dtypes
  - first 3 rows
  - value_counts() of the configured target_column (if it parses)

Use this output to correct config.STRUCTURED_SOURCES[...]['target_column']
and ['categorical_columns'] for the exact dataset versions you downloaded —
public dataset mirrors frequently rename columns between uploads.
"""

import os
import pandas as pd
from config import STRUCTURED_SOURCES, URINALYSIS_TABULAR_FILE, URINE_DETECTION_DATASETS


def inspect_csv(name, path, target_column=None):
    print("=" * 70)
    print(f"SOURCE: {name}")
    print(f"PATH:   {path}")
    if not os.path.exists(path):
        print("  !! FILE NOT FOUND. Check config.py RAW_DATA_DIR / filename.")
        return
    df = pd.read_csv(path)
    print(f"  shape: {df.shape}")
    print("  columns + dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"    - {col}: {dtype}")
    print("  head(3):")
    print(df.head(3).to_string())
    if target_column and target_column in df.columns:
        print(f"  value_counts('{target_column}'):")
        print(df[target_column].value_counts(dropna=False).to_string())
    elif target_column:
        print(f"  !! configured target_column '{target_column}' NOT in columns — update config.py")
    print()


def inspect_folder(name, path):
    print("=" * 70)
    print(f"IMAGE DATASET: {name}")
    print(f"PATH: {path}")
    if not os.path.exists(path):
        print("  !! FOLDER NOT FOUND.")
        return
    for root, dirs, files in os.walk(path):
        depth = root.replace(path, "").count(os.sep)
        if depth > 2:
            continue
        print(f"  {root}/  ->  {len(files)} files" + (f" (e.g. {files[:2]})" if files else ""))
    print()


if __name__ == "__main__":
    for name, cfg in STRUCTURED_SOURCES.items():
        inspect_csv(name, cfg["file"], cfg.get("target_column"))

    inspect_csv("urinalysis (tabular)", URINALYSIS_TABULAR_FILE)

    for name, path in URINE_DETECTION_DATASETS.items():
        inspect_folder(name, path)

    print("Done. Fix any '!!' warnings in config.py before running 02_train_all_sources.py")
