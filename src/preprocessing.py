"""
Generic, reusable preprocessing for every structured source (kidney, liver,
diabetes, heart). Each source gets its own fitted preprocessing pipeline,
saved alongside its model — sources are NEVER merged into one table.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def _clean_messy_values(X: pd.DataFrame, categorical_columns) -> pd.DataFrame:
    """Real-world CSVs often store missing values as '?', '\\t?', or extra
    whitespace/tabs instead of a truly empty cell. Normalize these to NaN,
    and coerce columns NOT marked categorical into actual numbers so a
    numeric column doesn't stay stuck as text just because a few rows are
    messy placeholders."""
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = (
                X[col].astype(str)
                .str.strip()
                .str.replace("\t", "", regex=False)
                .replace({"?": np.nan, "nan": np.nan, "": np.nan, "None": np.nan})
            )
    for col in X.columns:
        if col not in categorical_columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X

def load_and_split_xy(cfg: dict):
    """Load one source's CSV and return (X: DataFrame, y: Series[0/1])."""
    df = pd.read_csv(cfg["file"])

    # drop id-like columns that must never be used as features
    drop_cols = [c for c in cfg.get("id_columns", []) if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    target_col = cfg["target_column"]
    if target_col not in df.columns:
        raise ValueError(
            f"target_column '{target_col}' not found in {cfg['file']}. "
            f"Available columns: {list(df.columns)}. Run 01_inspect_data.py and fix config.py."
        )

    y_raw = df[target_col]
    X = df.drop(columns=[target_col])
    X = _clean_messy_values(X, cfg.get("categorical_columns", []))

    negative_label = cfg.get("negative_label")
    if negative_label is not None:
        y = (y_raw != negative_label).astype(int)
        return X, y

    positive_label = cfg.get("positive_label")
    if positive_label is not None:
        y = (y_raw == positive_label).astype(int)
    else:
        # assume already binary-ish; coerce common string labels
        y = y_raw.replace({"ckd": 1, "notckd": 0, "yes": 1, "no": 0,
                            "Yes": 1, "No": 0, "positive": 1, "negative": 0})
        y = pd.to_numeric(y, errors="coerce")
        if y.isna().any():
            raise ValueError(
                f"Could not coerce target column '{target_col}' to 0/1 automatically. "
                f"Set cfg['positive_label'] explicitly in config.py. Unique raw values: {y_raw.unique()}"
            )
        y = y.astype(int)

    return X, y


def build_preprocessor(X: pd.DataFrame, categorical_columns=None):
    """Build a ColumnTransformer: median-impute + scale numeric, impute + one-hot categorical."""
    categorical_columns = categorical_columns or []
    categorical_columns = [c for c in categorical_columns if c in X.columns]
    numeric_columns = [c for c in X.columns if c not in categorical_columns]

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_columns),
        ("cat", categorical_pipeline, categorical_columns),
    ])

    return preprocessor, numeric_columns, categorical_columns
