r"""
Central configuration for the Smart Hospital & Laboratory Management ML pipeline.

HOW TO USE
----------
1. Either copy your dataset files/folders into ./data/raw/ using the exact
   names below, OR change RAW_DATA_DIR to point straight at your existing
   folder, e.g.:

   RAW_DATA_DIR = r"C:\Users\Rahul\Desktop\KJSCE\TY\ML mini project"

2. Run `python 01_inspect_data.py` FIRST. It prints the columns of every
   structured CSV so you can confirm/correct TARGET_COLUMN and
   CATEGORICAL_COLUMNS below for each source (every public dataset names
   its target/label column slightly differently depending on which exact
   version you downloaded).

3. Then run `python 02_train_all_sources.py`, `python 03_train_fusion_demo.py`,
   `python 04_train_clusters.py`, and finally `python api.py`.
"""

import os

# ---------------------------------------------------------------------------
# 1. Paths — matches the folder structure in your screenshot
# ---------------------------------------------------------------------------

# Point this at your existing "ML mini project" folder if you don't want to
# copy files. Otherwise leave as-is and drop your files into data/raw/.
RAW_DATA_DIR = os.environ.get("HOSPITAL_ML_RAW_DIR", os.path.join(os.path.dirname(__file__), "data", "raw"))

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

STRUCTURED_SOURCES = {
    "kidney": {
    "file": os.path.join(RAW_DATA_DIR, "kidney_disease_dataset.csv"),
    "target_column": "classification",
    "positive_label": None,
    "negative_label": "notckd",
    "id_columns": ["id"],
    "categorical_columns": ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"],
    },
    "liver": {
        "file": os.path.join(RAW_DATA_DIR, "indian_liver_patient.csv"),
        "target_column": "Dataset",           # ILPD: 1 = liver patient, 2 = not -> remapped to 1/0 in preprocessing
        "positive_label": 1,
        "id_columns": [],
        "categorical_columns": ["Gender"],
    },
    "diabetes": {
        "file": os.path.join(RAW_DATA_DIR, "diabetes.csv"),
        "target_column": "Outcome",           # Pima Indians Diabetes
        "positive_label": 1,
        "id_columns": [],
        "categorical_columns": [],
    },
    "heart": {
        # 'Heart disease' is a folder in your screenshot — point this at the
        # actual csv inside it once you check the name.
        "file": os.path.join(RAW_DATA_DIR, "Heart disease", "heart.csv"),
        "target_column": "target",            # common UCI heart dataset column name
        "positive_label": 1,
        "id_columns": [],
        "categorical_columns": ["cp", "restecg", "slope", "thal"],
        "optional": True,
    },
}

# Tabular urinalysis dataset — used to validate the urine evidence *format*,
# and as a fallback/manual-entry path when no strip photo is available.
URINALYSIS_TABULAR_FILE = os.path.join(RAW_DATA_DIR, "urinalysis_tests.csv")

# Roboflow-style folders (images + annotations) for the strip-reading module.
URINE_DETECTION_DATASETS = {
    "urinalyze": os.path.join(RAW_DATA_DIR, "urinalyze-dataset"),
    "urine": os.path.join(RAW_DATA_DIR, "urine-dataset"),
    "urine_pad": os.path.join(RAW_DATA_DIR, "urine-pad-dataset"),
}

# ---------------------------------------------------------------------------
# 2. Modeling
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
SVM_MAX_TRAIN_SAMPLES = 3000  # rbf-kernel SVC scales very poorly past a few thousand rows
TEST_SIZE = 0.2
CV_FOLDS = 5

MODEL_TYPES = ["naive_bayes", "decision_tree", "svm"]

# ---------------------------------------------------------------------------
# 3. Fusion / risk calibration
# ---------------------------------------------------------------------------

# Baseline static reliability weight per source (can be replaced later with
# each model's validation AUC — see fusion.py: weights_from_auc()).
SOURCE_BASE_RELIABILITY = {
    "kidney": 1.0,
    "liver": 0.9,
    "diabetes": 1.0,
    "heart": 0.8,
    "urinalysis": 0.85,
}

RISK_THRESHOLDS = {
    # NOT clinically validated — placeholders to be tuned during evaluation.
    "low": 0.33,
    "moderate": 0.66,
    # anything above 'moderate' => "High"
}

FOLLOWUP_ABNORMAL_COUNT_HIGH = 3  # >= this many abnormal findings -> bump priority
