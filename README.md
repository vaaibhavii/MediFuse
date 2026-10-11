# MediFuse

**Multi-source disease risk screening with context-aware evidence fusion.**

MediFuse is an ML-based Smart Hospital & Laboratory Management System. It predicts kidney, liver, diabetes and heart disease risk from patient and lab data, reads urine test strips from a photo, and fuses all of that evidence into one unified risk report. The models are Naive Bayes, Decision Tree and SVM.

> **Disclaimer:** This is an academic project for research and education only. The risk thresholds, abnormal-value ranges and the urine colour chart are illustrative and have not been clinically validated. Do not use it for real medical decisions.

---

## How it works

```
Patient / lab data ──► Kidney · Liver · Diabetes · Heart models (NB / DT / SVM)
                                   │
                                   ▼
                       standardized evidence
                  (probability + confidence + severity)
                                   │
Urine strip photo ──► pad detection (YOLO) ──► colour matching ──┤
Manual urine entry ──────────────────────────────────────────────┤
                                   ▼
          Context-Aware Multi-Source Evidence Fusion
                                   │
                                   ▼
   unified risk score · risk category · abnormal findings · follow-up priority

Structured data ──► K-Means (separate feature: "similar patient profiles")
```

### Key components

| Component | What it does |
|---|---|
| **Source models** | One model set per disease (kidney, liver, diabetes, heart). Naive Bayes, Decision Tree and SVM are each trained and evaluated. The best one is selected per source. |
| **Urine module** | A YOLOv8 detector finds the reagent pads on a strip photo. Each pad's average colour is matched against a reference chart. Confidence comes from colour distance, not a classifier. |
| **Fusion layer** | Each source contributes `probability × confidence × reliability × relevance × (1 + severity bonus)`. It compares three strategies: individual sources, simple average, and the proposed context-aware fusion. |
| **Abnormal rules** | Clinical reference-range checks that produce readable findings and a severity score. |
| **Clustering** | K-Means per source, used only as a supporting "similar profile" label. It is never used as a diagnosis. |
| **API** | FastAPI backend. The frontend only talks to HTTP endpoints. |
| **Frontend** | Static HTML/JS pages for patient records, new screening and the results report. |

---

## Machine learning workflow

1. **Inspect:** check dataset structure, feature types, target labels and class balance.
2. **Preprocess:** impute missing values, scale numeric features and encode categoricals. Each source has its own pipeline, and the datasets are never merged into one table.
3. **Train:** Naive Bayes, Decision Tree and SVM, independently per source.
4. **Evaluate:** accuracy, precision, recall, F1, ROC-AUC and confusion matrices. Medical data is often imbalanced, so accuracy alone is not enough.
5. **Select:** the best model per source is chosen by F1, with ROC-AUC as the tie-breaker.
6. **Fuse:** combine the available sources using reliability, confidence, patient context and abnormality severity. The two baselines are individual-source predictions and simple average fusion.
7. **Categorize:** convert the fused score to Low / Moderate / High using configurable thresholds.

---

## Project structure

```
.
├── api.py                      # FastAPI app (endpoints below)
├── run_api.py                  # Click-to-run launcher for the server
├── config.py                   # Paths, dataset columns, thresholds, reliability weights
├── 01_inspect_data.py          # Prints CSV columns so config.py can be verified
├── 02_train_all_sources.py     # Trains NB/DT/SVM per source, saves models + metrics
├── 04_train_clusters.py        # Fits K-Means per source
├── train_pad_detector.py       # (Optional) fine-tunes YOLOv8 to find urine pads
├── requirements.txt
├── src/
│   ├── preprocessing.py        # Loading, cleaning, encoding, splitting
│   ├── train_source_models.py  # Model training / evaluation / selection
│   ├── inference.py            # predict_patient(): evidence -> fusion -> report
│   ├── fusion.py               # Individual / average / context-aware fusion
│   ├── urine_module.py         # Strip image -> urine evidence
│   ├── abnormal_rules.py       # Reference ranges and severity
│   ├── recommendations.py      # Follow-up recommendations
│   └── clustering.py           # K-Means helpers
├── frontend/                   # index.html, screening.html, results.html + JS/CSS
├── models/                     # Trained models, preprocessors, metrics (.pkl / .json / .pt)
└── data/raw/                   # Datasets (see below)
```

---

## Getting started

### 1. Install

Python 3.10+ is recommended (developed on 3.13).

```bash
git clone <YOUR_REPOSITORY_URL>
cd <repo-folder>
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

### 2. Datasets

Put the datasets in `data/raw/` with these names, or point to another folder with `HOSPITAL_ML_RAW_DIR`.

| Path | Used for |
|---|---|
| `kidney_disease_dataset.csv` | Kidney model |
| `indian_liver_patient.csv` | Liver model (ILPD) |
| `diabetes.csv` | Diabetes model (Pima) |
| `Heart disease/heart.csv` | Heart model |
| `urinalysis_tests.csv` | Urine evidence format and manual-entry path |
| `urinalyze-dataset/`, `urine-pad-dataset/`, `urine-dataset/` | Pad-detector training (Roboflow YOLO exports) |

The four CSVs are already in the repo. The image datasets are large and are **not** in the repo, so you only need them to retrain the pad detector. Trained models are already in `models/`.

```powershell
# optional: use a dataset folder outside the project
$env:HOSPITAL_ML_RAW_DIR="C:\path\to\datasets"
```

### 3. Train

```bash
python 01_inspect_data.py       # check column names; fix config.py if it prints "!!"
python 02_train_all_sources.py  # train and evaluate NB/DT/SVM per source
python 04_train_clusters.py     # K-Means per source

# optional, for the photo-based urine reader:
python train_pad_detector.py --dataset urinalyze-dataset
```

Metrics are saved next to the models, for example `models/kidney_metrics.json`.

### 4. Run the API

```bash
uvicorn api:app --reload --port 8000
```

Or open `run_api.py` in VS Code and press Run. Then go to **http://127.0.0.1:8000/docs** for interactive Swagger docs.

### 5. Open the frontend

Open `frontend/index.html` in a browser while the API is running.

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Patient and lab data (plus optional manual urine readings) → unified risk JSON |
| `POST` | `/urine/analyze` | Strip photo upload → urine evidence JSON |
| `POST` | `/predict/full` | One call: patient and lab data as form-field JSON, plus an optional strip photo |
| `GET` | `/health` | Health check |

Example `/predict` body:

```json
{
  "patient":   { "...": "kidney / liver / diabetes / heart fields" },
  "laboratory": { "...": "additional lab values" },
  "urine_parameters": { "...": "manual strip readings (optional)" },
  "known_conditions": ["diabetes"]
}
```

The response contains the unified risk score and category, per-source evidence, abnormal findings, and a follow-up priority. If `models/pad_detector.pt` is missing, `/urine/analyze` returns an error message. `/predict/full` then continues without urine evidence.

---

## Limitations

- **Not clinically validated.** `config.RISK_THRESHOLDS` and the ranges in `src/abnormal_rules.py` are starting points.
- **Urine colour chart is illustrative.** The RGB values in `src/urine_module.py` must be calibrated against your strip brand, camera and lighting before the strip reader can be trusted.
- **Public datasets differ.** Target and categorical column names vary between mirrors, so run `01_inspect_data.py` and correct `config.py` if you swap datasets.
- **Small datasets.** High metrics (for example a ROC-AUC near 1.0 on kidney) should be read with caution.
- Reliability weights are static guesses by default. `fusion.weights_from_metrics()` can derive them from validation AUC instead.
- Class imbalance can hurt detection of minority classes.
- Results may not generalize to other hospitals, populations or lab equipment without external validation.

## Future improvements

- Better class-imbalance handling, hyperparameter tuning and probability calibration.
- Validation on independent clinical datasets, and evaluating fusion against ground-truth labels.
- Improved urine-strip colour calibration.
- Explainability and fuller API documentation.

## Tech stack

Python · scikit-learn · pandas · NumPy · OpenCV · Ultralytics YOLOv8 · FastAPI · Uvicorn · HTML/CSS/JavaScript

## License

No license has been chosen yet. Add one (for example MIT) before others reuse the code.
