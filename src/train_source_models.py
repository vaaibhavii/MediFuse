"""
Train Naive Bayes, Decision Tree, and SVM for ONE structured source, evaluate
with accuracy/precision/recall/F1/ROC-AUC/confusion matrix, and persist:
  models/{source}_preprocessor.pkl
  models/{source}_naive_bayes.pkl
  models/{source}_decision_tree.pkl
  models/{source}_svm.pkl
  models/{source}_metrics.json          (all three models' scores)
  models/{source}_selected_model.json   (which one fusion should call by default, and why)
"""

import os
import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from src.preprocessing import load_and_split_xy, build_preprocessor
from config import MODELS_DIR, RANDOM_STATE, TEST_SIZE, CV_FOLDS, SVM_MAX_TRAIN_SAMPLES

def _make_model(model_type: str):
    if model_type == "naive_bayes":
        return GaussianNB()
    if model_type == "decision_tree":
        return DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=RANDOM_STATE)
    if model_type == "svm":
        # probability=True needed so we get calibrated-ish probabilities for fusion
        return SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)
    raise ValueError(model_type)


def _evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if len(set(y_test)) > 1 else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def train_source(source_name: str, cfg: dict):
    print(f"\n=== Training source: {source_name} ===")
    X, y = load_and_split_xy(cfg)
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X, cfg.get("categorical_columns"))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y if y.nunique() > 1 else None
    )

    # fit preprocessor once on train, reuse transformed arrays for all 3 models
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    all_metrics = {}
    fitted_models = {}

    for model_type in ["naive_bayes", "decision_tree", "svm"]:
        model = _make_model(model_type)
        # GaussianNB / SVC need dense arrays
        Xtr = X_train_t.toarray() if hasattr(X_train_t, "toarray") else X_train_t
        Xte = X_test_t.toarray() if hasattr(X_test_t, "toarray") else X_test_t

        fit_Xtr, fit_ytr = Xtr, y_train
        if model_type == "svm" and Xtr.shape[0] > SVM_MAX_TRAIN_SAMPLES:
            print(f"  svm: subsampling {Xtr.shape[0]} -> {SVM_MAX_TRAIN_SAMPLES} training rows (rbf SVC doesn't scale to large datasets)")
            sample_idx, _ = train_test_split(
                np.arange(Xtr.shape[0]), train_size=SVM_MAX_TRAIN_SAMPLES,
                random_state=RANDOM_STATE, stratify=y_train
            )
            fit_Xtr, fit_ytr = Xtr[sample_idx], y_train.iloc[sample_idx]

        model.fit(fit_Xtr, fit_ytr)
        metrics = _evaluate(model, Xte, y_test)

        try:
            cv_scores = cross_val_score(model, fit_Xtr, fit_ytr, cv=min(CV_FOLDS, fit_ytr.value_counts().min()), scoring="f1")
            metrics["cv_f1_mean"] = float(np.mean(cv_scores))
        except Exception as e:
            metrics["cv_f1_mean"] = None

        all_metrics[model_type] = metrics
        fitted_models[model_type] = model
        print(f"  {model_type:15s} acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f} "
              f"recall={metrics['recall']:.3f} auc={metrics['roc_auc']}")

    # model selection: don't just pick highest accuracy — prioritize F1 (imbalance-aware),
    # tie-break on ROC-AUC
    def score_key(item):
        m = item[1]
        return (m["f1"], m["roc_auc"] or 0)

    best_model_type, best_metrics = max(all_metrics.items(), key=score_key)

    # persist everything
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, f"{source_name}_preprocessor.pkl"))
    for model_type, model in fitted_models.items():
        joblib.dump(model, os.path.join(MODELS_DIR, f"{source_name}_{model_type}.pkl"))

    with open(os.path.join(MODELS_DIR, f"{source_name}_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    with open(os.path.join(MODELS_DIR, f"{source_name}_selected_model.json"), "w") as f:
        json.dump({
            "selected_model": best_model_type,
            "reason": "highest F1-score (ties broken by ROC-AUC) — not raw accuracy, "
                      "because these datasets are class-imbalanced",
            "metrics": best_metrics,
        }, f, indent=2)

    print(f"  -> selected for fusion: {best_model_type} (F1={best_metrics['f1']:.3f})")
    return all_metrics, best_model_type
