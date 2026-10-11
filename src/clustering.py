"""
K-Means clustering — a SEPARATE supporting feature, not part of core risk
prediction. Fit one KMeans model per structured source (on that source's own
preprocessed feature space), so "similar patient profile" is meaningful
within that source's population rather than an artificial mash-up of
unrelated datasets.

Clusters are used only for a "similar profiles" UI label and as one signal
into follow-up priority — never as a stand-alone diagnosis.
"""

import os
import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.preprocessing import load_and_split_xy, build_preprocessor
from config import MODELS_DIR, RANDOM_STATE


def train_clusters_for_source(source_name: str, cfg: dict, k_range=range(2, 7)):
    X, _y = load_and_split_xy(cfg)
    preprocessor, _, _ = build_preprocessor(X, cfg.get("categorical_columns"))
    X_t = preprocessor.fit_transform(X)
    X_t = X_t.toarray() if hasattr(X_t, "toarray") else X_t

    best_k, best_score, best_model = None, -1, None
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_t)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X_t, labels)
        if score > best_score:
            best_k, best_score, best_model = k, score, km

    joblib.dump(preprocessor, os.path.join(MODELS_DIR, f"{source_name}_cluster_preprocessor.pkl"))
    joblib.dump(best_model, os.path.join(MODELS_DIR, f"{source_name}_kmeans.pkl"))
    print(f"  [{source_name}] best k={best_k} silhouette={best_score:.3f}")
    return best_k, best_score


def assign_cluster(source_name: str, patient_row_df) -> int:
    """patient_row_df: single-row DataFrame with the same columns used in training."""
    preprocessor = joblib.load(os.path.join(MODELS_DIR, f"{source_name}_cluster_preprocessor.pkl"))
    km = joblib.load(os.path.join(MODELS_DIR, f"{source_name}_kmeans.pkl"))
    X_t = preprocessor.transform(patient_row_df)
    X_t = X_t.toarray() if hasattr(X_t, "toarray") else X_t
    return int(km.predict(X_t)[0])
