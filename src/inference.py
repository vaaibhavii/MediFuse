"""
predict_patient() is the ONE function the API layer (and eventually the
frontend, indirectly, through the API) depends on. Everything upstream —
which models were used, how fusion was weighted, whether urine came from an
image or manual entry — is invisible past this boundary.
"""

import os
import json
import joblib
import pandas as pd

from config import MODELS_DIR, STRUCTURED_SOURCES
from src.abnormal_rules import evaluate_lab_values, urine_abnormal_findings
from src.fusion import context_aware_fusion, risk_category, follow_up_priority
from src.recommendations import generate_recommendations


def _load_selected_model(source_name: str):
    sel_path = os.path.join(MODELS_DIR, f"{source_name}_selected_model.json")
    if not os.path.exists(sel_path):
        return None
    with open(sel_path) as f:
        selected = json.load(f)["selected_model"]
    preprocessor = joblib.load(os.path.join(MODELS_DIR, f"{source_name}_preprocessor.pkl"))
    model = joblib.load(os.path.join(MODELS_DIR, f"{source_name}_{selected}.pkl"))
    return preprocessor, model, selected


def _run_structured_source(source_name: str, patient_row: dict, abnormal_keys: dict):
    """Returns a standardized evidence dict for one structured source, or
    {"available": False} if the caller didn't supply data for it."""
    if patient_row is None:
        return {"source": source_name, "available": False}

    loaded = _load_selected_model(source_name)
    if loaded is None:
        return {"source": source_name, "available": False, "note": "model not trained yet"}
    preprocessor, model, selected = loaded

    X = pd.DataFrame([patient_row])
    X_t = preprocessor.transform(X)
    X_t = X_t.toarray() if hasattr(X_t, "toarray") else X_t

    probability = float(model.predict_proba(X_t)[:, 1][0])
    prediction = int(probability >= 0.5)

    relevant_values = {k: patient_row.get(k) for k in abnormal_keys.get(source_name, []) if k in patient_row}
    findings = evaluate_lab_values(relevant_values)
    severity = max([f["severity"] for f in findings], default=0.0)

    return {
        "source": source_name,
        "available": True,
        "prediction": prediction,
        "probability": round(probability, 4),
        "confidence": round(max(probability, 1 - probability), 4),  # distance from 0.5, as a simple self-confidence proxy
        "abnormal": len(findings) > 0,
        "severity": severity,
        "model_used": selected,
        "findings": findings,
    }


# map each structured source to which raw fields abnormal_rules.py understands
# (extend this as you add more CHECKS in abnormal_rules.py)
ABNORMAL_KEYS_BY_SOURCE = {
    "kidney": ["bu", "sc", "hemo", "bp"],
    "liver": ["Total_Bilirubin", "Alamine_Aminotransferase", "Aspartate_Aminotransferase"],
    "diabetes": ["Glucose"],
    "heart": [],
}


def predict_patient(patient_data: dict = None, lab_data: dict = None, urine_evidence: dict = None,
                     patient_context: dict = None) -> dict:
    """
    patient_data / lab_data: flat dicts merged into one row per structured
        source. In practice you'll likely pass one combined dict and let each
        source's preprocessor pick out the columns it needs — sklearn's
        ColumnTransformer ignores extra columns it wasn't fit on... actually
        it will error on missing expected columns, so make sure the caller
        supplies at least the columns each source's model was trained on.
        (See api.py for how the /predict endpoint assembles this per source.)
    urine_evidence: output of urine_module.analyze_urine_strip() or
        urine_module.manual_entry_to_evidence(), or None if not available.
    patient_context: optional dict, e.g. {"known_conditions": ["diabetes"]}
    """
    combined_row = {**(patient_data or {}), **(lab_data or {})}

    evidence = {}
    for source_name in STRUCTURED_SOURCES:
        row = combined_row if combined_row else None
        evidence[source_name] = _run_structured_source(source_name, row, ABNORMAL_KEYS_BY_SOURCE)

    abnormal_findings = []
    for source_name, ev in evidence.items():
        for f in ev.get("findings", []):
            abnormal_findings.append(f["finding"])

    if urine_evidence:
        urine_findings = urine_abnormal_findings(urine_evidence)
        abnormal_findings.extend([f["finding"] for f in urine_findings])
        urine_severity = max([f["severity"] for f in urine_findings], default=0.0)
        # crude proxy probability for urine "risk contribution": fraction of abnormal analytes
        urine_abnormal_fraction = len(urine_findings) / max(len(urine_evidence), 1)
        evidence["urinalysis"] = {
            "source": "urinalysis",
            "available": True,
            "prediction": int(urine_abnormal_fraction > 0.2),
            "probability": round(urine_abnormal_fraction, 4),
            "confidence": round(
                sum(r.get("confidence", 0.5) for r in urine_evidence.values()) / max(len(urine_evidence), 1), 4
            ),
            "abnormal": len(urine_findings) > 0,
            "severity": urine_severity,
        }
    else:
        evidence["urinalysis"] = {"source": "urinalysis", "available": False}

    fusion_result = context_aware_fusion(evidence, patient_context=patient_context)
    risk_score = fusion_result["risk_score"]
    category = risk_category(risk_score)
    priority = follow_up_priority(risk_score, len(abnormal_findings))

    explanation = []
    for source_name, contrib in fusion_result["per_source_contribution"].items():
        if contrib.get("used"):
            direction = "increased" if contrib["probability"] >= 0.5 else "did not increase"
            explanation.append(
                f"{source_name.capitalize()}-related evidence {direction} risk "
                f"(probability={contrib['probability']:.2f}, weight={contrib['effective_weight']:.2f})"
            )

    source_evidence_summary = {
        name: ev.get("probability") for name, ev in evidence.items() if ev.get("available")
    }

    recommendations = generate_recommendations(evidence, category)

    return {
        "risk_score": risk_score,
        "risk_category": category,
        "abnormal_findings": abnormal_findings,
        "source_evidence": source_evidence_summary,
        "follow_up_priority": priority,
        "explanation": explanation,
        "recommended_actions": recommendations["recommended_actions"],
        "disclaimer": recommendations["disclaimer"],
        "_debug_per_source_contribution": fusion_result["per_source_contribution"],
    }
