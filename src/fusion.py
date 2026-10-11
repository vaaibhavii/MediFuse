"""
Evidence fusion layer.

Standardized evidence format expected from every source, structured or urine:

    {
      "source": "kidney",
      "available": True,
      "prediction": 1,
      "probability": 0.82,     # model's P(condition present)
      "confidence": 0.82,      # how sure the source itself is (may differ
                                # from probability for rule/color-matched
                                # sources like urine, where confidence comes
                                # from color-distance, not a classifier)
      "abnormal": True,        # did this source's raw values breach a
                                # clinical threshold, independent of the model?
      "severity": 0.6,         # 0-1, how far past threshold (0 if normal)
    }

Three fusion strategies are implemented so they can be compared empirically:
  1. individual_sources()      -- baseline 1: no fusion at all
  2. simple_average_fusion()   -- baseline 2: naive mean of probabilities
  3. context_aware_fusion()    -- proposed method
"""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np

from config import SOURCE_BASE_RELIABILITY, RISK_THRESHOLDS


def individual_sources(evidence: Dict[str, dict]) -> Dict[str, float]:
    """Baseline 1: just expose each source's own probability, unfused."""
    return {name: ev["probability"] for name, ev in evidence.items() if ev.get("available", True)}


def simple_average_fusion(evidence: Dict[str, dict]) -> float:
    """Baseline 2: unweighted mean of available sources' probabilities."""
    probs = [ev["probability"] for ev in evidence.values() if ev.get("available", True)]
    return float(np.mean(probs)) if probs else 0.0


def weights_from_metrics(metrics_by_source: Dict[str, dict]) -> Dict[str, float]:
    """Optional helper: derive reliability weights from each source model's
    validation ROC-AUC (models/{source}_selected_model.json), instead of the
    static SOURCE_BASE_RELIABILITY guesses in config.py. Call this once after
    training and feed the result into context_aware_fusion(reliability=...)."""
    weights = {}
    for source, m in metrics_by_source.items():
        auc = m.get("metrics", {}).get("roc_auc") or 0.5
        weights[source] = max(0.1, auc)  # floor so a weak source isn't zeroed out
    return weights


def context_aware_fusion(
    evidence: Dict[str, dict],
    patient_context: dict = None,
    reliability: Dict[str, float] = None,
) -> dict:
    """
    Proposed method: Context-Aware Multi-Source Evidence Fusion.

    Each available source contributes:
        contribution_i = probability_i * confidence_i * reliability_i * relevance_i * (1 + severity_bonus_i)

    - probability_i:   model's own P(condition present) for that source
    - confidence_i:    the source's self-reported certainty
    - reliability_i:   how trustworthy this source model has proven to be
                        historically (defaults to config.SOURCE_BASE_RELIABILITY,
                        or pass in AUC-derived weights via weights_from_metrics())
    - relevance_i:      contextual relevance multiplier — e.g. if patient_context
                        flags a pre-existing condition matching this source
                        (documented diabetic -> diabetes evidence weighted up),
                        or if the source is currently unavailable (relevance
                        drops to 0 rather than silently imputing a value)
    - severity_bonus_i: abnormal-value severity adds extra weight, so a source
                        that is both "positive" AND far outside normal range
                        counts more than a borderline positive

    Final risk_score = weighted average of contributions, renormalized by the
    total weight actually used (so missing sources don't silently drag the
    score toward 0 — this is the key difference vs. simple_average_fusion,
    which treats "missing" the same as "not calculated").
    """
    reliability = reliability or SOURCE_BASE_RELIABILITY
    patient_context = patient_context or {}

    weighted_sum = 0.0
    total_weight = 0.0
    per_source_contribution = {}

    for source, ev in evidence.items():
        if not ev.get("available", True):
            per_source_contribution[source] = {"used": False, "reason": "source not available for this patient"}
            continue

        probability = ev.get("probability", 0.0)
        confidence = ev.get("confidence", 1.0)
        base_reliability = reliability.get(source, 0.75)

        # contextual relevance: boost a source if the patient's known history
        # makes it especially pertinent right now (documented, not inferred)
        relevance = 1.0
        known_conditions = patient_context.get("known_conditions", [])
        source_relevance_map = {
            "diabetes": "diabetes", "kidney": "kidney_disease", "liver": "liver_disease",
            "heart": "cardiovascular_disease", "urinalysis": "kidney_disease",
        }
        if source_relevance_map.get(source) in known_conditions:
            relevance = 1.25  # modest boost, not a hard override

        severity = ev.get("severity", 0.0)  # 0..1
        severity_bonus = 1.0 + (0.3 * severity)  # up to +30% weight for severe abnormal values

        weight = confidence * base_reliability * relevance * severity_bonus
        contribution = probability * weight

        weighted_sum += contribution
        total_weight += weight

        per_source_contribution[source] = {
            "used": True,
            "probability": probability,
            "confidence": confidence,
            "reliability": base_reliability,
            "relevance": relevance,
            "severity_bonus": severity_bonus,
            "effective_weight": round(weight, 4),
            "weighted_contribution": round(contribution, 4),
        }

    risk_score = (weighted_sum / total_weight) if total_weight > 0 else 0.0

    return {
        "risk_score": round(float(risk_score), 4),
        "per_source_contribution": per_source_contribution,
    }


def risk_category(risk_score: float) -> str:
    if risk_score <= RISK_THRESHOLDS["low"]:
        return "Low"
    if risk_score <= RISK_THRESHOLDS["moderate"]:
        return "Moderate"
    return "High"


def follow_up_priority(risk_score: float, abnormal_count: int) -> str:
    from config import FOLLOWUP_ABNORMAL_COUNT_HIGH
    category = risk_category(risk_score)
    if category == "High" or abnormal_count >= FOLLOWUP_ABNORMAL_COUNT_HIGH:
        return "High"
    if category == "Moderate":
        return "Medium"
    return "Low"
