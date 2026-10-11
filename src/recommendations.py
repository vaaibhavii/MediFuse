"""
Rule-based next-step suggestions — NOT a diagnosis or treatment plan.
Maps which sources are abnormal + the overall risk category to standard
"who to see next" signposting, the same way a triage nurse would.
"""

SOURCE_SPECIALIST_MAP = {
    "kidney": "a nephrologist (kidney specialist)",
    "liver": "a hepatologist or gastroenterologist (liver specialist)",
    "diabetes": "an endocrinologist or your primary physician for diabetes management",
    "heart": "a cardiologist",
    "urinalysis": "your primary physician for a follow-up urinalysis",
}

RISK_TIMEFRAME_MAP = {
    "High": "as soon as possible (within a few days)",
    "Moderate": "within the next few weeks",
    "Low": "at your next routine check-up",
}


def generate_recommendations(evidence: dict, risk_category: str) -> dict:
    actions = []

    for source, ev in evidence.items():
        if ev.get("available") and ev.get("abnormal"):
            specialist = SOURCE_SPECIALIST_MAP.get(source)
            if specialist:
                actions.append(f"Consider consulting {specialist}, based on abnormal {source} findings.")

    timeframe = RISK_TIMEFRAME_MAP.get(risk_category, "at your next routine check-up")
    actions.append(f"Overall risk category is '{risk_category}' — recommended follow-up: {timeframe}.")

    return {
        "recommended_actions": actions,
        "disclaimer": (
            "These are general next-step suggestions generated from rule-based "
            "thresholds, not a medical diagnosis or treatment plan. Always consult "
            "a qualified healthcare professional for actual medical advice."
        ),
    }