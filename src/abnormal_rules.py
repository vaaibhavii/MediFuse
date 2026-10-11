"""
Simple clinical-threshold checks used for:
  (a) the human-readable "abnormal_findings" list in the API response
  (b) the "severity" input to context_aware_fusion()

These are illustrative common reference ranges, NOT validated clinical
guidance — a domain expert should review/adjust before any real use.
Each check returns (is_abnormal: bool, severity: float 0..1, label: str).
"""

def _severity(value, low, high):
    if value < low:
        span = max(low - value, 1e-6)
        return min(1.0, span / max(low, 1e-6))
    if value > high:
        span = max(value - high, 1e-6)
        return min(1.0, span / max(high, 1e-6))
    return 0.0


CHECKS = {
    "bu":                       {"range": (7, 20),   "label": "Elevated blood urea"},
    "sc":                       {"range": (0.6, 1.3), "label": "Elevated creatinine"},
    "hemo":                     {"range": (12, 17),  "label": "Reduced hemoglobin"},
    "bp":                       {"range": (60, 120), "label": "Abnormal blood pressure (kidney reading)"},
    "Total_Bilirubin":          {"range": (0.1, 1.2), "label": "Elevated bilirubin"},
    "Alamine_Aminotransferase": {"range": (7, 56),   "label": "Elevated liver enzyme (ALT)"},
    "Aspartate_Aminotransferase": {"range": (10, 40), "label": "Elevated liver enzyme (AST)"},
    "Glucose":                  {"range": (70, 140), "label": "Elevated glucose"},
}


def evaluate_lab_values(lab_values: dict) -> list:
    """
    lab_values: flat dict of {check_name: numeric_value}, e.g.
        {"glucose_fasting": 145, "creatinine": 1.6}
    Returns a list of {"finding": str, "severity": float} for anything abnormal.
    """
    findings = []
    for key, value in lab_values.items():
        check = CHECKS.get(key)
        if check is None or value is None:
            continue
        low, high = check["range"]
        severity = _severity(value, low, high)
        if severity > 0:
            findings.append({"finding": check["label"], "severity": round(severity, 3)})
    return findings


def urine_abnormal_findings(urine_evidence: dict) -> list:
    """Flags any non-negative/non-normal urine strip reading."""
    normal_values = {"negative", "normal", "trace"}  # trace is borderline-normal
    findings = []
    for analyte, result in urine_evidence.items():
        value = str(result.get("value", "")).lower()
        if value not in normal_values and value not in ("", "none"):
            findings.append({
                "finding": f"Abnormal urinalysis: {analyte} = {result.get('value')}",
                "severity": round(1.0 - result.get("confidence", 0.5) * 0.3, 3),  # rough proxy
            })
    return findings
