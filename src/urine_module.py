"""
Urine-strip evidence module.

Pipeline: image -> pad detection (bounding boxes) -> per-pad average color ->
nearest-match against a manufacturer-style reference chart -> parameter value
+ confidence (based on color distance, NOT a trained classifier — there is no
public dataset mapping strip photos to lab-verified concentrations, so this
mirrors how real strip-reader systems work).

Two ways to get pad bounding boxes:
  1. `detector`: a loaded object-detection model (e.g. an Ultralytics YOLO
     model fine-tuned on data/raw/urinalyze-dataset or urine-pad-dataset,
     see train_pad_detector.py). Pass class-name -> analyte mapping.
  2. `manual_layout`: if you photograph strips with a fixed jig/holder, pass
     fixed relative bounding boxes instead of running a detector at all.

Either way the output format is identical, so the fusion layer never needs
to know which one produced it.
"""

from dataclasses import dataclass
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


# ---------------------------------------------------------------------------
# Reference chart: approximate RGB centroids per analyte per reading level.
# THESE ARE ILLUSTRATIVE PLACEHOLDERS. For a real deployment, calibrate this
# chart against your actual strip brand's printed color chart under
# controlled, consistent lighting — color reproduction varies by camera/light.
# ---------------------------------------------------------------------------

REFERENCE_CHART = {
    "glucose": {
        "negative": (0, 120, 180), "trace": (0, 150, 140), "+1": (60, 170, 90),
        "+2": (150, 160, 40), "+3": (190, 120, 20), "+4": (170, 60, 10),
    },
    "protein": {
        "negative": (230, 230, 120), "trace": (190, 210, 110),
        "+1": (140, 190, 110), "+2": (80, 170, 120), "+3": (40, 140, 130),
    },
    "pH": {
        "5.0": (220, 140, 40), "6.0": (220, 180, 40), "6.5": (200, 200, 60),
        "7.0": (140, 200, 90), "7.5": (80, 190, 130), "8.5": (40, 160, 170),
    },
    "ketones": {
        "negative": (250, 230, 220), "trace": (240, 190, 200),
        "+1": (220, 140, 180), "+2": (190, 80, 150), "+3": (150, 40, 120),
    },
    "blood": {
        "negative": (230, 200, 60), "trace": (190, 190, 80),
        "+1": (120, 170, 100), "+2": (60, 130, 110), "+3": (30, 90, 100),
    },
    "bilirubin": {
        "negative": (250, 245, 235), "+1": (230, 210, 170),
        "+2": (200, 170, 110), "+3": (170, 130, 70),
    },
    "urobilinogen": {
        "normal": (240, 190, 170), "+1": (220, 150, 150),
        "+2": (190, 100, 130), "+3": (150, 60, 110),
    },
    "nitrite": {"negative": (255, 250, 245), "positive": (250, 200, 200)},
    "leukocytes": {
        "negative": (245, 235, 210), "trace": (230, 210, 160),
        "+1": (210, 180, 110), "+2": (180, 140, 90),
    },
    "specific_gravity": {
        "1.000": (30, 90, 150), "1.010": (40, 120, 130), "1.020": (60, 150, 100),
        "1.030": (100, 170, 70),
    },
}


def _color_distance(c1, c2):
    return float(np.linalg.norm(np.array(c1, dtype=float) - np.array(c2, dtype=float)))


def match_color_to_level(analyte: str, rgb_color) -> dict:
    """Nearest-neighbor match against the reference chart. Confidence is a
    simple inverse-distance score in [0, 1] — NOT a model probability."""
    chart = REFERENCE_CHART.get(analyte)
    if chart is None:
        return {"value": None, "confidence": 0.0, "note": f"no reference chart for '{analyte}'"}

    distances = {level: _color_distance(rgb_color, ref) for level, ref in chart.items()}
    best_level = min(distances, key=distances.get)
    best_dist = distances[best_level]

    max_possible_dist = 441.67  # sqrt(255^2 * 3), worst case
    confidence = max(0.0, 1.0 - (best_dist / max_possible_dist))

    return {"value": best_level, "confidence": round(confidence, 3)}


def extract_average_color(image, bbox):
    """bbox = (x1, y1, x2, y2) pixel coords. Returns (R, G, B)."""
    if cv2 is None:
        raise ImportError("opencv-python is required: pip install opencv-python")
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"Empty crop for bbox {bbox}")
    mean_bgr = crop.reshape(-1, 3).mean(axis=0)
    b, g, r = mean_bgr
    return (float(r), float(g), float(b))


def analyze_urine_strip(image_path: str, detections: list) -> dict:
    """
    detections: list of {"analyte": str, "bbox": (x1,y1,x2,y2)} — typically
    produced by a pad-detector model (see train_pad_detector.py) or by a
    fixed manual layout if you use a physical strip holder/jig.

    Returns standardized evidence dict, ready for the fusion layer:
        {
          "glucose": {"value": "negative", "confidence": 0.91},
          "protein": {"value": "trace", "confidence": 0.77},
          ...
        }
    """
    if cv2 is None:
        raise ImportError("opencv-python is required: pip install opencv-python")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)

    results = {}
    for det in detections:
        analyte = det["analyte"]
        bbox = det["bbox"]
        color = extract_average_color(image, bbox)
        results[analyte] = match_color_to_level(analyte, color)

    return results


def manual_entry_to_evidence(manual_values: dict) -> dict:
    """
    Fallback path: if no photo is available and a lab tech types in strip
    readings directly (matches urinalysis_tests.csv columns), wrap them in
    the same {value, confidence} shape so fusion doesn't need two code paths.
    Manual entry is treated as maximum confidence (a human read the strip).
    """
    return {k: {"value": v, "confidence": 1.0} for k, v in manual_values.items()}
