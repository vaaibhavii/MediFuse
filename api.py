"""
Backend API. Run with:  uvicorn api:app --reload --port 8000

Endpoints match the handoff spec:
  POST /predict         -> unified risk JSON
  POST /urine/analyze    -> urine parameters + confidence from a strip photo

The frontend team should ONLY ever talk to these two endpoints — never
import sklearn/joblib/model files directly.
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import json
from pydantic import BaseModel
from typing import Optional, Dict, Any
import shutil
import tempfile
import os

from src.inference import predict_patient
from src.urine_module import analyze_urine_strip, manual_entry_to_evidence

app = FastAPI(title="Smart Hospital Risk Screening API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    patient: Optional[Dict[str, Any]] = None
    laboratory: Optional[Dict[str, Any]] = None
    urine_parameters: Optional[Dict[str, Any]] = None  # manual strip readings, if no photo
    known_conditions: Optional[list] = None


@app.post("/predict")
def predict(req: PredictRequest):
    urine_evidence = manual_entry_to_evidence(req.urine_parameters) if req.urine_parameters else None
    result = predict_patient(
        patient_data=req.patient,
        lab_data=req.laboratory,
        urine_evidence=urine_evidence,
        patient_context={"known_conditions": req.known_conditions or []},
    )
    return result


def _run_urine_image_through_detector(upload_file: UploadFile) -> dict:
    """Shared by /urine/analyze and /predict/full. Saves the upload to a temp
    file, runs the pad detector + colorimetric matching, and always cleans up
    the temp file afterward. Returns either the evidence dict or an
    {"error": ...} dict if no trained detector is available yet."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(upload_file.filename)[1]) as tmp:
        shutil.copyfileobj(upload_file.file, tmp)
        tmp_path = tmp.name

    try:
        pad_detector_path = os.path.join("models", "pad_detector.pt")
        if not os.path.exists(pad_detector_path):
            return {
                "error": "No trained pad detector found at models/pad_detector.pt. "
                         "Run train_pad_detector.py first, or submit readings manually via "
                         "POST /predict's urine_parameters field instead."
            }
        from train_pad_detector import PadDetector
        detector = PadDetector(pad_detector_path)
        detections = detector.detect(tmp_path)
        return analyze_urine_strip(tmp_path, detections)
    finally:
        os.remove(tmp_path)


@app.post("/urine/analyze")
def urine_analyze(file: UploadFile = File(...)):
    """
    Accepts a strip photo, runs the pad detector + colorimetric matching, and
    returns urine evidence in the same {analyte: {value, confidence}} shape
    used everywhere else in the system.
    """
    return _run_urine_image_through_detector(file)

@app.post("/predict/full")
def predict_full(
    patient: str = Form(..., description="JSON string of patient fields, e.g. kidney/liver/diabetes values"),
    laboratory: str = Form("{}", description="JSON string of additional lab values"),
    known_conditions: str = Form("[]", description="JSON array of strings, e.g. [\"diabetes\"]"),
    urine_image: UploadFile | None = File(None),
):
    """
    One-call version for the frontend: patient/laboratory data as JSON
    strings (multipart form fields, since a file upload forces multipart
    rather than a plain JSON body) plus a strip photo, all in a single
    request. Internally runs the same urine detector as /urine/analyze and
    the same fusion as /predict, so results are identical to calling both
    endpoints separately and chaining them by hand.
    """
    patient_data = json.loads(patient)
    from fastapi import HTTPException

    def _safe_json(field_name: str, raw: str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Field '{field_name}' is not valid JSON: {e}. You sent: {raw!r}"
            )

    def _parse_conditions(raw: str) -> list:
        raw = raw.strip()
        if not raw or raw == "[]":
            return []
        if raw.startswith("["):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
        # fallback: accept plain comma-separated text too, e.g. "diabetes" or "diabetes, hypertension"
        return [item.strip() for item in raw.split(",") if item.strip()]

    patient_data = _safe_json("patient", patient)
    lab_data = _safe_json("laboratory", laboratory)
    conditions = _parse_conditions(known_conditions)

    urine_evidence = _run_urine_image_through_detector(urine_image) if urine_image else None
    if urine_evidence and "error" in urine_evidence:
        # no trained detector available — fall through with no urine evidence
        # rather than failing the whole request
        urine_evidence = None

    result = predict_patient(
        patient_data=patient_data,
        lab_data=lab_data,
        urine_evidence=urine_evidence,
        patient_context={"known_conditions": conditions},
    )
    return result


@app.get("/health")
def health():
    return {"status": "ok"}
