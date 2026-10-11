"""
OPTIONAL. Fine-tunes a small YOLOv8 object detector on one of your Roboflow
exports (urinalyze-dataset / urine-pad-dataset / urine-dataset) to localize
reagent pads in a strip photo.

Requirements: pip install ultralytics
Your Roboflow export must include a data.yaml (standard Roboflow YOLO export
format) at the dataset root — Roboflow generates this automatically when you
export as "YOLOv8".

Usage:
    python train_pad_detector.py --dataset urinalyze-dataset

The resulting weights (best.pt) go to models/pad_detector.pt, and get loaded
by src/urine_module.py's inference wrapper (see PadDetector class below).
"""

import argparse
import os
import shutil
from config import RAW_DATA_DIR, MODELS_DIR


def main(dataset_folder_name: str, epochs: int = 30, imgsz: int = 640):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("pip install ultralytics  # then re-run this script")

    dataset_path = os.path.join(RAW_DATA_DIR, dataset_folder_name)
    data_yaml = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(data_yaml):
        raise SystemExit(
            f"No data.yaml found at {data_yaml}. Make sure you exported the "
            f"Roboflow dataset in YOLOv8 format (Roboflow > Export > YOLOv8)."
        )

    model = YOLO("yolov8n.pt")  # nano — fast to fine-tune on small datasets
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=8, project=MODELS_DIR, name="pad_detector_run", exist_ok=True)

    best_weights = os.path.join(model.trainer.save_dir, "weights", "best.pt")

    best_weights = os.path.join(MODELS_DIR, "pad_detector_run", "weights", "best.pt")
    final_path = os.path.join(MODELS_DIR, "pad_detector.pt")
    shutil.copy(best_weights, final_path)
    print(f"Saved fine-tuned detector to {final_path}")


class PadDetector:
    """Thin inference wrapper so urine_module.analyze_urine_strip() can take
    an image path straight through, without the API layer knowing about YOLO."""

    def __init__(self, weights_path=None, class_to_analyte=None):
        from ultralytics import YOLO
        weights_path = weights_path or os.path.join(MODELS_DIR, "pad_detector.pt")
        self.model = YOLO(weights_path)
        # map the detector's trained class names -> analyte names used in
        # urine_module.REFERENCE_CHART. Edit this after checking data.yaml's
        # 'names' list for your specific dataset export.
        self.class_to_analyte = class_to_analyte or {
            "bilirubin": "bilirubin",
            "blood": "blood",
            "glucose": "glucose",
            "ketone": "ketones",
            "leukocyte": "leukocytes",
            "nitrite": "nitrite",
            "pH": "pH",
            "protein": "protein",
            "specific-gravity": "specific_gravity",
            "urobilinogen": "urobilinogen",
            # "test-strip" intentionally omitted — it's the whole strip's bounding box, not an analyte pad
        }

    def detect(self, image_path: str, conf: float = 0.4):
        result = self.model.predict(image_path, conf=conf, verbose=False)[0]
        detections = []
        for box in result.boxes:
            cls_name = result.names[int(box.cls[0])]
            analyte = self.class_to_analyte.get(cls_name)
            if analyte is None:
                continue  # skip non-analyte classes like "strip" or "color card"
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({"analyte": analyte, "bbox": (x1, y1, x2, y2)})
        return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="urinalyze-dataset",
                         choices=["urinalyze-dataset", "urine-pad-dataset", "urine-dataset"])
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    main(args.dataset, args.epochs)
