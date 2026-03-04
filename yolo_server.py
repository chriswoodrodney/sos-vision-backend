# yolo_server.py
# Full Flask app with /detect endpoint (YOLO + EasyOCR + CORS + robust error handling)

import os
import base64
import traceback
from typing import List, Dict, Any

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np

from ultralytics import YOLO
import easyocr
import cv2

# ---------------- Config ----------------
ALLOWED_ORIGINS = [
    "https://sos-frontend-woad-gamma.vercel.app",
    "https://sos-frontend-git-main-chriswoodrodney.vercel.app",
    "https://sos-frontend-ozxeq29u8-chriswoodrodney.vercel.app",
]

YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolov8n.pt")
OCR_LANGS = os.environ.get("OCR_LANGS", "en").split(",")

# Tunables
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.30"))
OCR_MIN_CONF = float(os.environ.get("OCR_MIN_CONF", "0.30"))  # lowered for noisy live camera frames

# ---------------- App init ----------------
app = Flask(__name__, static_folder="build", static_url_path="")
CORS(app, resources={r"/detect": {"origins": ALLOWED_ORIGINS}})

# ---------------- Model loading ----------------
try:
    model = YOLO(YOLO_WEIGHTS)
    try:
        model.to("cpu")
    except Exception as e:
        print("Warning: couldn't set model device explicitly:", e)
    print("YOLO model loaded.")
except Exception as e:
    model = None
    print("Failed loading YOLO model:", e)
    traceback.print_exc()

try:
    reader = easyocr.Reader(OCR_LANGS, gpu=False)
    print("EasyOCR reader initialized.")
except Exception as e:
    reader = None
    print("Failed to initialize EasyOCR:", e)
    traceback.print_exc()

# ---------------- Helpers ----------------
def safe_cast_cls(box):
    try:
        cls_val = getattr(box, "cls", None)
        if cls_val is None:
            return None
        try:
            return int(cls_val)
        except Exception:
            return int(np.array(cls_val).item())
    except Exception:
        return None


def preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    """
    Stronger OCR preprocessing for live camera frames:
    - BGR -> GRAY
    - upscale 2x (helps small text)
    - denoise
    - contrast boost (CLAHE)
    - adaptive threshold (often helps packaging text)
    Returns a single-channel (gray/binary) image that EasyOCR can read.
    """
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # upscale
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        # denoise + edge preservation
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        # CLAHE contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # adaptive threshold to make text stand out
        th = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 8
        )
        return th
    except Exception:
        return img_bgr


def extract_text_easyocr(img_for_ocr: np.ndarray) -> List[str]:
    """
    Runs EasyOCR and returns cleaned text list.
    Filters by confidence and basic cleanup.
    """
    if reader is None:
        return []

    ocr_texts: List[str] = []
    try:
        raw_results = reader.readtext(img_for_ocr)
        for item in raw_results:
            if not item or len(item) < 2:
                continue
            text = str(item[1]).strip()
            conf = float(item[2]) if len(item) > 2 and item[2] is not None else 0.0

            if conf < OCR_MIN_CONF:
                continue

            # basic cleanup
            text = " ".join(text.split())
            if text:
                ocr_texts.append(text)
    except Exception as oe:
        print("OCR inference error:", oe)

    return ocr_texts


def apply_ocr_keyword_mapping(texts: List[str], detections: List[Dict[str, Any]]) -> None:
    """
    If OCR sees keywords, add/boost a detection label.
    """
    try:
        text_combined = " ".join(texts).lower()

        keyword_map = {
            "gown": ["gown", "surgical gown", "isolation gown"],
            "mask": ["mask", "n95", "respirator", "surgical mask", "kn95"],
            "gloves": ["glove", "gloves", "latex", "nitrile", "vinyl"],
            "bandage": ["bandage", "gauze", "dressing", "wound dressing"],
            "syringe": ["syringe", "needle", "luer", "ml"],
            "catheter": ["catheter", "foley", "urinary catheter"],
        }

        def has_label(lbl: str) -> bool:
            return any(str(d.get("label", "")).lower() == lbl for d in detections)

        for label, phrases in keyword_map.items():
            if any(p in text_combined for p in phrases):
                if not has_label(label):
                    detections.append({"label": label, "confidence": 0.95})
                else:
                    # boost existing
                    for d in detections:
                        if str(d.get("label", "")).lower() == label:
                            d["confidence"] = max(float(d.get("confidence", 0.0)), 0.95)
    except Exception as e:
        print("Hybrid logic error:", e)


# ---------------- Routes ----------------
@app.route("/", methods=["GET"])
def serve_index():
    try:
        return send_from_directory(app.static_folder, "index.html")
    except Exception:
        return "OK", 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "yolo": bool(model is not None), "ocr": bool(reader is not None)})


@app.route("/detect", methods=["POST"])
def detect():
    """
    Expect JSON: { "imageBase64": "<base64-encoded-image-or-dataurl>" }
    Returns JSON: { "detections": [...], "ocr_text": [...] }
    """
    try:
        payload = request.get_json(force=True, silent=True) or {}
        image_b64 = payload.get("imageBase64", "")

        if "," in image_b64:
            image_b64 = image_b64.split(",")[-1]

        if not image_b64:
            return jsonify({"error": "no imageBase64 provided", "detections": [], "ocr_text": []}), 400

        try:
            img_bytes = base64.b64decode(image_b64)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("cv2.imdecode returned None")
        except Exception as e:
            print("Image decode error:", e)
            return jsonify({"error": "invalid image payload", "detections": [], "ocr_text": []}), 400

        detections: List[Dict[str, Any]] = []

        # YOLO inference
        if model is not None:
            try:
                results = model(img_bgr, conf=YOLO_CONF, verbose=False)
                for r in results:
                    for box in getattr(r, "boxes", []) or []:
                        cls_idx = safe_cast_cls(box)
                        if cls_idx is None:
                            continue

                        # resolve label name
                        label = str(cls_idx)
                        try:
                            names = getattr(model, "names", None)
                            if isinstance(names, dict):
                                label = names.get(cls_idx, str(cls_idx))
                            elif isinstance(names, (list, tuple)) and 0 <= cls_idx < len(names):
                                label = names[cls_idx]
                        except Exception:
                            pass

                        confidence = float(getattr(box, "conf", 0.0))
                        detections.append({"label": str(label), "confidence": round(confidence, 2)})
            except Exception as ye:
                print("YOLO inference error:", ye)

        # OCR inference (improved)
        ocr_texts: List[str] = []
        if reader is not None:
            img_for_ocr = preprocess_for_ocr(img_bgr)
            ocr_texts = extract_text_easyocr(img_for_ocr)

        # Hybrid mapping: OCR text -> category label
        apply_ocr_keyword_mapping(ocr_texts, detections)

        return jsonify({"detections": detections, "ocr_text": ocr_texts}), 200

    except Exception as e:
        print("Top-level /detect error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e), "detections": [], "ocr_text": []}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)