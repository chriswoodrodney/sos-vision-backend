# yolo_server.py
# Full Flask app with /detect endpoint (YOLO + EasyOCR + CORS + robust error handling)

import os
import base64
import traceback
from typing import List, Dict, Any, Tuple

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
OCR_MIN_CONF = float(os.environ.get("OCR_MIN_CONF", "0.15"))  # LOWERED for live camera frames
OCR_MAX_LINES = int(os.environ.get("OCR_MAX_LINES", "30"))

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
    Safer OCR preprocessing for packaging:
    - upscale
    - denoise lightly
    - contrast boost (CLAHE)
    NOTE: We DO NOT always threshold, because threshold can destroy colored/faint text.
    """
    try:
        # Convert to gray
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Upscale (helps small text)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        # Light denoise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # CLAHE contrast boost
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        return gray
    except Exception:
        return img_bgr


def _clean_text(t: str) -> str:
    t = str(t or "").strip()
    t = " ".join(t.split())
    return t


def _looks_like_useful_text(t: str) -> bool:
    """
    Keep short but meaningful OCR outputs.
    """
    if not t:
        return False
    # must have at least 2 alnum chars
    alnum = sum(ch.isalnum() for ch in t)
    return alnum >= 2


def extract_text_easyocr(img_for_ocr: np.ndarray) -> List[str]:
    """
    Runs EasyOCR and returns cleaned text list.
    Filters by confidence, but allows lower confidence if text looks useful.
    """
    if reader is None:
        return []

    out: List[str] = []
    try:
        raw_results = reader.readtext(img_for_ocr)
        for item in raw_results:
            if not item or len(item) < 2:
                continue

            text = _clean_text(item[1])
            conf = float(item[2]) if len(item) > 2 and item[2] is not None else 0.0

            if not _looks_like_useful_text(text):
                continue

            # Keep high-confidence always; keep low-confidence if text seems informative
            if conf >= OCR_MIN_CONF or len(text) >= 6:
                out.append(text)

        # Limit lines to avoid huge payloads
        if len(out) > OCR_MAX_LINES:
            out = out[:OCR_MAX_LINES]

    except Exception as oe:
        print("OCR inference error:", oe)

    return out


def merge_text_lists(a: List[str], b: List[str]) -> List[str]:
    """
    Merge, de-dup (case-insensitive), preserve order.
    """
    seen = set()
    merged: List[str] = []
    for t in (a or []) + (b or []):
        key = _clean_text(t).lower()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        merged.append(_clean_text(t))
    return merged


def apply_ocr_keyword_mapping(texts: List[str], detections: List[Dict[str, Any]]) -> None:
    """
    OCR keyword -> add/boost a detection label.
    Also handles common OCR mistakes (0/O, 1/I, missing spaces).
    """
    try:
        raw = " ".join(texts).lower()

        # normalize common OCR confusions
        normalized = raw.replace("0", "o").replace("|", "i").replace("l", "l")
        normalized = normalized.replace("-", " ").replace("_", " ")
        normalized = " ".join(normalized.split())

        keyword_map = {
            "gown": ["gown", "isolation gown", "surgical gown"],
            "mask": ["mask", "n95", "kn95", "respirator", "surgical mask"],
            "gloves": ["glove", "gloves", "latex", "nitrile", "vinyl"],
            "bandage": ["bandage", "gauze", "dressing", "wound"],
            "syringe": ["syringe", "needle", "luer", "ml", "cc"],
            "catheter": ["catheter", "foley", "urinary"],
        }

        def has_label(lbl: str) -> bool:
            return any(str(d.get("label", "")).lower() == lbl for d in detections)

        for label, phrases in keyword_map.items():
            if any(p in normalized for p in phrases):
                if not has_label(label):
                    detections.append({"label": label, "confidence": 0.95})
                else:
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

        # OCR inference (DUAL PASS)
        ocr_texts: List[str] = []
        if reader is not None:
            try:
                # pass 1: original (often best for colored text)
                ocr_raw = extract_text_easyocr(img_bgr)

                # pass 2: preprocessed (often best for small/faint text)
                img_proc = preprocess_for_ocr(img_bgr)
                ocr_proc = extract_text_easyocr(img_proc)

                ocr_texts = merge_text_lists(ocr_raw, ocr_proc)
            except Exception as oe:
                print("OCR inference error:", oe)
                ocr_texts = []

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