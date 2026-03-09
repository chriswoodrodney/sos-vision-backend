# yolo_server.py
import os
import base64
import traceback
from typing import List, Dict, Any

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
import torch

from ultralytics import YOLO
import easyocr

ALLOWED_ORIGINS = [
    "https://sos-frontend-woad-gamma.vercel.app",
    "https://sos-frontend-git-main-chriswoodrodney.vercel.app",
    "https://sos-frontend-ozxeq29u8-chriswoodrodney.vercel.app",
]

YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolov8n.pt")
OCR_LANGS = os.environ.get("OCR_LANGS", "en").split(",")

YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.30"))
OCR_MIN_CONF = float(os.environ.get("OCR_MIN_CONF", "0.08"))
OCR_MAX_LINES = int(os.environ.get("OCR_MAX_LINES", "30"))

app = Flask(__name__, static_folder="build", static_url_path="")
CORS(app, resources={r"/detect": {"origins": ALLOWED_ORIGINS}})

print("NumPy version:", np.__version__)
print("OpenCV version:", cv2.__version__)
print("Torch version:", torch.__version__)
print("Torch CUDA available:", torch.cuda.is_available())


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
    OCR-friendly preprocessing:
    - BGR -> GRAY
    - upscale 2x
    - denoise
    - CLAHE contrast boost
    - adaptive threshold
    """
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.fastNlMeansDenoising(gray, h=10)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        return thresh
    except Exception as e:
        print("OCR preprocess error:", e)
        return img_bgr


def _clean_text(t: str) -> str:
    t = str(t or "").strip()
    return " ".join(t.split())


def _is_meaningful_text(text: str) -> bool:
    if not text:
        return False

    stripped = "".join(ch for ch in text if ch.isalnum())
    return len(stripped) >= 2


def extract_text_easyocr(img: np.ndarray) -> List[str]:
    if reader is None:
        return []

    out: List[str] = []
    try:
        results = reader.readtext(
            img,
            paragraph=False,
            detail=1,
            decoder="greedy",
        )

        for item in results:
            if not item or len(item) < 2:
                continue

            text = _clean_text(item[1])
            conf = float(item[2]) if len(item) > 2 and item[2] is not None else 0.0

            if not _is_meaningful_text(text):
                continue

            # keep more OCR output, even for lower-confidence live camera frames
            if conf >= OCR_MIN_CONF or len(text) >= 2:
                out.append(text)

        if len(out) > OCR_MAX_LINES:
            out = out[:OCR_MAX_LINES]

    except Exception as e:
        print("OCR inference error:", e)

    return out


def merge_text_lists(a: List[str], b: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []

    for t in (a or []) + (b or []):
        cleaned = _clean_text(t)
        key = cleaned.lower()

        if not cleaned or key in seen:
            continue

        seen.add(key)
        merged.append(cleaned)

    return merged


@app.route("/", methods=["GET"])
def serve_index():
    try:
        return send_from_directory(app.static_folder, "index.html")
    except Exception:
        return "OK", 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "yolo": bool(model is not None),
        "ocr": bool(reader is not None),
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "torch_version": torch.__version__,
    })


@app.route("/detect", methods=["POST"])
def detect():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        image_b64 = payload.get("imageBase64", "")

        if "," in image_b64:
            image_b64 = image_b64.split(",")[-1]

        if not image_b64:
            return jsonify({
                "error": "no imageBase64 provided",
                "detections": [],
                "ocr_text": []
            }), 400

        try:
            img_bytes = base64.b64decode(image_b64)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img_bgr is None:
                raise ValueError("cv2.imdecode returned None")
        except Exception as e:
            print("Image decode error:", e)
            return jsonify({
                "error": "invalid image payload",
                "detections": [],
                "ocr_text": []
            }), 400

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
                        detections.append({
                            "label": str(label),
                            "confidence": round(confidence, 2)
                        })

            except Exception as ye:
                print("YOLO inference error:", ye)

        # OCR inference: return raw readable text only
        ocr_texts: List[str] = []
        if reader is not None:
            try:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                ocr_raw = extract_text_easyocr(img_rgb)

                img_proc = preprocess_for_ocr(img_bgr)
                ocr_proc = extract_text_easyocr(img_proc)

                ocr_texts = merge_text_lists(ocr_raw, ocr_proc)

                print("RAW OCR:", ocr_texts)
                print("RAW DETECTIONS:", detections)

            except Exception as e:
                print("OCR inference error:", e)
                ocr_texts = []

        # response shape unchanged
        return jsonify({
            "detections": detections,
            "ocr_text": ocr_texts
        }), 200

    except Exception as e:
        print("Top-level /detect error:", e)
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "detections": [],
            "ocr_text": []
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)