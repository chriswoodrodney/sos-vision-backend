# yolo-server.py
# Full Flask app with /detect endpoint (YOLO + EasyOCR + CORS + robust error handling)

import os
import base64
import io
import traceback
from typing import List

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageOps, ImageFilter
import numpy as np

# YOLO/EasyOCR imports
from ultralytics import YOLO
import easyocr
import cv2

# ---------------- Config ----------------
# Adjust these to your Vercel preview / production domains if desired.
ALLOWED_ORIGINS = [
    "https://sos-frontend-woad-gamma.vercel.app",
    "https://sos-frontend-git-main-chriswoodrodney.vercel.app",
    "https://sos-frontend-ozxeq29u8-chriswoodrodney.vercel.app",
    "https://sos-frontend-git-main-chriswoodrodney.vercel.app"
]

# If you want to allow all origins while debugging, replace the above with:
# ALLOWED_ORIGINS = ["*"]

# Path to YOLO weights you included in the project or container
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolov8n.pt")

# OCR config
OCR_LANGS = os.environ.get("OCR_LANGS", "en").split(",")  # e.g. "en" or "en,fr"

# ---------------- App init ----------------
app = Flask(__name__, static_folder="build", static_url_path="")
CORS(app, resources={r"/detect": {"origins": ALLOWED_ORIGINS}})

# ---------------- Model loading ----------------
# Load YOLO model and force CPU to avoid GPU requirements in the cloud
try:
    model = YOLO(YOLO_WEIGHTS)
    try:
        # Move to CPU explicitly
        model.to("cpu")
    except Exception as e:
        print("Warning: couldn't set model device explicitly:", e)
    print("YOLO model loaded.")
except Exception as e:
    model = None
    print("Failed loading YOLO model:", e)
    traceback.print_exc()

# Load EasyOCR reader once (CPU). If Railway offers GPU and you want it, set gpu=True.
try:
    reader = easyocr.Reader(OCR_LANGS, gpu=False)
    print("EasyOCR reader initialized.")
except Exception as e:
    reader = None
    print("Failed to initialize EasyOCR:", e)
    traceback.print_exc()

# ---------------- Helpers ----------------
def preprocess_for_ocr(img_rgb: np.ndarray) -> np.ndarray:
    """
    Accepts an RGB numpy array (H,W,3) or BGR and returns a numpy array suitable for EasyOCR.
    Steps: convert to PIL, grayscale, autocontrast, optional upscale, median denoise, return RGB array.
    """
    try:
        pil = Image.fromarray(img_rgb)
        # convert to L (grayscale) then autocontrast for better OCR
        proc = pil.convert("L")
        proc = ImageOps.autocontrast(proc)
        w, h = proc.size
        scale = max(1, 1024 // max(w, h))
        if scale > 1:
            proc = proc.resize((w * scale, h * scale), Image.LANCZOS)
        proc = proc.filter(ImageFilter.MedianFilter(size=3))
        # EasyOCR can work with RGB or grayscale; return RGB to be safe
        return np.array(proc.convert("RGB"))
    except Exception:
        # fallback: return original array
        return img_rgb

def safe_cast_cls(box):
    """
    Extract integer class index from YOLO box representation.
    """
    try:
        # some ultralytics versions store box.cls as a scalar-like
        cls_val = getattr(box, "cls", None)
        if cls_val is None:
            return None
        # try direct int cast
        try:
            return int(cls_val)
        except Exception:
            # try converting via numpy
            return int(np.array(cls_val).item())
    except Exception:
        return None

# ---------------- Routes ----------------
@app.route("/", methods=["GET"])
def serve_index():
    # Serve static frontend (if build exists)
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
        payload = request.get_json(force=True)
        image_b64 = payload.get("imageBase64", "")
        # allow data URLs or plain base64
        if "," in image_b64:
            image_b64 = image_b64.split(",")[-1]

        if not image_b64:
            return jsonify({"error": "no imageBase64 provided", "detections": [], "ocr_text": []}), 400

        # decode image bytes -> cv2 BGR
        try:
            img_bytes = base64.b64decode(image_b64)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("cv2.imdecode returned None")
        except Exception as e:
            print("Image decode error:", e)
            return jsonify({"error": "invalid image payload", "detections": [], "ocr_text": []}), 400

        detections = []
        # YOLO inference (defensive)
        if model is not None:
            try:
                # ultralytics accepts numpy images directly
                results = model(img_bgr, conf=0.3, verbose=False)
                for r in results:
                    # r.boxes container
                    try:
                        for box in getattr(r, "boxes", []):
                            cls_idx = safe_cast_cls(box)
                            if cls_idx is None:
                                continue
                            # model.names might be list or dict; handle both
                            label = None
                            try:
                                names = getattr(model, "names", None)
                                if isinstance(names, dict):
                                    label = names.get(cls_idx, str(cls_idx))
                                elif isinstance(names, (list, tuple)):
                                    label = names[cls_idx] if 0 <= cls_idx < len(names) else str(cls_idx)
                                else:
                                    label = str(cls_idx)
                            except Exception:
                                label = str(cls_idx)
                            confidence = float(getattr(box, "conf", 0.0))
                            detections.append({"label": str(label), "confidence": round(confidence, 2)})
                    except Exception as inner:
                        print("Per-result parsing error:", inner)
            except Exception as ye:
                print("YOLO inference error:", ye)
        else:
            print("YOLO model not initialized; skipping detection")

        # OCR inference
        ocr_texts: List[str] = []
        if reader is not None:
            try:
                # convert BGR -> RGB for PIL / easyocr
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                proc = preprocess_for_ocr(img_rgb)
                raw_results = reader.readtext(proc)
                for item in raw_results:
                    if not item or len(item) < 2:
                        continue
                    text = item[1].strip()
                    conf = item[2] if len(item) > 2 else None
                    # optional confidence filter (skip very low)
                    if text:
                        ocr_texts.append(text)
            except Exception as oe:
                print("OCR inference error:", oe)
        else:
            print("EasyOCR reader not initialized; skipping OCR")

        # HYBRID LOGIC: if OCR found certain keywords, add or boost detections
        try:
            text_combined = " ".join(ocr_texts).lower()
            keywords = ["mask", "glove", "syringe", "bandage", "scalpel", "catheter", "gown"]
            for kw in keywords:
                if kw in text_combined:
                    if not any(d["label"].lower() == kw for d in detections):
                        detections.append({"label": kw, "confidence": 0.95})
        except Exception as hybrid_err:
            print("Hybrid logic error:", hybrid_err)

        return jsonify({"detections": detections, "ocr_text": ocr_texts}), 200

    except Exception as e:
        print("Top-level /detect error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e), "detections": [], "ocr_text": []}), 500

# ---------------- Run ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)