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

ALLOWED_ORIGINS = [
    "https://sos-frontend-woad-gamma.vercel.app",
    "https://sos-frontend-git-main-chriswoodrodney.vercel.app",
    "https://sos-frontend-ozxeq29u8-chriswoodrodney.vercel.app",
]

YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolov8n.pt")
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.35"))

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
        "ocr": False,
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

                print("RAW DETECTIONS:", detections)

            except Exception as e:
                print("YOLO inference error:", e)

        return jsonify({
            "detections": detections,
            "ocr_text": []
        }), 200

    except Exception as e:
        print("Top-level /detect error:", e)
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "detections": [],
            "ocr_text": []
        }), 500