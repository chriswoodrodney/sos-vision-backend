from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import easyocr
import base64
import numpy as np
import cv2
import os

app = Flask(__name__, static_folder="build", static_url_path="")
CORS(app)

# Load YOLO model
model = YOLO("yolov8n.pt")

# Load OCR model (loads once at startup)
reader = easyocr.Reader(['en'])

@app.route("/")
def serve():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json()
    image_base64 = data.get("imageBase64")

    if not image_base64:
        return jsonify({"detections": [], "ocr_text": []})

    try:
        # Decode image
        img_bytes = base64.b64decode(image_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # ---------------- YOLO DETECTION ----------------
        results = model(img, conf=0.3)

        detections = []

        for r in results:
            for box in r.boxes:
                detections.append({
                    "label": model.names[int(box.cls)],
                    "confidence": float(box.conf)
                })

        # ---------------- OCR TEXT EXTRACTION ----------------
        ocr_results = reader.readtext(img)

        extracted_text = []
        for (bbox, text, confidence) in ocr_results:
            if confidence > 0.5:
                extracted_text.append(text)

        # ---------------- HYBRID LOGIC ----------------
        text_combined = " ".join(extracted_text).lower()

        # Override using keywords
        if "mask" in text_combined:
            detections.append({
                "label": "mask",
                "confidence": 0.99
            })

        if "glove" in text_combined:
            detections.append({
                "label": "gloves",
                "confidence": 0.99
            })

        if "syringe" in text_combined:
            detections.append({
                "label": "syringe",
                "confidence": 0.99
            })

        if "bandage" in text_combined:
            detections.append({
                "label": "bandage",
                "confidence": 0.99
            })

        return jsonify({
            "detections": detections,
            "ocr_text": extracted_text
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"detections": [], "ocr_text": []})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
app.run(host="0.0.0.0", port=port)

