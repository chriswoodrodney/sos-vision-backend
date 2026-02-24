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

# 1. Load YOLO model and FORCE CPU (essential for cloud builds)
model = YOLO("yolov8n.pt")
model.to('cpu') 

# 2. Load OCR model - disable GPU here too
reader = easyocr.Reader(['en'], gpu=False)

@app.route("/")
def serve():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json()
    image_base64 = data.get("imageBase64", "").split(",")[-1] # Robust base64 stripping

    if not image_base64:
        return jsonify({"detections": [], "ocr_text": []})

    try:
        # Decode image
        img_bytes = base64.b64decode(image_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # ---------------- YOLO DETECTION ----------------
        # Using persist=True can help if you're doing video frames later
        results = model(img, conf=0.3, verbose=False) 

        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "label": model.names[int(box.cls)],
                    "confidence": round(float(box.conf), 2)
                })

        # ---------------- OCR TEXT EXTRACTION ----------------
        ocr_results = reader.readtext(img)
        extracted_text = [text for (bbox, text, confidence) in ocr_results if confidence > 0.5]
        text_combined = " ".join(extracted_text).lower()

        # ---------------- HYBRID LOGIC ----------------
        # Keywords for medical supplies
        keywords = ["mask", "glove", "syringe", "bandage", "scalpel", "catheter"]
        
        for kw in keywords:
            if kw in text_combined:
                # Only add if it's not already detected by YOLO to avoid duplicates
                if not any(d['label'] == kw for d in detections):
                    detections.append({
                        "label": kw,
                        "confidence": 0.95
                    })

        return jsonify({
            "detections": detections,
            "ocr_text": extracted_text
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e), "detections": [], "ocr_text": []}), 500

if __name__ == "__main__":
    # Ensure port is handled correctly for Railway/Heroku/Render
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)