import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

import torch
import torchvision.transforms as transforms
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from ultralytics import YOLO

from src.model import SimpleCNN


ROOT_DIR = Path(__file__).parent
MODEL_DIR = ROOT_DIR / "models"
CLASSES_PATH = MODEL_DIR / "classes.json"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
DETECTION_MODEL_NAME = "yolov8n.pt"

app = Flask(__name__)
model_cache = {"model": None, "classes": None, "checkpoint": None}
detection_model_cache = {"model": None}


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def find_latest_checkpoint():
    checkpoints = list(MODEL_DIR.glob("lightning_logs/**/checkpoints/*.ckpt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def load_classes():
    if not CLASSES_PATH.exists():
        return None
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def format_label_name(class_name):
    return class_name.replace("_", " ").replace("-", " ").title()


def build_labels(classes):
    return [
        {
            "id": index,
            "name": class_name,
            "displayName": format_label_name(class_name),
        }
        for index, class_name in enumerate(classes)
    ]


def get_model():
    classes = load_classes()
    checkpoint = find_latest_checkpoint()
    if classes is None or checkpoint is None:
        return None, None, None

    if (
        model_cache["model"] is None
        or model_cache["checkpoint"] != str(checkpoint)
        or model_cache["classes"] != classes
    ):
        model = SimpleCNN.load_from_checkpoint(
            str(checkpoint),
            num_classes=len(classes),
            map_location="cpu",
        )
        model.eval()
        model_cache["model"] = model
        model_cache["classes"] = classes
        model_cache["checkpoint"] = str(checkpoint)

    return model_cache["model"], model_cache["classes"], model_cache["checkpoint"]


def preprocess_image(image):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transform(image).unsqueeze(0)


def is_allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_detection_model():
    if detection_model_cache["model"] is None:
        detection_model_cache["model"] = YOLO(DETECTION_MODEL_NAME)
    return detection_model_cache["model"]


def image_to_data_url(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def draw_detections(image, detections):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        label = f'{detection["label"]} {detection["confidencePercent"]}%'
        draw.rectangle((x1, y1, x2, y2), outline="#ffff00", width=4)
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        label_y = max(0, y1 - text_height - 8)
        draw.rectangle(
            (x1, label_y, x1 + text_width + 8, label_y + text_height + 6),
            fill="#ffff00",
        )
        draw.text((x1 + 4, label_y + 3), label, fill="#111111", font=font)

    return annotated


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    model, classes, checkpoint = get_model()
    return jsonify(
        {
            "ready": model is not None,
            "classes": classes or [],
            "labels": build_labels(classes or []),
            "checkpoint": checkpoint,
        }
    )


@app.post("/predict")
def predict():
    model, classes, checkpoint = get_model()
    if model is None:
        return (
            jsonify(
                {
                    "error": (
                        "Model not found. Train first with: python src/train.py"
                    )
                }
            ),
            400,
        )

    if "image" not in request.files:
        return jsonify({"error": "Please upload an image file."}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "Please choose an image file."}), 400

    if not is_allowed_file(image_file.filename):
        return jsonify({"error": "Use jpg, jpeg, png, or webp images only."}), 400

    try:
        image = Image.open(image_file.stream).convert("RGB")
    except UnidentifiedImageError:
        return jsonify({"error": "Could not read this image file."}), 400

    image_tensor = preprocess_image(image)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_index = torch.max(probabilities, dim=0)

    labels = build_labels(classes)
    predicted_label = labels[predicted_index.item()]

    probability_list = [
        {
            "labelId": labels[index]["id"],
            "className": class_name,
            "displayName": labels[index]["displayName"],
            "probability": round(float(probabilities[index].item()), 6),
            "percent": round(float(probabilities[index].item()) * 100, 2),
        }
        for index, class_name in enumerate(classes)
    ]

    return jsonify(
        {
            "uploadedFilename": image_file.filename,
            "prediction": classes[predicted_index.item()],
            "predictedLabel": predicted_label,
            "confidence": round(float(confidence.item()), 6),
            "confidencePercent": round(float(confidence.item()) * 100, 2),
            "probabilities": probability_list,
            "checkpoint": checkpoint,
        }
    )


@app.post("/detect")
def detect():
    if "image" not in request.files:
        return jsonify({"error": "Please upload an image file."}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "Please choose an image file."}), 400

    if not is_allowed_file(image_file.filename):
        return jsonify({"error": "Use jpg, jpeg, png, or webp images only."}), 400

    try:
        image = Image.open(image_file.stream).convert("RGB")
    except UnidentifiedImageError:
        return jsonify({"error": "Could not read this image file."}), 400

    detection_model = get_detection_model()
    result = detection_model.predict(image, conf=0.25, verbose=False)[0]
    names = result.names
    detections = []

    for box in result.boxes:
        x1, y1, x2, y2 = [round(float(value), 2) for value in box.xyxy[0].tolist()]
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        detections.append(
            {
                "labelId": class_id,
                "label": names[class_id],
                "confidence": round(confidence, 6),
                "confidencePercent": round(confidence * 100, 2),
                "box": [x1, y1, x2, y2],
            }
        )

    annotated_image = draw_detections(image, detections)

    return jsonify(
        {
            "uploadedFilename": image_file.filename,
            "detectionCount": len(detections),
            "detections": detections,
            "annotatedImage": image_to_data_url(annotated_image),
            "model": DETECTION_MODEL_NAME,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)