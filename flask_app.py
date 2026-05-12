import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms as transforms
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from src.model import SimpleCNN


ROOT_DIR = Path(__file__).parent
MODEL_DIR = ROOT_DIR / "models"
CLASSES_PATH = MODEL_DIR / "classes.json"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

app = Flask(__name__)
model_cache = {"model": None, "classes": None, "checkpoint": None}


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

    probability_list = [
        {
            "className": class_name,
            "probability": round(float(probabilities[index].item()), 6),
            "percent": round(float(probabilities[index].item()) * 100, 2),
        }
        for index, class_name in enumerate(classes)
    ]

    return jsonify(
        {
            "uploadedFilename": image_file.filename,
            "prediction": classes[predicted_index.item()],
            "confidence": round(float(confidence.item()), 6),
            "confidencePercent": round(float(confidence.item()) * 100, 2),
            "probabilities": probability_list,
            "checkpoint": checkpoint,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=True)
