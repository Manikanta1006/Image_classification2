import json
from pathlib import Path

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

from src.model import SimpleCNN


ROOT_DIR = Path(__file__).parent
MODEL_DIR = ROOT_DIR / "models"
CLASSES_PATH = MODEL_DIR / "classes.json"


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


@st.cache_resource
def load_model(checkpoint_path, num_classes):
    model = SimpleCNN.load_from_checkpoint(
        checkpoint_path,
        num_classes=num_classes,
        map_location="cpu",
    )
    model.eval()
    return model


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


st.set_page_config(page_title="Image Classifier")
st.title("Image Classifier")

classes = load_classes()
checkpoint_path = find_latest_checkpoint()

if classes is None or checkpoint_path is None:
    st.info("Train the model first, then refresh this app.")
    st.code("python src/train.py", language="bash")
    st.stop()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input image", use_container_width=True)

    model = load_model(str(checkpoint_path), len(classes))
    image_tensor = preprocess_image(image)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_index = torch.max(probabilities, dim=0)

    predicted_class = classes[predicted_index.item()]
    st.metric("Prediction", predicted_class)
    st.metric("Confidence", f"{confidence.item() * 100:.2f}%")

    st.subheader("Class probabilities")
    probability_data = {
        class_name: float(probabilities[index].item())
        for index, class_name in enumerate(classes)
    }
    st.bar_chart(probability_data)
