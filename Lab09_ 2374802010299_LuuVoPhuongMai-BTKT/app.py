import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import torchvision.transforms as transforms

from models import CatDog_CNN, CIFAR10_CNN, PlantVillage_CNN

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Labels =====
CATDOG_LABELS = ["cat", "dog"]

CIFAR10_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

with open("plantvillage_classes.json", "r", encoding="utf-8") as f:
    PLANT_LABELS = json.load(f)

# ===== Load models =====
catdog_model = CatDog_CNN().to(device)
catdog_model.load_state_dict(torch.load("weights/best_catdog_cnn.pt", map_location=device))
catdog_model.eval()

cifar_model = CIFAR10_CNN().to(device)
cifar_model.load_state_dict(torch.load("weights/best_cifar10_cnn.pt", map_location=device))
cifar_model.eval()

plant_model = PlantVillage_CNN(num_classes=len(PLANT_LABELS)).to(device)
plant_model.load_state_dict(torch.load("weights/best_plantvillage_cnn.pt", map_location=device))
plant_model.eval()

# ===== Transform cho từng model =====
MODEL_CONFIGS = {
    "catdog": {
        "model": catdog_model,
        "labels": CATDOG_LABELS,
        "transform": transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    },
    "cifar10": {
        "model": cifar_model,
        "labels": CIFAR10_LABELS,
        "transform": transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ])
    },
    "plantvillage": {
        "model": plant_model,
        "labels": PLANT_LABELS,
        "transform": transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }
}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image_path, model_key):
    config = MODEL_CONFIGS[model_key]
    model = config["model"]
    labels = config["labels"]
    transform = config["transform"]

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_idx].item() * 100

    predicted_label = labels[predicted_idx]
    return predicted_label, confidence


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None
    selected_model = "catdog"
    error = None

    if request.method == "POST":
        selected_model = request.form.get("model_type", "catdog")
        file = request.files.get("image")

        if file is None or file.filename == "":
            error = "Bồ chưa chọn ảnh."
        elif not allowed_file(file.filename):
            error = "Định dạng ảnh không hợp lệ."
        else:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            try:
                prediction, confidence = predict_image(filepath, selected_model)
                image_path = filepath.replace("\\", "/")
            except Exception as e:
                error = f"Lỗi dự đoán: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        selected_model=selected_model,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)