import os
import torch
import albumentations
import numpy as np
from flask import Flask, request, render_template
from PIL import Image
from timm import create_model
from torch.nn import functional as F
import torch.nn as nn

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class SEResnext50_32x4d(nn.Module):
    def __init__(self, pretrained="imagenet"):
        super(SEResnext50_32x4d, self).__init__()
        self.base_model = create_model("seresnext50_32x4d", pretrained=True)
        self.l0 = nn.Linear(2048, 1)

    def forward(self, image):
        batch_size, _, _, _ = image.shape
        x = self.base_model.forward_features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        out = self.l0(x)
        return out

# Load your trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SEResnext50_32x4d(pretrained=None)
model.load_state_dict(torch.load("model_fold_0.bin", map_location=device))
model.to(device)
model.eval()

def preprocess_image(image_path):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    aug = albumentations.Compose(
        [albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)]
    )

    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))  # Resize to the input size expected by the model
    image = np.array(image)
    augmented = aug(image=image)
    image = augmented["image"]
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.tensor(image, dtype=torch.float).unsqueeze(0)
    return image

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            # Preprocess and predict
            image = preprocess_image(file_path)
            image = image.to(device)
            with torch.no_grad():
                predictions = model(image)
            prediction = torch.sigmoid(predictions).item()
            os.remove(file_path)

            # Interpret prediction
            result = "Melanoma Detected" if prediction > 0.5 else "No Melanoma Detected"
            return f"<h1>{result}</h1><p>Prediction Score: {prediction:.2f}</p>"

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
