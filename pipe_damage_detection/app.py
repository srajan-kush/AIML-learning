from flask import Flask, render_template, request
import torch
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import os

app = Flask(__name__)

os.makedirs("static", exist_ok=True)

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.eval()

classes = ["Damaged", "Undamaged"]

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET","POST"])
def index():
    prediction = None
    error = None
    image_path = None

    if request.method == "POST":
        if "file" not in request.files:
            error = "No file uploaded."
            return render_template("index.html", error=error)

        file = request.files["file"]

        if file.filename == "":
            error = "Please select an image."
            return render_template("index.html", error=error)

        if not allowed_file(file.filename):
            error = "Only PNG, JPG, JPEG, WEBP files are allowed."
            return render_template("index.html", error=error)

        try:
            filepath = os.path.join("static", file.filename)
            file.save(filepath)
            image_path = filepath

            image = Image.open(filepath).convert("RGB")
            image = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(image)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            prediction = classes[predicted.item()]
            confidence = round(confidence.item()*100,2)

            return render_template("index.html",
                                   prediction=prediction,
                                   confidence=confidence,
                                   image_path=image_path)

        except Exception as e:
            error = f"Prediction failed: {str(e)}"

    return render_template("index.html", error=error)

if __name__ == "__main__":
    app.run(debug=True)