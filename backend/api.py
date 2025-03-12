import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from flask import Flask, request, jsonify
import io
from PIL import Image
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define Fashion MNIST Model using ResNet18
class FashionMNIST_ResNet(nn.Module):
    def __init__(self):
        super(FashionMNIST_ResNet, self).__init__()
        self.model = models.resnet18(weights=None)  # No pre-trained weights
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Adjust for grayscale input
        self.model.fc = nn.Linear(512, 10)  # 10 classes

    def forward(self, x):
        return self.model(x)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionMNIST_ResNet().to(device)
try:
    model.load_state_dict(torch.load("./best_model.pth", map_location=device)) # change to ../best_model.pth if running explicitly
    model.eval()
    logging.info("Model loaded successfully!")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise SystemExit("Failed to load model. Check model path or format.")

# Define Flask App
app = Flask(__name__)

# Authentication Credentials (Modify for security)
USERNAME = "admin"
PASSWORD = "password"

dataMean = 0.2860
dataStd = 0.3530

# Preprocessing for Fashion MNIST images
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure 1 channel
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((dataMean,), (dataStd,))
])

# Load dataset to get class labels
try:
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    class_labels = test_dataset.classes
except Exception as e:
    logging.error(f"Error loading dataset: {str(e)}")
    raise SystemExit("Failed to load dataset.")

# Class Labels for Fashion MNIST
# class_labels = [
#     "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
#     "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
# ]

# Authentication Middleware
def check_auth(username, password):
    return username == USERNAME and password == PASSWORD

@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint for making predictions on Fashion MNIST images."""
    try:
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            logging.warning("Unauthorized access attempt")
            return jsonify({"error": "Unauthorized"}), 401

        if "file" not in request.files:
            logging.error("No file uploaded in request")
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        image = Image.open(io.BytesIO(file.read())).convert("L")  # Convert to grayscale
        image = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            output = model(image)
            predicted_class = torch.argmax(output, dim=1).item()

        logging.info(f"Prediction made: {class_labels[predicted_class]}")
        return jsonify({"class": class_labels[predicted_class]})

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

# Run Flask App
if __name__ == "__main__":
    logging.info("Starting Flask API on port 5000...")
    app.run(host="0.0.0.0", port=5000)
