from flask import Flask, request, jsonify
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import io

# ✅ Initialize Flask App
app = Flask(__name__)

# ✅ Load the trained ResNet-50 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=None)  # Load ResNet-50 architecture
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),  # Keep dropout from training
    torch.nn.Linear(2048, 7)  # ✅ Adjust output to 7 classes
)

# ✅ Load trained weights
checkpoint_path = "../models/best_model_final_optimized.pth"  # Adjust path if needed
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()  # ✅ Set to evaluation mode

print("✅ Model Loaded Successfully!")

# ✅ Define the image transform (must match training preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ✅ Match training size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ✅ Keep consistent with training
])

@app.route("/predict", methods=["POST"])
def predict():
    """Receives an X-ray image from the frontend, runs inference, and returns a prediction with confidence score."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")  # Open image

    image = transform(image).unsqueeze(0).to(device)  # Apply transforms
    with torch.no_grad():
        output = model(image)  # Get model output (logits)
        probabilities = F.softmax(output, dim=1)  # Convert logits to probabilities

        # ✅ Get prediction and confidence
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = float(probabilities[0][prediction].cpu().numpy())  

    # ✅ Map numerical class to fracture type
    class_mapping = {
        0: "Avulsion Fracture",
        1: "Comminuted Fracture",
        2: "Fracture Dislocation",
        3: "Curved Fractures",
        4: "Linear Fractures",
        5: "Internal Fractures",
        6: "Hairline Fracture"
    }
    result = {
        "prediction": class_mapping[prediction],
        "confidence": round(confidence * 100, 2)  # Convert confidence to percentage
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)  # ✅ Runs on localhost:5001
