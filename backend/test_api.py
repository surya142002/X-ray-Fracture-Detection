import requests

# ✅ Replace with your actual test X-ray image path
image_path = "/Users/surya/Desktop/X_ray_Detection/preprocessing/preprocessed_dataset/val/fractured/images6.jpg"  # Change to a real X-ray file

# ✅ Open the image file and send it to the backend API
with open(image_path, "rb") as img:
    response = requests.post("http://127.0.0.1:5001/predict", files={"file": img})

# ✅ Print the response (Prediction)
if response.status_code == 200:
    print("✅ API Response:", response.json())  # Expected: "Fracture Detected" or "No Fracture"
else:
    print("❌ Error:", response.text)
