### **Updated Frontend `app.py` with Fracture Type Descriptions**
import streamlit as st
import requests
import io
from PIL import Image

# ✅ Title of the Web App
st.title("🦴 X-Ray Fracture Classification AI")
st.write("Upload an X-ray image, and the model will classify the type of fracture.")
st.write("Only 55 percent accurate right now, but it's a start!")

# ✅ Display classification explanations
st.subheader("🔍 Possible Fracture Classifications:")
fracture_info = {
    "Avulsion Fracture": "A fragment of bone is pulled off by a tendon or ligament.",
    "Comminuted Fracture": "The bone is broken into multiple pieces.",
    "Fracture Dislocation": "A joint is dislocated along with a fracture.",
    "Curved Fractures": "Includes spiral and oblique fractures caused by twisting forces.",
    "Linear Fractures": "A simple break along the length of the bone.",
    "Internal Fractures": "Fractures occurring within the bone, such as impacted or pathological fractures.",
    "Hairline Fracture": "A thin crack in the bone that may not fully break it apart."
}
for fracture, description in fracture_info.items():
    st.markdown(f"**{fracture}**: {description}")

# ✅ Upload an X-ray
uploaded_file = st.file_uploader("Choose an X-ray...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # ✅ Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    # ✅ Send image to Flask API
    with st.spinner("Analyzing X-ray..."):
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="JPEG")
        
        api_url = "http://127.0.0.1:5001/predict"  # ✅ Ensure backend is running on this port
        try:
            response = requests.post(api_url, files={"file": ("image.jpg", image_bytes.getvalue(), "image/jpeg")})

            # ✅ Get the response from Flask API
            if response.status_code == 200:
                result = response.json()
                st.success(f"**Prediction: {result['prediction']}**")
                st.info(f"Confidence: {result['confidence']}%")
            else:
                st.error("⚠️ Server error! Please try again.")

        except requests.exceptions.ConnectionError:
            st.error("⚠️ Could not connect to the backend! Make sure the Flask API is running.")
