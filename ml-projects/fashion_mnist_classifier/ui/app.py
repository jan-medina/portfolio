# ui/app.py

import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://localhost:8000"

st.set_page_config(page_title="FashionMNIST Classifier", layout="centered")

st.title("👕 FashionMNIST Classifier")
st.write("Upload an image (28x28 grayscale or RGB) and get the predicted clothing class.")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=200)

    if st.button("🔍 Predict"):
        with st.spinner("Classifying..."):
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(f"{API_URL}/predict", files={"file": uploaded_file})
            if response.status_code == 200:
                result = response.json()
                st.success(f"🧠 Predicted: **{result['predicted_class']}** (class index {result['class_index']})")
            else:
                st.error(f"Error: {response.status_code} - {response.json().get('detail')}")

st.markdown("---")
st.header("📊 Confusion Matrix")

if st.button("📈 Show Metrics"):
    response = requests.get(f"{API_URL}/metrics/plot")
    if response.status_code == 200:
        image = Image.open(io.BytesIO(response.content))
        st.image(image, caption="Normalized Confusion Matrix", use_container_width=True)
    else:
        st.error("Failed to fetch metrics image.")
