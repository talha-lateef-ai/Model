import streamlit as st
import joblib
import pickle
import numpy as np
import cv2
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

# Page setup
st.set_page_config(page_title="Image Classification App", page_icon="🖼️")
st.title("🖼️ Image Classification App")
st.write("Apni image upload karein — model khud features extract kar ke prediction dega:")

# Model load karein
@st.cache_resource
def load_model():
    try:
        return joblib.load("model.pkl")
    except Exception:
        with open("model.pkl", "rb") as file:
            return pickle.load(file)

model = load_model()

# Exact feature extraction function (same as training)
def extract_features(pil_image):
    # Convert PIL image to RGB numpy array
    image = np.array(pil_image.convert("RGB"))

    # Resize image
    image = cv2.resize(image, (224, 224))

    # RGB color features
    red_mean = np.mean(image[:, :, 0])
    red_std = np.std(image[:, :, 0])

    green_mean = np.mean(image[:, :, 1])
    green_std = np.std(image[:, :, 1])

    blue_mean = np.mean(image[:, :, 2])
    blue_std = np.std(image[:, :, 2])

    # Convert RGB to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Quantize grayscale to 8 levels
    gray_quantized = (gray // 32).astype(np.uint8)

    # GLCM
    glcm = graycomatrix(
        gray_quantized,
        distances=[1],
        angles=[0],
        levels=8,
        symmetric=True,
        normed=True
    )

    # Texture features
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    # Brightness features
    brightness_mean = np.mean(gray)
    brightness_std = np.std(gray)
    brightness_range = np.percentile(gray, 95) - np.percentile(gray, 5)

    # Final 13 features vector
    features = [
        red_mean,
        red_std,
        green_mean,
        green_std,
        blue_mean,
        blue_std,
        contrast,
        homogeneity,
        energy,
        correlation,
        brightness_mean,
        brightness_std,
        brightness_range
    ]

    return np.array([features])

# --- IMAGE UPLOADER ---
uploaded_file = st.file_uploader("Photo choose karein (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict", type="primary"):
        with st.spinner("Image analyze ho rahi hai..."):
            try:
                features = extract_features(image)
                prediction = model.predict(features)
                st.success(f"### Prediction Result: {prediction[0]}")
            except Exception as e:
                st.error(f"Error: {e}")
