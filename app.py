
import streamlit as st
import joblib
import pickle
import numpy as np
import cv2
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="DiabeticRetina AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #07111f 0%, #0b1726 50%, #102435 100%);
    color: #eaf4ff;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

[data-testid="stSidebar"] {
    background-color: #0d1b2a;
}

.hero {
    background: linear-gradient(120deg, #112b3c, #123e4c);
    padding: 35px;
    border-radius: 22px;
    border: 1px solid rgba(45, 212, 191, 0.25);
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    color: #ffffff;
    margin-bottom: 8px;
}

.hero p {
    color: #b9d4e5;
    font-size: 17px;
}

.tag {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 30px;
    background: rgba(45, 212, 191, 0.12);
    color: #5eead4;
    border: 1px solid rgba(45, 212, 191, 0.25);
    font-size: 13px;
    margin-bottom: 15px;
}

.panel {
    background: #102131;
    padding: 24px;
    border-radius: 18px;
    border: 1px solid #20384b;
    margin-bottom: 20px;
}

.result-card {
    background: linear-gradient(135deg, #143448, #102b3a);
    padding: 30px;
    border-radius: 22px;
    border: 1px solid rgba(45, 212, 191, 0.35);
    text-align: center;
}

.result-number {
    font-size: 70px;
    font-weight: 800;
    color: #5eead4;
    margin: 0;
}

.result-label {
    font-size: 23px;
    font-weight: 700;
    color: white;
}

.result-sub {
    color: #b9d4e5;
    font-size: 15px;
}

.footer {
    text-align: center;
    padding: 25px;
    color: #91a9bc;
    font-size: 13px;
}

div[data-testid="stFileUploader"] {
    background: #102131;
    border-radius: 15px;
    padding: 15px;
    border: 1px dashed #3a6077;
}

.stButton > button {
    background: linear-gradient(90deg, #14b8a6, #0d9488);
    color: white;
    border: none;
    border-radius: 12px;
    height: 50px;
    font-size: 17px;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover {
    background: #0f766e;
    color: white;
    border: 1px solid #5eead4;
}

div[data-testid="stMetric"] {
    background: #102131;
    padding: 15px;
    border-radius: 15px;
    border: 1px solid #20384b;
}

hr {
    border-color: #20384b;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================
st.markdown("""
<div class="hero">
    <div class="tag">AI-POWERED RETINAL SCREENING</div>
    <h1>🩺 DiabeticRetina AI</h1>
    <p>
        Retinal Image-Based Diabetic Retinopathy Classification
    </p>
    <p>
        An experimental machine learning screening prototype
        designed to analyze retinal fundus images.
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# DR CLASS INFORMATION
# ==========================================
class_names = {
    0: "No Diabetic Retinopathy",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

class_descriptions = {
    0: "No diabetic retinopathy detected by the model.",
    1: "The model classified this image as mild DR.",
    2: "The model classified this image as moderate DR.",
    3: "The model classified this image as severe DR.",
    4: "The model classified this image as proliferative DR."
}

class_colors = {
    0: "#34d399",
    1: "#facc15",
    2: "#fb923c",
    3: "#f87171",
    4: "#e879f9"
}

# ==========================================
# MODEL LOADING
# ==========================================
@st.cache_resource
def load_model():

    try:
        return joblib.load("best_svm_pipeline.pkl")

    except Exception:

        with open("best_svm_pipeline.pkl", "rb") as file:
            return pickle.load(file)

try:
    model = load_model()
    model_loaded = True

except Exception as e:

    model = None
    model_loaded = False

# ==========================================
# FEATURE EXTRACTION
# ==========================================
def extract_features(pil_image):

    image = np.array(pil_image.convert("RGB"))

    image = cv2.resize(image, (224, 224))

    # RGB features
    red_mean = np.mean(image[:, :, 0])
    red_std = np.std(image[:, :, 0])

    green_mean = np.mean(image[:, :, 1])
    green_std = np.std(image[:, :, 1])

    blue_mean = np.mean(image[:, :, 2])
    blue_std = np.std(image[:, :, 2])

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Quantization
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

    contrast = graycoprops(glcm, "contrast")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]

    # Brightness
    brightness_mean = np.mean(gray)
    brightness_std = np.std(gray)

    brightness_range = (
        np.percentile(gray, 95) -
        np.percentile(gray, 5)
    )

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

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:

    st.markdown("## 🩺 DiabeticRetina AI")

    st.markdown("---")

    st.markdown("### About")

    st.write(
        "This experimental AI application uses "
        "machine learning to classify retinal images "
        "into five diabetic retinopathy severity classes."
    )

    st.markdown("---")

    st.markdown("### Classification")

    for i in range(5):

        st.markdown(
            f"**Class {i}:** {class_names[i]}"
        )

    st.markdown("---")

    st.caption("Prototype | Academic AI Project")

# ==========================================
# MODEL STATUS
# ==========================================
if model_loaded:

    st.success("AI Model loaded successfully.")

else:

    st.error(
        "Model could not be loaded. "
        "Ensure best_svm_pipeline.pkl is in the app folder."
    )

# ==========================================
# MAIN CONTENT
# ==========================================
col1, col2 = st.columns([1, 1])

with col1:

    st.markdown("### 📤 Upload Retinal Image")

    st.write(
        "Upload a retinal fundus photograph "
        "to perform an experimental classification."
    )

    uploaded_file = st.file_uploader(
        "Choose retinal image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG"
    )

with col2:

    st.markdown("### 🔬 Analysis Overview")

    st.markdown("""
    - **Input:** Retinal fundus image
    - **Processing:** Image resizing and feature extraction
    - **Features:** 13 numerical features
    - **Classifier:** Support Vector Machine (SVM)
    - **Output:** DR severity class (0–4)
    """)

# ==========================================
# IMAGE PREVIEW
# ==========================================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("---")

    left, right = st.columns([1, 1])

    with left:

        st.markdown("### 🖼️ Retinal Image")

        st.image(
            image,
            use_container_width=True
        )

    with right:

        st.markdown("### 📋 Image Information")

        st.metric("Image Width", f"{image.width} px")

        st.metric("Image Height", f"{image.height} px")

        st.metric("Image Format", uploaded_file.type)

    st.markdown("---")

    if st.button(
        "🔍 Analyze Retinal Image",
        type="primary",
        use_container_width=True
    ):

        if not model_loaded:

            st.error("Model is not available.")

        else:

            with st.spinner(
                "Extracting features and analyzing image..."
            ):

                try:

                    features = extract_features(image)

                    prediction = model.predict(features)

                    predicted_class = int(prediction[0])

                    # Validate predicted class
                    if predicted_class not in class_names:

                        raise ValueError(
                            "Model returned an unknown DR class."
                        )

                    st.markdown("## 🧠 AI Analysis Result")

                    color = class_colors[predicted_class]

                    st.markdown(
                        f"""
                        <div class="result-card">
                            <p class="result-sub">
                                Predicted DR Severity
                            </p>
                            <p class="result-number"
                               style="color:{color}">
                                Class {predicted_class}
                            </p>
                            <p class="result-label">
                                {class_names[predicted_class]}
                            </p>
                            <p class="result-sub">
                                {class_descriptions[predicted_class]}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown("")

                    # Confidence, if supported
                    if hasattr(model, "predict_proba"):

                        probabilities = model.predict_proba(features)[0]

                        model_classes = model.classes_

                        class_index = np.where(
                            model_classes == predicted_class
                        )[0][0]

                        confidence = probabilities[class_index] * 100

                        st.metric(
                            "Model Probability for Predicted Class",
                            f"{confidence:.2f}%"
                        )

                        st.caption(
                            "This is the model's probability estimate, "
                            "not a clinically validated diagnostic certainty."
                        )

                    else:

                        st.info(
                            "This model does not expose probability "
                            "estimates. Prediction is shown without confidence."
                        )

                    # All class probabilities if available
                    if hasattr(model, "predict_proba"):

                        st.markdown("### 📊 Class Probability Estimates")

                        probability_data = {
                            class_names[int(cls)]: float(prob) * 100
                            for cls, prob in zip(
                                model_classes,
                                probabilities
                            )
                        }

                        st.bar_chart(probability_data)

                    # Severity guide
                    st.markdown("### 📚 DR Severity Reference")

                    for i in range(5):

                        st.markdown(
                            f"""
                            <div class="panel">
                                <b style="color:{class_colors[i]}">
                                    Class {i} — {class_names[i]}
                                </b>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.markdown("### ⚠️ Medical Disclaimer")

                    st.warning(
                        "This application is an experimental academic "
                        "prototype. It is not a medical device and "
                        "cannot diagnose diabetic retinopathy. "
                        "Its predictions are not clinically validated. "
                        "Consult a qualified eye-care professional "
                        "for examination and diagnosis."
                    )

                except Exception as e:

                    st.error(
                        f"Image analysis failed: {str(e)}"
                    )

# ==========================================
# FOOTER
# ==========================================
st.markdown("""
<div class="footer">
    <hr>
    <b>DiabeticRetina AI</b><br>
    Retinal Image-Based Diabetic Retinopathy Classification<br>
    Academic Machine Learning Prototype | Not for clinical use
</div>
""", unsafe_allow_html=True)
