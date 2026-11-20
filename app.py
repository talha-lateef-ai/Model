
import streamlit as st
import numpy as np
import cv2
import joblib
import pickle
import pandas as pd

from pathlib import Path
from PIL import Image
from skimage.feature import graycomatrix, graycoprops


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="DiabeticRetina AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #07111f 0%, #0b1730 55%, #10102a 100%);
    color: #f1f5f9;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

[data-testid="stSidebar"] {
    background: #0b1426;
    border-right: 1px solid #1e3150;
}

[data-testid="stSidebar"] * {
    color: #e2e8f0;
}

.hero {
    background: linear-gradient(110deg, #142c50, #202052);
    border: 1px solid #29466e;
    border-radius: 22px;
    padding: 30px;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 38px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 8px;
}

.hero p {
    color: #b7c9e5;
    font-size: 15px;
}

.section-title {
    color: #ffffff;
    font-size: 22px;
    font-weight: 700;
    margin: 20px 0 15px 0;
}

.metric-card {
    background: #111f36;
    border: 1px solid #263e60;
    border-radius: 16px;
    padding: 20px;
    min-height: 115px;
}

.metric-label {
    color: #91a9c9;
    font-size: 13px;
    margin-bottom: 10px;
}

.metric-value {
    color: #ffffff;
    font-size: 25px;
    font-weight: 800;
}

.result-card {
    background: linear-gradient(120deg, #142b47, #1c2250);
    border: 1px solid #3b5e91;
    border-radius: 20px;
    padding: 25px;
    margin-top: 15px;
}

.result-grade {
    font-size: 42px;
    font-weight: 800;
    color: #70d6ff;
}

.result-name {
    font-size: 23px;
    font-weight: 700;
    color: white;
}

.info-card {
    background: #111f36;
    border: 1px solid #263e60;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 15px;
}

.info-card h3 {
    color: #76cfff;
    font-size: 18px;
}

.info-card p {
    color: #b6c5da;
    font-size: 14px;
    line-height: 1.7;
}

div.stButton > button {
    width: 100%;
    border: none;
    border-radius: 12px;
    height: 50px;
    font-weight: 700;
    color: white;
    background: linear-gradient(90deg, #1677e8, #7555e8);
    transition: 0.3s;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #278cff, #8b6aff);
    border: none;
    color: white;
    transform: translateY(-2px);
}

.stFileUploader {
    background: #101e34;
    border: 1px dashed #41648e;
    border-radius: 15px;
    padding: 15px;
}

.footer {
    text-align: center;
    color: #7186a5;
    font-size: 12px;
    padding: 25px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# MODEL LOADING
# ==========================================

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "model.pkl file not found. Place it in the same folder as app.py."
        )

    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        with open(MODEL_PATH, "rb") as file:
            return pickle.load(file)


try:
    model = load_model()
    model_loaded = True
except Exception as error:
    model = None
    model_loaded = False
    st.error(f"Model loading error: {error}")


# ==========================================
# DR CLASS INFORMATION
# ==========================================

DR_CLASSES = {
    0: {
        "name": "No Diabetic Retinopathy",
        "severity": "No DR",
        "description": "No visible signs of diabetic retinopathy in the predicted class.",
        "color": "#35d39a"
    },
    1: {
        "name": "Mild Non-Proliferative DR",
        "severity": "Mild",
        "description": "The predicted class indicates mild retinal changes.",
        "color": "#80d8ff"
    },
    2: {
        "name": "Moderate Non-Proliferative DR",
        "severity": "Moderate",
        "description": "The predicted class indicates moderate diabetic retinal changes.",
        "color": "#f5c451"
    },
    3: {
        "name": "Severe Non-Proliferative DR",
        "severity": "Severe",
        "description": "The predicted class indicates severe diabetic retinal changes.",
        "color": "#ff9b54"
    },
    4: {
        "name": "Proliferative Diabetic Retinopathy",
        "severity": "Proliferative",
        "description": "The predicted class indicates proliferative diabetic retinopathy.",
        "color": "#ff657a"
    }
}


# ==========================================
# FEATURE EXTRACTION
# Must match training preprocessing exactly
# ==========================================

def extract_features(pil_image):

    image = np.array(pil_image.convert("RGB"))
    image = cv2.resize(image, (224, 224))

    red_mean = np.mean(image[:, :, 0])
    red_std = np.std(image[:, :, 0])

    green_mean = np.mean(image[:, :, 1])
    green_std = np.std(image[:, :, 1])

    blue_mean = np.mean(image[:, :, 2])
    blue_std = np.std(image[:, :, 2])

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    gray_quantized = (gray // 32).astype(np.uint8)

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

    brightness_mean = np.mean(gray)
    brightness_std = np.std(gray)

    brightness_range = (
        np.percentile(gray, 95) - np.percentile(gray, 5)
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

    st.markdown("## 👁️ DiabeticRetina AI")
    st.caption("AI-powered retinal screening prototype")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "🔬 DR Classification",
            "📊 DR Severity Guide",
            "ℹ️ About Project"
        ]
    )

    st.divider()

    if model_loaded:
        st.success("Model Loaded Successfully")
    else:
        st.error("Model Not Available")

    st.markdown("---")

    st.caption("Developed as an academic AI/ML project.")
    st.caption("Version 1.0")


# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <h1>DiabeticRetina AI 👁️</h1>
    <p>Retinal Image-Based Diabetic Retinopathy Classification</p>
    <p>AI-assisted screening prototype using Machine Learning.</p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# DASHBOARD
# ==========================================

if page == "🏠 Dashboard":

    st.markdown('<div class="section-title">System Overview</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Classification Classes</div>
            <div class="metric-value">05 Grades</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Model Type</div>
            <div class="metric-value">SVM</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Input Image Size</div>
            <div class="metric-value">224 × 224</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">What does this system do?</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:
        st.markdown("""
        <div class="info-card">
            <h3>🔬 Retinal Image Analysis</h3>
            <p>Upload a retinal fundus image. The system extracts
            color, brightness and texture-based features for
            machine learning classification.</p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="info-card">
            <h3>🧠 AI Classification</h3>
            <p>The trained Support Vector Machine model predicts
            one of five diabetic retinopathy grades, from Grade 0
            to Grade 4.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Workflow</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">
        <p>
        <b>01</b> &nbsp; Upload retinal image
        &nbsp; → &nbsp;
        <b>02</b> &nbsp; Extract 13 features
        &nbsp; → &nbsp;
        <b>03</b> &nbsp; SVM prediction
        &nbsp; → &nbsp;
        <b>04</b> &nbsp; Display predicted DR grade
        </p>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# CLASSIFICATION PAGE
# ==========================================

elif page == "🔬 DR Classification":

    st.markdown(
        '<div class="section-title">Retinal Image Screening</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload a retinal fundus image to generate an AI-based DR class prediction."
    )

    uploaded_file = st.file_uploader(
        "Upload Fundus Image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG"
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        image_col, details_col = st.columns([1.1, 1])

        with image_col:

            st.markdown("### Retinal Image Preview")

            st.image(
                image,
                caption="Uploaded retinal fundus image",
                use_container_width=True
            )

            st.caption(
                f"Image dimensions: {image.width} × {image.height}"
            )

        with details_col:

            st.markdown("### Image Information")

            st.markdown(f"""
            <div class="info-card">
                <p><b>File name:</b> {uploaded_file.name}</p>
                <p><b>Image format:</b> {image.format or "Converted to RGB"}</p>
                <p><b>Color mode:</b> {image.mode}</p>
                <p><b>Processing size:</b> 224 × 224</p>
                <p><b>Feature count:</b> 13</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        if st.button("🚀 Analyze Retinal Image", use_container_width=True):

            if not model_loaded:
                st.error("Please ensure model.pkl is available.")

            else:
                try:
                    with st.spinner("Analyzing retinal image..."):

                        features = extract_features(image)

                        prediction = int(model.predict(features)[0])

                        if prediction not in DR_CLASSES:
                            raise ValueError(
                                f"Unexpected model output: {prediction}"
                            )

                    result = DR_CLASSES[prediction]

                    st.markdown(
                        '<div class="section-title">AI Prediction Result</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(f"""
                    <div class="result-card">
                        <div class="metric-label">PREDICTED DR GRADE</div>
                        <div class="result-grade">Grade {prediction}</div>
                        <div class="result-name">{result["name"]}</div>
                        <p style="color:#c1d0e5;">{result["description"]}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("### Severity Indicator")

                    st.progress((prediction + 1) / 5)

                    st.caption(
                        f"Predicted category: {result['severity']}"
                    )

                    # Probability chart, if supported
                    if hasattr(model, "predict_proba"):

                        probabilities = model.predict_proba(features)[0]

                        if hasattr(model, "classes_"):
                            class_labels = [
                                f"Grade {int(c)}" for c in model.classes_
                            ]
                        else:
                            class_labels = [
                                f"Class {i}" for i in range(len(probabilities))
                            ]

                        probability_df = pd.DataFrame({
                            "DR Grade": class_labels,
                            "Model Probability (%)": probabilities * 100
                        })

                        st.markdown("### Model Probability Distribution")

                        st.bar_chart(
                            probability_df.set_index("DR Grade")
                        )

                        st.caption(
                            "These are model output probabilities, not clinically "
                            "validated diagnostic confidence scores."
                        )

                    st.warning(
                        "This result is an AI model prediction, not a medical "
                        "diagnosis. Please consult a qualified eye-care professional."
                    )

                except Exception as error:
                    st.error(f"Prediction failed: {error}")


# ==========================================
# DR SEVERITY GUIDE
# ==========================================

elif page == "📊 DR Severity Guide":

    st.markdown(
        '<div class="section-title">Diabetic Retinopathy Grades</div>',
        unsafe_allow_html=True
    )

    for grade, info in DR_CLASSES.items():

        st.markdown(f"""
        <div class="info-card">
            <h3 style="color:{info['color']}">
                Grade {grade} — {info['severity']}
            </h3>
            <p><b>{info['name']}</b></p>
            <p>{info['description']}</p>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# ABOUT PROJECT
# ==========================================

elif page == "ℹ️ About Project":

    st.markdown(
        '<div class="section-title">About DiabeticRetina AI</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">
        <h3>Project Overview</h3>
        <p>
        DiabeticRetina AI is an academic machine learning project
        designed to classify diabetic retinopathy severity from
        retinal fundus images.
        </p>
    </div>

    <div class="info-card">
        <h3>Machine Learning Approach</h3>
        <p>
        The current prototype extracts 13 handcrafted image features,
        including RGB statistics, brightness statistics and GLCM
        texture features. These features are passed to a trained
        Support Vector Machine (SVM) model.
        </p>
    </div>

    <div class="info-card">
        <h3>Classification Output</h3>
        <p>
        The system predicts five categories: No DR, Mild DR,
        Moderate DR, Severe DR and Proliferative DR.
        </p>
    </div>

    <div class="info-card">
        <h3>Important Medical Disclaimer</h3>
        <p>
        This application is an academic prototype and has not been
        validated as a clinical diagnostic device. It must not be
        used to diagnose, treat or rule out diabetic retinopathy.
        A qualified eye-care professional should assess the patient.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    DiabeticRetina AI | Academic Machine Learning Project<br>
    AI-assisted retinal image classification prototype
</div>
""", unsafe_allow_html=True)
