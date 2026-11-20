
import streamlit as st
import joblib
import pickle
import numpy as np
import cv2

from pathlib import Path
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="DiabeticRetina AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# 2. CUSTOM CSS
# ==================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
    radial-gradient(circle at 80% 0%, #172b55 0%, transparent 35%),
    linear-gradient(135deg, #071225, #0b1428);
    color: #f1f5f9;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
}

[data-testid="stSidebar"] {
    background: #09162b;
    border-right: 1px solid #233d63;
}

.hero {
    background: linear-gradient(110deg, #102b50, #173c65, #27265c);
    border: 1px solid #315b91;
    padding: 32px;
    border-radius: 22px;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 40px;
    font-weight: 800;
    color: white;
}

.hero p {
    color: #c5d6ec;
    font-size: 16px;
}

.eyebrow {
    color: #54e6dc;
    letter-spacing: 2px;
    font-size: 12px;
    font-weight: 800;
}

.panel {
    background: #101f38;
    border: 1px solid #263e61;
    padding: 23px;
    border-radius: 18px;
    margin-bottom: 18px;
}

.panel-title {
    font-size: 19px;
    font-weight: 750;
    margin-bottom: 15px;
    color: #f8fafc;
}

.muted {
    color: #a7bad3;
    font-size: 14px;
}

.result-card {
    background: linear-gradient(120deg, #103c51, #25275b);
    border: 1px solid #397b9b;
    padding: 28px;
    border-radius: 20px;
    text-align: center;
    margin-top: 20px;
}

.result-grade {
    font-size: 58px;
    font-weight: 800;
    margin: 8px;
}

.result-name {
    color: white;
    font-size: 24px;
    font-weight: 700;
}

div.stButton > button {
    background: linear-gradient(90deg, #06b6d4, #2563eb, #7c3aed);
    color: white;
    border: none;
    border-radius: 12px;
    min-height: 48px;
    font-weight: 700;
    width: 100%;
}

div.stButton > button:hover {
    color: white;
    border: 1px solid #67e8f9;
    filter: brightness(1.12);
}

div[data-testid="stFileUploader"] {
    background: #101f38;
    border: 1px dashed #397b9b;
    border-radius: 15px;
    padding: 15px;
}

.footer {
    text-align: center;
    color: #8298b5;
    padding: 30px 0;
    font-size: 12px;
}

hr {
    border-color: #263e61;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# 3. DR CLASS INFORMATION
# ==================================================

DR_CLASSES = {
    0: {
        "name": "No DR",
        "description": "No diabetic retinopathy",
        "color": "#22c55e"
    },
    1: {
        "name": "Mild DR",
        "description": "Mild non-proliferative DR",
        "color": "#eab308"
    },
    2: {
        "name": "Moderate DR",
        "description": "Moderate non-proliferative DR",
        "color": "#f97316"
    },
    3: {
        "name": "Severe DR",
        "description": "Severe non-proliferative DR",
        "color": "#ef4444"
    },
    4: {
        "name": "Proliferative DR",
        "description": "Proliferative diabetic retinopathy",
        "color": "#c084fc"
    }
}

# ==================================================
# 4. MODEL LOADING
# ==================================================

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"model.pkl not found at: {MODEL_PATH}"
        )

    try:
        return joblib.load(MODEL_PATH)

    except Exception:
        with open(MODEL_PATH, "rb") as file:
            return pickle.load(file)


try:
    model = load_model()
    MODEL_LOADED = True
    MODEL_ERROR = None

except Exception as e:
    model = None
    MODEL_LOADED = False
    MODEL_ERROR = str(e)

# ==================================================
# 5. FEATURE EXTRACTION
# ==================================================

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

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

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

    contrast = graycoprops(
        glcm, "contrast"
    )[0, 0]

    homogeneity = graycoprops(
        glcm, "homogeneity"
    )[0, 0]

    energy = graycoprops(
        glcm, "energy"
    )[0, 0]

    correlation = graycoprops(
        glcm, "correlation"
    )[0, 0]

    # Brightness features

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

# ==================================================
# 6. SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center;padding:15px;">
            <div style="font-size:48px;">👁️</div>
            <h2 style="color:white;">DiabeticRetina AI</h2>
            <p style="color:#54e6dc;font-size:11px;">
                RETINAL AI SCREENING
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Home",
            "🧠 About Project",
            "⚙️ Model Information",
            "📊 DR Classification",
            "⚠️ Disclaimer"
        ],
        label_visibility="visible"
    )

    st.divider()

    if MODEL_LOADED:
        st.success("Model Connected")
    else:
        st.error("Model Not Connected")

    st.caption("Academic AI Project")

# ==================================================
# 7. HOME PAGE
# ==================================================

if page == "🏠 Home":

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">
                ARTIFICIAL INTELLIGENCE • HEALTHCARE
            </div>
            <h1>DiabeticRetina AI</h1>
            <p>
                Retinal Image-Based Diabetic Retinopathy Classification
            </p>
            <p>
                Upload a retinal fundus image and explore
                its predicted DR severity class.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if MODEL_LOADED:
        st.success("Machine Learning Model Loaded Successfully")
    else:
        st.error(f"Model Loading Error: {MODEL_ERROR}")

    upload_col, preview_col = st.columns([1, 1.15])

    with upload_col:

        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="panel-title">📤 Upload Retinal Image</div>',
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Select a fundus photograph",
            type=["jpg", "jpeg", "png"]
        )

        st.markdown(
            """
            <p class="muted">
                Supported formats: JPG, JPEG and PNG<br>
                Image will be resized to 224 × 224 pixels.
            </p>
            """,
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

    image = None

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file).convert("RGB")

        except Exception:
            st.error("Unable to open the uploaded image.")

    with preview_col:

        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="panel-title">🖼️ Image Preview</div>',
            unsafe_allow_html=True
        )

        if image is not None:

            st.image(
                image,
                use_container_width=True
            )

            st.caption(
                f"Dimensions: {image.width} × {image.height} pixels"
            )

        else:

            st.markdown(
                """
                <div style="
                    height:220px;
                    background:#09172d;
                    border:1px dashed #315b91;
                    border-radius:14px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    text-align:center;
                    color:#9bb1cc;
                ">
                    <div>
                        <div style="font-size:45px;">👁️</div>
                        Upload an image to preview
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    if image is not None:

        if st.button(
            "✦ ANALYZE RETINAL IMAGE",
            type="primary",
            use_container_width=True,
            disabled=not MODEL_LOADED
        ):

            with st.spinner(
                "Extracting retinal features and predicting DR class..."
            ):

                try:

                    features = extract_features(image)

                    # Feature count validation

                    if features.shape[1] != 13:
                        raise ValueError(
                            "Expected 13 features."
                        )

                    # Model feature count validation

                    expected = getattr(
                        model,
                        "n_features_in_",
                        None
                    )

                    if expected is not None and expected != 13:
                        raise ValueError(
                            f"Model expects {expected} features, "
                            "but app extracted 13."
                        )

                    # Prediction

                    prediction = model.predict(features)

                    predicted_class = int(prediction[0])

                    if predicted_class not in DR_CLASSES:
                        raise ValueError(
                            "Unexpected predicted class."
                        )

                    result = DR_CLASSES[predicted_class]

                    st.markdown("---")

                    st.markdown("## 🧠 Analysis Result")

                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div style="color:#65e9df;
                            font-weight:800;letter-spacing:2px;">
                                PREDICTED DR SEVERITY
                            </div>

                            <div class="result-grade"
                            style="color:{result['color']};">
                                CLASS {predicted_class}
                            </div>

                            <div class="result-name">
                                {result['name']}
                            </div>

                            <p style="color:#bdcce2;">
                                {result['description']}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Probability if available

                    if hasattr(model, "predict_proba"):

                        try:

                            probabilities = model.predict_proba(
                                features
                            )[0]

                            model_classes = list(model.classes_)

                            index = model_classes.index(
                                predicted_class
                            )

                            confidence = (
                                probabilities[index] * 100
                            )

                            st.metric(
                                "Model Probability Estimate",
                                f"{confidence:.2f}%"
                            )

                            st.caption(
                                "This probability is not a clinically "
                                "validated diagnostic confidence."
                            )

                        except Exception:
                            st.info(
                                "Probability estimates are unavailable."
                            )

                    # Severity scale

                    st.markdown("### DR Severity Scale")

                    cols = st.columns(5)

                    for i, data in DR_CLASSES.items():

                        with cols[i]:

                            st.markdown(
                                f"""
                                <div style="
                                    background:#101f38;
                                    border:1px solid {data['color']};
                                    border-radius:12px;
                                    padding:12px 5px;
                                    text-align:center;
                                ">
                                    <h2 style="color:{data['color']};">
                                        {i}
                                    </h2>
                                    <p style="font-size:12px;color:white;">
                                        {data['name']}
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                    st.markdown("")

                    st.warning(
                        "This is an experimental academic prediction, "
                        "not a medical diagnosis. Please consult a "
                        "qualified eye-care professional."
                    )

                except Exception as e:

                    st.error(
                        f"Prediction failed: {str(e)}"
                    )

    st.markdown(
        """
        <div class="footer">
            DiabeticRetina AI | Academic Machine Learning Prototype<br>
            Not intended for clinical diagnosis
        </div>
        """,
        unsafe_allow_html=True
    )

# ==================================================
# 8. ABOUT PROJECT
# ==================================================

elif page == "🧠 About Project":

    st.markdown("## About DiabeticRetina AI")

    st.write(
        """
        DiabeticRetina AI is an academic machine learning project
        designed to classify diabetic retinopathy severity from
        retinal fundus photographs.
        """
    )

    a, b = st.columns(2)

    with a:

        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">🎯 Project Objective</div>
                To explore automated classification of retinal images
                into five DR severity categories using machine learning.
            </div>
            """,
            unsafe_allow_html=True
        )

    with b:

        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">🔬 Project Approach</div>
                Image preprocessing, handcrafted feature extraction,
                and Support Vector Machine classification.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Project Workflow")

    st.markdown("""
    1. Retinal image upload
    2. Image resizing and preprocessing
    3. Extraction of 13 numerical features
    4. SVM model prediction
    5. Display of predicted DR class
    """)

# ==================================================
# 9. MODEL INFORMATION
# ==================================================

elif page == "⚙️ Model Information":

    st.markdown("## Model Information")

    c1, c2, c3 = st.columns(3)

    c1.metric("Algorithm", "SVM")
    c2.metric("Input Features", "13")
    c3.metric("Output Classes", "5")

    st.markdown("### Feature Engineering")

    st.markdown("""
    | Feature category | Extracted features |
    |---|---|
    | Color | RGB mean and standard deviation |
    | Texture | Contrast, Homogeneity, Energy, Correlation |
    | Brightness | Mean, Standard deviation, Range |
    """)

    st.markdown("### Model File")

    st.code("model.pkl")

    if MODEL_LOADED:
        st.success("Your model.pkl file is loaded.")
    else:
        st.error(f"Model unavailable: {MODEL_ERROR}")

# ==================================================
# 10. DR CLASSIFICATION GUIDE
# ==================================================

elif page == "📊 DR Classification":

    st.markdown("## DR Severity Classification")

    for i, data in DR_CLASSES.items():

        st.markdown(
            f"""
            <div class="panel"
            style="border-left:5px solid {data['color']};">
                <h3 style="color:{data['color']};">
                    Grade {i} — {data['name']}
                </h3>
                <p class="muted">{data['description']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ==================================================
# 11. DISCLAIMER
# ==================================================

elif page == "⚠️ Disclaimer":

    st.markdown("## Medical Disclaimer")

    st.error(
        """
        DiabeticRetina AI is an academic prototype.
        It is not a clinically validated medical device.
        It must not be used to diagnose, rule out, or treat
        diabetic retinopathy.
        """
    )

    st.write(
        """
        - Predictions may be incorrect.
        - Image quality can affect model output.
        - Model probabilities are not clinical certainty.
        - Always consult a qualified eye-care professional.
        """
    )

