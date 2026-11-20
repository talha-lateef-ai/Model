
import streamlit as st
import numpy as np
import cv2
import joblib
from pathlib import Path
from PIL import Image
from skimage.feature import graycomatrix, graycoprops


# -------------------------------
# 1. PAGE SETTINGS
# -------------------------------

st.set_page_config(
    page_title="DiabeticRetina AI",
    page_icon="👁️",
    layout="wide"
)


# -------------------------------
# 2. SIMPLE CSS DESIGN
# -------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #f4f8fc;
}

h1, h2, h3 {
    color: #12345a;
}

.header {
    background: #12345a;
    padding: 25px;
    border-radius: 12px;
    color: white;
    margin-bottom: 25px;
}

.header h1 {
    color: white;
    margin-bottom: 5px;
}

.header p {
    color: #dceafa;
}

.result {
    background: #e4f5f2;
    border-left: 6px solid #13a88a;
    padding: 20px;
    border-radius: 10px;
    margin-top: 15px;
}

.info {
    background: white;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #dce5ef;
    margin-bottom: 15px;
}

div.stButton > button {
    background-color: #087e8b;
    color: white;
    border-radius: 8px;
    border: none;
    height: 45px;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #05636d;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# -------------------------------
# 3. LOAD MODEL
# -------------------------------

MODEL_PATH = Path(__file__).parent / "model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False


# -------------------------------
# 4. DR CLASS NAMES
# -------------------------------

dr_classes = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

dr_descriptions = {
    0: "No diabetic retinopathy detected by the model.",
    1: "The model predicts mild diabetic retinopathy.",
    2: "The model predicts moderate diabetic retinopathy.",
    3: "The model predicts severe diabetic retinopathy.",
    4: "The model predicts proliferative diabetic retinopathy."
}


# -------------------------------
# 5. FEATURE EXTRACTION
# -------------------------------

def extract_features(image):

    image = np.array(image.convert("RGB"))
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
        red_mean, red_std,
        green_mean, green_std,
        blue_mean, blue_std,
        contrast, homogeneity,
        energy, correlation,
        brightness_mean, brightness_std,
        brightness_range
    ]

    return np.array([features])


# -------------------------------
# 6. SIDEBAR
# -------------------------------

st.sidebar.title("👁️ DiabeticRetina AI")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "DR Classification",
        "DR Grades",
        "About Project"
    ]
)

st.sidebar.markdown("---")

if model_loaded:
    st.sidebar.success("Model Ready")
else:
    st.sidebar.error("Model Not Loaded")


# -------------------------------
# 7. HEADER
# -------------------------------

st.markdown("""
<div class="header">
    <h1>👁️ DiabeticRetina AI</h1>
    <p>Retinal Image-Based Diabetic Retinopathy Classification</p>
</div>
""", unsafe_allow_html=True)


# -------------------------------
# 8. HOME PAGE
# -------------------------------

if page == "Home":

    st.title("Welcome to DiabeticRetina AI")

    st.write(
        "An AI-based academic prototype that classifies "
        "diabetic retinopathy using retinal fundus images."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("DR Classes", "5")
    col2.metric("ML Model", "SVM")
    col3.metric("Features", "13")

    st.subheader("How does it work?")

    st.markdown("""
    <div class="info">
    <h3>1. Upload Image</h3>
    Upload a retinal fundus image.
    </div>

    <div class="info">
    <h3>2. Feature Extraction</h3>
    The system extracts 13 image features.
    </div>

    <div class="info">
    <h3>3. AI Prediction</h3>
    The trained SVM model predicts the DR grade.
    </div>

    <div class="info">
    <h3>4. View Result</h3>
    See the predicted DR classification.
    </div>
    """, unsafe_allow_html=True)


# -------------------------------
# 9. CLASSIFICATION PAGE
# -------------------------------

elif page == "DR Classification":

    st.title("Retinal Image Classification")

    st.write("Upload your retinal image below.")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Image Information")
            st.write("File:", uploaded_file.name)
            st.write("Width:", image.width)
            st.write("Height:", image.height)
            st.write("Processing Size: 224 × 224")
            st.write("Features: 13")

        if st.button("Analyze Image"):

            if not model_loaded:
                st.error("Please put model.pkl in the project folder.")

            else:
                try:
                    with st.spinner("Analyzing image..."):

                        features = extract_features(image)

                        prediction = int(model.predict(features)[0])

                    if prediction not in dr_classes:
                        st.error("Unexpected prediction from model.")

                    else:
                        st.markdown("## Prediction Result")

                        st.markdown(
                            f"""
                            <div class="result">
                                <h2>Grade {prediction}</h2>
                                <h3>{dr_classes[prediction]}</h3>
                                <p>{dr_descriptions[prediction]}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.info(
                            "This is an AI model prediction, "
                            "not a medical diagnosis."
                        )

                        if hasattr(model, "predict_proba"):

                            probabilities = model.predict_proba(features)[0]

                            st.subheader("Model Probability")

                            st.bar_chart({
                                f"Grade {int(c)}": float(p * 100)
                                for c, p in zip(
                                    model.classes_,
                                    probabilities
                                )
                            })

                            st.caption(
                                "Model probabilities are not clinically "
                                "validated diagnostic confidence scores."
                            )

                except Exception as e:
                    st.error(f"Error: {e}")


# -------------------------------
# 10. DR GRADES PAGE
# -------------------------------

elif page == "DR Grades":

    st.title("Diabetic Retinopathy Grades")

    for grade, name in dr_classes.items():

        st.markdown(
            f"""
            <div class="info">
                <h3>Grade {grade}: {name}</h3>
                <p>{dr_descriptions[grade]}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# -------------------------------
# 11. ABOUT PAGE
# -------------------------------

elif page == "About Project":

    st.title("About DiabeticRetina AI")

    st.markdown("""
    <div class="info">
        <h3>Project Introduction</h3>
        DiabeticRetina AI is an academic machine learning
        project for classifying diabetic retinopathy from
        retinal fundus images.
    </div>

    <div class="info">
        <h3>Technology Used</h3>
        Python, Streamlit, OpenCV, GLCM and Support Vector Machine.
    </div>

    <div class="info">
        <h3>Classification</h3>
        The system predicts five DR grades, from Grade 0 to Grade 4.
    </div>

    <div class="info">
        <h3>Disclaimer</h3>
        This application is an academic prototype. It is not
        a clinically validated diagnostic system. Always consult
        a qualified eye-care professional.
    </div>
    """, unsafe_allow_html=True)


# -------------------------------
# 12. FOOTER
# -------------------------------

st.markdown("---")

st.caption(
    "DiabeticRetina AI | Academic Machine Learning Project"
)
