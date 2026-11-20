import streamlit as st
import joblib
import pickle
import numpy as np

# Page setup
st.set_page_config(page_title="ML Prediction App", page_icon="🤖")
st.title("🤖 Machine Learning Prediction App")
st.write("Neeche diye gaye inputs enter karein aur **Predict** button dabayein:")

# model.pkl load karein
@st.cache_resource
def load_model():
    try:
        return joblib.load("model.pkl")
    except Exception:
        with open("model.pkl", "rb") as file:
            return pickle.load(file)

model = load_model()

# --- INPUT FIELDS ---
col1, col2 = st.columns(2)

with col1:
    feature_1 = st.number_input("Feature 1", value=0.0)
    feature_2 = st.number_input("Feature 2", value=0.0)

with col2:
    feature_3 = st.number_input("Feature 3", value=0.0)
    feature_4 = st.number_input("Feature 4", value=0.0)

# --- PREDICTION ---
if st.button("Predict", type="primary"):
    try:
        input_data = np.array([[feature_1, feature_2, feature_3, feature_4]])
        prediction = model.predict(input_data)
        st.success(f"### Prediction Result: {prediction[0]}")
    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Check karein ke aap ke model ko jitne input features chahiye, utne hi yahan diye gaye hain.")
