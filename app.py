import streamlit as st
import joblib
import pickle
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    try:
        return joblib.load(MODEL_PATH)

    except Exception:
        with open(MODEL_PATH, "rb") as file:
            return pickle.load(file)


try:
    model = load_model()
    MODEL_LOADED = True
    st.sidebar.success("Model Loaded Successfully!")

except Exception as error:
    model = None
    MODEL_LOADED = False
    st.sidebar.error("Model Loading Failed")
    st.error(str(error))
