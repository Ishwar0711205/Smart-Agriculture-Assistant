## CROP RECOMMENDATION - Standalone Web App
## This file is a standalone version of the crop recommendation feature.
## The main integrated app is app.py in the project root.

import streamlit as st
import numpy as np
import pickle
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model from relative path
MODEL_PATH = os.path.join(BASE_DIR, "RF.pkl")
IMAGE_PATH = os.path.join(BASE_DIR, "crop.png")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

RF_Model_pkl = load_model()

def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    if RF_Model_pkl is None:
        return "Model not loaded"
    prediction = RF_Model_pkl.predict(
        np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]).reshape(1, -1)
    )
    return prediction[0]

def main():
    st.set_page_config(page_title="KRISHI – Crop Recommendation", page_icon="🌾", layout="wide")
    st.markdown("<h1 style='text-align: center;'>🌾 SMART CROP RECOMMENDATIONS</h1>", unsafe_allow_html=True)

    if os.path.exists(IMAGE_PATH):
        try:
            img = Image.open(IMAGE_PATH)
            st.image(img, use_container_width=True)
        except Exception:
            pass

    st.sidebar.title("KRISHI")
    if RF_Model_pkl is None:
        st.error("Model file not found. Please ensure RF.pkl is in the CROP-RECOMMENDATION folder.")
        return

    st.sidebar.header("Enter Crop Details")
    nitrogen    = st.sidebar.number_input("Nitrogen",    min_value=0.0, max_value=140.0, value=0.0, step=0.1)
    phosphorus  = st.sidebar.number_input("Phosphorus",  min_value=0.0, max_value=145.0, value=0.0, step=0.1)
    potassium   = st.sidebar.number_input("Potassium",   min_value=0.0, max_value=205.0, value=0.0, step=0.1)
    temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=51.0,  value=25.0, step=0.1)
    humidity    = st.sidebar.number_input("Humidity (%)",      min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    ph          = st.sidebar.number_input("pH Level",          min_value=0.0, max_value=14.0,  value=6.5,  step=0.01)
    rainfall    = st.sidebar.number_input("Rainfall (mm)",     min_value=0.0, max_value=500.0, value=100.0, step=0.1)

    inputs = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    if st.sidebar.button("Predict"):
        if (inputs == 0).all():
            st.error("Please fill in at least one input field with a valid non-zero value.")
        else:
            prediction = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
            st.success(f"🌱 The recommended crop is: **{str(prediction).capitalize()}**")

if __name__ == '__main__':
    main()
