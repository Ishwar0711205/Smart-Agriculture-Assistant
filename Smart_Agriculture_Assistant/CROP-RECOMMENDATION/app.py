"""Crop Recommendation Standalone Application

This is a lightweight standalone version of the crop recommendation feature.
For the full integrated application, use app.py in the parent directory.
"""

import logging
from pathlib import Path
from typing import Optional

import streamlit as st
import numpy as np
import pickle
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

APP_TITLE = "KRISHI – Crop Recommendation"
APP_ICON = "🌾"

# Paths relative to this file
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "RF.pkl"
IMAGE_PATH = BASE_DIR / "crop.png"

# Input validation ranges
NITROGEN_RANGE = (0.0, 140.0)
PHOSPHORUS_RANGE = (0.0, 145.0)
POTASSIUM_RANGE = (0.0, 205.0)
TEMPERATURE_RANGE = (0.0, 51.0)
HUMIDITY_RANGE = (0.0, 100.0)
PH_RANGE = (0.0, 14.0)
RAINFALL_RANGE = (0.0, 500.0)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# MODEL LOADING
# =============================================================================


@st.cache_resource(show_spinner="Loading crop recommendation model...")
def load_model() -> Optional[any]:
    """Load the Random Forest crop recommendation model.
    
    Returns:
        Loaded model or None if loading fails.
    """
    if not MODEL_PATH.exists():
        logger.error(f"Model file not found at {MODEL_PATH}")
        return None
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info("Successfully loaded crop recommendation model")
        return model
    except (pickle.PickleError, EOFError, ValueError) as e:
        logger.error(f"Error loading model: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading model: {e}")
        return None


rf_model = load_model()

# =============================================================================
# PREDICTION FUNCTION
# =============================================================================


def predict_crop(
    nitrogen: float,
    phosphorus: float,
    potassium: float,
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float
) -> str:
    """Predict the best crop using the Random Forest model.
    
    Args:
        nitrogen: Nitrogen content (0-140)
        phosphorus: Phosphorus content (0-145)
        potassium: Potassium content (0-205)
        temperature: Temperature in Celsius (0-51)
        humidity: Humidity percentage (0-100)
        ph: Soil pH level (0-14)
        rainfall: Rainfall in mm (0-500)
    
    Returns:
        Predicted crop name or error message.
    """
    if rf_model is None:
        return "Model not loaded. Please restart the application."
    
    try:
        features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        prediction = rf_model.predict(features)
        return str(prediction[0])
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "Error during prediction. Please check your inputs."


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide"
    )
    
    st.markdown("<h1 style='text-align: center;'>🌾 SMART CROP RECOMMENDATIONS</h1>", unsafe_allow_html=True)
    
    # Display image if available
    if IMAGE_PATH.exists():
        try:
            img = Image.open(IMAGE_PATH)
            st.image(img, use_container_width=True)
        except Exception as e:
            logger.warning(f"Could not load image: {e}")
    
    st.sidebar.title("🌾 KRISHI")
    st.sidebar.markdown("*Smart Crop Recommendation*")
    st.sidebar.markdown("---")
    
    # Check if model loaded
    if rf_model is None:
        st.error(
            "❌ Model file not found. Please ensure RF.pkl is in the same directory as this script and restart the app."
        )
        return
    
    st.sidebar.header("📊 Enter Crop Details")
    
    # Input fields
    nitrogen = st.sidebar.number_input(
        "Nitrogen (N)",
        min_value=NITROGEN_RANGE[0],
        max_value=NITROGEN_RANGE[1],
        value=0.0,
        step=0.1
    )
    
    phosphorus = st.sidebar.number_input(
        "Phosphorus (P)",
        min_value=PHOSPHORUS_RANGE[0],
        max_value=PHOSPHORUS_RANGE[1],
        value=0.0,
        step=0.1
    )
    
    potassium = st.sidebar.number_input(
        "Potassium (K)",
        min_value=POTASSIUM_RANGE[0],
        max_value=POTASSIUM_RANGE[1],
        value=0.0,
        step=0.1
    )
    
    temperature = st.sidebar.number_input(
        "Temperature (°C)",
        min_value=TEMPERATURE_RANGE[0],
        max_value=TEMPERATURE_RANGE[1],
        value=25.0,
        step=0.1
    )
    
    humidity = st.sidebar.number_input(
        "Humidity (%)",
        min_value=HUMIDITY_RANGE[0],
        max_value=HUMIDITY_RANGE[1],
        value=50.0,
        step=0.1
    )
    
    ph = st.sidebar.number_input(
        "pH Level",
        min_value=PH_RANGE[0],
        max_value=PH_RANGE[1],
        value=6.5,
        step=0.01
    )
    
    rainfall = st.sidebar.number_input(
        "Rainfall (mm)",
        min_value=RAINFALL_RANGE[0],
        max_value=RAINFALL_RANGE[1],
        value=100.0,
        step=0.1
    )
    
    # Prediction button
    if st.sidebar.button("🌾 Predict", type="primary", use_container_width=True):
        inputs = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        
        # Validation
        if (inputs == 0).all():
            st.error("⚠️ Please enter at least one non-zero value for soil parameters.")
        else:
            prediction = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
            
            if "Error" in prediction or "not loaded" in prediction:
                st.error(f"❌ {prediction}")
            else:
                st.success(f"🌱 **Recommended Crop: {str(prediction).capitalize()}**")
                st.info(f"""
                Based on your soil and climate data, **{str(prediction).capitalize()}** is the best crop for your farm.
                
                **Next Steps:**
                - Consult your local agricultural officer for seed varieties
                - Check weather forecast for optimal planting time
                - Monitor crop health regularly
                """)


if __name__ == '__main__':
    main()
