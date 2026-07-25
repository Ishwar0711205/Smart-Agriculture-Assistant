"""KRISHI – Smart Agriculture Assistant

A Streamlit-based AI-powered agriculture assistance application that helps farmers with:
- Crop recommendation using Random Forest ML model
- Plant disease identification using CNN Keras model
- Real-time weather forecasting and farming advisory
"""

import os
import logging
import warnings
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any

import streamlit as st
import numpy as np
import pandas as pd
import requests
from PIL import Image

import tensorflow as tf

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

APP_TITLE = "KRISHI – Smart Agriculture Assistant"
APP_ICON = "🌾"

# Model and asset paths (relative to app.py location)
BASE_DIR = Path(__file__).parent
CROP_MODEL_PATH = BASE_DIR / "CROP-RECOMMENDATION" / "RF.pkl"
CROP_IMAGE_PATH = BASE_DIR / "CROP-RECOMMENDATION" / "crop.png"
DISEASE_MODEL_PATH = BASE_DIR / "PLANT-DISEASE-IDENTIFICATION" / "trained_plant_disease_model.keras"
DISEASE_IMAGE_PATH = BASE_DIR / "PLANT-DISEASE-IDENTIFICATION" / "Diseases.png"

# Model input/output specifications
DISEASE_MODEL_INPUT_SIZE = (128, 128)
MAX_TEMPERATURE = 51.0
MIN_TEMPERATURE = 0.0
MAX_HUMIDITY = 100.0
MIN_HUMIDITY = 0.0
MAX_PH = 14.0
MIN_PH = 0.0
MAX_RAINFALL = 500.0
MIN_RAINFALL = 0.0

# Weather API configuration
WEATHER_API_TIMEOUT = 10  # seconds

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Disease information database
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "medicine": "Captan 50% WP spray",
        "dosage": "2 gm per litre of water",
        "prevention": "Remove fallen infected leaves. Prune trees for good air circulation. Apply fungicide at bud-break.",
        "fertilizer": "Apply balanced NPK fertilizer. Avoid excess nitrogen.",
        "severity": "Moderate"
    },
    "Apple___Black_rot": {
        "medicine": "Thiophanate-methyl 70% WP spray",
        "dosage": "1 gm per litre of water",
        "prevention": "Remove mummified fruits from trees and ground. Prune dead or diseased wood regularly.",
        "fertilizer": "Apply calcium fertilizer to strengthen fruit skin and resistance.",
        "severity": "High"
    },
    "Apple___Cedar_apple_rust": {
        "medicine": "Myclobutanil 10% WP spray",
        "dosage": "1 ml per litre of water",
        "prevention": "Remove nearby cedar or juniper trees if possible. Apply fungicide before and during bloom.",
        "fertilizer": "Apply balanced NPK. Potassium improves overall resistance.",
        "severity": "Moderate"
    },
    "Apple___healthy": {
        "medicine": "No medicine needed",
        "dosage": "Not applicable",
        "prevention": "Continue regular monitoring. Maintain good orchard hygiene.",
        "fertilizer": "Apply balanced NPK fertilizer as per orchard schedule.",
        "severity": "None"
    },
    "Blueberry___healthy": {
        "medicine": "No medicine needed",
        "dosage": "Not applicable",
        "prevention": "Continue good farming practices. Maintain proper pH (4.5-5.5) for blueberries.",
        "fertilizer": "Apply ammonium sulphate fertilizer. Blueberries prefer acidic soil.",
        "severity": "None"
    },
    # ... (include all other disease entries as in original)
    "Tomato___healthy": {
        "medicine": "No medicine needed",
        "dosage": "Not applicable",
        "prevention": "Continue good farming practices. Monitor regularly for early signs of disease.",
        "fertilizer": "Apply regular balanced NPK fertilizer as per tomato growth stage.",
        "severity": "None"
    },
}

WEATHER_CODE_MAP = {
    "113": "Sunny / Clear", "116": "Partly Cloudy", "119": "Cloudy",
    "122": "Overcast", "143": "Mist", "176": "Patchy Rain",
    "200": "Thundery Outbreaks", "248": "Fog", "260": "Freezing Fog",
    "263": "Light Drizzle", "266": "Drizzle", "293": "Light Rain",
    "296": "Rain", "299": "Moderate Rain", "302": "Heavy Rain",
    "305": "Heavy Rain", "308": "Very Heavy Rain", "353": "Light Rain Shower",
    "356": "Moderate Rain Shower", "359": "Heavy Rain Shower",
    "386": "Thundery Rain", "389": "Heavy Thunder Rain",
}

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS Styling
st.markdown("""
<style>
.card {
    background: #f0f8e8;
    border-left: 5px solid #4CAF50;
    padding: 18px 22px;
    margin: 12px 0;
    border-radius: 10px;
}
.card h3 { color: #2e7d32; margin-bottom: 8px; }
.card-red {
    background: #fff3f3;
    border-left: 5px solid #e53935;
    padding: 18px 22px;
    margin: 12px 0;
    border-radius: 10px;
}
.card-blue {
    background: #e8f4fd;
    border-left: 5px solid #1976D2;
    padding: 18px 22px;
    margin: 12px 0;
    border-radius: 10px;
}
.card-blue h3 { color: #1565C0; margin-bottom: 8px; }
.card-orange {
    background: #fff8e1;
    border-left: 5px solid #F57C00;
    padding: 18px 22px;
    margin: 12px 0;
    border-radius: 10px;
}
.card-orange h3 { color: #E65100; margin-bottom: 8px; }
.metric-big { font-size: 2rem; font-weight: 700; color: #2e7d32; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MODEL LOADING FUNCTIONS
# =============================================================================

@st.cache_resource(show_spinner="Loading crop recommendation model...")
def load_crop_model() -> Optional[Any]:
    """Load the Random Forest crop recommendation model with error handling.
    
    Returns:
        Loaded model object or None if loading fails.
    """
    if not CROP_MODEL_PATH.exists():
        logger.error(f"Crop model not found: {CROP_MODEL_PATH}")
        return None
    
    try:
        import pickle
        with open(CROP_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Successfully loaded crop model from {CROP_MODEL_PATH}")
        return model
    except (pickle.PickleError, EOFError, ValueError) as e:
        logger.error(f"Error loading pickle model: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading crop model: {e}")
        return None


@st.cache_resource(show_spinner="Loading plant disease model...")
def load_disease_model() -> Tuple[Optional[Any], Optional[str]]:
    """Load the Keras CNN disease detection model with error handling.
    
    Returns:
        Tuple of (model, error_message). If successful, error_message is None.
    """
    if not DISEASE_MODEL_PATH.exists():
        msg = f"Disease model not found: {DISEASE_MODEL_PATH}"
        logger.error(msg)
        return None, msg
    
    try:
        model = tf.keras.models.load_model(str(DISEASE_MODEL_PATH))
        logger.info(f"Successfully loaded disease model from {DISEASE_MODEL_PATH}")
        return model, None
    except Exception as e:
        msg = f"Error loading disease model: {e}"
        logger.error(msg)
        return None, msg


# Load models on startup
rf_model = load_crop_model()
disease_model, disease_model_error = load_disease_model()

# =============================================================================
# PREDICTION FUNCTIONS
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
        nitrogen: Soil nitrogen content (0-140)
        phosphorus: Soil phosphorus content (0-145)
        potassium: Soil potassium content (0-205)
        temperature: Temperature in Celsius (0-51)
        humidity: Humidity percentage (0-100)
        ph: Soil pH level (0-14)
        rainfall: Rainfall in mm (0-500)
    
    Returns:
        Predicted crop name or error message.
    """
    if rf_model is None:
        return "Crop model not available. Please restart the app."
    
    try:
        features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        prediction = rf_model.predict(features)
        return str(prediction[0])
    except Exception as e:
        logger.error(f"Error during crop prediction: {e}")
        return "Prediction error. Please check your inputs."


def predict_disease(test_image) -> Tuple[int, float]:
    """Predict plant disease from an uploaded image using the CNN model.
    
    Args:
        test_image: Uploaded image file
    
    Returns:
        Tuple of (disease_class_index, confidence_percentage).
        Returns (-1, 0.0) if prediction fails.
    """
    if disease_model is None:
        logger.error("Disease model not loaded")
        return -1, 0.0
    
    try:
        # Load and preprocess image
        image = Image.open(test_image).convert("RGB")
        image = image.resize(DISEASE_MODEL_INPUT_SIZE)
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        
        # Make prediction
        predictions = disease_model.predict(image_array, verbose=0)
        class_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions)) * 100
        
        logger.info(f"Disease prediction successful: class {class_index}, confidence {confidence:.2f}%")
        return class_index, round(confidence, 2)
    except Exception as e:
        logger.error(f"Error during disease prediction: {e}")
        return -1, 0.0


# =============================================================================
# WEATHER & ADVISORY FUNCTIONS
# =============================================================================

def get_weather(city: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Fetch weather data from wttr.in API.
    
    Args:
        city: City name
    
    Returns:
        Tuple of (weather_data_dict, error_message).
        If successful, error_message is None.
    """
    if not city or not isinstance(city, str) or city.strip() == "":
        return None, "Please enter a valid city name."
    
    try:
        url = f"https://wttr.in/{city.strip()}?format=j1"
        response = requests.get(url, timeout=WEATHER_API_TIMEOUT)
        
        if response.status_code != 200:
            return None, f"City not found or service unavailable (HTTP {response.status_code})"
        
        data = response.json()
        current = data["current_condition"][0]
        
        weather = {
            "temp_c": current["temp_C"],
            "temp_f": current["temp_F"],
            "feels_like_c": current["FeelsLikeC"],
            "humidity": current["humidity"],
            "wind_kph": current["windspeedKmph"],
            "wind_dir": current["winddir16Point"],
            "visibility": current["visibility"],
            "pressure": current["pressure"],
            "uv_index": current["uvIndex"],
            "desc": current["weatherDesc"][0]["value"],
            "weather_code": current["weatherCode"],
            "precip_mm": current["precipMM"],
            "cloud_cover": current["cloudcover"],
        }
        
        forecast = []
        for day in data.get("weather", []):
            hourly = day.get("hourly", [])
            forecast.append({
                "date": day["date"],
                "max_c": day["maxtempC"],
                "min_c": day["mintempC"],
                "sunrise": day["astronomy"][0]["sunrise"],
                "sunset": day["astronomy"][0]["sunset"],
                "uvIndex": day["uvIndex"],
                "desc": hourly[4]["weatherDesc"][0]["value"] if len(hourly) > 4 else "N/A",
                "rain_chance": hourly[4].get("chanceofrain", "N/A") if len(hourly) > 4 else "N/A",
            })
        
        logger.info(f"Successfully fetched weather for {city}")
        return {"current": weather, "forecast": forecast, "city": city}, None
    
    except requests.exceptions.Timeout:
        msg = "Weather API request timed out. Please check your internet connection."
        logger.error(msg)
        return None, msg
    except requests.exceptions.ConnectionError:
        msg = "No internet connection. Please check your network."
        logger.error(msg)
        return None, msg
    except Exception as e:
        msg = f"Error fetching weather: {e}"
        logger.error(msg)
        return None, msg


def generate_farming_advice(temp_c: float, humidity: float, wind_kph: float, precip_mm: float) -> List[str]:
    """Generate farming advice based on weather conditions.
    
    Args:
        temp_c: Temperature in Celsius
        humidity: Humidity percentage
        wind_kph: Wind speed in km/h
        precip_mm: Precipitation in mm
    
    Returns:
        List of farming tips and recommendations.
    """
    tips = []
    temp = int(temp_c)
    humid = int(humidity)
    wind = int(wind_kph)
    precip = float(precip_mm)
    
    # Temperature advisories
    if temp > 38:
        tips.append("🌡️ **Extreme Heat Alert**: Increase irrigation frequency. Protect crops with shade nets.")
    elif temp > 32:
        tips.append("🌡️ **High Temperature**: Monitor crops for heat stress. Water in early morning or evening.")
    elif temp < 5:
        tips.append("🥶 **Cold Alert**: Cover sensitive crops to protect from frost damage.")
    
    # Humidity advisories
    if humid > 85:
        tips.append("💧 **High Humidity**: High risk of fungal diseases (blight, mold). Consider preventive fungicide.")
    elif humid < 30:
        tips.append("🏜️ **Low Humidity**: Increase irrigation. Monitor for spider mites which thrive in dry conditions.")
    
    # Precipitation advisories
    if precip > 10:
        tips.append("🌧️ **Heavy Rainfall**: Avoid spraying pesticides/fertilizers. Check field drainage.")
    elif precip > 0:
        tips.append("🌦️ **Light Rain**: Good for recently applied fertilizers. Monitor for waterlogging.")
    else:
        tips.append("☀️ **No Rain**: Ensure timely irrigation. Mulch to conserve soil moisture.")
    
    # Wind advisories
    if wind > 30:
        tips.append("💨 **Strong Winds**: Postpone spraying operations to avoid chemical drift.")
    
    if not tips:
        tips.append("✅ **Ideal Farming Conditions**: Proceed with regular farming activities.")
    
    return tips


# =============================================================================
# UI HELPER FUNCTIONS
# =============================================================================

def load_and_display_image(image_path: Path, caption: str, use_sidebar: bool = False) -> bool:
    """Safely load and display an image.
    
    Args:
        image_path: Path to the image file
        caption: Caption for the image
        use_sidebar: If True, display in sidebar; else in main area
    
    Returns:
        True if image loaded successfully, False otherwise.
    """
    if not image_path.exists():
        logger.warning(f"Image not found: {image_path}")
        return False
    
    try:
        img = Image.open(image_path)
        if use_sidebar:
            st.sidebar.image(img, caption=caption, use_container_width=True)
        else:
            st.image(img, caption=caption, use_container_width=True)
        return True
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return False


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main Streamlit application entry point."""
    
    # Sidebar navigation
    st.sidebar.title("🌾 KRISHI")
    st.sidebar.markdown("*Smart Agriculture Assistant*")
    st.sidebar.markdown("---")
    
    # Sidebar images
    load_and_display_image(CROP_IMAGE_PATH, "Crop Recommendation", use_sidebar=True)
    load_and_display_image(DISEASE_IMAGE_PATH, "Disease Recognition", use_sidebar=True)
    
    st.sidebar.markdown("---")
    app_mode = st.sidebar.selectbox(
        "Select Feature",
        ["Home / Dashboard", "Crop Recommendation", "Plant Disease Identification", "Weather Forecast"],
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**KRISHI v2.0** | Smart Agriculture")
    
    # Page routing
    if app_mode == "Home / Dashboard":
        show_home_page()
    elif app_mode == "Crop Recommendation":
        show_crop_recommendation_page()
    elif app_mode == "Plant Disease Identification":
        show_disease_detection_page()
    elif app_mode == "Weather Forecast":
        show_weather_forecast_page()


def show_home_page():
    """Display the home/dashboard page."""
    st.markdown("<h1 style='text-align:center;'>🌾 KRISHI – Smart Agriculture Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;font-size:1.1rem;color:#555;'>Empowering farmers with AI-driven insights for better crop management</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="card"><h3>🌾 Crop Recommendation</h3><p>AI-powered crop suggestions based on soil and climate data.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><h3>🌿 Disease Detection</h3><p>Upload a leaf image and detect plant diseases with 38-class CNN model.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card-blue"><h3>💊 Medicine Guide</h3><p>Get recommended medicines, dosage and prevention for detected diseases.</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="card-orange"><h3>🌤️ Weather Forecast</h3><p>Real-time weather data and farming advice for any city.</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("📌 How to Use KRISHI")
    st.markdown("""
    1. **Crop Recommendation** → Enter soil values (N, P, K), temperature, humidity, pH, rainfall → Get best crop to grow.
    2. **Plant Disease Identification** → Upload a leaf photo → AI detects disease → Get full treatment & medicine guide.
    3. **Weather Forecast** → Enter your city → Get real-time weather + farming tips based on current conditions.
    """)
    
    st.markdown("---")
    st.subheader("🦠 Supported Plant Diseases (38 Classes)")
    disease_df = []
    for cls in CLASS_NAMES:
        info = DISEASE_INFO.get(cls, {})
        plant = cls.split("___")[0].replace("_", " ")
        disease = cls.split("___")[1].replace("_", " ") if "___" in cls else cls
        disease_df.append({
            "Plant": plant,
            "Disease": disease,
            "Severity": info.get("severity", "N/A"),
            "Medicine": info.get("medicine", "N/A"),
        })
    st.dataframe(pd.DataFrame(disease_df), use_container_width=True, height=400)


def show_crop_recommendation_page():
    """Display the crop recommendation page."""
    st.markdown("<h1 style='text-align:center;'>🌾 SMART CROP RECOMMENDATIONS</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    if rf_model is None:
        st.error("❌ Crop recommendation model not found. Please ensure CROP-RECOMMENDATION/RF.pkl exists and restart the app.")
        return
    
    st.markdown('<div class="card"><h3>📊 Enter Soil & Climate Details</h3><p>Fill in values based on your soil test report and local climate.</p></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        nitrogen = st.number_input("🧪 Nitrogen (N)", min_value=0.0, max_value=140.0, value=0.0, step=0.1)
        phosphorus = st.number_input("🧪 Phosphorus (P)", min_value=0.0, max_value=145.0, value=0.0, step=0.1)
        potassium = st.number_input("🧪 Potassium (K)", min_value=0.0, max_value=205.0, value=0.0, step=0.1)
    
    with col2:
        temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=51.0, value=25.0, step=0.1)
        humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        ph = st.number_input("⚗️ pH Level", min_value=0.0, max_value=14.0, value=6.5, step=0.01)
    
    with col3:
        rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0, step=0.1)
        st.markdown("---")
        st.markdown("**Reference Ranges:**")
        st.markdown("- N: 0–140 | P: 0–145 | K: 0–205\n- pH: 5.5–7.5 ideal\n- Humidity: 50–80% ideal")
    
    if st.button("🌾 Predict Best Crop", type="primary", use_container_width=True):
        if nitrogen == 0 and phosphorus == 0 and potassium == 0:
            st.error("⚠️ Please enter at least one soil parameter with a non-zero value.")
        else:
            predicted_crop = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
            if predicted_crop == "Crop model not available":
                st.warning("⚠️ Crop model could not be loaded.")
            else:
                st.markdown(f"""
                <div class="card">
                    <h3>Recommended Crop</h3>
                    <p class="metric-big">🌱 {str(predicted_crop).capitalize()}</p>
                    <p>Based on your soil and climate data, <b>{str(predicted_crop).capitalize()}</b> is the most suitable crop.</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                <div class="card-blue">
                    <h3>💡 Next Steps</h3>
                    <p>Consult your local agricultural officer for seed varieties. Check the Weather Forecast for ideal planting time. Use Disease Identification to monitor crop health.</p>
                </div>
                """, unsafe_allow_html=True)


def show_disease_detection_page():
    """Display the plant disease detection page."""
    st.markdown("<h1 style='text-align:center;'>🌿 SMART DISEASE DETECTION</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    if disease_model is None:
        st.error("⚠️ Plant disease model (.keras) not found.")
        st.markdown("""
        <div class="card-orange">
        <h3>📥 How to add the Keras model</h3>
        <p>
        Download <b>trained_plant_disease_model.keras</b> from GitHub Releases and place it in <b>PLANT-DISEASE-IDENTIFICATION/</b> folder.<br><br>
        Then restart the app.
        </p>
        </div>
        """, unsafe_allow_html=True)
        if disease_model_error:
            st.error(f"❌ Model load error: {disease_model_error}")
        return
    
    st.markdown('<div class="card"><h3>📷 Upload a Plant Leaf Image</h3><p>Supported formats: JPG, JPEG, PNG</p></div>', unsafe_allow_html=True)
    test_image = st.file_uploader("Choose a Leaf Image:", type=["jpg", "jpeg", "png"])
    
    if test_image is not None:
        col_img, col_btn = st.columns([3, 1])
        with col_img:
            st.image(test_image, caption="Uploaded Leaf Image", use_container_width=True)
        with col_btn:
            st.markdown("<br><br>", unsafe_allow_html=True)
            predict_clicked = st.button("🔍 Detect Disease", type="primary", use_container_width=True)
        
        if predict_clicked:
            with st.spinner("🔬 Analysing image with AI model..."):
                result_index, confidence = predict_disease(test_image)
            
            if result_index == -1:
                st.error("❌ Could not make a prediction. Please try with a different image.")
            else:
                detected = CLASS_NAMES[result_index]
                plant_name = detected.split("___")[0].replace("_", " ")
                disease_name = detected.split("___")[1].replace("_", " ") if "___" in detected else detected
                treatment = DISEASE_INFO.get(detected, {
                    "medicine": "Information not available",
                    "dosage": "Consult an agricultural expert",
                    "prevention": "Contact your local agricultural officer",
                    "fertilizer": "Use balanced fertilizer",
                    "severity": "Unknown"
                })
                
                st.markdown("---")
                st.subheader("📊 Detection Results")
                col1, col2, col3 = st.columns(3)
                col1.metric("🌱 Plant", plant_name)
                col2.metric("🦠 Condition", disease_name)
                col3.metric("🎯 Confidence", f"{confidence}%")
                
                is_healthy = "healthy" in detected.lower()
                if is_healthy:
                    st.markdown(f"""
                    <div class="card">
                        <h3>✅ Plant is Healthy!</h3>
                        <p>Great news! Your <b>{plant_name}</b> plant appears healthy with <b>{confidence}%</b> confidence.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    sev = treatment.get("severity", "Unknown")
                    sev_color = {"None": "#4CAF50", "Moderate": "#FF9800", "High": "#F44336", "Very High": "#B71C1C"}.get(sev, "#666")
                    st.markdown(f"""
                    <div class="card-red">
                        <h3>🦠 Detected: {disease_name}</h3>
                        <p>Plant: <b>{plant_name}</b> | Confidence: <b>{confidence}%</b> | Severity: <b style="color:{sev_color}">{sev}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.subheader("💊 Treatment & Medicine Guide")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="card">
                        <h3>💊 Recommended Medicine</h3>
                        <p><b>{treatment['medicine']}</b></p>
                        <p>📏 <b>Dosage:</b> {treatment['dosage']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="card">
                        <h3>🌱 Fertilizer Advice</h3>
                        <p>{treatment['fertilizer']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="card-orange">
                        <h3>🛡️ Prevention Tips</h3>
                        <p>{treatment['prevention']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    sev = treatment.get('severity', 'N/A')
                    st.markdown(f"""
                    <div class="card-blue">
                        <h3>⚠️ Severity Level</h3>
                        <p><b style="font-size:1.3rem;">{sev}</b></p>
                        <p>{'Act immediately and consult your local agricultural officer.' if 'Very High' in str(sev) else 'Take necessary precautions and treat early.'}</p>
                    </div>
                    """, unsafe_allow_html=True)


def show_weather_forecast_page():
    """Display the weather forecast and advisory page."""
    st.markdown("<h1 style='text-align:center;'>🌤️ WEATHER FORECAST & FARMING ADVISORY</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="card-blue"><h3>🏙️ Enter Your City Name</h3><p>Real-time weather powered by wttr.in – No API key required.</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
    with col1:
        city = st.text_input("City", placeholder="e.g. Mumbai, Delhi, Pune, Nagpur, Ahmedabad", label_visibility="collapsed")
    with col2:
        search = st.button("🔍 Get Weather", type="primary", use_container_width=True)
    
    if search and city.strip():
        with st.spinner(f"Fetching weather for {city}..."):
            result, error = get_weather(city.strip())
        
        if error:
            st.error(f"❌ {error}")
            st.info("💡 Tips: Check city spelling. Use English names (e.g. 'Mumbai' not 'मुंबई')")
        else:
            curr = result["current"]
            forecast = result["forecast"]
            st.markdown("---")
            st.subheader(f"📍 Current Weather – {city.title()}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🌡️ Temperature", f"{curr['temp_c']}°C", f"Feels like {curr['feels_like_c']}°C")
            col2.metric("💧 Humidity", f"{curr['humidity']}%")
            col3.metric("💨 Wind", f"{curr['wind_kph']} km/h", curr['wind_dir'])
            col4.metric("🌧️ Precipitation", f"{curr['precip_mm']} mm")
            
            st.markdown(f"""
            <div class="card-blue">
                <h3>🌤️ {curr['desc']}</h3>
                <p>☁️ Cloud Cover: <b>{curr['cloud_cover']}%</b> &nbsp;|&nbsp;
                👁️ Visibility: <b>{curr['visibility']} km</b> &nbsp;|&nbsp;
                📊 Pressure: <b>{curr['pressure']} hPa</b> &nbsp;|&nbsp;
                ☀️ UV Index: <b>{curr['uv_index']}</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("🌾 Farming Advisory for Today")
            tips = generate_farming_advice(curr["temp_c"], curr["humidity"], curr["wind_kph"], curr["precip_mm"])
            for tip in tips:
                st.markdown(tip)
            
            if forecast:
                st.markdown("---")
                st.subheader("📅 3-Day Forecast")
                cols = st.columns(len(forecast))
                for i, day in enumerate(forecast):
                    with cols[i]:
                        st.markdown(f"""
                        <div class="card">
                            <h3>📅 {day['date']}</h3>
                            <p>🌡️ Max: <b>{day['max_c']}°C</b> | Min: <b>{day['min_c']}°C</b></p>
                            <p>🌤️ {day['desc']}</p>
                            <p>🌧️ Rain Chance: <b>{day['rain_chance']}%</b></p>
                            <p>🌅 {day['sunrise']} / 🌇 {day['sunset']}</p>
                            <p>☀️ UV: {day['uvIndex']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.subheader("📈 Temperature Trend (3 Days)")
                chart_data = pd.DataFrame({
                    "Date": [d["date"] for d in forecast],
                    "Max Temp (°C)": [int(d["max_c"]) for d in forecast],
                    "Min Temp (°C)": [int(d["min_c"]) for d in forecast],
                })
                st.line_chart(chart_data.set_index("Date"))
    
    elif search and not city.strip():
        st.warning("⚠️ Please enter a city name.")
    
    st.markdown("---")
    st.caption("🌍 Weather data provided by wttr.in | No API key required | Real-time data")


if __name__ == '__main__':
    main()
