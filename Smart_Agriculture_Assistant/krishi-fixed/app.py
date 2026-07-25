import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import requests
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# PAGE CONFIG
st.set_page_config(
    page_title="KRISHI – Smart Agriculture Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# PATHS
CROP_IMAGE_PATH    = "CROP-RECOMMENDATION/crop.png"
CROP_MODEL_PATH    = "CROP-RECOMMENDATION/RF.pkl"
DISEASE_MODEL_PATH = "PLANT-DISEASE-IDENTIFICATION/trained_plant_disease_model.keras"
DISEASE_IMAGE_PATH = "PLANT-DISEASE-IDENTIFICATION/Diseases.png"

# DISEASE → MEDICINE DICTIONARY
disease_info = {
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
    "Cherry_(including_sour)___Powdery_mildew": {
        "medicine": "Sulphur 80% WP spray",
        "dosage": "2 gm per litre of water",
        "prevention": "Prune infected shoots immediately. Avoid overhead irrigation. Improve air circulation.",
        "fertilizer": "Reduce nitrogen. Apply potassium fertilizer to strengthen the plant.",
        "severity": "Moderate"
    },
    "Cherry_(including_sour)___healthy": {
        "medicine": "No medicine needed",
        "dosage": "Not applicable",
        "prevention": "Continue regular monitoring. Maintain proper pruning for air circulation.",
        "fertilizer": "Apply balanced NPK fertilizer as per cherry orchard schedule.",
        "severity": "None"
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "medicine": "Azoxystrobin 23% SC spray",
        "dosage": "1 ml per litre of water",
        "prevention": "Crop rotation. Avoid planting corn in the same field every year. Use resistant varieties.",
        "fertilizer": "Apply balanced nitrogen. Avoid excess nitrogen as it worsens the disease.",
        "severity": "High"
    },
    "Corn_(maize)___Common_rust_": {
        "medicine": "Propiconazole 25% EC spray",
        "dosage": "1 ml per litre of water",
        "prevention": "Use resistant varieties. Early planting helps avoid peak infection period.",
        "fertilizer": "Apply nitrogen fertilizer carefully. Excess nitrogen worsens rust.",
        "severity": "Moderate"
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "medicine": "Mancozeb 75% WP + Propiconazole 25% EC",
        "dosage": "2 gm + 0.5 ml per litre of water",
        "prevention": "Use resistant hybrids. Rotate crops. Remove crop debris after harvest.",
        "fertilizer": "Apply balanced NPK. Ensure good potassium for disease resistance.",
        "severity": "High"
    },
    "Corn_(maize)___healthy": {
        "medicine": "No medicine needed",
        "dosage": "Not applicable",
        "prevention": "Continue good farming practices. Monitor regularly for early signs.",
        "fertilizer": "Apply urea (nitrogen) at recommended doses for maize growth stages.",
        "severity": "None"
    },
    "Grape___Black_rot": {
        "medicine": "Myclobutanil 10% WP spray",
        "dosage": "1 ml per litre of water",
        "prevention": "Remove infected berries and leaves. Improve air circulation. Apply fungicide before rain events.",
        "fertilizer": "Apply potassium fertilizer before the fruiting stage.",
        "severity": "High"
    },
    "Grape___Esca_(Black_Measles)": {
        "medicine": "Trichoderma-based biocontrol as alternative",
        "dosage": "As per expert recommendation",
        "prevention": "Prune during dry weather. Seal pruning wounds immediately. Remove and destroy infected wood.",
        "fertilizer": "Apply balanced NPK. Maintain vine vigour with proper fertilization.",
        "severity": "Very High"
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "medicine": "Copper oxychloride 50% WP spray",
        "dosage": "3 gm per litre of water",
        "prevention": "Remove infected leaves. Ensure good air circulation between vines.",
        "fertilizer": "Apply potassium and calcium fertilizer for stronger cell walls.",
        "severity": "Moderate"
    },
    "Grape___healthy": {
        "medicine": "No medicine needed",
        "dosage": "Not applicable",
        "prevention": "Continue good vineyard management. Maintain pruning and canopy management.",
        "fertilizer": "Apply balanced NPK fertilizer as per grape growth stage.",
        "severity": "None"
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "medicine": "Oxytetracycline antibiotic injection (trunk injection method)",
        "dosage": "As per expert recommendation",
        "prevention": "Control Asian citrus psyllid (vector insect). Remove and destroy infected trees immediately.",
        "fertilizer": "Apply zinc and micronutrient fertilizer to slow symptom progression.",
        "severity": "Very High"
    },
    "Peach___Bacterial_spot": {
        "medicine": "Copper hydroxide 77% WP spray",
        "dosage": "2 gm per litre of water",
        "prevention": "Use disease-resistant varieties. Avoid overhead watering. Apply copper sprays in spring.",
        "fertilizer": "Avoid excess nitrogen. Apply potassium and calcium fertilizer.",
        "severity": "Moderate"
    },
    "Peach___healthy": {
        "medicine": "No medicine needed",
        "dosage": "Not applicable",
        "prevention": "Continue regular monitoring. Maintain proper pruning and orchard sanitation.",
        "fertilizer": "Apply balanced NPK fertilizer as per peach orchard schedule.",
        "severity": "None"
    },
    "Pepper,_bell___Bacterial_spot": {
        "medicine": "Copper oxychloride 50% WP + Mancozeb 75% WP spray",
        "dosage": "2.5 gm per litre of water",
        "prevention": "Use disease-free transplants. Avoid overhead watering. Rotate crops regularly.",
        "fertilizer": "Avoid excess nitrogen. Apply calcium fertilizer for cell wall strength.",
        "severity": "High"
    },
    "Pepper,_bell___healthy": {
        "medicine": "No medicine needed",
        "dosage": "Not applicable",
        "prevention": "Continue good farming practices. Monitor regularly for early signs.",
        "fertilizer": "Apply balanced NPK. Calcium-rich fertilizer improves fruit quality.",
        "severity": "None"
    },
    "Potato___Early_blight": {
        "medicine": "Chlorothalonil 75% WP spray",
        "dosage": "2 gm per litre of water",
        "prevention": "Crop rotation. Remove and destroy infected leaves immediately.",
        "fertilizer": "Apply balanced NPK before planting. Ensure adequate potassium.",
        "severity": "Moderate"
    },
    "Potato___Late_blight": {
        "medicine": "Metalaxyl 8% + Mancozeb 64% WP spray",
        "dosage": "2.5 gm per litre of water",
        "prevention": "Use certified disease-free seeds. Destroy infected plants completely. Avoid overhead irrigation.",
        "fertilizer": "Apply potassium fertilizer to boost plant resistance.",
        "severity": "Very High"
    },
    "Potato___healthy": {
        "medicine": "No medicine needed",
        "dosage": "Not applicable",
        "prevention": "Continue good farming practices. Monitor regularly.",
        "fertilizer": "Apply balanced NPK as per potato growth stage.",
        "severity": "None"
    },
    "Raspberry___healthy": {
        "medicine": "No medicine needed",
        "dosage": "Not applicable",
        "prevention": "Continue regular monitoring. Maintain proper pruning and air circulation.",
        "fertilizer": "Apply nitrogen-rich fertilizer in spring. Potassium improves fruit quality.",
        "severity": "None"
    },
    "Soybean___healthy": {
        "medicine": "No medicine needed",
        "dosage": "Not applicable",
        "prevention": "Continue good farming practices. Use good quality certified seeds.",
        "fertilizer": "Soybeans fix nitrogen. Apply phosphorus and potassium as needed.",
        "severity": "None"
    },
    "Squash___Powdery_mildew": {
        "medicine": "Sulphur 80% WP spray or Neem oil spray",
        "dosage": "2 gm per litre of water (Sulphur) or 5 ml per litre (Neem oil)",
        "prevention": "Improve air circulation. Avoid overhead watering. Remove heavily infected leaves.",
        "fertilizer": "Reduce nitrogen application. Apply potassium to strengthen plant resistance.",
        "severity": "Moderate"
    },
    "Strawberry___Leaf_scorch": {
        "medicine": "Copper hydroxide 77% WP spray",
        "dosage": "2 gm per litre of water",
        "prevention": "Remove old and infected leaves. Avoid overhead irrigation. Use certified planting material.",
        "fertilizer": "Apply balanced NPK fertilizer. Avoid excess nitrogen.",
        "severity": "Moderate"
    },
    "Strawberry___healthy": {
        "medicine": "No medicine needed",
        "dosage": "Not applicable",
        "prevention": "Continue regular monitoring. Maintain proper irrigation and mulching.",
        "fertilizer": "Apply balanced NPK. Calcium fertilizer improves fruit firmness.",
        "severity": "None"
    },
    "Tomato___Bacterial_spot": {
        "medicine": "Copper-based bactericide (Copper oxychloride 50% WP)",
        "dosage": "2.5 gm per litre of water",
        "prevention": "Use disease-free seeds. Avoid working in wet fields. Rotate crops annually.",
        "fertilizer": "Avoid excess nitrogen fertilizer. Apply potassium for resistance.",
        "severity": "High"
    },
    "Tomato___Early_blight": {
        "medicine": "Mancozeb 75% WP spray",
        "dosage": "2.5 gm per litre of water",
        "prevention": "Remove infected leaves immediately. Avoid overhead watering. Mulch around plants.",
        "fertilizer": "Apply potassium-rich fertilizer to strengthen the plant.",
        "severity": "Moderate"
    },
    "Tomato___Late_blight": {
        "medicine": "Metalaxyl + Mancozeb spray",
        "dosage": "2 gm per litre of water",
        "prevention": "Destroy infected plants. Use certified disease-free seeds. Improve drainage.",
        "fertilizer": "Apply balanced NPK fertilizer. Avoid waterlogged conditions.",
        "severity": "Very High"
    },
    "Tomato___Leaf_Mold": {
        "medicine": "Copper Oxychloride 50% WP spray",
        "dosage": "3 gm per litre of water",
        "prevention": "Improve air circulation. Reduce humidity. Avoid wetting foliage during irrigation.",
        "fertilizer": "Apply calcium and potassium fertilizer for stronger leaves.",
        "severity": "Moderate"
    },
    "Tomato___Septoria_leaf_spot": {
        "medicine": "Chlorothalonil 75% WP spray",
        "dosage": "2 gm per litre of water",
        "prevention": "Remove infected leaves promptly. Avoid overhead watering. Stake plants for air circulation.",
        "fertilizer": "Apply balanced NPK. Avoid waterlogged conditions.",
        "severity": "Moderate"
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "medicine": "Abamectin 1.8% EC spray (Miticide)",
        "dosage": "0.5 ml per litre of water",
        "prevention": "Maintain adequate irrigation. Avoid dusty conditions. Introduce predatory mites (biological control).",
        "fertilizer": "Apply balanced NPK. Avoid excess nitrogen which encourages mite multiplication.",
        "severity": "High"
    },
    "Tomato___Target_Spot": {
        "medicine": "Azoxystrobin 23% SC + Difenoconazole 11.4% SC",
        "dosage": "0.5 ml per litre of water",
        "prevention": "Remove infected leaves. Avoid overhead watering. Improve field drainage.",
        "fertilizer": "Apply potassium and calcium fertilizer for resistance.",
        "severity": "High"
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "medicine": "Imidacloprid 17.8% SL spray (controls whitefly vector)",
        "dosage": "0.5 ml per litre of water",
        "prevention": "Control whitefly population (vector). Use yellow sticky traps. Remove and destroy infected plants.",
        "fertilizer": "Apply balanced NPK. Avoid excess nitrogen which attracts whiteflies.",
        "severity": "Very High"
    },
    "Tomato___Tomato_mosaic_virus": {
        "medicine": "No direct cure. Control aphid vectors with Imidacloprid 17.8% SL",
        "dosage": "0.5 ml per litre of water",
        "prevention": "Remove and destroy infected plants. Disinfect tools. Use resistant varieties. Control aphids.",
        "fertilizer": "Maintain plant vigour with balanced NPK fertilizer.",
        "severity": "Very High"
    },
    "Tomato___healthy": {
        "medicine": "No medicine needed",
        "dosage": "Not applicable",
        "prevention": "Continue good farming practices. Monitor regularly for early signs of disease.",
        "fertilizer": "Apply regular balanced NPK fertilizer as per tomato growth stage.",
        "severity": "None"
    },
}

# CLASS NAMES
class_name = [
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

# === AUTO-EXTRACT MODEL ===
import os, zipfile, glob, shutil
_model_dest = "PLANT-DISEASE-IDENTIFICATION/trained_plant_disease_model.keras"
st.warning("Running Auto-Locator for .keras file...")
if not os.path.exists(_model_dest):
    found_keras = False
    _downloads_path = r"c:\Users\Jay Jangam\Downloads"
    # Search for any .keras file in Downloads
    st.info(f"Scanning {_downloads_path} for .keras files...")
    for root, dirs, files in os.walk(_downloads_path):
        for file in files:
            if file.endswith('.keras'):
                _src_path = os.path.join(root, file)
                st.success(f"Found model at: {_src_path}")
                os.makedirs("PLANT-DISEASE-IDENTIFICATION", exist_ok=True)
                shutil.copy(_src_path, _model_dest)
                st.write(f"✅ Copied to {_model_dest}!")
                found_keras = True
                break
        if found_keras: break
    
    # If not found directly, check inside ai_model.zip just in case
    if not found_keras:
        _zip_path = os.path.join(_downloads_path, "ai_model.zip")
        if os.path.exists(_zip_path):
            st.info("Searching inside ai_model.zip...")
            try:
                with zipfile.ZipFile(_zip_path, 'r') as z:
                    for _f in z.namelist():
                        if _f.endswith('.keras'):
                            with z.open(_f) as _src, open(_model_dest, 'wb') as _tgt:
                                shutil.copyfileobj(_src, _tgt)
                            st.success(f"Auto-extracted {_f} from ai_model.zip!")
                            found_keras = True
                            break
            except Exception as e:
                st.error(f"Auto-extract failed: {e}")
    if not found_keras:
        st.error("❌ No .keras files found in Downloads folder or ai_model.zip!")

# MODEL LOADING (cached)
@st.cache_resource(show_spinner="Loading crop recommendation model...")
def load_crop_model():
    if not os.path.exists(CROP_MODEL_PATH):
        return None
    try:
        with open(CROP_MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading crop model: {e}")
        return None

@st.cache_resource(show_spinner="Loading plant disease model...")
def load_disease_model():
    if not os.path.exists(DISEASE_MODEL_PATH):
        return None, "not_found"
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(DISEASE_MODEL_PATH)
        return model, "ok"
    except Exception as e:
        return None, str(e)

RF_Model_pkl = load_crop_model()
disease_model, disease_model_status = load_disease_model()

# PREDICT FUNCTIONS
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    if RF_Model_pkl is None:
        return "Model not loaded"
    pred = RF_Model_pkl.predict(
        np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]).reshape(1, -1)
    )
    return pred[0]

def model_prediction(test_image):
    if disease_model is None:
        return -1, 0.0
    try:
        import tensorflow as tf
        image = Image.open(test_image).convert("RGB")
        image = image.resize((128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.expand_dims(input_arr, axis=0)
        predictions = disease_model.predict(input_arr)
        index = int(np.argmax(predictions))
        confidence = float(np.max(predictions)) * 100
        return index, round(confidence, 2)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return -1, 0.0

# WEATHER FUNCTION
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

def get_weather(city):
    try:
        url = f"https://wttr.in/{city}?format=j1"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None, f"City not found or service unavailable (HTTP {resp.status_code})"
        data = resp.json()
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
        return {"current": weather, "forecast": forecast, "city": city}, None
    except requests.exceptions.ConnectionError:
        return None, "No internet connection. Please check your network."
    except Exception as e:
        return None, f"Error fetching weather: {e}"

def farming_advice_from_weather(temp_c, humidity, wind_kph, precip_mm):
    tips = []
    t = int(temp_c)
    h = int(humidity)
    w = int(wind_kph)
    p = float(precip_mm)
    if t > 38:
        tips.append("🌡️ **Extreme Heat Alert**: Increase irrigation frequency. Protect crops with shade nets.")
    elif t > 32:
        tips.append("🌡️ **High Temperature**: Monitor crops for heat stress. Water in early morning or evening.")
    elif t < 5:
        tips.append("🥶 **Cold Alert**: Cover sensitive crops to protect from frost damage.")
    if h > 85:
        tips.append("💧 **High Humidity**: High risk of fungal diseases (blight, mold). Consider preventive fungicide.")
    elif h < 30:
        tips.append("🏜️ **Low Humidity**: Increase irrigation. Monitor for spider mites which thrive in dry conditions.")
    if p > 10:
        tips.append("🌧️ **Heavy Rainfall**: Avoid spraying pesticides/fertilizers. Check field drainage.")
    elif p > 0:
        tips.append("🌦️ **Light Rain**: Good for recently applied fertilizers. Monitor for waterlogging.")
    else:
        tips.append("☀️ **No Rain**: Ensure timely irrigation. Mulch to conserve soil moisture.")
    if w > 30:
        tips.append("💨 **Strong Winds**: Postpone spraying operations to avoid chemical drift.")
    if not tips:
        tips.append("✅ **Ideal Farming Conditions**: Proceed with regular farming activities.")
    return tips

# STYLES
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

# SIDEBAR
st.sidebar.title("🌾 KRISHI")
st.sidebar.markdown("*Smart Agriculture Assistant*")
st.sidebar.markdown("---")
if os.path.exists(CROP_IMAGE_PATH):
    try:
        st.sidebar.image(Image.open(CROP_IMAGE_PATH), caption="Crop Recommendation", use_container_width=True)
    except Exception:
        pass
if os.path.exists(DISEASE_IMAGE_PATH):
    try:
        st.sidebar.image(Image.open(DISEASE_IMAGE_PATH), caption="Disease Recognition", use_container_width=True)
    except Exception:
        pass
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox(
    "Select Feature",
    ["Home / Dashboard", "Crop Recommendation", "Plant Disease Identification", "Weather Forecast"],
)
st.sidebar.markdown("---")
st.sidebar.markdown("**KRISHI v2.0** | Smart Agriculture")

# ═══════════ PAGE: HOME ═══════════
if app_mode == "Home / Dashboard":
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
    for cls in class_name:
        info = disease_info.get(cls, {})
        plant = cls.split("___")[0].replace("_", " ")
        disease = cls.split("___")[1].replace("_", " ") if "___" in cls else cls
        disease_df.append({
            "Plant": plant,
            "Disease": disease,
            "Severity": info.get("severity", "N/A"),
            "Medicine": info.get("medicine", "N/A"),
        })
    st.dataframe(pd.DataFrame(disease_df), use_container_width=True)

# ═══════════ PAGE: CROP RECOMMENDATION ═══════════
elif app_mode == "Crop Recommendation":
    st.markdown("<h1 style='text-align:center;'>🌾 SMART CROP RECOMMENDATIONS</h1>", unsafe_allow_html=True)
    st.markdown("---")
    if RF_Model_pkl is None:
        st.error("Crop recommendation model not found at CROP-RECOMMENDATION/RF.pkl. Please ensure the file exists and restart.")
    else:
        st.markdown('<div class="card"><h3>📊 Enter Soil & Climate Details</h3><p>Fill in values based on your soil test report and local climate.</p></div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            nitrogen   = st.number_input("🧪 Nitrogen (N)", min_value=0.0, max_value=140.0, value=0.0, step=0.1)
            phosphorus = st.number_input("🧪 Phosphorus (P)", min_value=0.0, max_value=145.0, value=0.0, step=0.1)
            potassium  = st.number_input("🧪 Potassium (K)", min_value=0.0, max_value=205.0, value=0.0, step=0.1)
        with col2:
            temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=51.0, value=25.0, step=0.1)
            humidity    = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
            ph          = st.number_input("⚗️ pH Level", min_value=0.0, max_value=14.0, value=6.5, step=0.01)
        with col3:
            rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0, step=0.1)
            st.markdown("---")
            st.markdown("**Reference Ranges:**")
            st.markdown("- N: 0–140 | P: 0–145 | K: 0–205\n- pH: 5.5–7.5 ideal\n- Humidity: 50–80% ideal")
        if st.button("🌾 Predict Best Crop", type="primary", use_container_width=True):
            inputs = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
            if (inputs == 0).all():
                st.error("Please fill in at least one input field with a non-zero value before predicting.")
            else:
                predicted_crop = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
                if predicted_crop == "Model not loaded":
                    st.warning("Crop recommendation model could not be loaded.")
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

# ═══════════ PAGE: PLANT DISEASE IDENTIFICATION ═══════════
elif app_mode == "Plant Disease Identification":
    st.markdown("<h1 style='text-align:center;'>🌿 SMART DISEASE DETECTION</h1>", unsafe_allow_html=True)
    st.markdown("---")
    if disease_model is None:
        st.error("⚠️ Plant disease model (.keras) not found.")
        st.markdown("""
        <div class="card-orange">
        <h3>📥 How to add the Keras model</h3>
        <p>
        Place <b>trained_plant_disease_model.keras</b> inside the <b>PLANT-DISEASE-IDENTIFICATION/</b> folder, then restart the app.<br><br>
        <b>Option A – Local (simplest):</b><br>
        Copy the .keras file to PLANT-DISEASE-IDENTIFICATION/ and restart.<br><br>
        <b>Option B – Google Drive:</b><br>
        Set env variable: DISEASE_MODEL_GDRIVE_ID=your_file_id<br><br>
        <b>Option C – Hugging Face Hub:</b><br>
        Set env variable: DISEASE_MODEL_HF_REPO=your-username/krishi-models
        </p>
        </div>
        """, unsafe_allow_html=True)
        if disease_model_status != "not_found":
            st.error(f"Model load error details: {disease_model_status}")
        st.markdown("---")
        st.subheader("⬇️ Download Model from Remote Source")
        gdrive_id = st.text_input("Google Drive File ID", placeholder="e.g. 1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs")
        hf_repo   = st.text_input("Hugging Face Repo", placeholder="e.g. your-username/krishi-models")
        if st.button("⬇️ Download Model", type="primary"):
            os.makedirs("PLANT-DISEASE-IDENTIFICATION", exist_ok=True)
            if hf_repo:
                try:
                    from huggingface_hub import hf_hub_download
                    with st.spinner("Downloading from Hugging Face..."):
                        hf_hub_download(repo_id=hf_repo, filename="trained_plant_disease_model.keras",
                                        local_dir="PLANT-DISEASE-IDENTIFICATION")
                    st.success("Downloaded! Please restart the app.")
                except Exception as e:
                    st.error(f"HF download failed: {e}")
            elif gdrive_id:
                try:
                    import gdown
                    url = f"https://drive.google.com/uc?id={gdrive_id}"
                    with st.spinner("Downloading from Google Drive..."):
                        gdown.download(url, DISEASE_MODEL_PATH, quiet=False)
                    if os.path.exists(DISEASE_MODEL_PATH):
                        st.success("Downloaded! Please restart the app.")
                    else:
                        st.error("Download finished but file not found.")
                except Exception as e:
                    st.error(f"GDrive download failed: {e}")
            else:
                st.warning("Please enter a Google Drive File ID or Hugging Face repo name.")
    else:
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
                    result_index, confidence = model_prediction(test_image)
                if result_index == -1:
                    st.error("Could not make a prediction. Model might not be loaded correctly.")
                else:
                    detected = class_name[result_index]
                    plant_name = detected.split("___")[0].replace("_", " ")
                    disease_name = detected.split("___")[1].replace("_", " ") if "___" in detected else detected
                    treatment = disease_info.get(detected, {
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
                        sev = treatment.get('severity','N/A')
                        st.markdown(f"""
                        <div class="card-blue">
                            <h3>⚠️ Severity Level</h3>
                            <p><b style="font-size:1.3rem;">{sev}</b></p>
                            <p>{'Act immediately and consult your local agricultural officer.' if 'Very High' in str(sev) else 'Take necessary precautions and treat early.'}</p>
                        </div>
                        """, unsafe_allow_html=True)

# ═══════════ PAGE: WEATHER FORECAST ═══════════
elif app_mode == "Weather Forecast":
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
            st.info("Tips: Check city spelling. Use English names (e.g. 'Mumbai' not 'मुंबई')")
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
            tips = farming_advice_from_weather(curr["temp_c"], curr["humidity"], curr["wind_kph"], curr["precip_mm"])
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
    st.caption("Weather data provided by wttr.in | No API key required | Real-time data")
