"""Constants and configuration for KRISHI application."""

from pathlib import Path

# Application metadata
APP_TITLE = "KRISHI – Smart Agriculture Assistant"
APP_VERSION = "2.0"
APP_ICON = "🌾"

# Directory paths (relative to app.py)
BASE_DIR = Path(__file__).parent
CROP_DIR = BASE_DIR / "CROP-RECOMMENDATION"
DISEASE_DIR = BASE_DIR / "PLANT-DISEASE-IDENTIFICATION"

# Model paths
CROP_MODEL_PATH = CROP_DIR / "RF.pkl"
DISEASE_MODEL_PATH = DISEASE_DIR / "trained_plant_disease_model.keras"

# Asset paths
CROP_IMAGE_PATH = CROP_DIR / "crop.png"
DISEASE_IMAGE_PATH = DISEASE_DIR / "Diseases.png"

# Model specifications
DISEASE_MODEL_INPUT_SIZE = (128, 128)
DISEASE_MODEL_CLASSES = 38
CROP_MODEL_CLASSES = 22

# Input validation ranges
NITROGEN_RANGE = (0.0, 140.0)
PHOSPHORUS_RANGE = (0.0, 145.0)
POTASSIUM_RANGE = (0.0, 205.0)
TEMPERATURE_RANGE = (0.0, 51.0)
HUMIDITY_RANGE = (0.0, 100.0)
PH_RANGE = (0.0, 14.0)
RAINFALL_RANGE = (0.0, 500.0)

# API configuration
WEATHER_API_TIMEOUT = 10  # seconds
WEATHER_API_BASE_URL = "https://wttr.in"

# Disease severity levels
SEVERITY_LEVELS = ["None", "Moderate", "High", "Very High"]
SEVERITY_COLORS = {
    "None": "#4CAF50",
    "Moderate": "#FF9800",
    "High": "#F44336",
    "Very High": "#B71C1C"
}
