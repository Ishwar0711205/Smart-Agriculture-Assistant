# 🌾 KRISHI – Smart Agriculture Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 👨‍💻 Team Members
- **Ishwar Garje** (Team Leader)
- **Jay Jangam**
- **Akash Misale**

An AI-powered Streamlit application that empowers farmers with intelligent agricultural solutions.

## ✨ Features

### 🌾 Crop Recommendation
- **AI-Powered Predictions**: Random Forest model trained on soil and climate data
- **22 Crop Types**: Supports diverse crop recommendations
- **Easy Input**: Soil parameters (N, P, K) and climate data (Temperature, Humidity, pH, Rainfall)
- **Instant Results**: Get crop recommendations in seconds

### 🌿 Plant Disease Detection
- **38-Class CNN Model**: Identifies plant diseases and health status
- **High Accuracy**: Deep learning model trained on leaf images
- **Comprehensive Treatment Guide**:
  - 💊 Recommended medicines & dosages
  - 🛡️ Prevention strategies
  - 🌱 Fertilizer recommendations
  - ⚠️ Severity assessment

### 🌤️ Weather Forecast & Advisory
- **Real-Time Data**: Current weather for any city worldwide
- **3-Day Forecast**: Temperature trends and predictions
- **Farming Tips**: AI-generated advisory based on weather conditions
- **No API Key Required**: Uses free wttr.in service

### 📊 Dashboard
- **Quick Overview**: All features at a glance
- **Disease Reference**: Complete table of 38 supported diseases
- **How-to Guide**: Step-by-step usage instructions

---

## 📋 Supported Crops & Diseases

### Supported Crops (22)
Rice, Maize, Jute, Cotton, Coconut, Sugarcane, Groundnut, Rapeseed, Mustard, Barley, Wheat, Millets, Oil Seeds, Pulses, Tea, Tobacco, Potato, Chicpea, Kidneybeans, Pigeonpeas, Mothbeans, Mungbean

### Supported Diseases (38)
Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato with various disease conditions

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/Ishwar0711205/Smart-Agriculture-Assistant.git
cd Smart-Agriculture-Assistant
```

**2. Install dependencies**
```bash
pip install -r Smart_Agriculture_Assistant/requirements.txt
```

**3. Add the Keras model** (⚠️ Important)

The Keras model file is large (~100MB+) and available separately:

#### Option A: Download from GitHub Releases (Recommended)
```bash
# Download trained_plant_disease_model.keras from Releases
# Extract to: Smart_Agriculture_Assistant/PLANT-DISEASE-IDENTIFICATION/
```

#### Option B: From Google Drive
```bash
# Set environment variable (Windows)
set DISEASE_MODEL_GDRIVE_ID=your_file_id

# Or (Linux/Mac)
export DISEASE_MODEL_GDRIVE_ID=your_file_id
```

#### Option C: From Hugging Face Hub
```bash
# Set environment variable (Windows)
set DISEASE_MODEL_HF_REPO=username/model-repo

# Or (Linux/Mac)
export DISEASE_MODEL_HF_REPO=username/model-repo
```

**4. Run the application**
```bash
cd Smart_Agriculture_Assistant
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📁 Project Structure

```
Smart-Agriculture-Assistant/
├── README.md                               # Main documentation
├── Smart_Agriculture_Assistant/
│   ├── app.py                             # Main Streamlit application
│   ├── requirements.txt                   # Python dependencies
│   ├── model_loader.py                   # Model loading utilities
│   ├── model_downloader.py               # GitHub Release downloader
│   ├── constants.py                      # Centralized constants
│   ├── CROP-RECOMMENDATION/
│   │   ├── app.py                        # Crop recommendation app (refactored)
│   │   ├── RF.pkl                        # Random Forest model
│   │   ├── crop.png                      # UI image
│   │   └── Crop_recommendation.csv       # Training data reference
│   └── PLANT-DISEASE-IDENTIFICATION/
│       ├── trained_plant_disease_model.keras  # CNN model (download separately)
│       └── Diseases.png                       # UI image
└── CODE_REVIEW.md                         # Code review & improvements
```

---

## 🔧 Technical Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **Crop Model** | Random Forest (scikit-learn) |
| **Disease Model** | CNN (TensorFlow/Keras) |
| **Image Processing** | PIL/Pillow |
| **Weather API** | wttr.in (Free) |
| **Model Caching** | Streamlit `@st.cache_resource` |

---

## 📊 Model Specifications

### Crop Recommendation Model
- **Type**: Random Forest Classifier
- **File**: `RF.pkl`
- **Input Features**: 7 (N, P, K, Temperature, Humidity, pH, Rainfall)
- **Output Classes**: 22 crops
- **Training Data**: Indian crop recommendation dataset

### Disease Detection Model
- **Type**: Convolutional Neural Network (CNN)
- **File**: `trained_plant_disease_model.keras`
- **Input Size**: 128×128 RGB images
- **Output Classes**: 38 disease states
- **Framework**: TensorFlow/Keras
- **File Size**: ~100MB+

---

## 🎯 How to Use

### Crop Recommendation
1. Navigate to **"Crop Recommendation"** from the sidebar
2. Enter your soil parameters:
   - Nitrogen (N): 0-140 mg/kg
   - Phosphorus (P): 0-145 mg/kg
   - Potassium (K): 0-205 mg/kg
3. Enter climate data:
   - Temperature: 0-51°C
   - Humidity: 0-100%
   - pH Level: 0-14
   - Rainfall: 0-500 mm
4. Click **"Predict Best Crop"**
5. Get instant AI recommendation with guidance

### Disease Detection
1. Go to **"Plant Disease Identification"**
2. Upload a leaf image (JPG, JPEG, or PNG)
3. Click **"Detect Disease"**
4. View results with:
   - Detected disease name
   - Confidence percentage
   - Severity level
   - Treatment recommendations

### Weather Forecast
1. Select **"Weather Forecast"** from sidebar
2. Enter your city name
3. Click **"Get Weather"**
4. View:
   - Current weather conditions
   - 3-day forecast
   - Temperature trends
   - Farming advisory tips

---

## ⚠️ Troubleshooting

### "Keras model not found"
- **Solution**: Download `trained_plant_disease_model.keras` from Releases and place it in `PLANT-DISEASE-IDENTIFICATION/`
- The model file is too large for GitHub and must be downloaded separately

### "Crop model error"
- **Solution**: Ensure `CROP-RECOMMENDATION/RF.pkl` exists
- Run from the correct directory: `cd Smart_Agriculture_Assistant`

### "City not found" (Weather)
- **Solution**: Use English city names (e.g., "Mumbai" not "मुंबई")
- Check spelling carefully

### App runs slowly
- **Solution**: First load is slow as models are cached. Subsequent interactions are fast.
- Close and restart if models don't load

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📝 Code Quality

This project includes:
- ✅ PEP 8 compliant code
- ✅ Comprehensive docstrings
- ✅ Error handling for missing files
- ✅ Input validation
- ✅ Logging for debugging
- ✅ Environment variable support

See `CODE_REVIEW.md` for detailed improvements and refactoring notes.

---

## 📄 License

MIT License – See LICENSE file for details.

---

## 👨‍💻 Author

**Ishwar** – Smart Agriculture Assistant (KRISHI)

---

## 🙏 Acknowledgments

- Dataset: Indian Crop Recommendation Dataset
- Model Training: TensorFlow/scikit-learn
- Weather Data: wttr.in
- UI Framework: Streamlit

---

## 📞 Support

For issues, questions, or suggestions:
- Open an [GitHub Issue](https://github.com/Ishwar0711205/Smart-Agriculture-Assistant/issues)
- Check [CODE_REVIEW.md](./CODE_REVIEW.md) for technical details

---

**Last Updated**: 2026-07-25 | **Version**: 2.0
