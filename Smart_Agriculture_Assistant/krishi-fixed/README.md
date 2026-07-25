# 🌾 KRISHI – Smart Agriculture Assistant

A Streamlit-based AI-powered agriculture assistant that helps farmers with:
- **Crop Recommendation** using a trained Random Forest model
- **Plant Disease Detection** using a 38-class Keras CNN model
- **Medicine & Treatment Guide** for detected diseases
- **Real-time Weather Forecast** with farming advisory

---

## 📁 Project Structure

```
krishi-cloned/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── CROP-RECOMMENDATION/
│   ├── RF.pkl                      # Random Forest model (crop recommendation)
│   ├── Crop_recommendation.csv     # Dataset
│   ├── crop.png                    # Sidebar image
│   └── webapp.py                   # Standalone crop webapp (legacy)
├── PLANT-DISEASE-IDENTIFICATION/
│   ├── trained_plant_disease_model.keras  # ⚠️ You must add this file!
│   └── Diseases.png                       # Sidebar image
└── Dataset1/
    └── valid/                      # Sample validation images
```

---

## ⚙️ Setup & Installation

### Step 1 – Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2 – Add the Keras Disease Model

The Keras model file (`trained_plant_disease_model.keras`) is large and **not included** in the zip.
You must add it manually using one of these options:

#### Option A – Local (Simplest)
Copy `trained_plant_disease_model.keras` to `PLANT-DISEASE-IDENTIFICATION/` folder.

#### Option B – Google Drive
1. Upload `trained_plant_disease_model.keras` to Google Drive and make it public
2. Get the file ID from the sharing link
3. Set the environment variable:
   ```bash
   set DISEASE_MODEL_GDRIVE_ID=your_google_drive_file_id  # Windows
   export DISEASE_MODEL_GDRIVE_ID=your_google_drive_file_id  # Linux/Mac
   ```

#### Option C – Hugging Face Hub
1. Upload the model to Hugging Face Hub
2. Set the environment variable:
   ```bash
   set DISEASE_MODEL_HF_REPO=your-username/krishi-models  # Windows
   export DISEASE_MODEL_HF_REPO=your-username/krishi-models  # Linux/Mac
   ```

You can also download it directly from the app's Disease Identification page using the download form.

### Step 3 – Run the App

```bash
streamlit run app.py
```

Open your browser at: `http://localhost:8501`

---

## 🚀 Features

### 🌾 Crop Recommendation
- Enter soil parameters: Nitrogen (N), Phosphorus (P), Potassium (K)
- Enter climate data: Temperature, Humidity, pH, Rainfall
- Get AI-powered crop recommendation using Random Forest model
- Supports 22 different crops

### 🌿 Plant Disease Identification
- Upload a leaf photo (JPG/PNG)
- CNN model detects disease from 38 classes
- Full treatment guide including:
  - 💊 Recommended medicine & dosage
  - 🛡️ Prevention tips
  - 🌱 Fertilizer advice
  - ⚠️ Severity level

### 🌤️ Weather Forecast
- Enter any city name (works worldwide)
- Real-time weather: temperature, humidity, wind, UV index, etc.
- 3-day forecast with temperature trend chart
- **Farming advisory** based on current weather conditions
- No API key required (uses wttr.in)

### 🏠 Home / Dashboard
- Overview of all features
- Complete disease reference table with 38 diseases

---

## 🦠 Supported Plant Diseases (38 Classes)

| Plant | Diseases |
|-------|----------|
| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| Blueberry | Healthy |
| Cherry | Powdery Mildew, Healthy |
| Corn (Maize) | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Esca, Leaf Blight, Healthy |
| Orange | Haunglongbing (Citrus Greening) |
| Peach | Bacterial Spot, Healthy |
| Pepper (Bell) | Bacterial Spot, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery Mildew |
| Strawberry | Leaf Scorch, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## 🛠️ Technical Details

- **Frontend:** Streamlit
- **Crop Model:** Random Forest (sklearn) – `RF.pkl`
- **Disease Model:** CNN (TensorFlow/Keras) – `trained_plant_disease_model.keras`
  - Input size: 128×128 RGB
  - 38 output classes
- **Weather API:** wttr.in (free, no key needed)
- **Model Caching:** `@st.cache_resource` used for both models (loads once, no reload on interaction)

---

## ⚠️ Common Issues & Fixes

### Keras model not loading
- Ensure the file `trained_plant_disease_model.keras` exists in `PLANT-DISEASE-IDENTIFICATION/`
- The model uses `@st.cache_resource` – it will only load once on startup
- If you see a model load error, check the error message on the Disease page

### Wrong weather information
- The app uses wttr.in for real-time weather – results are always current
- Ensure the city name is spelled correctly in English
- For Indian cities: use "Mumbai" not "मुंबई"

### Crop model error
- Ensure `CROP-RECOMMENDATION/RF.pkl` exists
- Run the app from the project root directory (same level as `app.py`)

---

## 👨‍💻 Team

**KRISHI Project** – Smart Agriculture Assistant  


## Additional Files
Large files (trained model, training notebook) are available in the Releases section of this repo due to GitHub's file size limits.
Developed for Minor Project submission.

---

## 📄 License

MIT License – See LICENSE file for details.
