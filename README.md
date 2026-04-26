# Costco Route & Weather Delay Forecasting

A data pipeline and ML system for **Costco delivery routes**: from depot to warehouses, with **segment-level weather**, **weather-based delay prediction**, **severe weather classification**, and **revenue loss estimation**.

---

## Pipeline Overview

```
Weather Data → Pre-Model (Severe weather classification)
                 ├─ No severe weather → delay = 0, skip Model 1
                 └─ Severe weather detected ↓
               Model 1 (Delay % prediction: Ridge, Random Forest, XGBoost)
                 ↓
               Model 2 (Revenue loss prediction: XGBoost)
                 ↓
               Supply risk assessment & recommendations
```

| Component | What it does | Trained on |
|-----------|-------------|------------|
| **Pre-Model** | Binary classification: will severe weather occur? | 2.3M real NOAA weather station records (2021-2025) |
| **Model 1** | Regression: how much delay %? | 120K route segments with synthetic delay formula |
| **Model 2** | Regression: what is the lettuce spoilage loss rate? | 50K synthetic samples with sigmoid-based loss formula |

---

## Project Structure

```
costco_forecasting/
├── backend/
│   ├── app.py                      # Flask API server (port 5000)
│   ├── predict_has_delay.py        # Pre-Model: severe weather classifier
│   ├── predict_delay.py            # Model 1: delay % prediction
│   ├── predict_loss.py             # Model 2: loss rate prediction
│   ├── pipeline.py                 # End-to-end pipeline (Pre-Model → Model 1 → Model 2)
│   ├── recommendations.py          # Supply risk & recommendations
│   ├── build_classifier.py         # Train Pre-Model (3 classifiers)
│   ├── build_delay_model.py        # Train Model 1 (3 regressors)
│   ├── build_loss_model.py         # Train Model 2 (XGBoost)
│   ├── classifier_models/          # Pre-Model saved models (.joblib)
│   ├── delay_model_*.joblib        # Model 1 saved models
│   ├── loss_model_xgb.joblib       # Model 2 saved model
│   ├── route_utils.py              # Route/weather helpers, Google Maps integration
│   ├── data/                       # Routes, weather, source data
│   ├── .env                        # Google Maps API key
│   ├── requirements.txt            # Model 1 Python dependencies
│   ├── requirements_model2.txt     # Model 2 Python dependencies
│   ├── requirements_premodel.txt   # Pre-Model Python dependencies
│   ├── package.json                # Node.js dependencies
│   ├── README.md                   # Model 1 documentation
│   ├── README_model2.md            # Model 2 documentation
│   └── README_premodel.md          # Pre-Model documentation
├── frontend/
│   ├── index.html                  # Main page (routes + delays + classification)
│   ├── demo.html                   # Weather demo (manual input + classification)
│   ├── loss.html                   # Revenue loss dashboard
│   └── static/css/, static/js/     # Styles and scripts
├── README.md                       # This file
└── PROJECT_SUMMARY.md              # Technical summary
```

---

## Quick Start

### 1. Install dependencies

```bash
cd backend

# Python
pip install -r requirements.txt
pip install -r requirements_model2.txt
pip install -r requirements_premodel.txt

# Node.js (for route data pipeline, optional)
npm install
```

### 2. Configure Google Maps API Key

Edit `backend/.env`:
```
GOOGLE_MAPS_API_KEY=your_api_key_here
ROUTES_API_KEY=your_api_key_here
```

Required for the main page (route selection + map). The demo page works without it.

### 3. Start the server

```bash
cd backend
python3 app.py
```

Open **http://localhost:5000** in your browser.

### Pages

| URL | Page | Google Maps needed? |
|-----|------|---------------------|
| `/` | Main: select route → classify weather → predict delay → estimate loss | Yes |
| `/demo` | Demo: manually input weather → classify + predict delay | No |
| `/loss` | Loss dashboard: input delay params → predict revenue loss | No |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/classify` | POST | Pre-Model: classify weather as severe or not |
| `/api/predict` | POST | Model 1: predict delay % from weather |
| `/api/predict-loss` | POST | Model 2: predict loss rate from delay + cargo |
| `/api/route-loss` | POST | End-to-end: weather → classification → delay → loss |
| `/api/routes` | POST | Get routes via Google Directions API |
| `/api/route-delays` | POST | Predict delay along route (with classification) |
| `/api/warehouses` | GET | List Costco warehouse destinations |
| `/api/source` | GET | Tracy Depot source location |
| `/api/config` | GET | Google Maps API key for frontend |

---

## Retrain Models

```bash
cd backend

# Pre-Model (reads weather_data_2021_2025.csv, trains 3 classifiers)
python3 build_classifier.py

# Model 1 (reads route/weather data, trains Ridge/RF/XGBoost)
python3 build_delay_model.py

# Model 2 (generates synthetic data, trains XGBoost)
python3 build_loss_model.py
```

## Run Tests

```bash
cd backend

# Pre-Model tests (26 scenarios)
python3 test_classifier.py

# Pipeline tests
python3 test_pipeline.py
```

---

## Key Files Reference

| Purpose | Location |
|---------|----------|
| Flask API server | `backend/app.py` |
| Pre-Model classifier | `backend/predict_has_delay.py` |
| Delay prediction (Model 1) | `backend/predict_delay.py` |
| Loss prediction (Model 2) | `backend/predict_loss.py` |
| End-to-end pipeline | `backend/pipeline.py` |
| Pre-Model documentation | `backend/README_premodel.md` |
| Model 2 documentation | `backend/README_model2.md` |
| Technical summary | `PROJECT_SUMMARY.md` |
