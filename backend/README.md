# Backend

API, delay prediction models, and data pipeline for Costco Route Delay Forecasting.

## Structure

| Item | Description |
|------|-------------|
| **app.py** | Flask app: serves the frontend and exposes `GET /api/warehouses`, `GET /api/source`, `GET /api/config`, `POST /api/routes`, `POST /api/predict`. |
| **route_utils.py** | Loads warehouses + point weather; fetches Google Directions; computes delay per route. |
| **predict_delay.py** | Loads Ridge / Random Forest / XGBoost models; `predict_delay_pct()`, `predict_all_models()`. |
| **requirements.txt** | Python deps: numpy, scikit-learn, joblib, xgboost, flask, python-dotenv. |
| **.env** | Set `GOOGLE_MAPS_API_KEY` or `ROUTES_API_KEY` for Directions API and optional frontend map. Node pipeline also uses `ROUTES_API_KEY`. |

## Run (from this directory)

```bash
# Python API + frontend
pip install -r requirements.txt
python build_delay_model.py   # once, to create .joblib files
python app.py                 # serves UI at http://127.0.0.1:5000

# Node pipeline (routes + weather)
npm install
# Then run in order: get_routes.js → decode/simplify → add_weather_to_routes.js → add_segment_time_to_routes.js
```

## Test models

```bash
python predict_delay.py
```

Or from Python: `from predict_delay import predict_all_models; predict_all_models(weather_dict)`.
