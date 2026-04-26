# Costco Routes & Weather Delay: Project Summary

## Where We Started
- **Three routes** from source (depot) to each destination (Costco warehouse).
- Routes were represented as full paths only — no segment-level detail.

---

## What We Built

### 1. Path as 20 km segments
- Broke each route into **pitstops ~20 km apart** along the path.
- Output: path coordinates as a sequence of lat/lng points (e.g. in `routes_output_readable.json`).
- Enables segment-level analysis and weather/duration per chunk.

### 2. Weather at each pitstop
- For every pitstop, found the **nearest weather station** and attached its weather.
- To keep data size small, we store a **`weather_key`** (e.g. rounded lat_lng) in the routes file.
- **Weather data** lives in separate files; routes reference it by `weather_key`.
- **Output:** `data/routes/routes_with_weather.json` — each path point has `weather_key` to look up weather (e.g. in `data/weather/point_weekly_weather.json`).

### 3. Segment distance and time (Google Routes API)
- Used **Google Routes API** again to get **distance and duration** between **consecutive pitstops** (segment A → B).
- Each point now has `distance_to_next_km` and `duration_to_next_min` (last point has `null`).
- **Output:** `data/routes/routes_with_weather_and_substation_time.json` — routes with weather keys and segment-level travel time/distance.

### 4. Regression models for weather-based delay %
- Built **three models**: **Ridge**, **Random Forest**, and **XGBoost** using weather parameters (temp, snow, precipitation, visibility, wind, etc.) from the weekly weather at each point.
- Each model predicts **delay %** relative to baseline (e.g. how much longer travel takes under given weather).
- **Output:** One .joblib file per model: `delay_model_ridge.joblib`, `delay_model_random_forest.joblib`, `delay_model_xgboost.joblib` (each with model + scaler + feature names).

### 5. Testing the models
- **`predict_delay.py`** loads the joblib artifact and can run any of the three models.
- **`predict_delay_pct(weather_dict, model_name="ridge"|"random_forest"|"xgboost")`** returns delay % for one model; **`predict_all_models(weather_dict)`** returns all three.
- Run as script to test on several weather scenarios; delay % changes as inputs change.

---

## Next Step
- **Use the model in practice:** for any segment (or route), look up weather by `weather_key`, pass the weather features into the model, get **delay %**, then adjust baseline duration:  
  **adjusted_duration = baseline_duration × (1 + delay_pct / 100)**.

---

## Key Files (reference)

All backend paths are under **`backend/`**; the demo UI is under **`frontend/`**.

| Purpose | File |
|--------|------|
| Routes with 20 km–style points + weather key | `backend/data/routes/routes_with_weather.json` |
| Routes + segment distance/time | `backend/data/routes/routes_with_weather_and_substation_time.json` |
| Weather by key (e.g. weekly) | `backend/data/weather/point_weekly_weather.json` |
| Ridge / Random Forest / XGBoost models | `backend/delay_model_*.joblib` |
| Test / use the models | `backend/predict_delay.py` |
| API + serve demo UI | `backend/app.py` → serves `frontend/` at http://127.0.0.1:5000 |
