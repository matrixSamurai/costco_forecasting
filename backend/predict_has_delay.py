"""
Pre-Model 1 Prediction Module: Predict whether severe weather will cause delay.

Loads trained classifiers and provides prediction interface compatible with
Model 1's weather_dict format.

Usage:
    from predict_has_delay import predict_has_delay, predict_has_delay_all

    result = predict_has_delay({"temp_mean": 35, "visibility": 2, ...})
    # {"has_delay": True, "probability": 0.92, "model_name": "XGBoost"}

    all_results = predict_has_delay_all({"temp_mean": 35, "visibility": 2, ...})
    # {"XGBoost": {...}, "Random Forest": {...}, "LightGBM": {...}}
"""

import os
import numpy as np
from datetime import date
from joblib import load

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "classifier_models")

MODEL_FILES = {
    "xgboost": os.path.join(MODEL_DIR, "classifier_xgboost.joblib"),
    "random_forest": os.path.join(MODEL_DIR, "classifier_random_forest.joblib"),
    "lightgbm": os.path.join(MODEL_DIR, "classifier_lightgbm.joblib"),
}

FEATURE_NAMES = [
    "temp_mean",
    "temp_min",
    "temp_max",
    "prcp_total",
    "snow_depth",
    "visibility",
    "wind_speed_mean",
    "wind_gust_max",
    "month",
    "day_of_year",
]

# Mapping from Model 1 field names to our field names
MODEL1_FIELD_MAP = {
    "temp_mean_mean": "temp_mean",
    "temp_min_mean": "temp_min",
    "temp_max_mean": "temp_max",
    "prcp_total_mean": "prcp_total",
    "snow_depth_mean": "snow_depth",
    "visibility_mean": "visibility",
    "wind_speed_mean": "wind_speed_mean",  # same name
    "wind_gust_max_mean": "wind_gust_max",
}

# Default values when features are missing
DEFAULTS = {
    "temp_mean": 60.0,
    "temp_min": 50.0,
    "temp_max": 70.0,
    "prcp_total": 0.0,
    "snow_depth": 0.0,
    "visibility": 10.0,
    "wind_speed_mean": 8.0,
    "wind_gust_max": 15.0,
    "month": 6,
    "day_of_year": 172,
}

# Cache loaded artifacts
_artifacts = {}


def _load_artifact(model_name):
    """Load a model artifact from .joblib file, with caching."""
    if model_name in _artifacts:
        return _artifacts[model_name]
    path = MODEL_FILES.get(model_name)
    if path is None:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(MODEL_FILES.keys())}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}. Run build_classifier.py first.")
    artifact = load(path)
    _artifacts[model_name] = artifact
    return artifact


def _normalize_weather_dict(weather_dict):
    """Map Model 1 field names to our field names."""
    normalized = {}
    for key, val in weather_dict.items():
        mapped_key = MODEL1_FIELD_MAP.get(key, key)
        normalized[mapped_key] = val
    return normalized


def _build_feature_vector(weather_dict, query_date=None):
    """Build feature vector from weather dict and optional date."""
    w = _normalize_weather_dict(weather_dict)

    # Extract month and day_of_year from date if provided
    if query_date is not None:
        if isinstance(query_date, str):
            query_date = date.fromisoformat(query_date)
        month = query_date.month
        day_of_year = query_date.timetuple().tm_yday
    else:
        month = w.get("month", DEFAULTS["month"])
        day_of_year = w.get("day_of_year", DEFAULTS["day_of_year"])

    vec = np.array(
        [
            float(w.get("temp_mean", DEFAULTS["temp_mean"])),
            float(w.get("temp_min", DEFAULTS["temp_min"])),
            float(w.get("temp_max", DEFAULTS["temp_max"])),
            float(w.get("prcp_total", DEFAULTS["prcp_total"])),
            float(w.get("snow_depth", DEFAULTS["snow_depth"])),
            float(w.get("visibility", DEFAULTS["visibility"])),
            float(w.get("wind_speed_mean", DEFAULTS["wind_speed_mean"])),
            float(w.get("wind_gust_max", DEFAULTS["wind_gust_max"])),
            float(month),
            float(day_of_year),
        ],
        dtype=np.float64,
    ).reshape(1, -1)

    return vec


def predict_has_delay(weather_dict, model_name="xgboost", query_date=None):
    """
    Predict whether severe weather events will occur, indicating likely delay.

    Args:
        weather_dict: Dict with weather values. Accepts both CSV field names
                      (temp_mean, visibility) and Model 1 field names
                      (temp_mean_mean, visibility_mean).
        model_name: "xgboost", "random_forest", or "lightgbm"
        query_date: Optional date (str "YYYY-MM-DD" or date object) for month/day_of_year.

    Returns:
        {
            "has_delay": bool,
            "probability": float (0-1),
            "model_name": str,
        }
    """
    def _predict_with(model_key):
        artifact = _load_artifact(model_key)
        model = artifact["model"]
        scaler = artifact["scaler"]
        display_name = artifact["model_display_name"]
        vec = _build_feature_vector(weather_dict, query_date)
        vec_scaled = scaler.transform(vec)
        prob = model.predict_proba(vec_scaled)[0][1]
        return {
            "has_delay": bool(prob >= 0.5),
            "probability": round(float(prob), 4),
            "model_name": display_name,
        }

    # Prefer requested model, but gracefully fall back if it is unavailable at runtime.
    fallback_order = [model_name, "random_forest", "lightgbm", "xgboost"]
    seen = set()
    for key in fallback_order:
        if key in seen or key not in MODEL_FILES:
            continue
        seen.add(key)
        try:
            return _predict_with(key)
        except Exception:
            continue
    raise RuntimeError("No classifier model available for prediction")


def predict_has_delay_all(weather_dict, query_date=None):
    """
    Run all 3 models and return comparison.

    Returns:
        {
            "XGBoost": {"has_delay": bool, "probability": float, "model_name": str},
            "Random Forest": {...},
            "LightGBM": {...},
        }
    """
    results = {}
    for model_name in MODEL_FILES:
        try:
            r = predict_has_delay(weather_dict, model_name=model_name, query_date=query_date)
            results[r["model_name"]] = r
        except Exception:
            continue
    return results


# --- Demo ---
if __name__ == "__main__":
    print("=== Pre-Model 1 Classifier Demo ===\n")

    scenarios = [
        {
            "name": "Clear sky (no delay expected)",
            "weather": {
                "temp_mean": 70, "temp_min": 60, "temp_max": 80,
                "prcp_total": 0, "snow_depth": 0, "visibility": 10,
                "wind_speed_mean": 5, "wind_gust_max": 10,
            },
            "date": "2024-07-15",
        },
        {
            "name": "Blizzard (delay expected)",
            "weather": {
                "temp_mean": 20, "temp_min": 10, "temp_max": 25,
                "prcp_total": 3, "snow_depth": 8, "visibility": 1,
                "wind_speed_mean": 30, "wind_gust_max": 45,
            },
            "date": "2024-01-15",
        },
        {
            "name": "Dense fog (delay expected)",
            "weather": {
                "temp_mean": 50, "temp_min": 45, "temp_max": 55,
                "prcp_total": 0, "snow_depth": 0, "visibility": 0.5,
                "wind_speed_mean": 3, "wind_gust_max": 5,
            },
            "date": "2024-11-10",
        },
    ]

    for s in scenarios:
        print(f"Scenario: {s['name']}")
        results = predict_has_delay_all(s["weather"], query_date=s["date"])
        for model_name, r in results.items():
            label = "DELAY" if r["has_delay"] else "NO DELAY"
            print(f"  {model_name:<15s}: {label} (probability: {r['probability']:.2%})")
        print()
