"""
Predict delay % from baseline for a segment using weather, date of year, and time of day.
Supports Ridge, Random Forest, and XGBoost (each stored in its own .joblib file).

Features: 9 weather + journey_start_hour, journey_start_day_of_week, journey_start_month, week_of_year,
         day_of_year (1-365), segment_start_hour (0-24, time when driver starts this segment).
Date/time allow higher delay in winter (e.g. Jan 12) vs spring (Mar 28) and at night vs daylight.

Usage:
  from predict_delay import predict_delay_pct, predict_all_models
  predict_delay_pct(weather_dict, model_name="ridge", journey_start=...)
  predict_all_models(weather_dict, journey_start=...)  # returns dict of model_name -> delay_pct
"""
import os
from datetime import datetime
import numpy as np
from joblib import load

ROOT = os.path.dirname(os.path.abspath(__file__))

# Each model in its own .joblib file; display names for output
MODEL_FILES = {
    "ridge": os.path.join(ROOT, "delay_model_ridge.joblib"),
    "random_forest": os.path.join(ROOT, "delay_model_random_forest.joblib"),
    "xgboost": os.path.join(ROOT, "delay_model_xgboost.joblib"),
}
MODEL_DISPLAY_NAMES = {
    "ridge": "Ridge",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

_artifacts = {}  # model_key -> {model, scaler, feature_names, model_display_name}


def _load_artifact(model_name):
    """Load one model artifact from its .joblib file."""
    if model_name not in MODEL_FILES:
        raise ValueError("model_name must be one of %s" % list(MODEL_FILES.keys()))
    if model_name not in _artifacts:
        path = MODEL_FILES[model_name]
        if not os.path.isfile(path):
            raise FileNotFoundError("Model file not found: %s (run build_delay_model.py)" % path)
        _artifacts[model_name] = load(path)
    return _artifacts[model_name]


DEFAULT_JOURNEY_HOUR = 12
DEFAULT_JOURNEY_DAY_OF_WEEK = 0
DEFAULT_JOURNEY_MONTH = 6
DEFAULT_WEEK_OF_YEAR = 26
DEFAULT_DAY_OF_YEAR = 183  # mid-year


def _winter_severity(day_of_year):
    """0-1: peak winter (Dec/Jan) = 1, late Feb/Mar = 0.6, spring/summer = 0. Must match build_delay_model."""
    if day_of_year <= 45 or day_of_year >= 320:
        return 1.0
    if 46 <= day_of_year <= 80 or 280 <= day_of_year <= 319:
        return 0.6
    if 81 <= day_of_year <= 120 or 245 <= day_of_year <= 279:
        return 0.3
    return 0.0


def parse_journey_start(journey_start):
    """
    Parse journey_start to (hour 0-23, day_of_week 0-6 Monday=0, month 1-12, week_of_year 1-52, day_of_year 1-365).
    journey_start can be: None, datetime, or dict with date/time or hour/day_of_week/month/week_of_year/day_of_year.
    """
    if journey_start is None:
        return (
            DEFAULT_JOURNEY_HOUR,
            DEFAULT_JOURNEY_DAY_OF_WEEK,
            DEFAULT_JOURNEY_MONTH,
            DEFAULT_WEEK_OF_YEAR,
            DEFAULT_DAY_OF_YEAR,
        )
    if isinstance(journey_start, datetime):
        iso = journey_start.isocalendar()
        doy = journey_start.timetuple().tm_yday
        return journey_start.hour, journey_start.weekday(), journey_start.month, iso[1], doy
    if isinstance(journey_start, dict):
        if "hour" in journey_start and "day_of_week" in journey_start and "month" in journey_start and "week_of_year" in journey_start:
            doy = int(journey_start.get("day_of_year", 183))
            doy = max(1, min(365, doy))
            return (
                int(journey_start["hour"]) % 24,
                int(journey_start["day_of_week"]) % 7,
                max(1, min(12, int(journey_start["month"]))),
                max(1, min(52, int(journey_start["week_of_year"]))),
                doy,
            )
        date_str = journey_start.get("date") or journey_start.get("journey_start_date")
        time_str = journey_start.get("time") or journey_start.get("journey_start_time")
        if date_str and time_str and str(time_str).strip():
            try:
                dt = datetime.strptime(str(date_str).strip() + " " + str(time_str).strip()[:5], "%Y-%m-%d %H:%M")
                iso = dt.isocalendar()
                doy = dt.timetuple().tm_yday
                return dt.hour, dt.weekday(), dt.month, iso[1], doy
            except (ValueError, TypeError):
                pass
        elif date_str and str(date_str).strip():
            try:
                dt = datetime.strptime(str(date_str).strip(), "%Y-%m-%d")
                iso = dt.isocalendar()
                doy = dt.timetuple().tm_yday
                return DEFAULT_JOURNEY_HOUR, dt.weekday(), dt.month, iso[1], doy
            except (ValueError, TypeError):
                pass
    return (
        DEFAULT_JOURNEY_HOUR,
        DEFAULT_JOURNEY_DAY_OF_WEEK,
        DEFAULT_JOURNEY_MONTH,
        DEFAULT_WEEK_OF_YEAR,
        DEFAULT_DAY_OF_YEAR,
    )


def _features_to_vector(feature_dict, feature_names):
    """Build feature vector from dict (weather + journey_start_*)."""
    out = []
    for n in feature_names:
        if n in feature_dict and feature_dict[n] is not None:
            out.append(float(feature_dict[n]))
        else:
            out.append(0.0)
    return np.array([out], dtype=np.float64)


def predict_delay_pct(weather_dict, model_name="ridge", journey_start=None, segment_start_hour=None):
    """
    Predict delay % for a segment using one of the saved models.

    weather_dict: dict with temp_min_mean, temp_max_mean, temp_mean_mean, snow_depth_mean,
                  prcp_total_mean, visibility_mean, wind_speed_mean, wind_speed_max_mean, wind_gust_max_mean.
    model_name: "ridge", "random_forest", or "xgboost".
    journey_start: optional datetime or dict with "date" (YYYY-MM-DD) and "time" (HH:MM),
                   or dict with hour, day_of_week, month, week_of_year, day_of_year.
    segment_start_hour: optional 0-24, time when driver *starts* this segment (for multi-segment routes).
                         If None, journey start hour is used.
    Returns: predicted delay % (typically 0--50).
    """
    art = _load_artifact(model_name)
    feat_dict = _weather_to_feature_dict(weather_dict, journey_start, segment_start_hour=segment_start_hour)
    vec = _features_to_vector(feat_dict, art["feature_names"])
    vec_scaled = art["scaler"].transform(vec)
    return float(art["model"].predict(vec_scaled)[0])


def _weather_to_feature_dict(weather_dict, journey_start, segment_start_hour=None):
    """
    Build full feature dict from weather dict + journey_start + optional segment_start_hour.
    Adds hour, day_of_week, month, week_of_year, day_of_year. Uses segment_start_hour for
    time-of-day at this segment (if provided), else journey_start hour.
    """
    h, dow, mo, woy, doy = parse_journey_start(journey_start)
    out = dict(weather_dict)
    out["journey_start_hour"] = h
    out["journey_start_day_of_week"] = dow
    out["journey_start_month"] = mo
    out["week_of_year"] = woy
    out["day_of_year"] = doy
    out["winter_severity"] = _winter_severity(doy)  # colder day (Dec/Jan) = 1 → more delay
    # Per-segment time: when the driver starts this segment (can be later than journey start)
    out["segment_start_hour"] = segment_start_hour if segment_start_hour is not None else (h + 0.0)
    return out


def predict_all_models(weather_dict, journey_start=None, segment_start_hour=None):
    """
    Run all available models and return {model_name: delay_pct}.
    weather_dict: 9 weather features (temp_min_mean, temp_max_mean, temp_mean_mean, snow_depth_mean,
                  prcp_total_mean, visibility_mean, wind_speed_mean, wind_speed_max_mean, wind_gust_max_mean).
    journey_start: optional date/time for journey start (see parse_journey_start).
    segment_start_hour: optional 0-24, time when driver starts this segment (for per-segment delay).
    """
    out = {}
    feat_dict = _weather_to_feature_dict(weather_dict, journey_start, segment_start_hour=segment_start_hour)
    for key in MODEL_FILES:
        path = MODEL_FILES[key]
        if not os.path.isfile(path):
            continue
        try:
            art = _load_artifact(key)
            vec = _features_to_vector(feat_dict, art["feature_names"])
            vec_scaled = art["scaler"].transform(vec)
            display_name = art.get("model_display_name") or MODEL_DISPLAY_NAMES[key]
            out[display_name] = float(art["model"].predict(vec_scaled)[0])
        except Exception:
            # Skip models that cannot be loaded at runtime (e.g., XGBoost missing libomp on macOS).
            continue
    return out


if __name__ == "__main__":
    test_cases = [
        {
            "name": "Moderate wind, low visibility",
            "weather": {
                "temp_min_mean": 38, "temp_max_mean": 60, "snow_depth_mean": 0,
                "prcp_total_mean": 0.1, "visibility_mean": 2, "wind_speed_mean": 15,
                "wind_gust_max_mean": 29,
            },
        },
        {
            "name": "Clear, mild (low delay expected)",
            "weather": {
                "temp_min_mean": 45, "temp_max_mean": 72, "snow_depth_mean": 0,
                "prcp_total_mean": 0, "visibility_mean": 10, "wind_speed_mean": 5,
                "wind_gust_max_mean": 12,
            },
        },
        {
            "name": "Snow and precip (high delay expected)",
            "weather": {
                "temp_min_mean": 28, "temp_max_mean": 35, "snow_depth_mean": 5,
                "prcp_total_mean": 2.0, "visibility_mean": 3, "wind_speed_mean": 12,
                "wind_gust_max_mean": 35,
            },
        },
    ]

    try:
        print("Testing all models (Ridge, Random Forest, XGBoost):\n")
        for tc in test_cases:
            print("  Scenario: %s" % tc["name"])
            results = predict_all_models(tc["weather"], journey_start=None)
            for display_name, pct in results.items():
                print("    %s: delay %% = %s" % (display_name, round(pct, 2)))
            print()
        # With journey start
        results = predict_all_models(
            test_cases[0]["weather"],
            journey_start={"date": "2024-06-15", "time": "08:00"},
        )
        print("  With journey_start 2024-06-15 08:00:", results)
        print("Single-model: predict_delay_pct(weather_dict, model_name='ridge', journey_start={'date':'...','time':'...'})")
    except FileNotFoundError as e:
        print("Run build_delay_model.py first to create the .joblib model files")
        print(e)
