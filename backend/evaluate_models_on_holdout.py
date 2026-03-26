"""
Evaluate the trained delay models on the 20% holdout from synthetic_historical_route_delays.json.

Uses the same 80/20 split (random_state=42) as build_delay_model_from_synthetic.py,
loads each saved model, and computes Mean Squared Error (MSE) on the holdout set.

Usage:
  cd backend
  python evaluate_models_on_holdout.py

Requires: synthetic_historical_route_delays.json and the three delay_model_*.joblib files.
"""

import json
import os
import numpy as np
from joblib import load

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

ROOT = os.path.dirname(os.path.abspath(__file__))
SYNTHETIC_DELAYS_PATH = os.path.join(ROOT, "data", "synthetic_historical_route_delays.json")
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

# Must match build_delay_model_from_synthetic.py (15 features including day_of_year, segment_start_hour)
WEATHER_KEYS = [
    "temp_min_mean", "temp_max_mean", "temp_mean_mean",
    "snow_depth_mean", "prcp_total_mean", "visibility_mean",
    "wind_speed_mean", "wind_speed_max_mean", "wind_gust_max_mean",
]


def _month_to_day_of_year(month):
    if month < 1 or month > 12:
        return 183
    mid_days = [15, 44, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
    return mid_days[month - 1]


def _winter_severity(day_of_year):
    if day_of_year <= 45 or day_of_year >= 320:
        return 1.0
    if 46 <= day_of_year <= 80 or 280 <= day_of_year <= 319:
        return 0.6
    if 81 <= day_of_year <= 120 or 245 <= day_of_year <= 279:
        return 0.3
    return 0.0


# Same random state as in build_delay_model_from_synthetic.py
SPLIT_RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_synthetic_data(path):
    """Load historical_delays from synthetic_historical_route_delays.json."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("historical_delays") or []


def records_to_xy(records):
    """Build X and y from historical_delays (same as build script, 16 features)."""
    X_rows = []
    y_list = []
    for r in records:
        w = r.get("weather") or {}
        if not w:
            continue
        row = []
        for k in WEATHER_KEYS:
            val = w.get(k)
            row.append(float(val) if val is not None else 0.0)
        hour = int(r.get("hour", 0)) % 24
        row.append(hour)
        row.append(int(r.get("day_of_week", 0)) % 7)
        month = max(1, min(12, int(r.get("month", 1))))
        row.append(month)
        row.append(max(1, min(52, int(r.get("week_of_year", 1)))))
        day_of_year = int(r.get("day_of_year", 0)) or _month_to_day_of_year(month)
        row.append(max(1, min(365, day_of_year)))
        row.append(_winter_severity(day_of_year))
        seg_hour = r.get("segment_start_hour")
        row.append((float(seg_hour) % 24.0) if seg_hour is not None else float(hour))
        X_rows.append(row)
        y_list.append(float(r.get("delay_pct", 0)))
    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)
    return X, y


def main():
    if not HAS_SKLEARN:
        raise SystemExit("scikit-learn is required. pip install scikit-learn")

    if not os.path.isfile(SYNTHETIC_DELAYS_PATH):
        raise SystemExit(
            "File not found: %s\nGenerate it first with: python generate_synthetic_historical_delays.py"
            % SYNTHETIC_DELAYS_PATH
        )

    print("Loading data:", SYNTHETIC_DELAYS_PATH)
    records = load_synthetic_data(SYNTHETIC_DELAYS_PATH)
    if not records:
        raise SystemExit("No records in historical_delays.")

    X, y = records_to_xy(records)
    print("Total records:", len(y))

    # Same 80/20 split as build_delay_model_from_synthetic.py
    _, X_holdout, _, y_holdout = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SPLIT_RANDOM_STATE
    )
    print("Holdout (20%) size:", len(y_holdout))
    print()

    results = []
    for key, path in MODEL_FILES.items():
        if not os.path.isfile(path):
            print("%s: model file not found, skip" % MODEL_DISPLAY_NAMES[key])
            continue
        art = load(path)
        model = art["model"]
        scaler = art["scaler"]
        X_holdout_scaled = scaler.transform(X_holdout)
        y_pred = model.predict(X_holdout_scaled)
        mse = mean_squared_error(y_holdout, y_pred)
        rmse = np.sqrt(mse)
        display_name = art.get("model_display_name") or MODEL_DISPLAY_NAMES[key]
        results.append((display_name, mse, rmse))
        print("%s:" % display_name)
        print("  MSE  (holdout): %.6f" % mse)
        print("  RMSE (holdout): %.4f" % rmse)
        print()

    if results:
        print("Summary (20% holdout):")
        print("-" * 50)
        for name, mse, rmse in results:
            print("  %-16s  MSE = %.6f   RMSE = %.4f" % (name, mse, rmse))
    else:
        print("No models found. Run build_delay_model_from_synthetic.py first.")


if __name__ == "__main__":
    main()
