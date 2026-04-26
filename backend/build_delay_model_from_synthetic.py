"""
Build delay prediction models from synthetic_historical_route_delays.json.

Uses weather parameters and time-of-year (hour, day_of_week, month, week_of_year)
from each record. Trains on 80% of the data and holds out 20% for validation.
Saves the same artifact format as build_delay_model.py so predict_delay.py works unchanged.

Usage:
  pip install -r requirements.txt
  python build_delay_model_from_synthetic.py

Output:
  - delay_model_ridge.joblib
  - delay_model_random_forest.joblib
  - delay_model_xgboost.joblib  (each: model + scaler fit on train 80% + feature_names)
"""

import json
import os
import numpy as np
from joblib import dump

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

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

# Must match predict_delay.py and build_delay_model.py (date + winter_severity + per-segment time)
FEATURE_NAMES = [
    "temp_min_mean",
    "temp_max_mean",
    "temp_mean_mean",
    "snow_depth_mean",
    "prcp_total_mean",
    "visibility_mean",
    "wind_speed_mean",
    "wind_speed_max_mean",
    "wind_gust_max_mean",
    "journey_start_hour",
    "journey_start_day_of_week",
    "journey_start_month",
    "week_of_year",
    "day_of_year",
    "winter_severity",
    "segment_start_hour",
]

# Keys in synthetic JSON: weather.* and hour, day_of_week, month, week_of_year, day_of_year (optional), segment_start_hour (optional)
WEATHER_KEYS = [
    "temp_min_mean", "temp_max_mean", "temp_mean_mean",
    "snow_depth_mean", "prcp_total_mean", "visibility_mean",
    "wind_speed_mean", "wind_speed_max_mean", "wind_gust_max_mean",
]


def _month_to_day_of_year(month):
    """Approximate day-of-year at mid-month (1-365)."""
    if month < 1 or month > 12:
        return 183
    mid_days = [15, 44, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
    return mid_days[month - 1]


def _winter_severity(day_of_year):
    """0-1: peak winter = 1, March = 0.6, summer = 0. Must match build_delay_model."""
    if day_of_year <= 45 or day_of_year >= 320:
        return 1.0
    if 46 <= day_of_year <= 80 or 280 <= day_of_year <= 319:
        return 0.6
    if 81 <= day_of_year <= 120 or 245 <= day_of_year <= 279:
        return 0.3
    return 0.0


def load_synthetic_data(path):
    """Load historical_delays from synthetic_historical_route_delays.json."""
    print("Loading", path, "...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("historical_delays") or []
    print("  Loaded", len(records), "historical delay records")
    return records


def records_to_xy(records):
    """
    Build feature matrix X and target y from historical_delays records.
    X: [weather_9, journey_start_hour, day_of_week, month, week_of_year, day_of_year, segment_start_hour] per record.
    y: delay_pct.
    """
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
        # segment_start_hour: use hour from record (trip time) or same as journey_start_hour
        seg_hour = r.get("segment_start_hour")
        if seg_hour is not None:
            row.append(float(seg_hour) % 24.0)
        else:
            row.append(float(hour))
        X_rows.append(row)
        y_list.append(float(r.get("delay_pct", 0)))
    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)
    return X, y


def main():
    if not os.path.isfile(SYNTHETIC_DELAYS_PATH):
        raise SystemExit(
            "File not found: %s\nGenerate it first with: python generate_synthetic_historical_delays.py"
            % SYNTHETIC_DELAYS_PATH
        )

    records = load_synthetic_data(SYNTHETIC_DELAYS_PATH)
    if not records:
        raise SystemExit("No records in historical_delays.")

    X, y = records_to_xy(records)
    print("Features:", X.shape[1], "(", ", ".join(FEATURE_NAMES), ")")
    print("Target delay_pct: min=%.2f max=%.2f mean=%.2f" % (y.min(), y.max(), y.mean()))

    # 80% train, 20% holdout
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("Train size: %d (80%%)" % len(y_train))
    print("Holdout size: %d (20%%)" % len(y_holdout))

    if not HAS_SKLEARN:
        raise SystemExit("scikit-learn is required. pip install scikit-learn")

    # Fit scaler on train only (no leakage from holdout)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_holdout_scaled = scaler.transform(X_holdout)

    # 1. Ridge — train on 80% only
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    print("Ridge          R² train=%.4f holdout=%.4f" % (
        ridge.score(X_train_scaled, y_train),
        ridge.score(X_holdout_scaled, y_holdout),
    ))
    dump({
        "model": ridge,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "model_display_name": MODEL_DISPLAY_NAMES["ridge"],
    }, MODEL_FILES["ridge"])
    print("  Saved", MODEL_FILES["ridge"])

    # 2. Random Forest — train on 80% only
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    print("Random Forest  R² train=%.4f holdout=%.4f" % (
        rf.score(X_train_scaled, y_train),
        rf.score(X_holdout_scaled, y_holdout),
    ))
    dump({
        "model": rf,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "model_display_name": MODEL_DISPLAY_NAMES["random_forest"],
    }, MODEL_FILES["random_forest"])
    print("  Saved", MODEL_FILES["random_forest"])

    # 3. XGBoost — train on 80% only
    if HAS_XGB:
        xgb_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1
        )
        xgb_model.fit(X_train_scaled, y_train)
        print("XGBoost        R² train=%.4f holdout=%.4f" % (
            xgb_model.score(X_train_scaled, y_train),
            xgb_model.score(X_holdout_scaled, y_holdout),
        ))
        dump({
            "model": xgb_model,
            "scaler": scaler,
            "feature_names": FEATURE_NAMES,
            "model_display_name": MODEL_DISPLAY_NAMES["xgboost"],
        }, MODEL_FILES["xgboost"])
        print("  Saved", MODEL_FILES["xgboost"])
    else:
        print("XGBoost        skipped (pip install xgboost to include)")

    print("Done. Models trained on 80%% of synthetic_historical_route_delays.json.")


if __name__ == "__main__":
    main()
