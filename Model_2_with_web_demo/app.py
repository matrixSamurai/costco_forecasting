#!/usr/bin/env python3
"""
Flask app for Revenue Loss Prediction.

Standalone web server in pipeline/ — does NOT modify Model 1's original files.
Imports predict_loss.py (local) and optionally Model 1's predict_delay for
end-to-end route-loss predictions.

Usage:
    cd pipeline/
    python3 app.py
    # Open http://localhost:5001
"""

import os
import sys

from flask import Flask, request, jsonify, render_template

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(SCRIPT_DIR, "frontend")
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Add Model 1 backend to path (for optional end-to-end route-loss)
MODEL1_BACKEND = os.path.join(
    PROJECT_ROOT, "model 1", "ver2", "costco_forecasting-master", "backend"
)

# Ensure local imports work
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from predict_loss import LossPredictor
from recommendations import get_supply_risk, generate_recommendations

app = Flask(
    __name__,
    static_folder=os.path.join(FRONTEND_DIR, "static"),
    template_folder=FRONTEND_DIR,
)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
_loss_predictor = None

BASE_PRICES = {
    ("iceberg", 0): 1.84, ("iceberg", 1): 5.14,
    ("romaine", 0): 3.04, ("romaine", 1): 6.65,
    ("butterhead", 0): 3.34, ("butterhead", 1): 7.32,
    ("leaf", 0): 2.89, ("leaf", 1): 6.33,
    ("spring_mix", 0): 3.95, ("spring_mix", 1): 8.66,
}


def _get_predictor():
    global _loss_predictor
    if _loss_predictor is None:
        _loss_predictor = LossPredictor()
    return _loss_predictor


def _get_model1():
    """Try to import Model 1's predict_all_models. Returns None if unavailable."""
    try:
        if MODEL1_BACKEND not in sys.path:
            sys.path.insert(0, MODEL1_BACKEND)
        from predict_delay import predict_all_models
        return predict_all_models
    except (ImportError, FileNotFoundError):
        return None


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
@app.route("/api/predict-loss", methods=["POST"])
def api_predict_loss():
    """Predict loss_rate and revenue_loss from delay + cargo parameters."""
    try:
        data = request.get_json() or {}
        delay_hours = float(data.get("delay_hours", 0))
        delay_temp_f = float(data.get("delay_temp_f", 41))
        lettuce_type = str(data.get("lettuce_type", "iceberg"))
        organic = int(data.get("organic", 0))
        quantity_lb = float(data.get("quantity_lb", 1000))
        month = int(data.get("month", 6))
        transit_distance_km = float(data.get("transit_distance_km", 1000))
        pre_delay = float(data.get("pre_delay_consumption_pct", 0))
    except (TypeError, ValueError) as e:
        return jsonify({"error": "Invalid input: %s" % str(e)}), 400

    try:
        predictor = _get_predictor()
        result = predictor.predict(
            delay_hours=delay_hours,
            delay_temp_f=delay_temp_f,
            lettuce_type=lettuce_type,
            organic=organic,
            month=month,
            transit_distance_km=transit_distance_km,
            pre_delay_consumption_pct=pre_delay,
        )
        price = BASE_PRICES.get((lettuce_type, organic), 3.04)
        total_value = quantity_lb * price
        revenue_loss = total_value * result["loss_rate"]
        revenue_loss_seasonal = revenue_loss * result["seasonal_price_index"]

        # Supply risk & recommendations
        supply_risk = get_supply_risk(
            lettuce_type, month, delay_hours=delay_hours, temp_f=delay_temp_f
        )
        recs = generate_recommendations(
            result["loss_rate"], delay_hours, delay_temp_f,
            supply_risk, quantity_lb
        )

        return jsonify({
            "loss_rate": round(result["loss_rate"], 6),
            "sigmoid_loss_rate": round(result["sigmoid_loss_rate"], 6),
            "shelf_life_days": round(result["shelf_life_days"], 2),
            "consumption_pct": round(result["consumption_pct"], 2),
            "seasonal_price_index": round(result["seasonal_price_index"], 4),
            "price_per_lb": price,
            "total_value": round(total_value, 2),
            "revenue_loss": round(revenue_loss, 2),
            "revenue_loss_seasonal": round(revenue_loss_seasonal, 2),
            "lettuce_type": lettuce_type,
            "organic": organic,
            "quantity_lb": quantity_lb,
            "delay_hours": delay_hours,
            "delay_temp_f": delay_temp_f,
            "month": month,
            "supply_risk": supply_risk,
            "recommendations": recs,
        })
    except FileNotFoundError as e:
        return jsonify({
            "error": "Loss model not found. Run build_loss_model.py first.",
            "detail": str(e),
        }), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/route-loss", methods=["POST"])
def api_route_loss():
    """End-to-end: weather → Model 1 delay → Model 2 loss → revenue_loss."""
    try:
        data = request.get_json() or {}
        weather = data.get("weather_dict") or {}
        journey_start = data.get("journey_start")
        lettuce_type = str(data.get("lettuce_type", "iceberg"))
        organic = int(data.get("organic", 0))
        quantity_lb = float(data.get("quantity_lb", 1000))
        month = int(data.get("month", 6))
        baseline_hours = float(data.get("baseline_hours", 24))
        transit_distance_km = float(data.get("transit_distance_km", 1000))
    except (TypeError, ValueError) as e:
        return jsonify({"error": "Invalid input: %s" % str(e)}), 400

    predict_all = _get_model1()
    if predict_all is None:
        return jsonify({
            "error": "Model 1 not available. Use /api/predict-loss with manual delay_hours instead.",
        }), 503

    try:
        # Derive missing weather features
        if "temp_mean_mean" not in weather:
            weather["temp_mean_mean"] = (
                float(weather.get("temp_min_mean", 0)) +
                float(weather.get("temp_max_mean", 0))
            ) / 2
        if "wind_speed_max_mean" not in weather:
            weather["wind_speed_max_mean"] = (
                float(weather.get("wind_speed_mean", 0)) * 1.5 or 10.0
            )

        # Model 1: delay prediction
        m1_results = predict_all(weather, journey_start=journey_start)
        delay_pct = m1_results.get("XGBoost") or next(iter(m1_results.values()), 0)
        delay_hours = baseline_hours * (delay_pct / 100.0)
        delay_temp_f = float(weather.get("temp_max_mean", 41))

        # Model 2: loss prediction
        predictor = _get_predictor()
        result = predictor.predict(
            delay_hours=delay_hours,
            delay_temp_f=delay_temp_f,
            lettuce_type=lettuce_type,
            organic=organic,
            month=month,
            transit_distance_km=transit_distance_km,
        )
        price = BASE_PRICES.get((lettuce_type, organic), 3.04)
        total_value = quantity_lb * price
        revenue_loss = total_value * result["loss_rate"]

        return jsonify({
            "model1_delays": m1_results,
            "delay_pct": round(delay_pct, 2),
            "delay_hours": round(delay_hours, 4),
            "delay_temp_f": delay_temp_f,
            "loss_rate": round(result["loss_rate"], 6),
            "sigmoid_loss_rate": round(result["sigmoid_loss_rate"], 6),
            "shelf_life_days": round(result["shelf_life_days"], 2),
            "consumption_pct": round(result["consumption_pct"], 2),
            "price_per_lb": price,
            "total_value": round(total_value, 2),
            "revenue_loss": round(revenue_loss, 2),
        })
    except FileNotFoundError as e:
        return jsonify({
            "error": "Model not found. Run build scripts first.",
            "detail": str(e),
        }), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001)
