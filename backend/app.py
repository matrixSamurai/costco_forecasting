"""
Flask API for the delay prediction demo.
Serves the frontend (from ../frontend) and exposes:
  GET  /api/warehouses  - list of Costco destinations
  GET  /api/source      - Tracy depot (fixed source)
  POST /api/predict     - delay % from weather dict
  POST /api/routes        - routes from origin to destination (no delay)
  POST /api/route-delays  - delay predictions per route from pitstops (small chunks) then aggregated
"""
import os
from flask import Flask, request, jsonify, render_template

ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(ROOT, "..", "frontend")

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

if ROOT not in __import__("sys").path:
    __import__("sys").path.insert(0, ROOT)

from predict_delay import predict_all_models, parse_journey_start
from route_utils import (
    load_warehouses,
    TRACY_DEPOT,
    decode_polyline,
    sample_route_points_by_distance,
    get_weather_features_for_pitstop,
    fetch_directions,
)

app = Flask(
    __name__,
    static_folder=os.path.join(FRONTEND_DIR, "static"),
    template_folder=FRONTEND_DIR,
)

FEATURE_KEYS = [
    "temp_min_mean", "temp_max_mean", "snow_depth_mean", "prcp_total_mean",
    "visibility_mean", "wind_speed_mean", "wind_gust_max_mean",
]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/demo")
def demo():
    """Weather-only demo page: manual inputs and Predict delay."""
    return render_template("demo.html")


@app.route("/api/source")
def api_source():
    """Fixed source: Tracy depot, California."""
    return jsonify({"name": "Tracy Depot, CA", **TRACY_DEPOT})


@app.route("/api/warehouses")
def api_warehouses():
    """List of Costco warehouse destinations (from costco_warehouses_full.json)."""
    try:
        warehouses = load_warehouses()
        return jsonify(warehouses)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/config")
def api_config():
    """Optional: Maps API key for frontend map display (same key as Directions)."""
    key = os.environ.get("GOOGLE_MAPS_API_KEY") or os.environ.get("ROUTES_API_KEY")
    return jsonify({"mapsApiKey": key or ""})


@app.route("/api/routes", methods=["POST"])
def api_routes():
    """
    Get routes from origin to destination via Google Directions API only (no delay prediction).
    Body: { "origin": { "lat", "lng" }, "destination": { "lat", "lng" } }
    """
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY") or os.environ.get("ROUTES_API_KEY")
    if not api_key:
        return jsonify({"error": "GOOGLE_MAPS_API_KEY or ROUTES_API_KEY not set in environment"}), 503
    try:
        data = request.get_json() or {}
        o = data.get("origin") or {}
        d = data.get("destination") or {}
        o_lat = o.get("lat") if "lat" in o else o.get("latitude")
        o_lng = o.get("lng") if "lng" in o else o.get("longitude")
        d_lat = d.get("lat") if "lat" in d else d.get("latitude")
        d_lng = d.get("lng") if "lng" in d else d.get("longitude")
        if o_lat is None or o_lng is None:
            return jsonify({"error": "Invalid origin (need lat, lng)"}), 400
        if d_lat is None or d_lng is None:
            return jsonify({"error": "Invalid destination (need lat, lng)"}), 400
        o_lat = float(o_lat)
        o_lng = float(o_lng)
        d_lat = float(d_lat)
        d_lng = float(d_lng)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid origin or destination (need numeric lat, lng)"}), 400

    try:
        routes, directions_error = fetch_directions(o_lat, o_lng, d_lat, d_lng, api_key)
    except Exception as e:
        return jsonify({"error": "Directions API failed", "detail": str(e)}), 502

    if not routes:
        err_msg = directions_error or "No routes found"
        resp = jsonify({"error": err_msg, "routes": []})
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        return resp, 404

    out_routes = []
    for idx, r in enumerate(routes):
        out_routes.append({
            "route_index": idx + 1,
            "polyline": r.get("polyline") or "",
            "distance_m": r.get("distance_m"),
            "duration_s": r.get("duration_s"),
            "distance_text": r.get("distance_text", ""),
            "duration_text": r.get("duration_text", ""),
        })
    resp = jsonify({"origin": {"lat": o_lat, "lng": o_lng}, "destination": {"lat": d_lat, "lng": d_lng}, "routes": out_routes})
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return resp


@app.route("/api/route-delays", methods=["POST"])
def api_route_delays():
    """
    For each route: pitstops every ~20 miles, nearest weather station per pitstop,
    run all three models (with optional journey start date/time) per chunk, aggregate per route.
    Body: { "routes": [ ... ], "journey_start_date": "YYYY-MM-DD", "journey_start_time": "HH:MM" }
    """
    try:
        data = request.get_json() or {}
        routes_in = data.get("routes") or []
        journey_start = {}
        if data.get("journey_start_date"):
            journey_start["date"] = data.get("journey_start_date")
        if data.get("journey_start_time") is not None:
            journey_start["time"] = data.get("journey_start_time")
        if not journey_start:
            journey_start = None
    except Exception as e:
        return jsonify({"error": "Invalid JSON body", "detail": str(e)}), 400
    if not routes_in:
        return jsonify({"error": "No routes provided", "routes": []}), 400
    if not isinstance(routes_in, list):
        return jsonify({"error": "routes must be an array"}), 400

    try:
        # Parse journey start hour/minute for per-segment time (each segment starts at journey_start + cumulative minutes)
        journey_hour, journey_minute = 12, 0
        if journey_start:
            h, _, _, _, _ = parse_journey_start(journey_start)
            journey_hour = h
            if journey_start.get("time"):
                try:
                    t = str(journey_start["time"]).strip()[:5]
                    if ":" in t:
                        parts = t.split(":")
                        journey_minute = int(parts[1]) if len(parts) > 1 else 0
                except (ValueError, TypeError):
                    pass

        out_routes = []
        for idx, r in enumerate(routes_in):
            if not isinstance(r, dict):
                continue
            polyline = r.get("polyline") or ""
            points = decode_polyline(polyline)
            sampled = sample_route_points_by_distance(points)
            total_duration_min = (float(r.get("duration_s") or 0) / 60.0)
            n_pitstops = len(sampled)
            seg_count = max(1, n_pitstops - 1)
            delays_per_point = []
            for i, (lat, lng) in enumerate(sampled):
                feats = get_weather_features_for_pitstop(lat, lng)
                if feats:
                    # Time when driver starts this segment = journey start + cumulative minutes so far
                    cumulative_min = (i * (total_duration_min / seg_count)) if seg_count else 0.0
                    segment_start_hour = (journey_hour + journey_minute / 60.0 + cumulative_min / 60.0) % 24.0
                    preds = predict_all_models(
                        feats, journey_start=journey_start, segment_start_hour=segment_start_hour
                    )
                    delays_per_point.append(preds)
            if not delays_per_point:
                avg_delays = {"Ridge": 0, "Random Forest": 0, "XGBoost": 0}
            else:
                avg_delays = {}
                for model in ("Ridge", "Random Forest", "XGBoost"):
                    vals = [p.get(model) for p in delays_per_point if p.get(model) is not None]
                    avg_delays[model] = round(sum(vals) / len(vals), 2) if vals else 0
            out_routes.append({
                "route_index": (r.get("route_index") or idx + 1),
                "polyline": polyline,
                "distance_m": r.get("distance_m"),
                "duration_s": r.get("duration_s"),
                "distance_text": r.get("distance_text", ""),
                "duration_text": r.get("duration_text", ""),
                "delays": avg_delays,
                "pitstop_count": len(delays_per_point),
            })
        return jsonify({"routes": out_routes})
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Route delay prediction failed",
            "detail": str(e),
            "traceback": traceback.format_exc(),
        }), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() or {}
        weather = {}
        for k in FEATURE_KEYS:
            v = data.get(k)
            if v is None:
                return jsonify({"error": "Missing field: %s" % k}), 400
            try:
                weather[k] = float(v)
            except (TypeError, ValueError):
                return jsonify({"error": "Invalid number for %s" % k}), 400
        journey_start = None
        if data.get("journey_start_date") or data.get("journey_start_time") is not None:
            journey_start = {
                "date": data.get("journey_start_date") or "2024-01-01",
                "time": data.get("journey_start_time") if data.get("journey_start_time") is not None else "08:00",
            }
        # Model expects 9 weather features; form sends 7 - derive temp_mean_mean and wind_speed_max_mean
        if "temp_mean_mean" not in weather:
            weather["temp_mean_mean"] = (float(weather.get("temp_min_mean", 0)) + float(weather.get("temp_max_mean", 0))) / 2
        if "wind_speed_max_mean" not in weather:
            weather["wind_speed_max_mean"] = float(weather.get("wind_speed_mean", 0)) * 1.5 or 10.0
        results = predict_all_models(weather, journey_start=journey_start)
        out = {"predictions": results, "weather": weather}
        if journey_start:
            out["journey_start"] = journey_start
        return jsonify(out)
    except FileNotFoundError as e:
        return jsonify({"error": "Models not found. Run build_delay_model.py first.", "detail": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
