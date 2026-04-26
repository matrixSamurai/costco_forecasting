"""
Generate synthetic historical delay % for all routes (Tracy depot → each warehouse).

Uses route-level weather from path_coordinates + point_weekly_weather.json and
time-of-day estimates (rush hour vs night) to produce delay % per (route, week_of_year, hour).
Output is written to data/synthetic_historical_route_delays.json for use elsewhere.

Usage:
  python generate_synthetic_historical_delays.py
"""

import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
POINT_WEEKLY_WEATHER_PATH = os.path.join(ROOT, "data", "weather", "point_weekly_weather.json")
ROUTES_PATH = os.path.join(ROOT, "data", "routes", "routes_with_weather_and_substation_time.json")
OUTPUT_PATH = os.path.join(ROOT, "data", "synthetic_historical_route_delays.json")

# Origin for all routes in the current routes file (Tracy depot)
ORIGIN_NAME = "Tracy"
ORIGIN_ID = 0  # placeholder; depot not in warehouse list

# Time buckets to generate (hour of day). Use 0,6,12,18 to keep file size manageable.
# Set to range(24) for hourly history.
HOURS = [0, 6, 12, 18]
WEEKS = list(range(1, 53))

# Weather parameter keys included in route-level aggregates (same as build_delay_model / point_weekly_weather)
WEATHER_KEYS = [
    "temp_min_mean",
    "temp_max_mean",
    "temp_mean_mean",
    "snow_depth_mean",
    "prcp_total_mean",
    "visibility_mean",
    "wind_speed_mean",
    "wind_speed_max_mean",
    "wind_gust_max_mean",
]


def aggregate_weekly_weather(weekly_list):
    """Reduce 52-week list to mean features (same as build_delay_model)."""
    if not weekly_list or len(weekly_list) == 0:
        return None
    keys_map = [
        ("temp_min_avg", "temp_min_mean"),
        ("temp_max_avg", "temp_max_mean"),
        ("temp_mean_avg", "temp_mean_mean"),
        ("snow_depth_avg", "snow_depth_mean"),
        ("prcp_total_avg", "prcp_total_mean"),
        ("visibility_avg", "visibility_mean"),
        ("wind_speed_mean_avg", "wind_speed_mean"),
        ("wind_speed_max_avg", "wind_speed_max_mean"),
        ("wind_gust_max_avg", "wind_gust_max_mean"),
    ]
    out = {}
    for src, dst in keys_map:
        vals = [w[src] for w in weekly_list if isinstance(w, dict) and src in w and w[src] is not None]
        if not vals:
            return None
        out[dst] = float(np.mean(vals))
    return out


def synthetic_delay_pct_from_weather(agg):
    """
    Base delay % from weather only (same formula as build_delay_model.synthetic_delay_pct).
    """
    if agg is None:
        return 0.0
    delay = 0.0
    delay += min(20.0, 2.0 * (agg.get("snow_depth_mean") or 0))
    delay += min(15.0, 3.0 * (agg.get("prcp_total_mean") or 0))
    vis = agg.get("visibility_mean") or 10
    if vis < 10:
        delay += min(15.0, (10 - vis) * 2.0)
    wg = agg.get("wind_gust_max_mean")
    delay += min(10.0, 0.3 * (wg or 0))
    if (agg.get("temp_min_mean") or 40) < 32:
        delay += 12.0
    return min(50.0, max(0.0, delay))


def _round_weather(weather_dict):
    """Round weather values for JSON output (2 decimal places)."""
    if not weather_dict:
        return {}
    return {k: round(float(v), 2) for k, v in weather_dict.items() if v is not None}


def time_of_day_multiplier(hour):
    """
    Multiplier for delay based on time of day (estimates).
    - Rush hours (7-9, 16-18): more congestion -> higher delay (1.15).
    - Night (22-5): less traffic -> lower delay (0.9).
    - Otherwise: 1.0.
    """
    if (7 <= hour <= 9) or (16 <= hour <= 18):
        return 1.15
    if hour >= 22 or hour <= 5:
        return 0.9
    return 1.0


def load_weather_aggregates(path):
    """Load point_weekly_weather.json and return dict weather_key -> aggregated features."""
    print("Loading weather:", path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    agg = {}
    for key, val in data.items():
        if not isinstance(val, dict) or "weekly_weather" not in val:
            continue
        a = aggregate_weekly_weather(val["weekly_weather"])
        if a is not None:
            agg[key] = a
    print("  Aggregated weather for", len(agg), "points")
    return agg


def route_mean_weather(path_coordinates, weather_agg):
    """
    For a route's path_coordinates, average weather over all points that have a weather_key.
    Returns a single aggregate dict or None if no valid points.
    """
    aggs = []
    for pt in path_coordinates or []:
        wk = pt.get("weather_key")
        if wk and wk in weather_agg:
            aggs.append(weather_agg[wk])
    if not aggs:
        return None
    # Mean across keys
    keys = list(aggs[0].keys())
    out = {}
    for k in keys:
        out[k] = float(np.mean([a.get(k) for a in aggs if a.get(k) is not None]))
    return out


def main():
    weather_agg = load_weather_aggregates(POINT_WEEKLY_WEATHER_PATH)
    print("Loading routes:", ROUTES_PATH)
    with open(ROUTES_PATH, "r", encoding="utf-8") as f:
        warehouses = json.load(f)

    records = []
    routes_meta = []

    for wh in warehouses:
        dest_id = wh.get("warehouse_id")
        dest_name = wh.get("warehouse_name", "")
        for route_idx, route in enumerate(wh.get("routes") or []):
            path = route.get("path_coordinates") or []
            route_weather = route_mean_weather(path, weather_agg)
            if route_weather is None:
                print("  Skip route", dest_name, "(no weather for path)")
                continue
            base_delay = synthetic_delay_pct_from_weather(route_weather)
            route_id = f"{ORIGIN_NAME}_{dest_id}_{route_idx}"
            weather_rounded = _round_weather(route_weather)
            routes_meta.append({
                "route_id": route_id,
                "origin_name": ORIGIN_NAME,
                "origin_warehouse_id": ORIGIN_ID,
                "destination_warehouse_id": dest_id,
                "destination_warehouse_name": dest_name,
                "route_index": route_idx,
                "base_weather_delay_pct": round(base_delay, 2),
                "weather": weather_rounded,
            })
            for week in WEEKS:
                for hour in HOURS:
                    mult = time_of_day_multiplier(hour)
                    delay_pct = min(50.0, max(0.0, base_delay * mult))
                    # day_of_week and month from week for consistency (approximate)
                    day_of_week = (week * 7) % 7
                    month = min(12, max(1, (week // 4) + 1))
                    records.append({
                        "route_id": route_id,
                        "origin_warehouse_id": ORIGIN_ID,
                        "origin_name": ORIGIN_NAME,
                        "destination_warehouse_id": dest_id,
                        "destination_warehouse_name": dest_name,
                        "week_of_year": week,
                        "hour": hour,
                        "day_of_week": day_of_week,
                        "month": month,
                        "delay_pct": round(delay_pct, 2),
                        "base_weather_delay_pct": round(base_delay, 2),
                        "time_of_day_multiplier": round(mult, 2),
                        "weather": weather_rounded,
                    })

    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    output = {
        "description": "Synthetic historical delay % for all routes (weather + time-of-day). Each record includes route-level weather parameters (temp_min_mean, temp_max_mean, temp_mean_mean, snow_depth_mean, prcp_total_mean, visibility_mean, wind_speed_mean, wind_speed_max_mean, wind_gust_max_mean). No real delay data.",
        "origin_depot": ORIGIN_NAME,
        "weather_parameters_included": WEATHER_KEYS,
        "routes_meta": routes_meta,
        "historical_delays": records,
        "num_routes": len(routes_meta),
        "num_records": len(records),
        "hours_sampled": HOURS,
        "weeks_sampled": "1..52",
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print("Wrote", OUTPUT_PATH)
    print("  Routes:", len(routes_meta))
    print("  Historical delay records:", len(records))


if __name__ == "__main__":
    main()
