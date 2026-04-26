"""
Route and weather helpers for the delay-by-route API.
- Load warehouses, weather stations, point weekly weather (aggregated).
- Sample pitstops at ~20 mile intervals along the route.
- For each pitstop find nearest weather station, then weather features; predict delay per chunk; aggregate per route.
"""
import json
import os
import math
import urllib.request
import urllib.parse

ROOT = os.path.dirname(os.path.abspath(__file__))
WAREHOUSES_PATH = os.path.join(ROOT, "data", "source", "costco_warehouses_full.json")
POINT_WEEKLY_WEATHER_PATH = os.path.join(ROOT, "data", "weather", "point_weekly_weather.json")
WEATHER_STATIONS_PATH = os.path.join(ROOT, "data", "weather", "weather_stations.json")

# Tracy depot, California (Costco)
TRACY_DEPOT = {"lat": 37.7397, "lng": -121.4252}

_warehouses = None
_weather_agg = None
_weather_stations = None


def decode_polyline(encoded):
    """Decode Google encoded polyline to list of (lat, lng)."""
    if not encoded:
        return []
    inv = 1.0 / 1e5
    coords = []
    x = y = 0
    i = 0
    n = len(encoded)
    while i < n:
        b = shift = result = 0
        while True:
            b = ord(encoded[i]) - 63
            i += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if result & 1 else result >> 1
        x += dlat

        if i >= n:
            break
        shift = result = 0
        while True:
            b = ord(encoded[i]) - 63
            i += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if result & 1 else result >> 1
        y += dlng

        coords.append((x * inv, y * inv))
    return coords


def _aggregate_weekly(weekly_list):
    """
    One point's 52-week list -> single feature dict for prediction.
    Matches point_weekly_weather.json: temp_min_avg, temp_max_avg, temp_mean_avg,
    snow_depth_avg, prcp_total_avg, visibility_avg, wind_speed_mean_avg, wind_speed_max_avg, wind_gust_max_avg.
    """
    if not weekly_list:
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
        vals = [w[src] for w in weekly_list if isinstance(w, dict) and w.get(src) is not None]
        if not vals:
            return None
        out[dst] = sum(vals) / len(vals)
    return out


def load_warehouses():
    global _warehouses
    if _warehouses is None:
        with open(WAREHOUSES_PATH, "r", encoding="utf-8") as f:
            _warehouses = json.load(f)
    return _warehouses


def load_weather_stations():
    """Load weather_stations.json (list of {station_id, latitude, longitude, station_name})."""
    global _weather_stations
    if _weather_stations is not None:
        return _weather_stations
    if not os.path.isfile(WEATHER_STATIONS_PATH):
        return []
    with open(WEATHER_STATIONS_PATH, "r", encoding="utf-8") as f:
        _weather_stations = json.load(f)
    return _weather_stations if isinstance(_weather_stations, list) else []


def load_weather_aggregates():
    """Load point_weekly_weather and build key -> aggregated features."""
    global _weather_agg
    if _weather_agg is not None:
        return _weather_agg
    with open(POINT_WEEKLY_WEATHER_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    _weather_agg = {}
    for key, val in data.items():
        if not isinstance(val, dict) or "weekly_weather" not in val:
            continue
        agg = _aggregate_weekly(val["weekly_weather"])
        if agg is not None:
            _weather_agg[key] = agg
    return _weather_agg


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    a = min(1.0, max(0.0, a))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _parse_key(k):
    try:
        parts = k.split("_")
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
    except (ValueError, IndexError):
        pass
    return None


def nearest_weather_key(lat, lng, weather_agg):
    """Find closest weather key to (lat, lng)."""
    if not weather_agg:
        return None
    best_key = None
    best_d = float("inf")
    for key in weather_agg:
        pt = _parse_key(key)
        if pt is None:
            continue
        d = _haversine_km(lat, lng, pt[0], pt[1])
        if d < best_d:
            best_d = d
            best_key = key
    return best_key


def nearest_weather_station(lat, lng, stations=None):
    """Find closest weather station to (lat, lng). Returns station dict or None."""
    if stations is None:
        stations = load_weather_stations()
    if not stations:
        return None
    best = None
    best_d = float("inf")
    for s in stations:
        slat = float(s.get("latitude", s.get("lat", 0)))
        slng = float(s.get("longitude", s.get("lng", 0)))
        d = _haversine_km(lat, lng, slat, slng)
        if d < best_d:
            best_d = d
            best = s
    return best


def get_weather_features_for_point(lat, lng):
    """Return dict of 7 features for predict_delay_pct, or None (uses point_weekly_weather keys)."""
    agg = load_weather_aggregates()
    key = nearest_weather_key(lat, lng, agg)
    if key is None:
        return None
    return agg.get(key)


def get_weather_features_for_pitstop(lat, lng):
    """
    For a pitstop (lat, lng): find nearest weather station from weather_stations.json,
    then get weather features at that station's location from point_weekly_weather.
    """
    station = nearest_weather_station(lat, lng)
    if station is None:
        return get_weather_features_for_point(lat, lng)
    slat = float(station.get("latitude", station.get("lat", 0)))
    slng = float(station.get("longitude", station.get("lng", 0)))
    return get_weather_features_for_point(slat, slng)


def _segment_km(lat1, lon1, lat2, lon2):
    return _haversine_km(lat1, lon1, lat2, lon2)


# Pitstop interval: 20 miles (~32.2 km) so long routes have on the order of 100–200 pitstops
PITSTOP_INTERVAL_MILES = 20
KM_PER_MILE = 1.609344
PITSTOP_INTERVAL_KM = PITSTOP_INTERVAL_MILES * KM_PER_MILE


def sample_route_points_by_distance(points, interval_km=None):
    """
    Sample pitstops along the route at approximately interval_km (default 20 miles ≈ 32.2 km).
    Always includes first and last point; intermediate points every ~interval_km (interpolating along segments).
    """
    if interval_km is None:
        interval_km = PITSTOP_INTERVAL_KM
    if not points or len(points) < 2:
        return list(points) if points else []
    if interval_km <= 0:
        return [points[0], points[-1]] if points[0] != points[-1] else [points[0]]
    out = [points[0]]
    cum_km = 0.0
    for i in range(1, len(points)):
        lat1, lng1 = points[i - 1]
        lat2, lng2 = points[i]
        seg_km = _segment_km(lat1, lng1, lat2, lng2)
        if seg_km <= 0:
            continue
        added_any = False
        while cum_km + seg_km >= interval_km:
            t = (interval_km - cum_km) / seg_km
            lat = lat1 + t * (lat2 - lat1)
            lng = lng1 + t * (lng2 - lng1)
            out.append((lat, lng))
            added_any = True
            # Remainder of segment from new point to (lat2, lng2); next carry is 0 (we consumed interval_km)
            seg_km = (1 - t) * seg_km
            lat1, lng1 = lat, lng
            cum_km = 0.0
        if not added_any:
            cum_km += seg_km
        else:
            cum_km = seg_km  # carry remainder into next segment
    if points[-1] != out[-1]:
        out.append(points[-1])
    return out


def sample_route_points(points, max_points=25):
    """Sample up to max_points along the route (evenly). Kept for backward compatibility."""
    if not points or len(points) <= max_points:
        return list(points) if points else []
    step = (len(points) - 1) / (max_points - 1)
    return [points[int(round(i * step))] for i in range(max_points)]


def fetch_directions(origin_lat, origin_lng, dest_lat, dest_lng, api_key):
    """Call Google Directions API; return (list of routes, error_message or None)."""
    origin = "%s,%s" % (origin_lat, origin_lng)
    dest = "%s,%s" % (dest_lat, dest_lng)
    url = "https://maps.googleapis.com/maps/api/directions/json?origin=%s&destination=%s&alternatives=true&key=%s" % (
        urllib.parse.quote(origin),
        urllib.parse.quote(dest),
        urllib.parse.quote(api_key),
    )
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())
    status = data.get("status", "")
    if status != "OK":
        err_msg = data.get("error_message") or status
        return [], err_msg
    routes_out = []
    for r in data.get("routes", []):
        leg = (r.get("legs") or [{}])[0]
        polyline = (r.get("overview_polyline") or {}).get("points") or ""
        dist = (leg.get("distance") or {}).get("value")
        dur = (leg.get("duration") or {}).get("value")
        routes_out.append({
            "polyline": polyline,
            "distance_m": dist,
            "duration_s": dur,
            "distance_text": (leg.get("distance") or {}).get("text", ""),
            "duration_text": (leg.get("duration") or {}).get("text", ""),
        })
    return routes_out, None
