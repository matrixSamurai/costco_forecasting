"""
Generate purely synthetic historical_delays without needing any real data files.
Writes data/synthetic_historical_route_delays.json compatible with
build_delay_model_from_synthetic.py.
"""

import json
import os
import math
import random

random.seed(42)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(ROOT, "data", "synthetic_historical_route_delays.json")

WAREHOUSE_IDS = list(range(1, 21))   # 20 fictional warehouses
HOURS = [0, 6, 9, 12, 15, 18, 21]
WEEKS = list(range(1, 53))


def winter_severity(day_of_year):
    if day_of_year <= 45 or day_of_year >= 320:
        return 1.0
    if 46 <= day_of_year <= 80 or 280 <= day_of_year <= 319:
        return 0.6
    if 81 <= day_of_year <= 120 or 245 <= day_of_year <= 279:
        return 0.3
    return 0.0


def week_to_day_of_year(week):
    return min(365, max(1, (week - 1) * 7 + 4))


def synth_weather(week):
    """Return plausible weather dict for a given week of year."""
    doy = week_to_day_of_year(week)
    # Temperature: sinusoidal, cold in winter
    temp_mean = 10 + 15 * math.sin(math.pi * (doy - 15) / 182.5)
    temp_min = temp_mean - 5 + random.gauss(0, 1)
    temp_max = temp_mean + 5 + random.gauss(0, 1)
    snow_depth = max(0, (1.0 - winter_severity(doy)) * -1 + 1.0) * random.expovariate(2)
    prcp = max(0, random.expovariate(3) * (0.5 + 0.5 * (1 - abs(doy - 182) / 182)))
    visibility = max(1, 10 - snow_depth * 2 - prcp * 0.5 + random.gauss(0, 0.5))
    wind_speed = max(0, random.gauss(15, 5))
    wind_speed_max = wind_speed + random.expovariate(0.5)
    wind_gust_max = wind_speed_max + random.expovariate(0.3)
    return {
        "temp_min_mean": round(temp_min, 2),
        "temp_max_mean": round(temp_max, 2),
        "temp_mean_mean": round(temp_mean, 2),
        "snow_depth_mean": round(snow_depth, 3),
        "prcp_total_mean": round(prcp, 3),
        "visibility_mean": round(visibility, 2),
        "wind_speed_mean": round(wind_speed, 2),
        "wind_speed_max_mean": round(wind_speed_max, 2),
        "wind_gust_max_mean": round(wind_gust_max, 2),
    }


def synth_delay(weather, hour, week):
    doy = week_to_day_of_year(week)
    ws = winter_severity(doy)
    base = 0.0
    base += ws * 8
    base += weather["snow_depth_mean"] * 5
    base += weather["prcp_total_mean"] * 3
    base += max(0, 10 - weather["visibility_mean"]) * 1.5
    base += weather["wind_gust_max_mean"] * 0.3
    # Rush hour penalty
    if 7 <= hour <= 9 or 16 <= hour <= 18:
        base += 5
    # Night slight penalty
    if hour == 0 or hour == 21:
        base += 2
    base += random.gauss(0, 2)
    return round(max(0, min(40, base)), 2)


records = []
for wh in WAREHOUSE_IDS:
    for week in WEEKS:
        weather = synth_weather(week)
        doy = week_to_day_of_year(week)
        month = max(1, min(12, (doy - 1) // 30 + 1))
        for hour in HOURS:
            day_of_week = (week * 7 + hour) % 7
            delay_pct = synth_delay(weather, hour, week)
            records.append({
                "warehouse_id": wh,
                "week_of_year": week,
                "hour": hour,
                "day_of_week": day_of_week,
                "month": month,
                "day_of_year": doy,
                "segment_start_hour": (hour + 1) % 24,
                "weather": weather,
                "delay_pct": delay_pct,
            })

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump({"historical_delays": records}, f)

print(f"Wrote {len(records)} records to {OUTPUT_PATH}")
