#!/usr/bin/env python3
"""
End-to-end pipeline: Model 1 (delay prediction) → Model 2 (revenue loss prediction).

Model 1 outputs delay_pct (0-50%), which is converted to delay_hours.
Model 2 (XGBoost) predicts loss_rate from delay + environmental features.
Revenue loss = quantity_lb × price_per_lb × loss_rate × seasonal_index.

Usage:
    from pipeline import predict_revenue_loss
    result = predict_revenue_loss(
        weather_dict={...},
        journey_start={"date": "2024-12-15", "time": "06:00"},
        lettuce_type="romaine",
        organic=0,
        quantity_lb=1000,
        transit_distance_km=2000,
    )
"""

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from predict_loss import LossPredictor


# ---------------------------------------------------------------------------
# Base prices (from lettuce_properties.csv)
# ---------------------------------------------------------------------------
BASE_PRICES = {
    ("iceberg", 0): 1.84,
    ("iceberg", 1): 5.14,
    ("romaine", 0): 3.04,
    ("romaine", 1): 6.65,
    ("butterhead", 0): 3.34,
    ("butterhead", 1): 7.32,
    ("leaf", 0): 2.89,
    ("leaf", 1): 6.33,
    ("spring_mix", 0): 3.95,
    ("spring_mix", 1): 8.66,
}

# Baseline transit hours for converting delay_pct -> delay_hours
# Tracy, CA to average Costco warehouse ~24h baseline
DEFAULT_BASELINE_HOURS = 24.0


def _load_model1():
    """Import Model 1's predict_all_models. Returns None if unavailable."""
    try:
        from predict_delay import predict_all_models
        return predict_all_models
    except (ImportError, FileNotFoundError) as e:
        print(f"Warning: Model 1 not available ({e}). Use manual delay_hours instead.")
        return None


def predict_revenue_loss(
    weather_dict=None,
    journey_start=None,
    lettuce_type="iceberg",
    organic=0,
    quantity_lb=1000,
    transit_distance_km=1000.0,
    pre_delay_consumption_pct=0.0,
    baseline_hours=DEFAULT_BASELINE_HOURS,
    delay_hours_override=None,
    delay_temp_f_override=None,
    model1_name="xgboost",
    loss_model_path=None,
):
    """
    End-to-end prediction: weather → delay → loss_rate → revenue_loss.

    Parameters
    ----------
    weather_dict : dict
        9 weather features for Model 1 (temp_min_mean, temp_max_mean, etc.).
        Not needed if delay_hours_override is provided.
    journey_start : datetime or dict, optional
        Journey start date/time for Model 1 and month extraction.
    lettuce_type : str
        One of: iceberg, romaine, butterhead, leaf, spring_mix.
    organic : int
        0 or 1.
    quantity_lb : float
        Shipment weight in pounds.
    transit_distance_km : float
        Route distance in km.
    pre_delay_consumption_pct : float
        Shelf life already consumed before departure (%).
    baseline_hours : float
        Baseline transit time (hours) for delay_pct → delay_hours conversion.
    delay_hours_override : float, optional
        Skip Model 1 and use this delay directly.
    delay_temp_f_override : float, optional
        Override temperature (otherwise derived from weather_dict temp_max_mean).
    model1_name : str
        Which Model 1 model to use: "ridge", "random_forest", or "xgboost".
    loss_model_path : str, optional
        Path to loss_model_xgb.joblib.

    Returns
    -------
    dict with keys: delay_hours, delay_pct, delay_temp_f, loss_rate, sigmoid_loss_rate,
                    price_per_lb, seasonal_price_index, revenue_loss, revenue_loss_seasonal,
                    total_value, shelf_life_days, consumption_pct, model1_results
    """
    result = {}

    # --- Step 1: Get delay_hours ---
    if delay_hours_override is not None:
        delay_hours = float(delay_hours_override)
        delay_pct = delay_hours / baseline_hours * 100.0
        result["model1_results"] = None
    else:
        if weather_dict is None:
            raise ValueError("Provide weather_dict or delay_hours_override")
        predict_all = _load_model1()
        if predict_all is None:
            raise RuntimeError("Model 1 not available. Provide delay_hours_override.")
        m1_results = predict_all(weather_dict, journey_start=journey_start)
        result["model1_results"] = m1_results

        # Pick the requested model's delay_pct
        display_map = {"ridge": "Ridge", "random_forest": "Random Forest", "xgboost": "XGBoost"}
        display_name = display_map.get(model1_name, model1_name)
        if display_name in m1_results:
            delay_pct = m1_results[display_name]
        elif m1_results:
            delay_pct = list(m1_results.values())[0]
        else:
            raise RuntimeError("Model 1 returned no results")

        # delay_pct -> delay_hours: delay_hours = baseline_hours × (delay_pct / 100)
        delay_hours = baseline_hours * (delay_pct / 100.0)

    result["delay_hours"] = round(delay_hours, 4)
    result["delay_pct"] = round(delay_pct, 2)

    # --- Step 2: Get temperature ---
    if delay_temp_f_override is not None:
        delay_temp_f = float(delay_temp_f_override)
    elif weather_dict is not None:
        delay_temp_f = float(weather_dict.get("temp_max_mean", 41))
    else:
        delay_temp_f = 41.0
    result["delay_temp_f"] = delay_temp_f

    # --- Step 3: Extract month ---
    month = _extract_month(journey_start)

    # --- Step 4: Model 2 → loss_rate ---
    predictor = LossPredictor(model_path=loss_model_path)
    loss_result = predictor.predict(
        delay_hours=delay_hours,
        delay_temp_f=delay_temp_f,
        lettuce_type=lettuce_type,
        organic=organic,
        month=month,
        transit_distance_km=transit_distance_km,
        pre_delay_consumption_pct=pre_delay_consumption_pct,
    )
    result["loss_rate"] = loss_result["loss_rate"]
    result["sigmoid_loss_rate"] = loss_result["sigmoid_loss_rate"]
    result["shelf_life_days"] = loss_result["shelf_life_days"]
    result["consumption_pct"] = loss_result["consumption_pct"]
    result["seasonal_price_index"] = loss_result["seasonal_price_index"]

    # --- Step 5: Calculate revenue loss ---
    price = BASE_PRICES.get((lettuce_type, organic), 3.04)
    total_value = quantity_lb * price
    revenue_loss = total_value * loss_result["loss_rate"]
    revenue_loss_seasonal = revenue_loss * loss_result["seasonal_price_index"]

    result["price_per_lb"] = price
    result["total_value"] = round(total_value, 2)
    result["revenue_loss"] = round(revenue_loss, 2)
    result["revenue_loss_seasonal"] = round(revenue_loss_seasonal, 2)
    result["lettuce_type"] = lettuce_type
    result["organic"] = organic
    result["quantity_lb"] = quantity_lb
    result["month"] = month

    return result


def _extract_month(journey_start):
    """Extract month from journey_start (datetime, dict, or None)."""
    if journey_start is None:
        return 6
    from datetime import datetime
    if isinstance(journey_start, datetime):
        return journey_start.month
    if isinstance(journey_start, dict):
        if "month" in journey_start:
            return int(journey_start["month"])
        date_str = journey_start.get("date", "")
        if date_str:
            try:
                return datetime.strptime(str(date_str).strip(), "%Y-%m-%d").month
            except ValueError:
                pass
    return 6


def format_result(result):
    """Pretty-print a pipeline result dict."""
    lines = [
        f"  Lettuce: {result['lettuce_type']} ({'organic' if result['organic'] else 'conventional'})",
        f"  Quantity: {result['quantity_lb']} lb @ ${result['price_per_lb']:.2f}/lb = ${result['total_value']:.2f}",
        f"  Delay: {result['delay_hours']:.2f} hours @ {result['delay_temp_f']:.0f}°F",
        f"  Shelf life: {result['shelf_life_days']:.1f} days | Consumption: {result['consumption_pct']:.1f}%",
        f"  Loss rate: {result['loss_rate']:.4f} (XGB) vs {result['sigmoid_loss_rate']:.4f} (sigmoid)",
        f"  Revenue loss: ${result['revenue_loss']:.2f}",
        f"  Seasonal loss (month {result['month']}): ${result['revenue_loss_seasonal']:.2f} "
        f"(index={result['seasonal_price_index']:.4f})",
    ]
    if result.get("model1_results"):
        m1 = result["model1_results"]
        m1_str = ", ".join(f"{k}: {v:.2f}%" for k, v in m1.items())
        lines.insert(2, f"  Model 1 delays: {m1_str}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 60)
    print("Pipeline Demo (manual delay_hours)")
    print("=" * 60)

    scenarios = [
        {
            "name": "Mild delay, cold",
            "params": dict(
                delay_hours_override=0.5, delay_temp_f_override=32,
                lettuce_type="iceberg", organic=0, quantity_lb=1000,
                journey_start={"date": "2024-06-15"},
            ),
        },
        {
            "name": "Moderate delay, warm",
            "params": dict(
                delay_hours_override=24, delay_temp_f_override=45,
                lettuce_type="romaine", organic=0, quantity_lb=1000,
                journey_start={"date": "2024-01-15"},
            ),
        },
        {
            "name": "Severe delay, hot",
            "params": dict(
                delay_hours_override=72, delay_temp_f_override=55,
                lettuce_type="romaine", organic=1, quantity_lb=2000,
                journey_start={"date": "2024-12-01"},
            ),
        },
    ]

    for s in scenarios:
        print(f"\n--- {s['name']} ---")
        r = predict_revenue_loss(**s["params"])
        print(format_result(r))
