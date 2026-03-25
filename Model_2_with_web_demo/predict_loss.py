#!/usr/bin/env python3
"""
Predict loss_rate using the trained XGBoost model.

Usage:
    from predict_loss import predict_loss_rate, LossPredictor

    predictor = LossPredictor()  # loads model once
    loss_rate = predictor.predict(
        delay_hours=48.0,
        delay_temp_f=45.0,
        lettuce_type="romaine",
        organic=0,
        month=6,
        transit_distance_km=2000,
        pre_delay_consumption_pct=5.0,
    )
"""

import os
import numpy as np
from joblib import load

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "loss_model_xgb.joblib")

# Sigmoid fallback (same formula as v2)
def _sigmoid_loss_rate(consumption_pct, k, x0, L_min):
    x = np.asarray(consumption_pct, dtype=float)
    rate = L_min + (1.0 - L_min) / (1.0 + np.exp(-k * (x - x0)))
    return float(np.clip(rate, L_min, 1.0))


def _effective_consumption_pct(delay_hours, shelf_life_days, beta, pre_delay_pct=0.0):
    delay_days = delay_hours / 24.0
    ratio = delay_days / shelf_life_days
    return ratio ** beta * 100.0 + pre_delay_pct


def _interpolate_shelf_life(shelf_table, lettuce_type, temp_f):
    points = shelf_table.get(lettuce_type)
    if points is None:
        return 7.0
    temps = [p[0] for p in points]
    lives = [p[1] for p in points]
    if temp_f <= temps[0]:
        return float(lives[0])
    if temp_f >= temps[-1]:
        return float(lives[-1])
    for i in range(len(temps) - 1):
        if temps[i] <= temp_f <= temps[i + 1]:
            frac = (temp_f - temps[i]) / (temps[i + 1] - temps[i])
            return lives[i] + frac * (lives[i + 1] - lives[i])
    return float(lives[-1])


class LossPredictor:
    """Load XGBoost model and predict loss_rate."""

    LETTUCE_TYPES = ["iceberg", "romaine", "butterhead", "leaf", "spring_mix"]

    def __init__(self, model_path=None):
        path = model_path or DEFAULT_MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found: {path}\nRun build_loss_model.py first."
            )
        artifact = load(path)
        self.model = artifact["model"]
        self.feature_names = artifact["feature_names"]
        self.lettuce_encoding = artifact["lettuce_type_encoding"]
        self.sigmoid_params = artifact["sigmoid_params"]
        self.shelf_table = artifact["shelf_life_table"]
        self.seasonal_lookup = artifact["seasonal_lookup"]

    def predict(
        self,
        delay_hours,
        delay_temp_f,
        lettuce_type="iceberg",
        organic=0,
        month=6,
        transit_distance_km=1000.0,
        pre_delay_consumption_pct=0.0,
    ):
        """
        Predict loss_rate (0-1) using the XGBoost model.

        Returns dict with loss_rate and intermediate values.
        """
        lt_enc = self.lettuce_encoding.get(lettuce_type, 0)
        shelf_life = _interpolate_shelf_life(self.shelf_table, lettuce_type, delay_temp_f)
        p = self.sigmoid_params
        consumption = _effective_consumption_pct(
            delay_hours, shelf_life, p["beta"], pre_delay_consumption_pct
        )
        si = self.seasonal_lookup.get((lettuce_type, month), 1.0)

        features = np.array([[
            delay_hours,
            delay_temp_f,
            shelf_life,
            consumption,
            lt_enc,
            organic,
            month,
            si,
            transit_distance_km,
            pre_delay_consumption_pct,
        ]])

        loss_rate = float(np.clip(self.model.predict(features)[0], 0, 1))

        # Also compute sigmoid baseline for comparison
        sig_loss = _sigmoid_loss_rate(consumption, p["k"], p["x0"], p["L_min"])

        return {
            "loss_rate": loss_rate,
            "sigmoid_loss_rate": sig_loss,
            "shelf_life_days": shelf_life,
            "consumption_pct": consumption,
            "seasonal_price_index": si,
        }


def predict_loss_rate(delay_hours, delay_temp_f, lettuce_type="iceberg", organic=0,
                      month=6, transit_distance_km=1000.0, pre_delay_consumption_pct=0.0,
                      model_path=None):
    """Convenience function — creates predictor and runs one prediction."""
    predictor = LossPredictor(model_path=model_path)
    return predictor.predict(
        delay_hours=delay_hours,
        delay_temp_f=delay_temp_f,
        lettuce_type=lettuce_type,
        organic=organic,
        month=month,
        transit_distance_km=transit_distance_km,
        pre_delay_consumption_pct=pre_delay_consumption_pct,
    )


if __name__ == "__main__":
    print("Testing LossPredictor...\n")
    pred = LossPredictor()
    test_cases = [
        {"delay_hours": 0.5, "delay_temp_f": 32, "lettuce_type": "iceberg", "organic": 0, "month": 6},
        {"delay_hours": 24, "delay_temp_f": 41, "lettuce_type": "romaine", "organic": 0, "month": 1},
        {"delay_hours": 72, "delay_temp_f": 55, "lettuce_type": "romaine", "organic": 0, "month": 12},
        {"delay_hours": 168, "delay_temp_f": 55, "lettuce_type": "spring_mix", "organic": 0, "month": 7},
    ]
    for tc in test_cases:
        result = pred.predict(**tc)
        print(f"  {tc['lettuce_type']:>12s} | {tc['delay_hours']:5.1f}h @ {tc['delay_temp_f']}°F "
              f"| XGB loss={result['loss_rate']:.4f} | Sigmoid={result['sigmoid_loss_rate']:.4f} "
              f"| shelf={result['shelf_life_days']:.1f}d | cons={result['consumption_pct']:.1f}%")
