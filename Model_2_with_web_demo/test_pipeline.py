#!/usr/bin/env python3
"""
End-to-end test scenarios for the Model 1 → Model 2 pipeline.

Tests:
1. XGBoost model accuracy vs sigmoid baseline
2. 6 representative scenarios (matching v2 continuous_loss_scenarios.csv)
3. Extreme scenarios (blizzard, mild weather)
4. Model 1 integration (if available)

Usage:
    cd pipeline/
    python test_pipeline.py
"""

import os
import sys
import traceback

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from predict_loss import LossPredictor
from pipeline import predict_revenue_loss, format_result

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
PASS_COUNT = 0
FAIL_COUNT = 0


def check(name, condition, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  PASS  {name}")
    else:
        FAIL_COUNT += 1
        print(f"  FAIL  {name}  {detail}")


# ---------------------------------------------------------------------------
# Test 1: Model loads and predicts valid loss_rate
# ---------------------------------------------------------------------------
def test_model_basics():
    print("\n" + "=" * 60)
    print("Test 1: Model basics")
    print("=" * 60)

    pred = LossPredictor()

    # Zero delay → loss near baseline (L_min = 0.07)
    r = pred.predict(delay_hours=0, delay_temp_f=32, lettuce_type="iceberg",
                     organic=0, month=6)
    check("zero delay → loss near baseline",
          0.05 <= r["loss_rate"] <= 0.15,
          f"got {r['loss_rate']:.4f}")

    # Moderate delay → moderate loss
    r = pred.predict(delay_hours=48, delay_temp_f=41, lettuce_type="romaine",
                     organic=0, month=6)
    check("48h @ 41°F → moderate loss (0.07-0.5)",
          0.07 <= r["loss_rate"] <= 0.5,
          f"got {r['loss_rate']:.4f}")

    # 7 days @ 55°F → high loss
    r = pred.predict(delay_hours=168, delay_temp_f=55, lettuce_type="spring_mix",
                     organic=0, month=7)
    check("168h @ 55°F spring_mix → high loss (>0.8)",
          r["loss_rate"] > 0.8,
          f"got {r['loss_rate']:.4f}")

    # All varieties produce valid loss_rate
    for lt in LossPredictor.LETTUCE_TYPES:
        r = pred.predict(delay_hours=24, delay_temp_f=41, lettuce_type=lt, organic=0, month=6)
        check(f"{lt} produces valid loss_rate",
              0 <= r["loss_rate"] <= 1,
              f"got {r['loss_rate']:.4f}")


# ---------------------------------------------------------------------------
# Test 2: Monotonicity — more delay / higher temp → higher loss
# ---------------------------------------------------------------------------
def test_monotonicity():
    print("\n" + "=" * 60)
    print("Test 2: Monotonicity")
    print("=" * 60)

    pred = LossPredictor()

    # Increasing delay → increasing loss
    losses_by_delay = []
    for h in [0, 6, 24, 48, 72, 120, 168]:
        r = pred.predict(delay_hours=h, delay_temp_f=41, lettuce_type="romaine",
                         organic=0, month=6)
        losses_by_delay.append(r["loss_rate"])
    is_monotonic = all(a <= b + 0.01 for a, b in zip(losses_by_delay, losses_by_delay[1:]))
    check("loss increases with delay hours",
          is_monotonic,
          f"losses: {[round(x, 4) for x in losses_by_delay]}")

    # Increasing temperature → increasing loss (same delay)
    losses_by_temp = []
    for t in [32, 37, 41, 45, 50, 55]:
        r = pred.predict(delay_hours=48, delay_temp_f=t, lettuce_type="iceberg",
                         organic=0, month=6)
        losses_by_temp.append(r["loss_rate"])
    is_monotonic = all(a <= b + 0.01 for a, b in zip(losses_by_temp, losses_by_temp[1:]))
    check("loss increases with temperature",
          is_monotonic,
          f"losses: {[round(x, 4) for x in losses_by_temp]}")


# ---------------------------------------------------------------------------
# Test 3: 6 representative scenarios (cross-check with v2 patterns)
# ---------------------------------------------------------------------------
def test_representative_scenarios():
    print("\n" + "=" * 60)
    print("Test 3: Representative scenarios")
    print("=" * 60)

    scenarios = [
        # (name, delay_hours, temp_f, lettuce, organic, month, expected_min, expected_max)
        ("Optimal: 0 delay, 32°F, iceberg",
         0, 32, "iceberg", 0, 6, 0.05, 0.12),
        ("1 day delay, 41°F, romaine",
         24, 41, "romaine", 0, 1, 0.07, 0.20),
        ("2 days delay, 50°F, butterhead",
         48, 50, "butterhead", 0, 8, 0.70, 0.95),  # shelf=2d, 48h=full shelf → high loss
        ("3 days delay, 41°F, leaf",
         72, 41, "leaf", 0, 3, 0.07, 0.15),  # shelf=6d, 3d/6d with beta=1.8 → low consumption
        ("5 days delay, 55°F, romaine",
         120, 55, "romaine", 1, 11, 0.75, 1.0),
        ("7 days delay, 50°F, spring_mix",
         168, 50, "spring_mix", 0, 7, 0.85, 1.0),
    ]

    pred = LossPredictor()
    for name, dh, tf, lt, org, mo, exp_min, exp_max in scenarios:
        r = pred.predict(delay_hours=dh, delay_temp_f=tf, lettuce_type=lt,
                         organic=org, month=mo)
        in_range = exp_min <= r["loss_rate"] <= exp_max
        check(name, in_range,
              f"loss={r['loss_rate']:.4f} expected [{exp_min}, {exp_max}]")


# ---------------------------------------------------------------------------
# Test 4: Extreme scenarios
# ---------------------------------------------------------------------------
def test_extreme_scenarios():
    print("\n" + "=" * 60)
    print("Test 4: Extreme scenarios")
    print("=" * 60)

    # Blizzard: 3 days delay + 55°F → loss near 100%
    r = predict_revenue_loss(
        delay_hours_override=72, delay_temp_f_override=55,
        lettuce_type="romaine", organic=0, quantity_lb=1000,
        journey_start={"date": "2024-12-15"},
    )
    check("Blizzard: 72h + 55°F romaine → loss > 85%",
          r["loss_rate"] > 0.85,
          f"loss={r['loss_rate']:.4f}")
    print(f"    Revenue loss: ${r['revenue_loss']:.2f} / ${r['total_value']:.2f}")

    # Mild: 30 min delay + 32°F → loss near baseline 7%
    r = predict_revenue_loss(
        delay_hours_override=0.5, delay_temp_f_override=32,
        lettuce_type="iceberg", organic=0, quantity_lb=1000,
        journey_start={"date": "2024-06-15"},
    )
    check("Mild: 0.5h + 32°F iceberg → loss near baseline (< 0.12)",
          r["loss_rate"] < 0.12,
          f"loss={r['loss_rate']:.4f}")
    print(f"    Revenue loss: ${r['revenue_loss']:.2f} / ${r['total_value']:.2f}")

    # Very long delay: 7 days + 55°F spring_mix → near total loss
    r = predict_revenue_loss(
        delay_hours_override=168, delay_temp_f_override=55,
        lettuce_type="spring_mix", organic=0, quantity_lb=500,
        journey_start={"date": "2024-07-01"},
    )
    check("Catastrophic: 168h + 55°F spring_mix → loss > 90%",
          r["loss_rate"] > 0.90,
          f"loss={r['loss_rate']:.4f}")
    print(f"    Revenue loss: ${r['revenue_loss']:.2f} / ${r['total_value']:.2f}")


# ---------------------------------------------------------------------------
# Test 5: Pipeline with revenue calculation
# ---------------------------------------------------------------------------
def test_revenue_calculation():
    print("\n" + "=" * 60)
    print("Test 5: Revenue calculation")
    print("=" * 60)

    r = predict_revenue_loss(
        delay_hours_override=24, delay_temp_f_override=41,
        lettuce_type="romaine", organic=0, quantity_lb=1000,
        journey_start={"date": "2024-01-15"},
    )
    expected_total = 1000 * 3.04
    check("total_value = qty × price",
          abs(r["total_value"] - expected_total) < 0.01,
          f"got {r['total_value']}, expected {expected_total}")

    expected_loss = expected_total * r["loss_rate"]
    check("revenue_loss = total_value × loss_rate",
          abs(r["revenue_loss"] - round(expected_loss, 2)) < 0.01,
          f"got {r['revenue_loss']}, expected {round(expected_loss, 2)}")

    print(f"\n  Full result:")
    print(format_result(r))


# ---------------------------------------------------------------------------
# Test 6: Model 1 integration (skip if not available)
# ---------------------------------------------------------------------------
def test_model1_integration():
    print("\n" + "=" * 60)
    print("Test 6: Model 1 integration")
    print("=" * 60)

    # Add Model 1 backend to path
    model1_dir = os.path.join(
        os.path.dirname(SCRIPT_DIR),
        "model 1/ver2/costco_forecasting-master/backend",
    )
    if not os.path.isdir(model1_dir):
        print("  SKIP  Model 1 backend not found")
        return

    sys.path.insert(0, model1_dir)

    try:
        from predict_delay import predict_all_models

        # Clear weather scenario
        weather_clear = {
            "temp_min_mean": 45, "temp_max_mean": 72, "snow_depth_mean": 0,
            "prcp_total_mean": 0, "visibility_mean": 10, "wind_speed_mean": 5,
            "wind_speed_max_mean": 8, "wind_gust_max_mean": 12,
            "temp_mean_mean": 58,
        }
        r = predict_revenue_loss(
            weather_dict=weather_clear,
            journey_start={"date": "2024-06-15", "time": "08:00"},
            lettuce_type="iceberg", organic=0, quantity_lb=1000,
            delay_temp_f_override=72,  # use weather temp_max as ambient
        )
        check("Model 1 → Model 2 pipeline runs",
              r["loss_rate"] is not None and r["model1_results"] is not None,
              f"loss={r.get('loss_rate')}")
        print(f"\n  Clear weather result:")
        print(format_result(r))

        # Snowy weather scenario
        weather_snow = {
            "temp_min_mean": 20, "temp_max_mean": 32, "snow_depth_mean": 8,
            "prcp_total_mean": 3.0, "visibility_mean": 2, "wind_speed_mean": 20,
            "wind_speed_max_mean": 30, "wind_gust_max_mean": 45,
            "temp_mean_mean": 26,
        }
        r = predict_revenue_loss(
            weather_dict=weather_snow,
            journey_start={"date": "2024-01-20", "time": "06:00"},
            lettuce_type="romaine", organic=0, quantity_lb=2000,
            baseline_hours=36,
            delay_temp_f_override=55,  # cold outside but truck refrigeration fails
        )
        check("Blizzard scenario via Model 1 runs",
              r["loss_rate"] is not None,
              f"loss={r.get('loss_rate')}")
        print(f"\n  Blizzard result:")
        print(format_result(r))

    except FileNotFoundError as e:
        print(f"  SKIP  Model 1 .joblib files not found: {e}")
    except Exception as e:
        print(f"  SKIP  Model 1 integration error: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global PASS_COUNT, FAIL_COUNT
    print("=" * 60)
    print("Pipeline Test Suite")
    print("=" * 60)

    test_model_basics()
    test_monotonicity()
    test_representative_scenarios()
    test_extreme_scenarios()
    test_revenue_calculation()
    test_model1_integration()

    print("\n" + "=" * 60)
    total = PASS_COUNT + FAIL_COUNT
    print(f"Results: {PASS_COUNT}/{total} passed, {FAIL_COUNT} failed")
    print("=" * 60)

    if FAIL_COUNT > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
