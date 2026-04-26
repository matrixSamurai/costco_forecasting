"""
Test suite for Pre-Model 1 Binary Classifier.

Validates classifier predictions against known weather scenarios.

Usage:
    python test_classifier.py
"""

import sys
from predict_has_delay import predict_has_delay, predict_has_delay_all

SCENARIOS = [
    # --- Should predict NO delay ---
    {
        "name": "Clear sky, summer",
        "weather": {
            "temp_mean": 75, "temp_min": 65, "temp_max": 85,
            "prcp_total": 0, "snow_depth": 0, "visibility": 10,
            "wind_speed_mean": 5, "wind_gust_max": 8,
        },
        "date": "2024-07-15",
        "expected_delay": False,
    },
    {
        "name": "Light breeze, clear",
        "weather": {
            "temp_mean": 65, "temp_min": 55, "temp_max": 72,
            "prcp_total": 0, "snow_depth": 0, "visibility": 9,
            "wind_speed_mean": 8, "wind_gust_max": 12,
        },
        "date": "2024-05-20",
        "expected_delay": False,
    },
    {
        "name": "Warm, dry, good visibility",
        "weather": {
            "temp_mean": 80, "temp_min": 70, "temp_max": 90,
            "prcp_total": 0, "snow_depth": 0, "visibility": 10,
            "wind_speed_mean": 4, "wind_gust_max": 6,
        },
        "date": "2024-08-01",
        "expected_delay": False,
    },
    # --- Should predict YES delay ---
    {
        "name": "Blizzard",
        "weather": {
            "temp_mean": 20, "temp_min": 10, "temp_max": 25,
            "prcp_total": 3, "snow_depth": 8, "visibility": 1,
            "wind_speed_mean": 30, "wind_gust_max": 45,
        },
        "date": "2024-01-15",
        "expected_delay": True,
    },
    {
        "name": "Dense fog",
        "weather": {
            "temp_mean": 50, "temp_min": 45, "temp_max": 55,
            "prcp_total": 0, "snow_depth": 0, "visibility": 0.5,
            "wind_speed_mean": 3, "wind_gust_max": 5,
        },
        "date": "2024-11-10",
        "expected_delay": True,
    },
    {
        "name": "Thunderstorm",
        "weather": {
            "temp_mean": 85, "temp_min": 75, "temp_max": 95,
            "prcp_total": 2, "snow_depth": 0, "visibility": 5,
            "wind_speed_mean": 25, "wind_gust_max": 40,
        },
        "date": "2024-06-20",
        "expected_delay": True,
    },
    {
        "name": "Ice storm, freezing",
        "weather": {
            "temp_mean": 28, "temp_min": 22, "temp_max": 32,
            "prcp_total": 1.5, "snow_depth": 3, "visibility": 3,
            "wind_speed_mean": 15, "wind_gust_max": 25,
        },
        "date": "2024-02-05",
        "expected_delay": True,
    },
]


def test_individual_models():
    """Test each model individually on all scenarios."""
    print("=" * 60)
    print("  Test 1: Individual Model Predictions")
    print("=" * 60)

    passed = 0
    failed = 0

    for model_name in ["xgboost", "random_forest", "lightgbm"]:
        print(f"\n  Model: {model_name}")
        for s in SCENARIOS:
            result = predict_has_delay(
                s["weather"], model_name=model_name, query_date=s["date"]
            )
            match = result["has_delay"] == s["expected_delay"]
            status = "PASS" if match else "FAIL"
            if match:
                passed += 1
            else:
                failed += 1
            expected_str = "DELAY" if s["expected_delay"] else "NO DELAY"
            actual_str = "DELAY" if result["has_delay"] else "NO DELAY"
            print(
                f"    [{status}] {s['name']:<30s} "
                f"expected={expected_str:<10s} actual={actual_str:<10s} "
                f"prob={result['probability']:.2%}"
            )

    return passed, failed


def test_all_models_agree_on_extremes():
    """Test that all models agree on clear-cut extreme cases."""
    print("\n" + "=" * 60)
    print("  Test 2: All Models Agree on Extremes")
    print("=" * 60)

    extreme_scenarios = [
        {
            "name": "Perfect weather",
            "weather": {
                "temp_mean": 72, "temp_min": 62, "temp_max": 82,
                "prcp_total": 0, "snow_depth": 0, "visibility": 10,
                "wind_speed_mean": 3, "wind_gust_max": 5,
            },
            "date": "2024-06-15",
            "expected_delay": False,
        },
        {
            "name": "Severe blizzard",
            "weather": {
                "temp_mean": 15, "temp_min": 5, "temp_max": 20,
                "prcp_total": 5, "snow_depth": 12, "visibility": 0.5,
                "wind_speed_mean": 35, "wind_gust_max": 55,
            },
            "date": "2024-12-20",
            "expected_delay": True,
        },
    ]

    passed = 0
    failed = 0

    for s in extreme_scenarios:
        results = predict_has_delay_all(s["weather"], query_date=s["date"])
        predictions = [r["has_delay"] for r in results.values()]
        all_agree = len(set(predictions)) == 1
        correct = predictions[0] == s["expected_delay"] if all_agree else False

        status = "PASS" if (all_agree and correct) else "FAIL"
        if all_agree and correct:
            passed += 1
        else:
            failed += 1

        print(f"\n    [{status}] {s['name']}")
        for model_name, r in results.items():
            label = "DELAY" if r["has_delay"] else "NO DELAY"
            print(f"      {model_name:<15s}: {label} ({r['probability']:.2%})")

    return passed, failed


def test_probability_sanity():
    """Test that probabilities are reasonable for extreme cases."""
    print("\n" + "=" * 60)
    print("  Test 3: Probability Sanity Check")
    print("=" * 60)

    passed = 0
    failed = 0

    # Clear weather should have low probability
    clear = predict_has_delay(
        {
            "temp_mean": 72, "temp_min": 62, "temp_max": 82,
            "prcp_total": 0, "snow_depth": 0, "visibility": 10,
            "wind_speed_mean": 3, "wind_gust_max": 5,
        },
        query_date="2024-06-15",
    )
    ok = clear["probability"] < 0.3
    status = "PASS" if ok else "FAIL"
    passed += ok
    failed += not ok
    print(f"    [{status}] Clear weather probability < 0.3: {clear['probability']:.2%}")

    # Blizzard should have high probability
    blizzard = predict_has_delay(
        {
            "temp_mean": 15, "temp_min": 5, "temp_max": 20,
            "prcp_total": 5, "snow_depth": 12, "visibility": 0.5,
            "wind_speed_mean": 35, "wind_gust_max": 55,
        },
        query_date="2024-12-20",
    )
    ok = blizzard["probability"] > 0.7
    status = "PASS" if ok else "FAIL"
    passed += ok
    failed += not ok
    print(f"    [{status}] Blizzard probability > 0.7: {blizzard['probability']:.2%}")

    return passed, failed


def test_model1_field_compatibility():
    """Test that Model 1 field names work correctly."""
    print("\n" + "=" * 60)
    print("  Test 4: Model 1 Field Name Compatibility")
    print("=" * 60)

    # Use Model 1 field names (with _mean suffix)
    model1_weather = {
        "temp_mean_mean": 20,
        "temp_min_mean": 10,
        "temp_max_mean": 25,
        "prcp_total_mean": 3,
        "snow_depth_mean": 8,
        "visibility_mean": 1,
        "wind_speed_mean": 30,
        "wind_gust_max_mean": 45,
    }

    # Use our field names
    our_weather = {
        "temp_mean": 20,
        "temp_min": 10,
        "temp_max": 25,
        "prcp_total": 3,
        "snow_depth": 8,
        "visibility": 1,
        "wind_speed_mean": 30,
        "wind_gust_max": 45,
    }

    r1 = predict_has_delay(model1_weather, query_date="2024-01-15")
    r2 = predict_has_delay(our_weather, query_date="2024-01-15")

    match = r1["has_delay"] == r2["has_delay"] and abs(r1["probability"] - r2["probability"]) < 0.01
    status = "PASS" if match else "FAIL"
    passed = 1 if match else 0
    failed = 0 if match else 1

    print(f"    [{status}] Model 1 names: has_delay={r1['has_delay']}, prob={r1['probability']:.2%}")
    print(f"    [{status}] Our names:     has_delay={r2['has_delay']}, prob={r2['probability']:.2%}")

    return passed, failed


def main():
    print("\n" + "=" * 60)
    print("  PRE-MODEL 1 CLASSIFIER TEST SUITE")
    print("=" * 60)

    total_passed = 0
    total_failed = 0

    p, f = test_individual_models()
    total_passed += p
    total_failed += f

    p, f = test_all_models_agree_on_extremes()
    total_passed += p
    total_failed += f

    p, f = test_probability_sanity()
    total_passed += p
    total_failed += f

    p, f = test_model1_field_compatibility()
    total_passed += p
    total_failed += f

    print("\n" + "=" * 60)
    print(f"  TOTAL: {total_passed} passed, {total_failed} failed")
    print("=" * 60)

    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
