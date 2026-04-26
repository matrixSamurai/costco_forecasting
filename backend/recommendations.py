#!/usr/bin/env python3
"""
Supply risk assessment and recommendations engine.

Loads supply_risk_lookup.csv (built from USDA, FRED, FEMA data) and generates
actionable recommendations based on Model 2 predictions.

Usage:
    from recommendations import get_supply_risk, generate_recommendations
"""

import csv
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOOKUP_CSV = os.path.join(SCRIPT_DIR, "data", "supply_risk_lookup.csv")

# Cache for supply risk lookup
_supply_lookup = None


def _load_supply_lookup():
    """Load supply_risk_lookup.csv into memory."""
    global _supply_lookup
    if _supply_lookup is not None:
        return _supply_lookup

    _supply_lookup = {}
    if not os.path.exists(LOOKUP_CSV):
        return _supply_lookup

    with open(LOOKUP_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["lettuce_type"], int(row["month"]))
            _supply_lookup[key] = {
                "supply_index": float(row["supply_index"]),
                "source_region": row["source_region"],
                "disaster_frequency": float(row["disaster_frequency"]),
                "risk_baseline": float(row["risk_baseline"]),
                "risk_level": row["risk_level"],
            }

    return _supply_lookup


def get_supply_risk(lettuce_type, month, delay_hours=0, temp_f=41):
    """
    Assess supply risk for a given lettuce type and month.

    Uses USDA shipment data, FRED prices, and FEMA disaster history.
    Extreme weather conditions (high delay/temp) can escalate risk level.

    Returns dict:
        risk_level: "low" | "medium" | "high"
        supply_index: 0-1 (1 = peak supply)
        source_region: primary growing region this month
        details: human-readable explanation
    """
    lookup = _load_supply_lookup()
    entry = lookup.get((lettuce_type, month))

    if entry is None:
        # Fallback: try generic lookup or default
        for lt in ["iceberg", "romaine"]:
            entry = lookup.get((lt, month))
            if entry:
                break
        if entry is None:
            return {
                "risk_level": "medium",
                "supply_index": 0.8,
                "source_region": "Unknown",
                "details": "Insufficient data for risk assessment.",
            }

    supply_index = entry["supply_index"]
    risk_level = entry["risk_level"]
    source_region = entry["source_region"]
    disaster_freq = entry["disaster_frequency"]

    # Escalate risk if extreme weather conditions
    if delay_hours > 48 or temp_f > 50:
        if risk_level == "low":
            risk_level = "medium"
        elif risk_level == "medium":
            risk_level = "high"

    # Build details message
    details_parts = []
    details_parts.append(f"Primary source: {source_region}")
    details_parts.append(f"Supply index: {supply_index:.0%}")

    if supply_index < 0.7:
        details_parts.append("Supply is below average — production transition period")
    elif supply_index >= 0.9:
        details_parts.append("Supply is at peak levels")

    if disaster_freq > 2.0:
        details_parts.append(
            f"Historical disaster frequency: {disaster_freq:.1f}/yr in this month"
        )

    if delay_hours > 48:
        details_parts.append("Extended delay increases regional supply disruption risk")

    return {
        "risk_level": risk_level,
        "supply_index": round(supply_index, 3),
        "source_region": source_region,
        "disaster_frequency": round(disaster_freq, 2),
        "details": ". ".join(details_parts) + ".",
    }


def generate_recommendations(loss_rate, delay_hours, temp_f, supply_risk_info,
                              quantity_lb=1000):
    """
    Generate actionable supply chain recommendations.

    All recommendations assume Costco's standard reefer truck fleet.
    Focus on scheduling, sourcing, and inventory decisions.

    Args:
        loss_rate: predicted loss rate (0-1) from Model 2
        delay_hours: expected delay in hours
        temp_f: transit temperature in °F
        supply_risk_info: dict from get_supply_risk()
        quantity_lb: shipment weight in pounds

    Returns list of dicts:
        [{ "level": "critical"|"warning"|"info", "message": "..." }, ...]
    """
    recs = []
    risk_level = supply_risk_info.get("risk_level", "low")
    source_region = supply_risk_info.get("source_region", "")

    # --- Loss-rate based recommendations ---

    if loss_rate > 0.5:
        recs.append({
            "level": "critical",
            "message": (
                f"Expected loss exceeds {loss_rate:.0%}. "
                "Recommend postponing shipment until weather conditions improve."
            ),
        })
        recs.append({
            "level": "critical",
            "message": (
                "Alert destination warehouse to activate backup supplier "
                "and adjust receiving schedule."
            ),
        })

    elif loss_rate > 0.3:
        max_qty = int(quantity_lb * (1 - loss_rate))
        recs.append({
            "level": "warning",
            "message": (
                f"Expected loss is {loss_rate:.0%}. Consider reducing shipment "
                f"size to {max_qty:,} lb to limit financial exposure."
            ),
        })
        recs.append({
            "level": "warning",
            "message": (
                "Notify destination store to adjust inventory expectations "
                "and consider supplementary orders from nearby distribution centers."
            ),
        })

    elif loss_rate > 0.15:
        recs.append({
            "level": "warning",
            "message": (
                f"Moderate loss expected ({loss_rate:.0%}). "
                "Monitor weather updates closely and have contingency routing ready."
            ),
        })

    # --- Delay-based recommendations ---

    if delay_hours > 48:
        recs.append({
            "level": "warning",
            "message": (
                f"Extended delay of {delay_hours:.0f} hours. "
                "Consider alternate route to reduce transit time, "
                "or split shipment across multiple departure windows."
            ),
        })

    # --- Supply risk recommendations ---

    if risk_level == "high":
        recs.append({
            "level": "warning",
            "message": (
                f"Supply risk is HIGH from {source_region}. "
                "Consider sourcing from alternate growing region "
                "or pre-ordering additional inventory to buffer potential shortages."
            ),
        })

    elif risk_level == "medium":
        recs.append({
            "level": "info",
            "message": (
                f"Moderate supply risk from {source_region}. "
                "Monitor regional crop availability reports."
            ),
        })

    # --- All clear ---

    if not recs:
        recs.append({
            "level": "info",
            "message": (
                "Conditions favorable. Proceed with standard shipment schedule."
            ),
        })

    return recs


if __name__ == "__main__":
    print("Testing recommendations engine...\n")

    # Test 1: Severe scenario
    print("=== Severe (72h delay, 55°F, Jan) ===")
    sr = get_supply_risk("romaine", 1, delay_hours=72, temp_f=55)
    print(f"  Supply risk: {sr['risk_level']} (index={sr['supply_index']})")
    print(f"  Region: {sr['source_region']}")
    print(f"  Details: {sr['details']}")
    recs = generate_recommendations(0.95, 72, 55, sr, 1000)
    for r in recs:
        print(f"  [{r['level'].upper():>8s}] {r['message']}")

    # Test 2: Mild scenario
    print("\n=== Mild (0.5h delay, 32°F, Jun) ===")
    sr = get_supply_risk("iceberg", 6, delay_hours=0.5, temp_f=32)
    print(f"  Supply risk: {sr['risk_level']} (index={sr['supply_index']})")
    print(f"  Region: {sr['source_region']}")
    recs = generate_recommendations(0.08, 0.5, 32, sr, 1000)
    for r in recs:
        print(f"  [{r['level'].upper():>8s}] {r['message']}")

    # Test 3: Medium scenario, winter
    print("\n=== Moderate (24h delay, 41°F, Dec) ===")
    sr = get_supply_risk("romaine", 12, delay_hours=24, temp_f=41)
    print(f"  Supply risk: {sr['risk_level']} (index={sr['supply_index']})")
    print(f"  Region: {sr['source_region']}")
    recs = generate_recommendations(0.25, 24, 41, sr, 1000)
    for r in recs:
        print(f"  [{r['level'].upper():>8s}] {r['message']}")
