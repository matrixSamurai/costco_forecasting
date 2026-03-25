#!/usr/bin/env python3
"""
Build supply_risk_lookup.csv by combining:
1. USDA lettuce shipment data (monthly supply volumes)
2. FEMA disaster data (historical disruption events)

Price data is already in the existing seasonal_price_indices.csv
(from model 2 lettuce data), so not duplicated here.

Output: supply_risk_lookup.csv
  Columns: lettuce_type, month, supply_index, source_region,
           disaster_frequency, risk_baseline, risk_level

This lookup is used by recommendations.py to assess supply risk.
"""

import csv
import os
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

USDA_CSV = os.path.join(SCRIPT_DIR, "usda_lettuce_shipments.csv")
FEMA_CSV = os.path.join(SCRIPT_DIR, "fema_disasters.csv")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "supply_risk_lookup.csv")

LETTUCE_TYPES = ["iceberg", "romaine", "butterhead", "leaf", "spring_mix"]


def load_usda():
    """Load USDA shipment data with real production volumes and supply index."""
    if not os.path.exists(USDA_CSV):
        print(f"  WARNING: {USDA_CSV} not found, using defaults")
        return None

    combined = {}
    regions = {}
    with open(USDA_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            month = int(row["month"])
            combined[month] = float(row["shipment_index"])
            regions[month] = row["source_region"]

    return combined, regions


def load_fema():
    """Load FEMA data and compute disaster frequency per month."""
    if not os.path.exists(FEMA_CSV):
        print(f"  WARNING: {FEMA_CSV} not found, using defaults")
        return None

    # Count disasters per state per month (across all years)
    monthly_counts = defaultdict(int)
    total_years = set()

    with open(FEMA_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            month = int(row["month"])
            year = int(row["year"])
            state = row["state"]
            total_years.add(year)
            # Weight by relevance to lettuce production
            if state == "CA" and month in range(4, 11):
                monthly_counts[month] += 1.0  # CA disaster during CA production
            elif state == "AZ" and month in [11, 12, 1, 2, 3]:
                monthly_counts[month] += 1.0  # AZ disaster during AZ production
            else:
                monthly_counts[month] += 0.3  # Off-season/other impact

    n_years = max(len(total_years), 1)
    # Normalize: annual frequency of disruption per month
    freq = {}
    for month in range(1, 13):
        freq[month] = round(monthly_counts.get(month, 0) / n_years, 3)

    return freq


def compute_risk_baseline(supply_index, disaster_freq):
    """
    Compute baseline risk level from supply index and disaster frequency.

    risk_baseline is 0-1 where higher = more risky.
    """
    # Invert supply (low supply = high risk)
    supply_risk = max(0, 1.0 - supply_index)
    # Disaster frequency adds risk (capped at 0.3 contribution)
    disaster_risk = min(0.3, disaster_freq * 0.5)
    return round(supply_risk * 0.7 + disaster_risk, 3)


def main():
    print("=" * 50)
    print("Building Supply Risk Lookup")
    print("=" * 50)

    # Load data
    print("\n[1] Loading data sources...")
    usda = load_usda()
    fema = load_fema()

    if usda:
        supply_indices, regions = usda
        print(f"    USDA: {len(supply_indices)} monthly supply indices")
    else:
        supply_indices = {m: 0.8 for m in range(1, 13)}
        regions = {m: "Unknown" for m in range(1, 13)}

    disaster_freq = fema or {m: 0.1 for m in range(1, 13)}
    if fema:
        print(f"    FEMA: disaster frequency for {len(disaster_freq)} months")

    # Build lookup
    print("\n[2] Computing supply risk lookup...")
    records = []

    for lt in LETTUCE_TYPES:
        for month in range(1, 13):
            si = supply_indices.get(month, 0.8)
            region = regions.get(month, "Unknown")
            df = disaster_freq.get(month, 0.1)
            risk = compute_risk_baseline(si, df)

            if risk > 0.4:
                risk_level = "high"
            elif risk > 0.25:
                risk_level = "medium"
            else:
                risk_level = "low"

            records.append({
                "lettuce_type": lt,
                "month": month,
                "supply_index": si,
                "source_region": region,
                "disaster_frequency": df,
                "risk_baseline": risk,
                "risk_level": risk_level,
            })

    # Write CSV
    fieldnames = [
        "lettuce_type", "month", "supply_index", "source_region",
        "disaster_frequency", "risk_baseline", "risk_level",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\n[3] Saved -> {os.path.basename(OUTPUT_CSV)} ({len(records)} rows)")

    # Summary
    print("\n    Monthly supply risk summary:")
    for month in range(1, 13):
        si = supply_indices.get(month, 0)
        region = regions.get(month, "?")
        df = disaster_freq.get(month, 0)
        risk = compute_risk_baseline(si, df)
        level = "HIGH" if risk > 0.4 else ("MED" if risk > 0.25 else "LOW")
        print(f"      Month {month:2d}: supply={si:.2f}  disaster={df:.2f}  "
              f"risk={risk:.3f} [{level:>4s}]  region={region}")

    print("\nDone.")


if __name__ == "__main__":
    main()
