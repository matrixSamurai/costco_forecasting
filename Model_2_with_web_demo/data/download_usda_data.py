#!/usr/bin/env python3
"""
Download USDA NASS lettuce data using Quick Stats API.

Downloads:
1. Annual production (CWT) by state (CA, AZ) and lettuce type (head, leaf, romaine)
2. Monthly price received ($/CWT) - national

API key from: https://quickstats.nass.usda.gov/api

Output:
  - usda_lettuce_production.csv   (annual production by state & type)
  - usda_lettuce_prices_monthly.csv (monthly farm-gate prices)
  - usda_lettuce_shipments.csv    (monthly supply index, computed from above)
"""

import csv
import json
import os
import sys
import urllib.request
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
API_KEY = os.environ.get("USDA_API_KEY", "AF1A6892-360D-371D-B505-23812A59F67C")

OUT_PRODUCTION = os.path.join(SCRIPT_DIR, "usda_lettuce_production.csv")
OUT_PRICES = os.path.join(SCRIPT_DIR, "usda_lettuce_prices_monthly.csv")
OUT_SHIPMENTS = os.path.join(SCRIPT_DIR, "usda_lettuce_shipments.csv")

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# Mapping from USDA short_desc to our lettuce types
TYPE_MAP = {
    "LETTUCE, HEAD": "iceberg",
    "LETTUCE, LEAF": "leaf",
    "LETTUCE, ROMAINE": "romaine",
    "LETTUCE, ORGANIC": "organic",
}


def api_get(params):
    """Call USDA NASS Quick Stats API."""
    base = "https://quickstats.nass.usda.gov/api/api_GET/"
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{base}?{query}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
        return data.get("data", [])


def parse_value(val_str):
    """Parse USDA value string (may contain commas, (D), (Z))."""
    val = val_str.strip().replace(",", "")
    if val in ("", "(D)", "(Z)", "(NA)"):
        return None
    try:
        return float(val)
    except ValueError:
        return None


def download_production():
    """Download annual production data (CWT) for CA and AZ."""
    print("\n[1] Downloading annual production data...")
    records = []

    for state in ["CA", "AZ"]:
        print(f"    Fetching {state}...")
        data = api_get({
            "key": API_KEY,
            "commodity_desc": "LETTUCE",
            "state_alpha": state,
            "statisticcat_desc": "PRODUCTION",
            "unit_desc": "CWT",
            "freq_desc": "ANNUAL",
            "format": "JSON",
            "year__GE": "2015",
        })
        print(f"    Got {len(data)} raw records for {state}")

        for r in data:
            val = parse_value(r.get("Value", ""))
            if val is None:
                continue
            desc = r.get("short_desc", "")
            # Only total production, not subtypes
            if any(x in desc for x in ["UTILIZED", "NOT SOLD", "PROCESSING",
                                        "UNDER PROTECTION", "FRESH MARKET"]):
                continue

            # Map to our lettuce type
            lettuce_type = None
            for pattern, lt in TYPE_MAP.items():
                if desc.startswith(pattern):
                    lettuce_type = lt
                    break
            if lettuce_type is None:
                continue

            records.append({
                "state": state,
                "year": int(r["year"]),
                "lettuce_type": lettuce_type,
                "production_cwt": int(val),
                "short_desc": desc,
            })

    # Write CSV
    records.sort(key=lambda x: (x["state"], x["lettuce_type"], x["year"]))
    with open(OUT_PRODUCTION, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "state", "year", "lettuce_type", "production_cwt", "short_desc"
        ])
        writer.writeheader()
        writer.writerows(records)

    print(f"    Saved {len(records)} records -> {os.path.basename(OUT_PRODUCTION)}")
    return records


def download_monthly_prices():
    """Download monthly price received data ($/CWT)."""
    print("\n[2] Downloading monthly farm-gate prices...")
    data = api_get({
        "key": API_KEY,
        "commodity_desc": "LETTUCE",
        "statisticcat_desc": "PRICE%20RECEIVED",
        "freq_desc": "MONTHLY",
        "format": "JSON",
        "year__GE": "2019",
    })
    print(f"    Got {len(data)} raw records")

    records = []
    for r in data:
        val = parse_value(r.get("Value", ""))
        if val is None:
            continue
        period = r.get("reference_period_desc", "")
        month = MONTH_MAP.get(period)
        if month is None:
            continue

        desc = r.get("short_desc", "")
        # Only use total (not fresh market subset)
        if "FRESH MARKET" in desc:
            continue

        lettuce_type = None
        for pattern, lt in TYPE_MAP.items():
            if desc.startswith(pattern):
                lettuce_type = lt
                break
        if lettuce_type is None:
            continue

        records.append({
            "year": int(r["year"]),
            "month": month,
            "lettuce_type": lettuce_type,
            "price_per_cwt": round(val, 2),
            "price_per_lb": round(val / 100, 4),
        })

    records.sort(key=lambda x: (x["lettuce_type"], x["year"], x["month"]))
    with open(OUT_PRICES, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "year", "month", "lettuce_type", "price_per_cwt", "price_per_lb"
        ])
        writer.writeheader()
        writer.writerows(records)

    print(f"    Saved {len(records)} records -> {os.path.basename(OUT_PRICES)}")

    # Print monthly averages
    monthly_avg = defaultdict(list)
    for r in records:
        monthly_avg[(r["lettuce_type"], r["month"])].append(r["price_per_cwt"])

    print("\n    Monthly average farm-gate prices ($/CWT):")
    for lt in ["iceberg", "leaf", "romaine"]:
        prices = []
        for m in range(1, 13):
            vals = monthly_avg.get((lt, m), [])
            avg = sum(vals) / len(vals) if vals else 0
            prices.append(avg)
        print(f"      {lt:>10s}: " +
              " ".join(f"${p:5.1f}" for p in prices))

    return records


def compute_supply_index(production_records):
    """
    Compute monthly supply index from annual production data.

    Uses CA/AZ production ratio + known seasonal patterns:
    - CA (Salinas): Apr-Oct peak production
    - AZ (Yuma): Nov-Mar peak production
    """
    print("\n[3] Computing monthly supply index from production data...")

    # Average recent production by state
    ca_total = defaultdict(list)
    az_total = defaultdict(list)
    for r in production_records:
        if r["year"] >= 2020:
            if r["state"] == "CA":
                ca_total[r["lettuce_type"]].append(r["production_cwt"])
            else:
                az_total[r["lettuce_type"]].append(r["production_cwt"])

    ca_avg = sum(sum(v) / len(v) for v in ca_total.values()) if ca_total else 80000000
    az_avg = sum(sum(v) / len(v) for v in az_total.values()) if az_total else 20000000
    total_avg = ca_avg + az_avg
    ca_share = ca_avg / total_avg
    az_share = az_avg / total_avg

    print(f"    CA avg production: {ca_avg:,.0f} CWT ({ca_share:.0%} of total)")
    print(f"    AZ avg production: {az_avg:,.0f} CWT ({az_share:.0%} of total)")

    # Monthly production distribution (based on USDA shipping point reports)
    # CA produces Apr-Oct, AZ produces Nov-Mar
    ca_monthly_pct = {
        1: 0.02, 2: 0.01, 3: 0.04, 4: 0.08, 5: 0.14, 6: 0.16,
        7: 0.15, 8: 0.14, 9: 0.13, 10: 0.08, 11: 0.03, 12: 0.02,
    }
    az_monthly_pct = {
        1: 0.20, 2: 0.18, 3: 0.15, 4: 0.05, 5: 0.01, 6: 0.00,
        7: 0.00, 8: 0.00, 9: 0.01, 10: 0.05, 11: 0.15, 12: 0.20,
    }

    records = []
    monthly_supply = {}
    for month in range(1, 13):
        ca_prod = ca_avg * ca_monthly_pct[month]
        az_prod = az_avg * az_monthly_pct[month]
        total_prod = ca_prod + az_prod
        monthly_supply[month] = total_prod

        if ca_prod > az_prod:
            region = "Salinas Valley, CA"
        else:
            region = "Yuma, AZ"

        records.append({
            "state": "US",
            "month": month,
            "ca_production_cwt": int(ca_prod),
            "az_production_cwt": int(az_prod),
            "total_production_cwt": int(total_prod),
            "source_region": region,
        })

    # Normalize to 0-1 index
    max_prod = max(monthly_supply.values())
    for r in records:
        r["shipment_index"] = round(r["total_production_cwt"] / max_prod, 3)

    with open(OUT_SHIPMENTS, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "state", "month", "ca_production_cwt", "az_production_cwt",
            "total_production_cwt", "shipment_index", "source_region",
        ])
        writer.writeheader()
        writer.writerows(records)

    print(f"    Saved -> {os.path.basename(OUT_SHIPMENTS)}")
    print("\n    Monthly supply breakdown:")
    for r in records:
        print(f"      Month {r['month']:2d}: CA={r['ca_production_cwt']:>10,} "
              f"AZ={r['az_production_cwt']:>10,} "
              f"Total={r['total_production_cwt']:>10,} "
              f"Index={r['shipment_index']:.3f} "
              f"[{r['source_region']}]")

    return records


def main():
    print("=" * 60)
    print("USDA NASS Lettuce Data Download")
    print("=" * 60)
    print(f"API Key: {API_KEY[:8]}...")

    production = download_production()
    prices = download_monthly_prices()
    shipments = compute_supply_index(production)

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Production: {len(production)} records -> {os.path.basename(OUT_PRODUCTION)}")
    print(f"  Prices:     {len(prices)} records -> {os.path.basename(OUT_PRICES)}")
    print(f"  Shipments:  {len(shipments)} records -> {os.path.basename(OUT_SHIPMENTS)}")
    print("Done.")


if __name__ == "__main__":
    main()
