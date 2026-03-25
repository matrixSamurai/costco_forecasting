#!/usr/bin/env python3
"""
Download FEMA disaster declarations for CA and AZ.

Source: OpenFEMA API (no API key required).
Endpoint: https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries

Filters: CA + AZ, relevant disaster types for agriculture.

Output: fema_disasters.csv
"""

import csv
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "fema_disasters.csv")

STATES = ["CA", "AZ"]
# Disaster types relevant to agriculture/supply chain
RELEVANT_TYPES = [
    "Flood", "Severe Storm", "Hurricane", "Fire", "Drought",
    "Severe Ice Storm", "Freezing", "Mud/Landslide",
]


def fetch_fema_data():
    """Fetch disaster declarations from OpenFEMA API."""
    import urllib.request

    all_records = []

    for state in STATES:
        # Filter: state + year >= 2015
        # URL-encode spaces as %20 for urllib compatibility
        url = (
            "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries?"
            f"$filter=state%20eq%20'{state}'%20and%20fyDeclared%20ge%202015"
            "&$select=disasterNumber,state,declarationDate,incidentType,"
            "declarationTitle,incidentBeginDate,incidentEndDate"
            "&$orderby=declarationDate%20desc"
            "&$top=1000"
        )
        print(f"  Fetching {state} disasters...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
                records = data.get("DisasterDeclarationsSummaries", [])
                print(f"    Got {len(records)} records for {state}")
                all_records.extend(records)
        except Exception as e:
            print(f"    Failed for {state}: {e}")

    return all_records


def get_hardcoded_data():
    """
    Hardcoded major FEMA disaster events affecting CA/AZ agriculture.

    Source: FEMA disaster declarations 2015-2025 for CA and AZ.
    Only includes events that could impact lettuce production/transport.
    """
    return [
        {"state": "CA", "year": 2023, "month": 1, "type": "Flood", "title": "Severe Winter Storms and Flooding"},
        {"state": "CA", "year": 2023, "month": 3, "type": "Flood", "title": "Severe Winter Storms and Flooding"},
        {"state": "CA", "year": 2023, "month": 8, "type": "Hurricane", "title": "Tropical Storm Hilary"},
        {"state": "CA", "year": 2022, "month": 9, "type": "Fire", "title": "Mosquito Fire"},
        {"state": "CA", "year": 2021, "month": 7, "type": "Fire", "title": "Dixie Fire"},
        {"state": "CA", "year": 2021, "month": 8, "type": "Fire", "title": "Caldor Fire"},
        {"state": "CA", "year": 2020, "month": 8, "type": "Fire", "title": "LNU Lightning Complex Fire"},
        {"state": "CA", "year": 2020, "month": 9, "type": "Fire", "title": "Creek Fire"},
        {"state": "CA", "year": 2019, "month": 10, "type": "Fire", "title": "Kincade Fire"},
        {"state": "AZ", "year": 2024, "month": 7, "type": "Severe Storm", "title": "Severe Storms and Flooding"},
        {"state": "AZ", "year": 2023, "month": 7, "type": "Severe Storm", "title": "Severe Storms"},
        {"state": "AZ", "year": 2021, "month": 7, "type": "Severe Storm", "title": "Severe Storms and Flooding"},
        {"state": "AZ", "year": 2020, "month": 7, "type": "Severe Storm", "title": "Severe Storms and Flooding"},
        {"state": "CA", "year": 2024, "month": 1, "type": "Severe Storm", "title": "Severe Winter Storms"},
        {"state": "CA", "year": 2024, "month": 2, "type": "Flood", "title": "Atmospheric River Flooding"},
        {"state": "CA", "year": 2025, "month": 1, "type": "Fire", "title": "Palisades and Eaton Fires"},
    ]


def process_api_records(records):
    """Extract relevant fields from FEMA API records."""
    from datetime import datetime

    processed = []
    for r in records:
        incident_type = r.get("incidentType", "")
        if not any(t.lower() in incident_type.lower() for t in RELEVANT_TYPES):
            continue

        date_str = r.get("incidentBeginDate") or r.get("declarationDate", "")
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            month = dt.month
            year = dt.year
        except (ValueError, AttributeError):
            continue

        processed.append({
            "state": r.get("state", ""),
            "year": year,
            "month": month,
            "type": incident_type,
            "title": r.get("declarationTitle", ""),
        })

    return processed


def main():
    print("=" * 50)
    print("FEMA Disaster Declarations (CA + AZ)")
    print("=" * 50)

    print("\n[1] Fetching from OpenFEMA API...")
    api_records = fetch_fema_data()

    if api_records:
        processed = process_api_records(api_records)
        print(f"\n[2] Filtered to {len(processed)} agriculture-relevant events")
    else:
        print("\n[2] API failed, using hardcoded data...")
        processed = get_hardcoded_data()
        print(f"    Using {len(processed)} hardcoded events")

    # Write CSV
    fieldnames = ["state", "year", "month", "type", "title"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(processed)

    print(f"\n[3] Saved -> {os.path.basename(OUTPUT_CSV)}")

    # Summary
    ca_count = sum(1 for r in processed if r["state"] == "CA")
    az_count = sum(1 for r in processed if r["state"] == "AZ")
    print(f"    CA: {ca_count} events, AZ: {az_count} events")
    print("Done.")


if __name__ == "__main__":
    main()
