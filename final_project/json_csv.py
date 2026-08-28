#!/usr/bin/env python3
"""
Convert Flood and Non-Flood Rainfall JSON datasets into a tabular CSV dataset for machine learning.

This script processes `flood_rainfall_records.json` and `noflood_rainfall_records.json`,
filters out subsided records and the 2025-05-22 TOA PAYOH event, aggregates rainfall readings
(15-min sum, 30-min sum, 90-min sum, 5-min max), attaches elevation statistics and geospatial coordinates,
and labels the records (target 1 for flooded, 0 for non-flooded).
"""

import os
import sys
import json
import argparse
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np


def find_file(candidate_names: List[str], base_dirs: List[str]) -> Optional[str]:
    """Find the first existing file among candidate names across search directories."""
    for base_dir in base_dirs:
        for name in candidate_names:
            path = os.path.join(base_dir, name)
            if os.path.exists(path):
                return os.path.abspath(path)
    return None


def get_default_paths() -> Tuple[str, str, str]:
    """Determine default file paths based on script location and current working directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    search_dirs = [script_dir, cwd, os.path.join(cwd, "final_project"), os.path.join(script_dir, "final_project")]

    seen = set()
    unique_search_dirs = []
    for d in search_dirs:
        if d not in seen and os.path.isdir(d):
            seen.add(d)
            unique_search_dirs.append(d)

    flood_path = find_file(["flood_rainfall_records.json"], unique_search_dirs)
    noflood_path = find_file(["noflood_rainfall_records.json"], unique_search_dirs)

    if not flood_path:
        flood_path = os.path.join(script_dir, "flood_rainfall_records.json")
    if not noflood_path:
        noflood_path = os.path.join(script_dir, "noflood_rainfall_records.json")

    output_csv = os.path.join(script_dir, "training_dataset.csv")

    return flood_path, noflood_path, output_csv


def is_excluded_event(datetime_val: Optional[str], pln_area_val: Optional[str]) -> bool:
    """
    Check if an event matches the excluded 2025-05-22 at TOA PAYOH event.
    """
    dt_str = str(datetime_val) if datetime_val is not None else ""
    pln_str = str(pln_area_val).strip().upper() if pln_area_val is not None else ""
    return ("2025-05-22" in dt_str) and ("TOA PAYOH" in pln_str)


def process_dataset(
    flood_json_path: str,
    noflood_json_path: str,
    output_csv: str = "training_dataset.csv",
    min_readings: int = 18
) -> pd.DataFrame:
    """
    Process flood and non-flood JSON files into a consolidated Pandas DataFrame and CSV file.

    Args:
        flood_json_path: Path to flood_rainfall_records.json
        noflood_json_path: Path to noflood_rainfall_records.json
        output_csv: Path for the generated CSV output file
        min_readings: Minimum number of 5-minute readings required in the 105-minute window (default: 18)

    Returns:
        pd.DataFrame containing the processed dataset.
    """
    rows = []

    # 1. Process Flooded Events (Target = 1)
    if not os.path.exists(flood_json_path):
        raise FileNotFoundError(f"Flood JSON file not found: {flood_json_path}")

    with open(flood_json_path, "r", encoding="utf-8") as f:
        flood_data = json.load(f)

    excluded_flood_count = 0
    subsided_flood_count = 0
    flood_sensor_count = 0
    noflood_sensor_count = 0

    for event in flood_data:
        flood_event = event.get("flood_event", {})
        # Ignore subsided events if any remain
        if "subsided" in flood_event.get("description", "").lower():
            subsided_flood_count += 1
            continue

        dt = flood_event.get("datetime")
        pln_area = flood_event.get("PLN_AREA_N")

        # Exclude 2025-05-22 at TOA PAYOH event (zero recorded rainfall artifact)
        if is_excluded_event(dt, pln_area):
            excluded_flood_count += 1
            continue

        elev_mean = flood_event.get("elev_mean")
        elev_min = flood_event.get("elev_min")
        elev_std = flood_event.get("elev_std")

        # Map sensor locations
        sensor_locs = {
            s["id"]: (s["location"]["latitude"], s["location"]["longitude"])
            for s in event.get("pln_area_sensors", [])
            if "id" in s and "location" in s and "latitude" in s["location"] and "longitude" in s["location"]
        }

        # Accumulate time series per sensor
        sensor_rain_series = {s_id: [] for s_id in sensor_locs.keys()}
        for reading in event.get("rainfall_readings", []):
            for d in reading.get("data", []):
                sid = d.get("stationId")
                if sid in sensor_rain_series:
                    val = d.get("value")
                    sensor_rain_series[sid].append(0.0 if val is None else float(val))

        for s_id, rain_vals in sensor_rain_series.items():
            if len(rain_vals) >= min_readings:
                lat, lon = sensor_locs[s_id]
                max_rain = max(rain_vals) if rain_vals else 0.0
                # Temporal resistance check: if all past readings < 0.2mm, insufficient rainfall for flash flood
                target_label = 1 if max_rain >= 0.2 else 0

                rows.append({
                    "station_id": s_id,
                    "pln_area": pln_area,
                    "latitude": lat,
                    "longitude": lon,
                    "elev_mean": elev_mean,
                    "elev_min": elev_min,
                    "elev_std": elev_std,
                    "rain_sum_15m": sum(rain_vals[-3:]),
                    "rain_sum_30m": sum(rain_vals[-6:]),
                    "rain_sum_90m": sum(rain_vals[-18:]),
                    "rain_max_5m": max_rain,
                    "target": target_label
                })
                if target_label == 1:
                    flood_sensor_count += 1
                else:
                    noflood_sensor_count += 1

    # 2. Process Non-Flooded Events (Target = 0)
    if not os.path.exists(noflood_json_path):
        raise FileNotFoundError(f"No-flood JSON file not found: {noflood_json_path}")

    with open(noflood_json_path, "r", encoding="utf-8") as f:
        noflood_data = json.load(f)

    excluded_noflood_count = 0
    noflood_sensor_count = 0

    for event in noflood_data:
        ref = event.get("flood_event_reference", {})
        dt = ref.get("datetime")
        ref_pln = ref.get("flood_PLN_AREA_N") or ref.get("PLN_AREA_N")

        # Exclude records corresponding to 2025-05-22 at TOA PAYOH event
        if is_excluded_event(dt, ref_pln):
            excluded_noflood_count += 1
            continue

        sensor_metadata = {}
        sensor_rain_series = {}

        for reading in event.get("rainfall_readings", []):
            for d in reading.get("data", []):
                s_id = d.get("stationId")
                if not s_id:
                    continue
                if s_id not in sensor_metadata:
                    sensor_metadata[s_id] = {
                        "pln_area": d.get("PLN_AREA_N"),
                        "lat": d.get("latitude"),
                        "lon": d.get("longitude"),
                        "elev_mean": d.get("elev_mean"),
                        "elev_min": d.get("elev_min"),
                        "elev_std": d.get("elev_std")
                    }
                    sensor_rain_series[s_id] = []
                val = d.get("value")
                sensor_rain_series[s_id].append(0.0 if val is None else float(val))

        for s_id, rain_vals in sensor_rain_series.items():
            meta = sensor_metadata.get(s_id, {})
            if len(rain_vals) >= min_readings and meta.get("elev_mean") is not None:
                rows.append({
                    "station_id": s_id,
                    "pln_area": meta.get("pln_area"),
                    "latitude": meta.get("lat"),
                    "longitude": meta.get("lon"),
                    "elev_mean": meta.get("elev_mean"),
                    "elev_min": meta.get("elev_min"),
                    "elev_std": meta.get("elev_std"),
                    "rain_sum_15m": sum(rain_vals[-3:]),
                    "rain_sum_30m": sum(rain_vals[-6:]),
                    "rain_sum_90m": sum(rain_vals[-18:]),
                    "rain_max_5m": max(rain_vals) if rain_vals else 0.0,
                    "target": 0
                })
                noflood_sensor_count += 1

    df = pd.DataFrame(rows)

    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(output_csv))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_csv, index=False)
    print(f"Excluded 2025-05-22 TOA PAYOH events: {excluded_flood_count} flood event(s), {excluded_noflood_count} no-flood event group(s).")
    print(f"Total positive (flooded) sensor records:     {flood_sensor_count}")
    print(f"Total negative (non-flooded) sensor records: {noflood_sensor_count}")
    print(f"Generated {output_csv} with {len(df)} rows.")
    print("Target distribution:")
    print(df["target"].value_counts().to_string())
    return df


def main():
    default_flood, default_noflood, default_output = get_default_paths()

    parser = argparse.ArgumentParser(
        description="Transform flood_rainfall_records.json and noflood_rainfall_records.json into a combined CSV dataset (excluding 2025-05-22 TOA PAYOH)."
    )
    parser.add_argument(
        "-f", "--flood-json",
        type=str,
        default=default_flood,
        help=f"Path to flood_rainfall_records.json (default: {default_flood})"
    )
    parser.add_argument(
        "-n", "--noflood-json",
        type=str,
        default=default_noflood,
        help=f"Path to noflood_rainfall_records.json (default: {default_noflood})"
    )
    parser.add_argument(
        "-o", "--output-csv",
        type=str,
        default=default_output,
        help=f"Path for output CSV file (default: {default_output})"
    )
    parser.add_argument(
        "--min-readings",
        type=int,
        default=18,
        help="Minimum number of 5-minute sensor readings required in the 105-minute window (default: 18)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("TRANSFORMING RAINFALL JSON TO CSV DATASET")
    print("=" * 70)
    print(f"Flood JSON:    {args.flood_json}")
    print(f"No-flood JSON: {args.noflood_json}")
    print(f"Output CSV:    {args.output_csv}\n")

    process_dataset(
        flood_json_path=args.flood_json,
        noflood_json_path=args.noflood_json,
        output_csv=args.output_csv,
        min_readings=args.min_readings
    )


if __name__ == "__main__":
    main()
