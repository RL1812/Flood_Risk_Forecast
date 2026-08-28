#!/usr/bin/env python3
"""
Extract Non-Flood Planning Area Rainfall Records.

This script:
1. Reads flood_rainfall_records.json, rainfall_sensors.json, and enriched_planning_areas.geojson.
2. For each flood event in flood_rainfall_records.json, identifies all sensors NOT
   located within that flood event's planning area (PLN_AREA_N).
3. Fetches / retrieves rainfall data for the exact timestamps of the flood event's window.
4. Enriches each non-flood sensor reading with its respective coordinates (latitude, longitude),
   PLN_AREA_N, and elevation metrics (elev_mean, elev_min, elev_std).
5. Saves all extracted non-flood rainfall records into noflood_rainfall_records.json.
"""

import os
import sys
import json
import time
import argparse
import datetime
from typing import Optional, Dict, Any, List, Set
import requests
import geopandas as gpd

RAINFALL_API_URL = "https://api-open.data.gov.sg/v2/real-time/api/rainfall"


def load_env_file(env_path: Optional[str] = None):
    """Load key-value pairs from .env into os.environ if not already set."""
    search_paths = [
        env_path,
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        os.path.join(os.getcwd(), "final_project", ".env"),
    ]
    for path in search_paths:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k not in os.environ:
                            os.environ[k] = v
            break


def find_file(candidate_names: List[str], base_dirs: List[str]) -> Optional[str]:
    """Find the first existing file among candidate names across search directories."""
    for base_dir in base_dirs:
        for name in candidate_names:
            path = os.path.join(base_dir, name)
            if os.path.exists(path):
                return os.path.abspath(path)
    return None


def get_default_paths():
    """Determine default file paths based on script location and current working directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    search_dirs = [cwd, script_dir, os.path.join(cwd, "final_project"), os.path.join(script_dir, "final_project")]

    seen = set()
    unique_search_dirs = []
    for d in search_dirs:
        if d not in seen and os.path.isdir(d):
            seen.add(d)
            unique_search_dirs.append(d)

    flood_rainfall_candidates = ["flood_rainfall_records.json"]
    sensors_candidates = ["rainfall_sensors.json"]
    geojson_candidates = [
        "enriched_planning_areas.geojson",
        "enriched_plan_areas.geojson",
        "MasterPlan2019PlanningAreaBoundaryNoSea.geojson"
    ]

    flood_rf_path = find_file(flood_rainfall_candidates, unique_search_dirs)
    sensors_path = find_file(sensors_candidates, unique_search_dirs)
    geojson_path = find_file(geojson_candidates, unique_search_dirs)
    noflood_out = os.path.join(script_dir, "noflood_rainfall_records.json")

    return flood_rf_path, sensors_path, geojson_path, noflood_out


def get_api_headers() -> Dict[str, str]:
    """Get HTTP headers with API key if available."""
    api_key = os.environ.get("DATA_GOV_SG_API_KEY")
    headers = {
        "Accept": "application/json",
        "User-Agent": "NoFloodRainfallFetcher/1.0"
    }
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def load_planning_areas_elevation(geojson_path: str) -> Dict[str, Dict[str, Optional[float]]]:
    """Load planning areas and extract elevation statistics per PLN_AREA_N."""
    if not os.path.exists(geojson_path):
        return {}
    gdf = gpd.read_file(geojson_path)
    elev_map = {}
    for _, row in gdf.iterrows():
        name = row.get("PLN_AREA_N")
        if name:
            elev_map[str(name).upper()] = {
                "elev_mean": float(row["elev_mean"]) if row.get("elev_mean") is not None else None,
                "elev_min": float(row["elev_min"]) if row.get("elev_min") is not None else None,
                "elev_std": float(row["elev_std"]) if row.get("elev_std") is not None else None,
            }
    return elev_map


def fetch_day_rainfall_readings(
    date_str: str,
    headers: Dict[str, str],
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> List[Dict[str, Any]]:
    """Fetch all rainfall readings for a specific day from the API with pagination handling."""
    all_readings = []
    params = {"date": date_str}
    page = 1

    while True:
        success = False
        data = None

        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(RAINFALL_API_URL, headers=headers, params=params, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("code") == 0:
                        success = True
                        break
                    else:
                        print(f"  [Warn] API returned code {data.get('code')}: {data.get('errorMsg')}")
                elif resp.status_code == 429:
                    print(f"  [Warn] Rate limited (429). Retrying in {retry_delay * attempt}s...")
                    time.sleep(retry_delay * attempt)
                else:
                    print(f"  [Warn] HTTP {resp.status_code} for date {date_str}, attempt {attempt}/{max_retries}")
                    time.sleep(retry_delay)
            except requests.RequestException as e:
                print(f"  [Warn] Request failed: {e}, attempt {attempt}/{max_retries}")
                time.sleep(retry_delay)

        if not success or not data or "data" not in data:
            print(f"  [Error] Failed to fetch data for date {date_str} (page {page}).")
            break

        api_data = data.get("data", {})
        page_readings = api_data.get("readings", [])
        all_readings.extend(page_readings)

        pagination_token = api_data.get("paginationToken")
        if not pagination_token:
            break

        params["paginationToken"] = pagination_token
        page += 1
        time.sleep(0.05)

    return all_readings


def process_noflood_rainfall(
    flood_rainfall_path: str,
    sensors_path: str,
    geojson_path: str,
    output_path: str
):
    """Extract non-flood sensor rainfall data under same timestamps as flood events."""
    load_env_file()
    headers = get_api_headers()

    print("=" * 70)
    print("STEP 1: Loading input datasets")
    print("=" * 70)
    print(f"Loading flood rainfall records from: {flood_rainfall_path}")
    with open(flood_rainfall_path, "r", encoding="utf-8") as f:
        flood_records = json.load(f)
    print(f"Loaded {len(flood_records)} flood events.")

    print(f"Loading sensor metadata from: {sensors_path}")
    with open(sensors_path, "r", encoding="utf-8") as f:
        sensors_list = json.load(f)
    print(f"Loaded {len(sensors_list)} sensors.")

    # Sensor lookup dictionary
    sensor_info_map = {}
    for s in sensors_list:
        sid = s.get("id")
        loc = s.get("location") if isinstance(s.get("location"), dict) else {}
        pln = loc.get("PLN_AREA_N")
        lat = loc.get("latitude")
        lon = loc.get("longitude")
        sensor_info_map[sid] = {
            "id": sid,
            "deviceId": s.get("deviceId"),
            "name": s.get("name"),
            "latitude": lat,
            "longitude": lon,
            "location": loc,
            "PLN_AREA_N": pln
        }

    # Elevation metrics
    print(f"Loading planning area elevations from: {geojson_path}")
    pln_elev_map = load_planning_areas_elevation(geojson_path)
    print(f"Loaded elevation metrics for {len(pln_elev_map)} planning areas.")

    # Determine unique dates needed
    unique_dates = set()
    for ev in flood_records:
        readings = ev.get("rainfall_readings", [])
        for r in readings:
            ts = r.get("timestamp")
            if ts:
                d_str = ts.split("T")[0]
                unique_dates.add(d_str)

    print("\n" + "=" * 70)
    print("STEP 2: Fetching / Caching daily rainfall readings from API")
    print("=" * 70)
    print(f"Required dates: {sorted(unique_dates)}")
    day_readings_cache = {}
    for d_str in sorted(unique_dates):
        print(f"Fetching readings for date: {d_str}...")
        day_readings = fetch_day_rainfall_readings(d_str, headers=headers)
        # Index readings by timestamp for O(1) lookup
        ts_map = {}
        for r in day_readings:
            ts = r.get("timestamp")
            if ts:
                ts_map[ts] = r.get("data", [])
        day_readings_cache[d_str] = ts_map
        print(f"  Indexed {len(ts_map)} timestamps for {d_str}.")

    print("\n" + "=" * 70)
    print("STEP 3: Extracting non-flood sensor readings for matching timestamps")
    print("=" * 70)

    noflood_results = []
    for idx, ev in enumerate(flood_records):
        flood_fe = ev.get("flood_event", {})
        flood_pln = flood_fe.get("PLN_AREA_N")
        flood_pln_key = flood_pln.upper() if flood_pln else ""
        flood_dt = flood_fe.get("datetime")

        # Identify non-flood sensors
        non_flood_sensors = [
            s for s in sensors_list
            if not (s.get("location", {}).get("PLN_AREA_N") and s["location"]["PLN_AREA_N"].upper() == flood_pln_key)
        ]
        non_flood_sensor_ids = set(s["id"] for s in non_flood_sensors if "id" in s)

        # Unique non-flood planning areas
        non_flood_plns = sorted(list(set(
            s["location"]["PLN_AREA_N"] for s in non_flood_sensors
            if s.get("location", {}).get("PLN_AREA_N")
        )))

        print(f"\nProcessing Event #{idx+1}: Flood in {flood_pln} at {flood_dt}")
        print(f"  Non-flood sensors count: {len(non_flood_sensors)} across {len(non_flood_plns)} planning areas.")

        target_timestamps = [r["timestamp"] for r in ev.get("rainfall_readings", []) if "timestamp" in r]
        print(f"  Extracting across {len(target_timestamps)} timestamps...")

        event_noflood_readings = []
        for ts in target_timestamps:
            d_str = ts.split("T")[0]
            ts_data_list = day_readings_cache.get(d_str, {}).get(ts, [])

            # Filter for non-flood sensors and enrich
            filtered_sensor_data = []
            for item in ts_data_list:
                sid = item.get("stationId")
                if sid in non_flood_sensor_ids:
                    s_info = sensor_info_map.get(sid, {})
                    s_pln = s_info.get("PLN_AREA_N")
                    s_pln_key = s_pln.upper() if s_pln else ""
                    elev_info = pln_elev_map.get(s_pln_key, {"elev_mean": None, "elev_min": None, "elev_std": None})

                    sensor_entry = {
                        "stationId": sid,
                        "value": item.get("value"),
                        "stationName": s_info.get("name"),
                        "latitude": s_info.get("latitude"),
                        "longitude": s_info.get("longitude"),
                        "PLN_AREA_N": s_pln,
                        "elev_mean": elev_info["elev_mean"],
                        "elev_min": elev_info["elev_min"],
                        "elev_std": elev_info["elev_std"]
                    }
                    filtered_sensor_data.append(sensor_entry)

            event_noflood_readings.append({
                "timestamp": ts,
                "data": filtered_sensor_data
            })

        event_entry = {
            "flood_event_reference": {
                "datetime": flood_dt,
                "flood_PLN_AREA_N": flood_pln,
                "latitude": flood_fe.get("latitude"),
                "longitude": flood_fe.get("longitude")
            },
            "rainfall_window": ev.get("rainfall_window", {}),
            "non_flood_sensors_count": len(non_flood_sensors),
            "non_flood_planning_areas_count": len(non_flood_plns),
            "readings_count": len(event_noflood_readings),
            "rainfall_readings": event_noflood_readings
        }
        noflood_results.append(event_entry)

    # Save to output file
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(noflood_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"SUCCESS: Saved non-flood rainfall records to: {output_path}")
    print("=" * 70)


def main():
    flood_rf_def, sensors_def, geojson_def, noflood_def = get_default_paths()

    parser = argparse.ArgumentParser(
        description="Extract rainfall records for non-flood sensors under the same timestamps as flood events."
    )
    parser.add_argument(
        "-f", "--flood-rainfall",
        default=flood_rf_def,
        help=f"Path to flood_rainfall_records.json (default: {flood_rf_def})"
    )
    parser.add_argument(
        "-s", "--sensors",
        default=sensors_def,
        help=f"Path to rainfall_sensors.json (default: {sensors_def})"
    )
    parser.add_argument(
        "-g", "--geojson",
        default=geojson_def,
        help=f"Path to enriched_planning_areas.geojson (default: {geojson_def})"
    )
    parser.add_argument(
        "-o", "--output",
        default=noflood_def,
        help=f"Output path for non-flood rainfall records JSON (default: {noflood_def})"
    )

    args = parser.parse_args()

    if not args.flood_rainfall or not os.path.exists(args.flood_rainfall):
        print(f"Error: Flood rainfall JSON file not found: {args.flood_rainfall}", file=sys.stderr)
        sys.exit(1)

    if not args.sensors or not os.path.exists(args.sensors):
        print(f"Error: Sensors JSON file not found: {args.sensors}", file=sys.stderr)
        sys.exit(1)

    process_noflood_rainfall(
        flood_rainfall_path=args.flood_rainfall,
        sensors_path=args.sensors,
        geojson_path=args.geojson,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
