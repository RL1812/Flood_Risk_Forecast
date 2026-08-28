#!/usr/bin/env python3
"""
Fetch Rainfall Data for Flood Events in Singapore from Data.gov.sg API.

This script:
1. Reads flood_events_extracted.json to extract the flood occurrence time,
   PLN_AREA_N value, and coordinates (longitude & latitude).
2. Performs a trial API request using the date of the first flood event to obtain
   water/rainfall sensor metadata from the 'stations' parameter.
3. Enriches the sensor location data with PLN_AREA_N from enriched_planning_areas.geojson
   and saves them into a standalone JSON file (rainfall_sensors.json).
4. For each flood event, fetches rainfall readings for the time window 2 hours
   before the flooding alert until 15 minutes before the flooding alert, keeping only the
   records for sensors that fall within that event's planning area.
5. Adds elevation statistics (elev_mean, elev_min, elev_std) of the planning area (PLN_AREA_N)
   into the generated flood_rainfall_records.json.
6. Saves the filtered rainfall records into a new JSON file (flood_rainfall_records.json).
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
from shapely.geometry import Point

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

    flood_candidates = ["flood_events_extracted.json", "flood_events.json"]
    geojson_candidates = [
        "enriched_planning_areas.geojson",
        "enriched_plan_areas.geojson",
        "MasterPlan2019PlanningAreaBoundaryNoSea.geojson"
    ]

    flood_path = find_file(flood_candidates, unique_search_dirs)
    geojson_path = find_file(geojson_candidates, unique_search_dirs)
    sensors_out = os.path.join(script_dir, "rainfall_sensors.json")
    rainfall_out = os.path.join(script_dir, "flood_rainfall_records.json")

    return flood_path, geojson_path, sensors_out, rainfall_out


def get_api_headers() -> Dict[str, str]:
    """Get HTTP headers with API key if available."""
    api_key = os.environ.get("DATA_GOV_SG_API_KEY")
    headers = {
        "Accept": "application/json",
        "User-Agent": "RainfallDataFetcher/1.0"
    }
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def load_planning_areas(geojson_path: str) -> gpd.GeoDataFrame:
    """Load planning areas GeoDataFrame and ensure WGS84 CRS."""
    if not os.path.exists(geojson_path):
        raise FileNotFoundError(f"Planning areas GeoJSON file not found: {geojson_path}")

    gdf = gpd.read_file(geojson_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def find_planning_area_for_point(gdf: gpd.GeoDataFrame, lon: Optional[float], lat: Optional[float]) -> Optional[str]:
    """Find the planning area name (PLN_AREA_N) containing the point."""
    if lon is None or lat is None:
        return None
    try:
        lon_f = float(lon)
        lat_f = float(lat)
    except (ValueError, TypeError):
        return None

    pt = Point(lon_f, lat_f)
    matches = gdf[gdf.geometry.contains(pt)]
    if len(matches) == 0:
        matches = gdf[gdf.geometry.intersects(pt)]

    if len(matches) > 0:
        return str(matches.iloc[0].get("PLN_AREA_N", ""))
    return None


def extract_planning_area_elev_map(gdf: gpd.GeoDataFrame) -> Dict[str, Dict[str, Optional[float]]]:
    """Extract mapping of PLN_AREA_N to elevation metrics (elev_mean, elev_min, elev_std)."""
    elev_map = {}
    for _, row in gdf.iterrows():
        pln_name = row.get("PLN_AREA_N")
        if pln_name:
            elev_mean = float(row["elev_mean"]) if row.get("elev_mean") is not None else None
            elev_min = float(row["elev_min"]) if row.get("elev_min") is not None else None
            elev_std = float(row["elev_std"]) if row.get("elev_std") is not None else None
            elev_map[str(pln_name).upper()] = {
                "elev_mean": elev_mean,
                "elev_min": elev_min,
                "elev_std": elev_std
            }
    return elev_map


def fetch_day_rainfall_readings(
    date_str: str,
    headers: Dict[str, str],
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> Dict[str, Any]:
    """
    Fetch all rainfall readings for a specific day from the API with pagination handling.
    Returns a dict containing 'stations' and 'readings'.
    """
    all_readings = []
    stations = []
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
        if not stations and "stations" in api_data:
            stations = api_data.get("stations", [])

        page_readings = api_data.get("readings", [])
        all_readings.extend(page_readings)

        pagination_token = api_data.get("paginationToken")
        if not pagination_token:
            break

        params["paginationToken"] = pagination_token
        page += 1
        time.sleep(0.05)

    return {"stations": stations, "readings": all_readings}


def process_rainfall_pipeline(
    flood_json_path: str,
    geojson_path: str,
    sensors_out_path: str,
    rainfall_out_path: str,
    start_hours_before: float = 2.0,
    end_minutes_before: float = 15.0
):
    """Run full pipeline steps 1 to 4."""
    load_env_file()
    headers = get_api_headers()

    print("=" * 70)
    print("STEP 1: Reading extracted flood records")
    print("=" * 70)
    if not os.path.exists(flood_json_path):
        raise FileNotFoundError(f"Flood events JSON not found: {flood_json_path}")

    with open(flood_json_path, "r", encoding="utf-8") as f:
        flood_events = json.load(f)

    extracted_flood_info = []
    for idx, ev in enumerate(flood_events):
        dt_str = ev.get("datetime")
        if not dt_str:
            continue
        
        # Parse ISO datetime
        dt_obj = datetime.datetime.fromisoformat(dt_str)
        loc = ev.get("location", {})
        pln_area = loc.get("PLN_AREA_N") if isinstance(loc, dict) else None
        lat = loc.get("latitude") if isinstance(loc, dict) else ev.get("latitude")
        lon = loc.get("longitude") if isinstance(loc, dict) else ev.get("longitude")

        flood_entry = {
            "index": idx + 1,
            "datetime_str": dt_str,
            "datetime_obj": dt_obj,
            "date_str": dt_obj.strftime("%Y-%m-%d"),
            "PLN_AREA_N": pln_area,
            "latitude": lat,
            "longitude": lon,
            "raw_event": ev
        }
        extracted_flood_info.append(flood_entry)
        print(f"  [{idx+1}] Time: {dt_str} | PLN_AREA_N: {pln_area} | Lat: {lat}, Lon: {lon}")

    if not extracted_flood_info:
        print("No valid flood events found to process.")
        return

    print("\n" + "=" * 70)
    print("STEP 2: Trial API request using date of the first flood event")
    print("=" * 70)
    first_event_date = extracted_flood_info[0]["date_str"]
    print(f"Executing trial request for first flood date: {first_event_date}")
    trial_result = fetch_day_rainfall_readings(first_event_date, headers=headers)
    raw_stations = trial_result.get("stations", [])
    print(f"Successfully retrieved {len(raw_stations)} sensors from API 'stations' parameter.")

    print("\n" + "=" * 70)
    print("STEP 3: Enriching sensors with PLN_AREA_N and saving standalone JSON")
    print("=" * 70)
    print(f"Loading planning area boundaries from: {geojson_path}")
    gdf = load_planning_areas(geojson_path)
    print(f"Loaded {len(gdf)} planning area shapes.")

    pln_elev_map = extract_planning_area_elev_map(gdf)

    enriched_sensors = []
    sensor_id_to_pln = {}
    pln_to_sensor_ids: Dict[str, Set[str]] = {}

    for stn in raw_stations:
        stn_copy = dict(stn)
        loc = stn_copy.get("location")
        if isinstance(loc, dict):
            stn_loc = dict(loc)
        else:
            stn_loc = {}

        stn_lat = stn_loc.get("latitude")
        stn_lon = stn_loc.get("longitude")
        stn_pln = find_planning_area_for_point(gdf, stn_lon, stn_lat)

        stn_loc["PLN_AREA_N"] = stn_pln
        stn_copy["location"] = stn_loc
        enriched_sensors.append(stn_copy)

        sensor_id = stn_copy.get("id")
        if sensor_id:
            sensor_id_to_pln[sensor_id] = stn_pln
            if stn_pln:
                pln_to_sensor_ids.setdefault(stn_pln.upper(), set()).add(sensor_id)

    # Save sensors JSON
    sensors_dir = os.path.dirname(os.path.abspath(sensors_out_path))
    if sensors_dir:
        os.makedirs(sensors_dir, exist_ok=True)

    with open(sensors_out_path, "w", encoding="utf-8") as f:
        json.dump(enriched_sensors, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(enriched_sensors)} enriched sensors to: {sensors_out_path}")

    print("\n" + "=" * 70)
    print(f"STEP 4: Fetching rainfall records ({start_hours_before} hrs before to {end_minutes_before} mins before flood alerts)")
    print("=" * 70)

    # Cache for day readings to avoid redundant requests for events on the same date
    day_readings_cache: Dict[str, List[Dict[str, Any]]] = {
        first_event_date: trial_result.get("readings", [])
    }

    # Determine all unique dates required
    required_dates = set()
    for ev in extracted_flood_info:
        t_flood = ev["datetime_obj"]
        t_start = t_flood - datetime.timedelta(hours=start_hours_before)
        t_end = t_flood - datetime.timedelta(minutes=end_minutes_before)
        curr_date = t_start.date()
        while curr_date <= t_end.date():
            required_dates.add(curr_date.strftime("%Y-%m-%d"))
            curr_date += datetime.timedelta(days=1)

    print(f"Unique dates requiring rainfall data: {sorted(required_dates)}")
    for d_str in sorted(required_dates):
        if d_str not in day_readings_cache:
            print(f"Fetching rainfall readings for date: {d_str}...")
            day_data = fetch_day_rainfall_readings(d_str, headers=headers)
            day_readings_cache[d_str] = day_data.get("readings", [])
            print(f"  Retrieved {len(day_readings_cache[d_str])} readings for {d_str}.")

    # Now filter readings for each flood event
    flood_rainfall_results = []

    for ev in extracted_flood_info:
        flood_dt = ev["datetime_obj"]
        t_start = flood_dt - datetime.timedelta(hours=start_hours_before)
        t_end = flood_dt - datetime.timedelta(minutes=end_minutes_before)
        pln_area = ev["PLN_AREA_N"]
        pln_key = pln_area.upper() if pln_area else ""

        # Elevation stats for this planning area
        elev_stats = pln_elev_map.get(pln_key, {"elev_mean": None, "elev_min": None, "elev_std": None})

        # Sensors located in this flood event's planning area
        target_sensors = [
            s for s in enriched_sensors
            if s.get("location", {}).get("PLN_AREA_N") and s["location"]["PLN_AREA_N"].upper() == pln_key
        ]
        target_sensor_ids = set(s["id"] for s in target_sensors if "id" in s)

        print(f"\nProcessing Flood Event #{ev['index']}:")
        print(f"  Datetime: {ev['datetime_str']}")
        print(f"  Planning Area: {pln_area} (elev_mean={elev_stats['elev_mean']}m, elev_min={elev_stats['elev_min']}m, elev_std={elev_stats['elev_std']}m)")
        print(f"  Time Window: {t_start.isoformat()} to {t_end.isoformat()} ({start_hours_before}h before to {end_minutes_before}min before)")
        print(f"  Matching sensors in {pln_area}: {sorted(target_sensor_ids)}")

        # Collect readings from relevant dates
        event_readings = []
        dates_to_check = set()
        curr_d = t_start.date()
        while curr_d <= t_end.date():
            dates_to_check.add(curr_d.strftime("%Y-%m-%d"))
            curr_d += datetime.timedelta(days=1)

        for d_str in sorted(dates_to_check):
            readings_list = day_readings_cache.get(d_str, [])
            for r in readings_list:
                r_ts_str = r.get("timestamp")
                if not r_ts_str:
                    continue
                try:
                    r_dt = datetime.datetime.fromisoformat(r_ts_str)
                except (ValueError, TypeError):
                    continue

                # Filter within time window: start <= timestamp <= end
                if t_start <= r_dt <= t_end:
                    # Filter sensor readings within the flood planning area
                    sensor_readings = [
                        item for item in r.get("data", [])
                        if item.get("stationId") in target_sensor_ids
                    ]
                    if sensor_readings:
                        event_readings.append({
                            "timestamp": r_ts_str,
                            "data": sensor_readings
                        })

        # Sort chronologically
        event_readings.sort(key=lambda x: x["timestamp"])

        event_summary = {
            "flood_event": {
                "datetime": ev["datetime_str"],
                "PLN_AREA_N": pln_area,
                "elev_mean": elev_stats["elev_mean"],
                "elev_min": elev_stats["elev_min"],
                "elev_std": elev_stats["elev_std"],
                "latitude": ev["latitude"],
                "longitude": ev["longitude"],
                "description": ev["raw_event"].get("description"),
                "severity": ev["raw_event"].get("severity"),
                "urgency": ev["raw_event"].get("urgency"),
                "instruction": ev["raw_event"].get("instruction")
            },
            "rainfall_window": {
                "window_start": t_start.isoformat(),
                "window_end": t_end.isoformat()
            },
            "pln_area_sensors": target_sensors,
            "readings_count": len(event_readings),
            "rainfall_readings": event_readings
        }
        flood_rainfall_results.append(event_summary)
        print(f"  Extracted {len(event_readings)} rainfall readings for sensors in {pln_area}.")

    # Save to output file
    rainfall_dir = os.path.dirname(os.path.abspath(rainfall_out_path))
    if rainfall_dir:
        os.makedirs(rainfall_dir, exist_ok=True)

    with open(rainfall_out_path, "w", encoding="utf-8") as f:
        json.dump(flood_rainfall_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"SUCCESS: Saved all extracted rainfall records to: {rainfall_out_path}")
    print("=" * 70)


def main():
    flood_def, geojson_def, sensors_def, rainfall_def = get_default_paths()

    parser = argparse.ArgumentParser(
        description="Extract rainfall data from data.gov.sg API for flood event time windows and planning areas."
    )
    parser.add_argument(
        "-f", "--flood-json",
        default=flood_def,
        help=f"Path to flood_events_extracted.json (default: {flood_def})"
    )
    parser.add_argument(
        "-g", "--geojson",
        default=geojson_def,
        help=f"Path to enriched_planning_areas.geojson (default: {geojson_def})"
    )
    parser.add_argument(
        "-s", "--sensors-out",
        default=sensors_def,
        help=f"Output path for sensors JSON (default: {sensors_def})"
    )
    parser.add_argument(
        "-o", "--rainfall-out",
        default=rainfall_def,
        help=f"Output path for rainfall records JSON (default: {rainfall_def})"
    )
    parser.add_argument(
        "--hours-before",
        type=float,
        default=2.0,
        help="Hours before the flood alert for window start (default: 2.0)"
    )
    parser.add_argument(
        "--mins-before-end",
        type=float,
        default=15.0,
        help="Minutes before the flood alert for window end (default: 15.0)"
    )

    args = parser.parse_args()

    if not args.flood_json or not os.path.exists(args.flood_json):
        print(f"Error: Flood JSON file not found: {args.flood_json}", file=sys.stderr)
        sys.exit(1)

    if not args.geojson or not os.path.exists(args.geojson):
        print(f"Error: GeoJSON file not found: {args.geojson}", file=sys.stderr)
        sys.exit(1)

    process_rainfall_pipeline(
        flood_json_path=args.flood_json,
        geojson_path=args.geojson,
        sensors_out_path=args.sensors_out,
        rainfall_out_path=args.rainfall_out,
        start_hours_before=args.hours_before,
        end_minutes_before=args.mins_before_end
    )


if __name__ == "__main__":
    main()
