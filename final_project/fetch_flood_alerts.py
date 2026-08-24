#!/usr/bin/env python3
"""
Fetch Flood Alert Records across Singapore from Data.gov.sg API.
Saves flood alert warning records into a structured JSON file.
Handles rate limits, API keys, and query caching.
"""

import os
import json
import time
import argparse
import datetime
import requests

API_URL = "https://api-open.data.gov.sg/v2/real-time/api/weather/flood-alerts"


def load_env_file(env_path=".env"):
    """
    Loads key-value pairs from a .env file into os.environ if not already set.
    """
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k not in os.environ:
                        os.environ[k] = v


def extract_alert_records(date_list, cache):
    """
    Parses cached API records and extracts structured flood alert & observation records.
    Includes all records even when there are no flood readings (observation only).
    """
    alerts = []
    for d_str in date_list:
        records = cache.get(d_str, [])
        for rec in records:
            dt = rec.get("datetime")
            updated_ts = rec.get("updatedTimestamp")
            item = rec.get("item", {})
            item_type = item.get("type", "observation")
            is_station = item.get("isStationData", False)
            status = item.get("status")
            identifier = item.get("identifier")
            readings = item.get("readings", [])

            if readings:
                for reading in readings:
                    area = reading.get("area", {})
                    area_desc = area.get("areaDesc", "")
                    circle = area.get("circle", [])

                    lat, lon, radius = None, None, None
                    if circle and len(circle) >= 2:
                        try:
                            lat = float(circle[0])
                            lon = float(circle[1])
                            radius = float(circle[2]) if len(circle) > 2 else None
                        except (ValueError, TypeError):
                            pass

                    alert_record = {
                        "datetime": dt,
                        "updatedTimestamp": updated_ts,
                        "type": item_type,
                        "has_flood_alert": True,
                        "identifier": identifier,
                        "isStationData": is_station,
                        "status": status,
                        "event": reading.get("event", "Flood"),
                        "headline": reading.get("headline"),
                        "description": reading.get("description"),
                        "severity": reading.get("severity"),
                        "urgency": reading.get("urgency"),
                        "certainty": reading.get("certainty"),
                        "responseType": reading.get("responseType"),
                        "instruction": reading.get("instruction"),
                        "senderName": reading.get("senderName"),
                        "expires": reading.get("expires"),
                        "eventCode": reading.get("eventCode"),
                        "areaDesc": area_desc,
                        "latitude": lat,
                        "longitude": lon,
                        "radius_km": radius,
                        "location": {
                            "areaDesc": area_desc,
                            "latitude": lat,
                            "longitude": lon,
                            "radius_km": radius
                        }
                    }
                    alerts.append(alert_record)
            else:
                obs_record = {
                    "datetime": dt,
                    "updatedTimestamp": updated_ts,
                    "type": item_type,
                    "has_flood_alert": False,
                    "identifier": identifier,
                    "isStationData": is_station,
                    "status": status,
                    "event": None,
                    "headline": None,
                    "description": "No flood alert",
                    "severity": None,
                    "urgency": None,
                    "certainty": None,
                    "responseType": None,
                    "instruction": None,
                    "senderName": None,
                    "expires": None,
                    "eventCode": None,
                    "areaDesc": None,
                    "latitude": None,
                    "longitude": None,
                    "radius_km": None,
                    "location": None
                }
                alerts.append(obs_record)

    # Sort alerts chronologically
    alerts.sort(key=lambda x: x.get("datetime") or "")
    return alerts


def fetch_flood_alerts(
    start_date="2025-05-01",
    end_date="2026-08-24",
    api_key=None,
    env_file=".env",
    cache_file="flood_alerts_cache.json",
    output_json="flood_alert_records.json",
    rate_delay=3.4,
    refresh_cache=False
):
    """
    Fetches flood alert records within the specified date range using the Data.gov.sg API.
    Manages rate limits, caching, and saves output to a JSON file.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(base_dir, env_file)
    cache_path = os.path.join(base_dir, cache_file)
    output_json_path = os.path.join(base_dir, output_json)

    # 1. Load API Key from argument, .env, or environment
    if not api_key:
        load_env_file(env_path)
        api_key = os.environ.get("DATA_GOV_SG_API_KEY")

    if api_key:
        print("[*] API key loaded successfully.", flush=True)
    else:
        print(
            "[!] Warning: No DATA_GOV_SG_API_KEY found. Proceeding without API key.", flush=True)

    print(f"[*] API Endpoint: {API_URL}", flush=True)

    # 2. Generate Date Range (start_date to end_date)
    start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    total_days = (end_dt - start_dt).days + 1
    date_list = [(start_dt + datetime.timedelta(days=i)).isoformat()
                 for i in range(total_days)]
    print(
        f"[*] Querying {total_days} dates ({start_date} to {end_date})...", flush=True)

    # 3. Load Cache if exists to support resuming
    cache = {}
    if os.path.exists(cache_path) and not refresh_cache:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            print(
                f"[*] Loaded {len(cache)} cached date queries from {cache_file}.", flush=True)
        except Exception:
            cache = {}

    session = requests.Session()
    if api_key:
        session.headers.update({"x-api-key": api_key})

    # 4. Fetch records sequentially with strict rate-limit management & caching
    processed_count = 0
    alerts_found_count = 0
    save_counter = 0

    for idx, d_str in enumerate(date_list, 1):
        processed_count += 1

        if d_str in cache and not refresh_cache:
            records = cache[d_str]
        else:
            records = []
            pagination_token = None
            date_failed = False

            while True:
                # First call proceeds with only the 'date' param; subsequent calls include 'paginationToken'
                params = {"date": d_str}
                if pagination_token:
                    params["paginationToken"] = pagination_token

                print(
                    f"[*] Requesting date: {d_str} | paginationToken: {pagination_token}", flush=True)

                page_data = None
                # Query API with automatic backoff on HTTP 429
                while True:
                    try:
                        resp = session.get(
                            API_URL, params=params, timeout=15)
                        if resp.status_code == 200:
                            res_json = resp.json()
                            page_data = res_json.get("data") if isinstance(res_json, dict) else None
                            break
                        elif resp.status_code == 429:
                            print(
                                f"[!] HTTP 429 (Rate Limit) on {d_str}. Sleeping for 10.5s...", flush=True)
                            time.sleep(10.5)
                        else:
                            print(
                                f"[!] Warning: HTTP {resp.status_code} for {d_str}", flush=True)
                            date_failed = True
                            break
                    except Exception as ex:
                        print(
                            f"[!] Connection exception on {d_str}: {ex}, retrying...", flush=True)
                        time.sleep(3)

                if date_failed or not isinstance(page_data, dict):
                    break

                page_records = page_data.get("records", [])
                if isinstance(page_records, list) and page_records:
                    records.extend(page_records)

                # paginationToken is under the data object
                pagination_token = page_data.get("paginationToken")

                # Pacing delay between queries to respect API rate limits
                time.sleep(rate_delay)

                # Stop when paginationToken is missing or empty
                if not pagination_token:
                    break

            cache[d_str] = records
            save_counter += 1

        # Check if this date has any flood alert readings
        has_reading = any(r.get("item", {}).get("readings") for r in records)
        if has_reading:
            alerts_found_count += 1
            print(f"  [+] Alert found on date {d_str}!", flush=True)

        if idx % 50 == 0 or idx == total_days:
            print(
                f"    Progress: {idx}/{total_days} dates processed ({(idx/total_days)*100:.1f}%) | Alerts found so far: {alerts_found_count}", flush=True)

        # Periodically persist cache
        if save_counter >= 25 or idx == total_days:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f)
            save_counter = 0

    # 5. Extract structured alert records
    alerts = extract_alert_records(date_list, cache)
    print(
        f"\n[+] Processing complete! Total flood alert records found: {len(alerts)}", flush=True)

    # 6. Save records to JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(alerts, f, indent=2, ensure_ascii=False)
    print(
        f"[+] Saved structured flood alert records to: {output_json_path}", flush=True)

    return alerts


def main():
    parser = argparse.ArgumentParser(
        description="Fetch flood alert records across Singapore from Data.gov.sg API.")
    parser.add_argument("--start", "--start-date", dest="start_date",
                        default="2025-05-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", "--end-date", dest="end_date",
                        default="2026-08-24", help="End date (YYYY-MM-DD)")
    parser.add_argument("--api-key", dest="api_key", default=None,
                        help="Data.gov.sg API key (optional, can also be read from .env)")
    parser.add_argument("--delay", type=float, default=3.4,
                        help="Pacing delay in seconds between queries (default: 3.4)")
    parser.add_argument(
        "--cache-file", default="flood_alerts_cache.json", help="Cache JSON filename")
    parser.add_argument(
        "--output-json", default="flood_alert_records.json", help="Output JSON filename")
    parser.add_argument(
        "--refresh", action="store_true", help="Force refresh cached queries from API")

    args = parser.parse_args()

    alerts = fetch_flood_alerts(
        start_date=args.start_date,
        end_date=args.end_date,
        api_key=args.api_key,
        cache_file=args.cache_file,
        output_json=args.output_json,
        rate_delay=args.delay,
        refresh_cache=args.refresh
    )

    active_alerts = [a for a in alerts if a.get("has_flood_alert")]

    print("\n" + "=" * 70)
    print(f"FLOOD ALERTS & OBSERVATION SUMMARY ({args.start_date} to {args.end_date})")
    print(f"Total Records Extracted: {len(alerts)} | Active Flood Alerts: {len(active_alerts)}")
    print("=" * 70)
    if not active_alerts:
        print("No active flood alert warnings recorded in the specified date range.")
    else:
        for idx, alert in enumerate(active_alerts, 1):
            print(f"\nActive Alert #{idx}:")
            print(f"  • Date & Time:   {alert.get('datetime')}")
            print(f"  • Event:         {alert.get('event')}")
            print(
                f"  • Headline:      {alert.get('headline')} [Severity: {alert.get('severity')}, Urgency: {alert.get('urgency')}]")
            print(f"  • Status:        {alert.get('status')}")
            print(f"  • Location Desc: {alert.get('areaDesc')}")
            print(
                f"  • Coordinates:   Latitude {alert.get('latitude')}, Longitude {alert.get('longitude')}")
            print(f"  • Description:   {alert.get('description')}")
            print(f"  • Instruction:   {alert.get('instruction')}")
            if alert.get("expires"):
                print(f"  • Expires:       {alert.get('expires')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
