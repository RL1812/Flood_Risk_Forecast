#!/usr/bin/env python3
"""
Fetch and Filter Flood Event Day Records across Singapore from Data.gov.sg API.
Searches for dates between start_date and end_date that have active flood events,
and saves all records on those specific flood event days into a structured JSON file.
Handles pagination, rate limits, caching, and resumes.
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


def date_has_flood_events(records):
    """
    Checks if a list of API records for a single date contains any flood event readings.
    """
    for rec in records:
        item = rec.get("item", {})
        readings = item.get("readings", [])
        if readings:
            return True
    return False


def extract_day_records(d_str, records):
    """
    Extracts all structured records (both active flood alerts and observation records)
    for a given date.
    """
    day_records = []
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
                    "date": d_str,
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
                day_records.append(alert_record)
        else:
            obs_record = {
                "date": d_str,
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
            day_records.append(obs_record)

    day_records.sort(key=lambda x: x.get("datetime") or "")
    return day_records


def fetch_flood_event_days(
    start_date="2025-05-01",
    end_date="2026-08-24",
    api_key=None,
    env_file=".env",
    cache_file="flood_alerts_cache.json",
    output_json="flood_event_days_records.json",
    rate_delay=3.4,
    refresh_cache=False
):
    """
    Searches for dates with flood events within the specified date range.
    Saves all records on those flood event dates to output_json.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(base_dir, env_file)
    cache_path = os.path.join(base_dir, cache_file)
    output_json_path = os.path.join(base_dir, output_json)

    # 1. Load API Key
    if not api_key:
        load_env_file(env_path)
        api_key = os.environ.get("DATA_GOV_SG_API_KEY")

    if api_key:
        print("[*] API key loaded successfully.", flush=True)
    else:
        print(
            "[!] Warning: No DATA_GOV_SG_API_KEY found. Proceeding without API key.", flush=True)

    print(f"[*] API Endpoint: {API_URL}", flush=True)

    # 2. Generate Date Range
    start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    total_days = (end_dt - start_dt).days + 1
    date_list = [(start_dt + datetime.timedelta(days=i)).isoformat()
                 for i in range(total_days)]
    print(
        f"[*] Searching across {total_days} dates ({start_date} to {end_date}) for flood event days...", flush=True)

    # 3. Load Cache
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

    # 4. Process dates and find flood event days
    flood_event_dates = []
    all_flood_day_records = []
    save_counter = 0

    for idx, d_str in enumerate(date_list, 1):
        if d_str in cache and not refresh_cache:
            records = cache[d_str]
        else:
            records = []
            pagination_token = None
            date_failed = False

            while True:
                params = {"date": d_str}
                if pagination_token:
                    params["paginationToken"] = pagination_token

                print(
                    f"[*] Requesting date: {d_str} | paginationToken: {pagination_token}", flush=True)

                page_data = None
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

                pagination_token = page_data.get("paginationToken")
                time.sleep(rate_delay)

                if not pagination_token:
                    break

            cache[d_str] = records
            save_counter += 1

        # Check if date contains any flood event readings
        if date_has_flood_events(records):
            flood_event_dates.append(d_str)
            day_records = extract_day_records(d_str, records)
            all_flood_day_records.extend(day_records)
            alert_count = sum(1 for r in day_records if r.get("has_flood_alert"))
            print(
                f"  [+] FLOOD EVENT DAY DETECTED: {d_str} ({alert_count} alert(s), {len(day_records)} total records)", flush=True)

        if idx % 50 == 0 or idx == total_days:
            print(
                f"    Progress: {idx}/{total_days} dates checked ({(idx/total_days)*100:.1f}%) | Flood event days found: {len(flood_event_dates)}", flush=True)

        # Periodically persist cache
        if save_counter >= 25 or idx == total_days:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f)
            save_counter = 0

    # 5. Save all records for the flood event days to output JSON
    all_flood_day_records.sort(key=lambda x: x.get("datetime") or "")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_flood_day_records, f, indent=2, ensure_ascii=False)

    print(
        f"\n[+] Processing complete!", flush=True)
    print(
        f"    • Total flood event days found: {len(flood_event_dates)} ({', '.join(flood_event_dates) if flood_event_dates else 'None'})", flush=True)
    print(
        f"    • Total records on flood event days: {len(all_flood_day_records)}", flush=True)
    print(
        f"    • Saved to: {output_json_path}", flush=True)

    return flood_event_dates, all_flood_day_records


def main():
    parser = argparse.ArgumentParser(
        description="Search for dates with flood events across Singapore and save all records on those days.")
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
        "--output-json", default="flood_event_days_records.json", help="Output JSON filename for records on flood days")
    parser.add_argument(
        "--refresh", action="store_true", help="Force refresh cached queries from API")

    args = parser.parse_args()

    flood_dates, records = fetch_flood_event_days(
        start_date=args.start_date,
        end_date=args.end_date,
        api_key=args.api_key,
        cache_file=args.cache_file,
        output_json=args.output_json,
        rate_delay=args.delay,
        refresh_cache=args.refresh
    )

    active_alerts = [r for r in records if r.get("has_flood_alert")]

    print("\n" + "=" * 70)
    print(f"FLOOD EVENT DAYS SUMMARY ({args.start_date} to {args.end_date})")
    print(f"Flood Event Dates Found: {len(flood_dates)}")
    print(f"Total Records on Flood Dates: {len(records)} | Active Flood Alerts: {len(active_alerts)}")
    print("=" * 70)
    if not flood_dates:
        print("No flood event dates found in the specified date range.")
    else:
        for d in flood_dates:
            d_records = [r for r in records if r.get("date") == d]
            d_alerts = [r for r in d_records if r.get("has_flood_alert")]
            print(f"\n📅 Date: {d} ({len(d_alerts)} alert(s), {len(d_records)} total records)")
            for idx, alert in enumerate(d_alerts, 1):
                print(f"  Alert #{idx}:")
                print(f"    • Time:          {alert.get('datetime')}")
                print(f"    • Headline:      {alert.get('headline')}")
                print(f"    • Severity:      {alert.get('severity')}")
                print(f"    • Location:      {alert.get('areaDesc')}")
                print(f"    • Coordinates:   Lat {alert.get('latitude')}, Lon {alert.get('longitude')}")
                print(f"    • Description:   {alert.get('description')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
