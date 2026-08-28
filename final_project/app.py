#!/usr/bin/env python3
"""
Singapore Flood Risk & Elevation Explorer - Flask Application.

Provides two primary functions:
1. Flood Vulnerability Map in Singapore (Digital Elevation Model & Topographic sensitivity).
2. Flood Risk Forecast in 15 Mins (Real-time rainfall ingestion from Data.gov.sg for the
   last 1h 45m + XGBoost Machine Learning Ensemble inference + Tiered Alerting).
"""

import os
import sys
import json
import time
import datetime
import urllib.request
import requests
from typing import Dict, Any, Optional, List, Tuple
from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
import joblib

# Add current directory to path so that custom classes can unpickle cleanly
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from xgboost_training import BalancedXGBEnsemble, FocalLossObjective
    setattr(sys.modules["__main__"], "BalancedXGBEnsemble", BalancedXGBEnsemble)
    setattr(sys.modules["__main__"], "FocalLossObjective", FocalLossObjective)
except ImportError:
    pass

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Global dataset & model caches
GEOJSON_DATA: Optional[Dict[str, Any]] = None
RAINFALL_SENSORS: Optional[List[Dict[str, Any]]] = None
MODEL_ENSEMBLE = None
MODEL_CALIBRATOR = None
MODEL_CONFIG: Optional[Dict[str, Any]] = None
PLN_AREA_LOOKUP: Dict[str, Dict[str, Any]] = {}

# In-memory cache for real-time Data.gov.sg rainfall readings (60-second TTL)
RAINFALL_CACHE = {
    "timestamp": 0,
    "date_str": "",
    "readings": []
}


def load_api_key() -> str:
    """Read DATA_GOV_SG_API_KEY from .env file or environment."""
    env_key = os.getenv("DATA_GOV_SG_API_KEY", "")
    if env_key:
        return env_key
    env_path = os.path.join(script_dir, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("DATA_GOV_SG_API_KEY="):
                    return line.strip().split("=", 1)[1].strip("\"' ")
    return ""


def init_resources():
    """Load GeoJSON, Sensors, XGBoost Model, and Config into memory."""
    global GEOJSON_DATA, RAINFALL_SENSORS, MODEL_ENSEMBLE, MODEL_CALIBRATOR, MODEL_CONFIG, PLN_AREA_LOOKUP

    # 1. Load GeoJSON
    geojson_path = os.path.join(script_dir, "enriched_planning_areas.geojson")
    with open(geojson_path, "r", encoding="utf-8") as f:
        GEOJSON_DATA = json.load(f)

    # 2. Load Sensors
    sensors_path = os.path.join(script_dir, "rainfall_sensors.json")
    with open(sensors_path, "r", encoding="utf-8") as f:
        RAINFALL_SENSORS = json.load(f)

    # 3. Build Planning Area Lookup (Elevation & Centroid)
    for feat in GEOJSON_DATA.get("features", []):
        props = feat.get("properties", {})
        name = props.get("PLN_AREA_N", "").upper()
        if not name:
            continue

        coords = feat.get("geometry", {}).get("coordinates", [])

        def flatten_coords(c):
            pts = []
            if isinstance(c, (list, tuple)) and len(c) >= 2 and isinstance(c[0], (int, float)):
                return [c]
            for sub in c:
                pts.extend(flatten_coords(sub))
            return pts

        pts = flatten_coords(coords)
        if pts:
            avg_lon = sum(p[0] for p in pts) / len(pts)
            avg_lat = sum(p[1] for p in pts) / len(pts)
        else:
            avg_lon, avg_lat = 103.8198, 1.3521

        PLN_AREA_LOOKUP[name] = {
            "name": name,
            "region": props.get("REGION_N", "Singapore"),
            "elev_mean": float(props.get("elev_mean", 10.0)),
            "elev_min": float(props.get("elev_min", 0.0)),
            "elev_std": float(props.get("elev_std", 5.0)),
            "centroid_lat": avg_lat,
            "centroid_lon": avg_lon
        }

    # 4. Load XGBoost Ensemble & Calibrator
    model_path = os.path.join(script_dir, "xgboost_flood_ensemble.pkl")
    calibrator_path = os.path.join(script_dir, "xgboost_flood_calibrator.pkl")
    config_path = os.path.join(script_dir, "xgboost_flood_model_config.json")

    if os.path.exists(model_path):
        try:
            MODEL_ENSEMBLE = joblib.load(model_path)
        except Exception as e:
            print(f"Warning: Could not load ensemble pickle: {e}")

    if os.path.exists(calibrator_path):
        try:
            MODEL_CALIBRATOR = joblib.load(calibrator_path)
        except Exception as e:
            print(f"Warning: Could not load calibrator pickle: {e}")

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                MODEL_CONFIG = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config json: {e}")


init_resources()


def get_sensor_for_planning_area(pln_area_name: str) -> Dict[str, Any]:
    """
    Find the best rainfall sensor for a given planning area.
    If multiple sensors exist in the area, pick 1.
    If none exist in the area, pick the closest sensor geographically.
    """
    clean_name = pln_area_name.strip().upper()
    direct_sensors = [
        s for s in RAINFALL_SENSORS
        if (s.get("location", {}).get("PLN_AREA_N") or "").upper() == clean_name
    ]

    if direct_sensors:
        s = direct_sensors[0]
        return {
            "id": s["id"],
            "name": s.get("name", s["id"]),
            "latitude": s["location"]["latitude"],
            "longitude": s["location"]["longitude"],
            "direct_match": True
        }

    # Fallback: nearest sensor to planning area centroid
    area_info = PLN_AREA_LOOKUP.get(clean_name)
    target_lat = area_info["centroid_lat"] if area_info else 1.3521
    target_lon = area_info["centroid_lon"] if area_info else 103.8198

    def dist_sq(s):
        slat = s.get("location", {}).get("latitude", 1.3521)
        slon = s.get("location", {}).get("longitude", 103.8198)
        return (slat - target_lat)**2 + (slon - target_lon)**2

    closest = min(RAINFALL_SENSORS, key=dist_sq)
    return {
        "id": closest["id"],
        "name": closest.get("name", closest["id"]),
        "latitude": closest["location"]["latitude"],
        "longitude": closest["location"]["longitude"],
        "direct_match": False,
        "nearest_note": f"Nearest station to {clean_name}"
    }


def fetch_live_rainfall_data() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Fetch real-time rainfall data from Data.gov.sg API.
    Cached for 60 seconds to optimize latency and prevent 429 rate limits.
    Handles pagination, cross-midnight queries, timestamp deduplication, and chronological sorting.
    """
    global RAINFALL_CACHE
    now_ts = time.time()
    if RAINFALL_CACHE["readings"] and (now_ts - RAINFALL_CACHE["timestamp"] < 60):
        return RAINFALL_CACHE["readings"], {
            "source": "Data.gov.sg Real-Time Rainfall API v2 (cached)",
            "cache_age_seconds": round(now_ts - RAINFALL_CACHE["timestamp"], 1),
            "status": "LIVE_CACHED"
        }

    sg_tz = datetime.timezone(datetime.timedelta(hours=8))
    now = datetime.datetime.now(sg_tz)
    date_str = now.strftime("%Y-%m-%d")

    api_key = load_api_key()
    headers = {
        "User-Agent": "SingaporeFloodRiskExplorer/1.0",
        "Accept": "application/json"
    }
    if api_key:
        headers["x-api-key"] = api_key

    url = "https://api-open.data.gov.sg/v2/real-time/api/rainfall"
    params = {"date": date_str}

    all_readings = []
    fetch_status = "LIVE_SUCCESS"

    # Paginate through today's readings (up to 3 pages = 75 readings = 6+ hours)
    for _ in range(3):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                api_data = data.get("data", {})
                page_readings = api_data.get("readings", [])
                all_readings.extend(page_readings)
                token = api_data.get("paginationToken")
                if not token:
                    break
                params = {"date": date_str, "paginationToken": token}
            else:
                break
        except Exception as e:
            print(f"Data.gov.sg API request error: {e}")
            fetch_status = "ERROR_FALLBACK_CACHE"
            break

    # If crossing midnight and fewer than 21 readings for today, fetch yesterday's readings too
    if len(all_readings) < 21:
        yesterday_str = (now - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            resp_y = requests.get(url, headers=headers, params={"date": yesterday_str}, timeout=10)
            if resp_y.status_code == 200:
                all_readings.extend(resp_y.json().get("data", {}).get("readings", []))
        except Exception as ye:
            print(f"Yesterday rainfall fetch notice: {ye}")

    if all_readings:
        # Deduplicate readings by timestamp and sort chronologically from oldest to newest
        seen_ts = {}
        for r in all_readings:
            ts = r.get("timestamp")
            if ts and ts not in seen_ts:
                seen_ts[ts] = r

        sorted_readings = sorted(
            seen_ts.values(),
            key=lambda r: datetime.datetime.fromisoformat(r["timestamp"])
        )

        RAINFALL_CACHE = {
            "timestamp": now_ts,
            "date_str": date_str,
            "readings": sorted_readings
        }
    else:
        sorted_readings = RAINFALL_CACHE["readings"]

    telemetry_meta = {
        "source": "Data.gov.sg Real-Time Rainfall API v2",
        "endpoint": f"{url}?date={date_str}",
        "status": fetch_status,
        "total_readings_available": len(sorted_readings),
        "evaluated_at": now.isoformat()
    }

    return sorted_readings, telemetry_meta


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/geojson")
def get_geojson():
    if GEOJSON_DATA is None:
        init_resources()
    return jsonify(GEOJSON_DATA)


@app.route("/api/sensors")
def get_sensors():
    if RAINFALL_SENSORS is None:
        init_resources()
    return jsonify(RAINFALL_SENSORS)


@app.route("/api/forecast", methods=["GET", "POST"])
def get_forecast():
    """
    Real-time 15-Minute Flood Risk Forecast Endpoint.

    Workflow:
    1. Accepts `pln_area` parameter (e.g., 'BUKIT TIMAH').
    2. Identifies the representative rainfall sensor and coordinates.
    3. Retrieves DEM elevation metrics (elev_mean, elev_min, elev_std) from GeoJSON.
    4. Fetches real-time rainfall data recorded for the last 1hr 45 mins (21 readings) from Data.gov.sg API.
    5. Calculates `rain_sum_15m`, `rain_sum_30m`, `rain_sum_90m`, and `rain_max_5m`.
    6. Applies Temporal Resistance Check (< 0.2mm) & runs the XGBoost machine learning model.
    7. Returns comprehensive forecast, calibrated risk probability, tier alert, and operational response.
    """
    pln_area = request.args.get("pln_area") or ""
    if request.is_json and request.json:
        pln_area = request.json.get("pln_area", pln_area)

    pln_area = pln_area.strip().upper()
    if not pln_area or pln_area not in PLN_AREA_LOOKUP:
        return jsonify({
            "error": f"Invalid or missing planning area: '{pln_area}'. Available areas: {list(PLN_AREA_LOOKUP.keys())[:5]}..."
        }), 400

    area_info = PLN_AREA_LOOKUP[pln_area]
    sensor_info = get_sensor_for_planning_area(pln_area)
    sensor_id = sensor_info["id"]

    # 1. Fetch Real-time rainfall readings for the last 1hr 45 mins (105 mins)
    all_readings, telemetry_meta = fetch_live_rainfall_data()

    # all_readings is sorted chronologically (oldest -> newest).
    # Take the latest 21 readings (105 mins = 21 * 5min intervals up to the latest timestamp)
    chronological_readings = all_readings[-21:] if len(all_readings) >= 21 else all_readings

    series_data = []
    rain_values = []

    for r in chronological_readings:
        ts = r.get("timestamp", "")
        # Find sensor value in this reading
        val = 0.0
        for d in r.get("data", []):
            if d.get("stationId") == sensor_id:
                v = d.get("value")
                val = float(v) if v is not None else 0.0
                break
        rain_values.append(val)
        series_data.append({
            "timestamp": ts,
            "value": round(val, 2)
        })

    # If no values retrieved from live stream, default to minimal zero series (105m = 21 steps)
    if not rain_values:
        rain_values = [0.0] * 21
        series_data = [{"timestamp": f"T-{105-i*5}m", "value": 0.0} for i in range(21)]

    # 2. Compute Rainfall Features
    rain_sum_15m = float(sum(rain_values[-3:]))
    rain_sum_30m = float(sum(rain_values[-6:]))
    rain_sum_90m = float(sum(rain_values[-18:])) if len(rain_values) >= 18 else float(sum(rain_values))
    rain_max_5m = float(max(rain_values)) if rain_values else 0.0

    # 3. Features for XGBoost Model
    lat = float(sensor_info["latitude"])
    lon = float(sensor_info["longitude"])
    elev_mean = float(area_info["elev_mean"])
    elev_min = float(area_info["elev_min"])
    elev_std = float(area_info["elev_std"])

    feature_dict = {
        "latitude": lat,
        "longitude": lon,
        "elev_mean": elev_mean,
        "elev_min": elev_min,
        "elev_std": elev_std,
        "rain_sum_15m": rain_sum_15m,
        "rain_sum_30m": rain_sum_30m,
        "rain_sum_90m": rain_sum_90m,
        "rain_max_5m": rain_max_5m
    }

    # 4. Temporal Resistance Check & XGBoost Model Inference
    temporal_resistance_triggered = (rain_max_5m < 0.2)

    raw_prob_flood = 0.0
    cal_prob_flood = 0.0

    if temporal_resistance_triggered:
        raw_prob_flood = 0.0
        cal_prob_flood = 0.0
        alert_tier = "NORMAL"
        alert_label = "Normal (No Alert)"
        alert_code = 0
        action_recommendation = (
            "Routine weather and drainage monitoring. Temporal resistance check satisfied: "
            f"Peak rainfall is {rain_max_5m:.1f}mm (< 0.2mm threshold across past 105 mins)."
        )
    else:
        if MODEL_ENSEMBLE is not None:
            features_df = pd.DataFrame([feature_dict])
            try:
                raw_probas = MODEL_ENSEMBLE.predict_proba(features_df)[0]
                raw_prob_flood = float(raw_probas[1])
            except Exception as e:
                print(f"Model prediction error: {e}")
                raw_prob_flood = 0.05
        else:
            raw_prob_flood = 0.05

        if MODEL_CALIBRATOR is not None:
            try:
                cal_prob_flood = float(np.clip(MODEL_CALIBRATOR.predict(np.array([raw_prob_flood])), 0.0, 1.0)[0])
            except Exception as e:
                cal_prob_flood = raw_prob_flood
        else:
            cal_prob_flood = raw_prob_flood

        # Tiered Decision Thresholds: Watch >= 0.030, Warning >= 0.120
        if cal_prob_flood >= 0.120:
            alert_tier = "WARNING"
            alert_label = "Flood Warning"
            alert_code = 2
            action_recommendation = (
                "🚨 IMMEDIATE TACTICAL RESPONSE: High confidence flash flood risk within 15 minutes. "
                "Deploy mobile flood barriers, trigger traffic diversion advisories, and activate local stormwater pumps."
            )
        elif cal_prob_flood >= 0.030:
            alert_tier = "WATCH"
            alert_label = "Flood Watch"
            alert_code = 1
            action_recommendation = (
                "⚡ HEIGHTENED SITUATIONAL AWARENESS: Elevated flood likelihood detected (High Recall mode). "
                "Place incident quick-response crews on standby, increase sensor telemetry polling rate, and inspect critical culverts."
            )
        else:
            alert_tier = "NORMAL"
            alert_label = "Normal (No Alert)"
            alert_code = 0
            action_recommendation = (
                "🛡️ ROUTINE MONITORING: Drainage system capacity sufficient. Continue regular telemetry monitoring."
            )

    return jsonify({
        "status": "success",
        "pln_area": pln_area,
        "region_name": area_info["region"],
        "timestamp_evaluated": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        "sensor": sensor_info,
        "elevation": {
            "elev_mean": round(elev_mean, 2),
            "elev_min": round(elev_min, 2),
            "elev_std": round(elev_std, 2)
        },
        "rainfall_window_minutes": 105,
        "readings_count": len(rain_values),
        "rainfall_metrics": {
            "rain_sum_15m": round(rain_sum_15m, 2),
            "rain_sum_30m": round(rain_sum_30m, 2),
            "rain_sum_90m": round(rain_sum_90m, 2),
            "rain_max_5m": round(rain_max_5m, 2)
        },
        "readings_series": series_data,
        "prediction": {
            "raw_prob_flood": round(raw_prob_flood, 4),
            "calibrated_prob_flood": round(cal_prob_flood, 4),
            "prob_percentage": round(cal_prob_flood * 100, 1),
            "alert_tier": alert_tier,
            "alert_label": alert_label,
            "alert_code": alert_code,
            "action_recommendation": action_recommendation,
            "temporal_resistance_triggered": temporal_resistance_triggered,
            "thresholds": {
                "watch": 0.030,
                "warning": 0.120
            }
        },
        "features_vector": feature_dict,
        "telemetry_provenance": telemetry_meta
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
