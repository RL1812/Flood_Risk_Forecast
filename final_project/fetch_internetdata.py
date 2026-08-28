#!/usr/bin/env python3
"""
Fetch and Extract Verified Singapore Flash Flood Events from CNA News (Jan 2023 - May 2025)
using Google News RSS Feed, googlenewsdecoder, newspaper3k, and Google Gemini API (gemini-3.7-flash).

Features:
1. Dynamic Google News RSS feed search targeting CNA articles without paid search APIs.
2. Google News redirect decoding to obtain verified canonical CNA URLs.
3. Content extraction via newspaper3k (headline, body, publish date).
4. Information extraction via Gemini API with gemini-3.7-flash (and fast rate-limit fallbacks).
5. Output format strictly matches flood_rainfall_records.json:
   - datetime: ISO 8601 string with +08:00 Singapore timezone (e.g. 2024-11-17T17:00:00+08:00)
   - PLN_AREA_N: Official uppercase Singapore Planning Area (e.g. ROCHOR, TOA PAYOH, BUKIT TIMAH)
"""

import os
import sys
import json
import time
import re
import argparse
import urllib.parse
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import requests
import pandas as pd
import geopandas as gpd
import feedparser
from newspaper import Article, Config

try:
    from googlenewsdecoder import new_decoderv1
except ImportError:
    new_decoderv1 = None


def load_env_api_key(env_path: Optional[str] = None) -> Optional[str]:
    """Load GEMINI_FREE_API_KEY from .env file or environment variables."""
    if os.environ.get("GEMINI_FREE_API_KEY"):
        return os.environ["GEMINI_FREE_API_KEY"]

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
                        if k == "GEMINI_FREE_API_KEY":
                            return v
    return None


def get_planning_areas_metadata(geojson_path: Optional[str] = None) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """
    Load official Singapore planning area names, coordinates, and elevation stats from geojson.
    """
    if not geojson_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        geojson_path = os.path.join(script_dir, "enriched_planning_areas.geojson")

    if not os.path.exists(geojson_path):
        default_plns = [
            "ANG MO KIO", "BEDOK", "BISHAN", "BOON LAY", "BUKIT BATOK", "BUKIT MERAH",
            "BUKIT PANJANG", "BUKIT TIMAH", "CENTRAL WATER CATCHMENT", "CHANGI", "CHANGI BAY",
            "CHOA CHU KANG", "CLEMENTI", "DOWNTOWN CORE", "GEYLANG", "HOUGANG", "JURONG EAST",
            "JURONG WEST", "KALLANG", "LIM CHU KANG", "MANDAI", "MARINA EAST", "MARINA SOUTH",
            "MARINE PARADE", "MUSEUM", "NEWTON", "NORTH-EASTERN ISLANDS", "NOVENA", "ORCHARD",
            "OUTRAM", "PASIR RIS", "PAYA LEBAR", "PIONEER", "PUNGGOL", "QUEENSTOWN",
            "RIVER VALLEY", "ROCHOR", "SELETAR", "SEMBAWANG", "SENGKANG", "SERANGOON",
            "SIMPANG", "SINGAPORE RIVER", "SOUTHERN ISLANDS", "STRAITS VIEW", "SUNGEI KADUT",
            "TAMPINES", "TANGLIN", "TENGAH", "TOA PAYOH", "TUAS", "WESTERN ISLANDS",
            "WESTERN WATER CATCHMENT", "WOODLANDS", "YISHUN"
        ]
        return default_plns, {}

    gdf = gpd.read_file(geojson_path)
    pln_list = sorted(gdf["PLN_AREA_N"].unique().tolist())
    pln_meta = {}

    for _, row in gdf.iterrows():
        name = str(row["PLN_AREA_N"]).strip().upper()
        centroid = row.geometry.centroid if hasattr(row, "geometry") and row.geometry else None
        pln_meta[name] = {
            "PLN_AREA_N": name,
            "elev_mean": float(row.get("elev_mean")) if pd.notnull(row.get("elev_mean")) else None,
            "elev_min": float(row.get("elev_min")) if pd.notnull(row.get("elev_min")) else None,
            "elev_std": float(row.get("elev_std")) if pd.notnull(row.get("elev_std")) else None,
            "latitude": float(centroid.y) if centroid else None,
            "longitude": float(centroid.x) if centroid else None,
        }

    return pln_list, pln_meta


def decode_google_news_url(google_url: str) -> Optional[str]:
    """Decode a Google News redirect URL to the destination canonical news URL."""
    if new_decoderv1:
        try:
            res = new_decoderv1(google_url)
            if isinstance(res, dict) and res.get("decoded_url"):
                return res["decoded_url"]
            elif isinstance(res, str) and res.startswith("http"):
                return res
        except Exception:
            pass

    # Fallback to direct request follow
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        r = requests.get(google_url, headers=headers, allow_redirects=True, timeout=10)
        if "channelnewsasia.com" in r.url:
            return r.url
    except Exception:
        pass

    return None


def search_cna_flood_news_rss(
    start_date: str = "2023-01-01",
    end_date: str = "2025-06-01"
) -> List[Dict[str, Any]]:
    """
    Dynamically search Google News RSS for verified CNA flash flood articles.
    """
    search_queries = [
        f'site:channelnewsasia.com "flash flood" Singapore after:{start_date} before:{end_date}',
        f'site:channelnewsasia.com "PUB" ("flash flood" OR "flash floods" OR "flooding") Singapore after:{start_date} before:{end_date}',
        f'site:channelnewsasia.com/singapore ("flash flood" OR "flash floods") after:{start_date} before:{end_date}',
        f'site:channelnewsasia.com "flooding" "Singapore" "PUB" after:{start_date} before:{end_date}',
    ]

    seen_rss_links = set()
    discovered_articles = []

    print(f"Querying Google News RSS feed for CNA flash flood reports ({start_date} to {end_date})...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for query in search_queries:
        encoded_q = urllib.parse.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-SG&gl=SG&ceid=SG:en"

        try:
            r = requests.get(rss_url, headers=headers, timeout=15)
            if r.status_code == 200:
                feed = feedparser.parse(r.text)
                for entry in feed.entries:
                    google_link = entry.link
                    if google_link not in seen_rss_links:
                        seen_rss_links.add(google_link)
                        discovered_articles.append({
                            "title": entry.title,
                            "published": entry.get("published", ""),
                            "google_link": google_link
                        })
        except Exception as e:
            print(f"Error querying RSS with query '{query}': {e}")

    print(f"Found {len(discovered_articles)} candidate CNA articles from Google News RSS feed.")
    return discovered_articles


def fetch_and_parse_article(canonical_url: str, fallback_title: str = "") -> Dict[str, Any]:
    """
    Download and parse article content using newspaper3k.
    """
    config = Config()
    config.browser_user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    config.request_timeout = 15

    try:
        art = Article(canonical_url, config=config)
        art.download()
        art.parse()
        title = art.title if art.title else fallback_title
        text = art.text.strip()
        pub_date = str(art.publish_date) if art.publish_date else None
        return {
            "url": canonical_url,
            "title": title,
            "text": text,
            "publish_date": pub_date,
            "success": bool(text and len(text) > 80)
        }
    except Exception as e:
        return {
            "url": canonical_url,
            "title": fallback_title,
            "text": fallback_title,
            "publish_date": None,
            "success": False,
            "error": str(e)
        }


def extract_flood_events_with_gemini(
    article_title: str,
    article_text: str,
    article_pubdate: Optional[str],
    api_key: str,
    official_pln_areas: List[str],
    primary_model: str = "gemini-3.7-flash",
) -> List[Dict[str, Any]]:
    """
    Use Gemini API (primary: gemini-3.7-flash, with automatic fallback) to extract structured
    flash flood events from verified CNA news text.
    """
    prompt = f"""
You are an expert meteorological and GIS data analyst for Singapore.
Analyze this verified Channel NewsAsia (CNA) news article and determine if it reports any ACTUAL Singapore flash flood events between January 2023 and May 2025.

Article Title: {article_title}
Publication Date: {article_pubdate or 'Unknown'}
Article Body:
\"\"\"
{article_text[:3500]}
\"\"\"

Official Singapore Planning Areas:
{json.dumps(official_pln_areas)}

CRITICAL INSTRUCTIONS:
1. ONLY extract ACTUAL, SPECIFIC flash flood occurrences or heavy-rain flood warnings that occurred on a specific date in Singapore. Do NOT extract generic weather forecasts, future outlooks, general infrastructure explainers, or flooding outside Singapore.
2. For each verified flash flood event, extract:
   - "datetime": The exact ISO 8601 datetime with Singapore timezone offset "+08:00" (e.g. "YYYY-MM-DDTHH:MM:SS+08:00"). If the time is not stated in the article, use the afternoon peak rain time "15:30:00+08:00" or morning "09:00:00+08:00" inferred from the article's date.
   - "PLN_AREA_N": The EXACT matching planning area name from the Official Singapore Planning Areas list provided above (must be UPPERCASE, e.g. "ROCHOR", "BUKIT TIMAH", "TOA PAYOH", "YISHUN", "BEDOK", "JURONG WEST", "MARINE PARADE", "KALLANG", "TAMPINES", "SEMBAWANG", "PASIR RIS"). Map landmarks to their planning area (e.g. Ophir Rd/Bugis -> ROCHOR, Potong Pasir/Wan Tho Ave -> TOA PAYOH, Dunearn Rd -> BUKIT TIMAH, Jalan Seaview -> MARINE PARADE, Stevens Rd -> BUKIT TIMAH).
   - "location_name": Specific road, junction, or landmark mentioned (e.g. "Ophir Road near Bugis", "Wan Tho Avenue").
   - "description": Clear, concise description of the flood event from the article.
   - "severity": "Minor", "Moderate", or "Severe".

3. If the article does NOT contain any specific Singapore flash flood occurrences, return an empty array [].
4. Return ONLY a valid JSON array of objects. No markdown code blocks, no explanation.
"""

    models_to_try = [primary_model, "gemini-3.5-flash-lite"]
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.1
        }
    }

    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    text_res = candidates[0]["content"]["parts"][0]["text"].strip()
                    if text_res.startswith("```json"):
                        text_res = text_res[7:]
                    if text_res.endswith("```"):
                        text_res = text_res[:-3]
                    parsed_json = json.loads(text_res.strip())
                    if isinstance(parsed_json, list):
                        return parsed_json
                    elif isinstance(parsed_json, dict):
                        return [parsed_json]
            elif resp.status_code in (429, 503):
                # Rate limited or high demand, proceed to fallback model
                continue
        except Exception:
            continue

    return []


def process_cna_flood_news(
    output_json_path: str,
    geojson_path: Optional[str] = None,
    api_key: Optional[str] = None,
    start_date: str = "2023-01-01",
    end_date: str = "2025-06-01",
    primary_model: str = "gemini-3.7-flash"
) -> List[Dict[str, Any]]:
    """
    Main extraction pipeline:
    1. Search Google News RSS feed for CNA flash flood articles (2023-01 to 2025-05).
    2. Decode Google News redirect links to verified canonical CNA URLs.
    3. Extract article content via newspaper3k.
    4. Extract structured flood events via Gemini.
    5. Format datetimes and planning areas to match flood_rainfall_records.json.
    6. Save to output JSON file.
    """
    print("=" * 85)
    print(f"FETCHING & EXTRACTING VERIFIED CNA FLASH FLOOD EVENTS ({start_date} TO {end_date})")
    print("=" * 85)

    # 1. API Key
    key = api_key or load_env_api_key()
    if not key:
        raise ValueError("GEMINI_FREE_API_KEY not found in .env or environment!")
    print("Gemini API Key loaded successfully.")

    # 2. Official Planning Areas
    pln_list, pln_meta = get_planning_areas_metadata(geojson_path)
    print(f"Loaded {len(pln_list)} official Singapore planning areas for location mapping.\n")

    # 3. Discover CNA Articles via Google News RSS
    raw_rss_entries = search_cna_flood_news_rss(start_date=start_date, end_date=end_date)

    # 4. Decode Google News URLs to canonical CNA URLs
    print("\nDecoding Google News URLs to canonical CNA links...")
    verified_cna_articles = []
    seen_cna_urls = set()

    for idx, item in enumerate(raw_rss_entries):
        g_url = item["google_link"]
        cna_url = decode_google_news_url(g_url)
        if cna_url and "channelnewsasia.com" in cna_url:
            clean_url = cna_url.split("?")[0]
            if clean_url not in seen_cna_urls:
                seen_cna_urls.add(clean_url)
                verified_cna_articles.append({
                    "url": clean_url,
                    "title": item["title"].replace(" - CNA", "").strip(),
                    "published": item["published"]
                })
                print(f"  [{len(verified_cna_articles)}] Verified CNA URL: {clean_url}")

    print(f"\nDiscovered {len(verified_cna_articles)} unique verified CNA articles.")
    print("=" * 85)

    extracted_records = []
    seen_event_keys = set()

    for idx, art_meta in enumerate(verified_cna_articles):
        cna_url = art_meta["url"]
        headline = art_meta["title"]
        print(f"\n[{idx+1}/{len(verified_cna_articles)}] Parsing with newspaper3k: {headline[:60]}...")
        print(f"    URL: {cna_url}")

        # Download & parse with newspaper3k
        art_data = fetch_and_parse_article(cna_url, fallback_title=headline)
        if not art_data.get("success") or len(art_data["text"]) < 50:
            print("    Skipping article (insufficient text / non-article page).")
            continue

        print(f"    Extracting flood events via Gemini ({primary_model})...")
        events = extract_flood_events_with_gemini(
            article_title=art_data["title"],
            article_text=art_data["text"],
            article_pubdate=art_data["publish_date"] or art_meta["published"],
            api_key=key,
            official_pln_areas=pln_list,
            primary_model=primary_model
        )

        if not events:
            print("    No specific flash flood events found in this article.")
            continue

        for ev in events:
            raw_dt = ev.get("datetime", "")
            raw_pln = str(ev.get("PLN_AREA_N", "")).strip().upper()

            # Ensure PLN_AREA_N matches an official planning area
            matched_pln = raw_pln if raw_pln in pln_meta or raw_pln in pln_list else "BUKIT TIMAH"

            geo_info = pln_meta.get(matched_pln, {})
            lat = ev.get("latitude") or geo_info.get("latitude", 1.3320)
            lon = ev.get("longitude") or geo_info.get("longitude", 103.8701)

            record = {
                "flood_event": {
                    "datetime": raw_dt,
                    "PLN_AREA_N": matched_pln,
                    "elev_mean": geo_info.get("elev_mean", 14.0),
                    "elev_min": geo_info.get("elev_min", -5.0),
                    "elev_std": geo_info.get("elev_std", 7.0),
                    "latitude": lat,
                    "longitude": lon,
                    "location_name": ev.get("location_name", ""),
                    "description": ev.get("description", ev.get("location_name", "")),
                    "severity": ev.get("severity", "Moderate"),
                    "source_url": cna_url,
                    "source_title": art_data["title"]
                }
            }

            event_key = (raw_dt.split("T")[0] if "T" in raw_dt else raw_dt, matched_pln)
            if event_key not in seen_event_keys:
                seen_event_keys.add(event_key)
                extracted_records.append(record)
                print(f"    -> Extracted Event: {raw_dt} | {matched_pln} ({ev.get('location_name')})")

        time.sleep(1)

    # Sort chronologically
    extracted_records.sort(key=lambda r: r["flood_event"].get("datetime", ""))

    # Save to JSON
    output_dir = os.path.dirname(os.path.abspath(output_json_path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(extracted_records, f, indent=2)

    print("\n" + "=" * 85)
    print("EXTRACTION COMPLETE")
    print("=" * 85)
    print(f"Total Verified Flash Flood Events Extracted: {len(extracted_records)}")
    print(f"Saved records to: {output_json_path}")
    print("-" * 85)
    for idx, r in enumerate(extracted_records):
        fe = r["flood_event"]
        print(f"  {idx+1:2d}. {fe['datetime']} | {fe['PLN_AREA_N']:<20} | {fe['location_name']}")
        print(f"      Verified URL: {fe['source_url']}")
    print("=" * 85)

    return extracted_records


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.join(script_dir, "cna_flood_events_2023_2025.json")
    default_geojson = os.path.join(script_dir, "enriched_planning_areas.geojson")

    parser = argparse.ArgumentParser(
        description="Search Google News RSS for CNA flood articles and extract events using Gemini."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=default_out,
        help=f"Output JSON file path (default: {default_out})"
    )
    parser.add_argument(
        "-g", "--geojson",
        type=str,
        default=default_geojson,
        help=f"Path to enriched_planning_areas.geojson (default: {default_geojson})"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-01-01",
        help="Start date filter YYYY-MM-DD (default: 2023-01-01)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-06-01",
        help="End date filter YYYY-MM-DD (default: 2025-06-01)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="gemini-3.7-flash",
        help="Gemini model name (default: gemini-3.7-flash)"
    )

    args = parser.parse_args()

    process_cna_flood_news(
        output_json_path=args.output,
        geojson_path=args.geojson,
        start_date=args.start_date,
        end_date=args.end_date,
        primary_model=args.model
    )


if __name__ == "__main__":
    main()
