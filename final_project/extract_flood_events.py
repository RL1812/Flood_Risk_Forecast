#!/usr/bin/env python3
"""
Extract Flood Events and Enrich with Planning Area Names.

This script extracts records where event == 'flood' from flood event day records,
extracts the specified fields (datetime, identifier, event, description, severity,
urgency, instruction, location), and enriches the location object with the
corresponding planning area name (PLN_AREA_N) based on point-in-polygon spatial
matching against enriched_planning_areas.geojson.
"""

import os
import sys
import json
import argparse
from typing import Optional, Dict, Any, List
import geopandas as gpd
from shapely.geometry import Point


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
    
    # Deduplicate search dirs while preserving order
    seen = set()
    unique_search_dirs = []
    for d in search_dirs:
        if d not in seen and os.path.isdir(d):
            seen.add(d)
            unique_search_dirs.append(d)

    input_candidates = [
        "flood_eventdays_records.json",
        "flood_event_days_records.json",
        "flood_alert_records.json"
    ]
    geojson_candidates = [
        "enriched_planning_areas.geojson",
        "MasterPlan2019PlanningAreaBoundaryNoSea.geojson"
    ]

    input_path = find_file(input_candidates, unique_search_dirs)
    geojson_path = find_file(geojson_candidates, unique_search_dirs)
    output_path = os.path.join(script_dir, "flood_events_extracted.json")

    return input_path, geojson_path, output_path


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


def find_planning_area(gdf: gpd.GeoDataFrame, lon: Optional[float], lat: Optional[float]) -> Optional[str]:
    """Find the planning area name (PLN_AREA_N) containing the given lon/lat point."""
    if lon is None or lat is None:
        return None
    
    try:
        lon_f = float(lon)
        lat_f = float(lat)
    except (ValueError, TypeError):
        return None

    pt = Point(lon_f, lat_f)
    # Check polygons containing the point
    matches = gdf[gdf.geometry.contains(pt)]
    if len(matches) == 0:
        # Fallback to intersects (e.g. boundary points)
        matches = gdf[gdf.geometry.intersects(pt)]
    
    if len(matches) > 0:
        return str(matches.iloc[0].get("PLN_AREA_N", ""))
    return None


def extract_and_enrich_flood_records(
    input_path: str,
    geojson_path: str,
    output_path: str
) -> List[Dict[str, Any]]:
    """
    Extracts flood records and adds PLN_AREA_N into the location property.
    
    Requested fields:
    - datetime
    - identifier
    - event
    - description
    - severity
    - urgency
    - instruction
    - location (enriched with PLN_AREA_N)
    """
    print(f"Loading planning areas from: {geojson_path}")
    gdf = load_planning_areas(geojson_path)
    print(f"Loaded {len(gdf)} planning area polygons.")

    print(f"Loading flood event records from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        raw_records = json.load(f)

    print(f"Total raw records loaded: {len(raw_records)}")

    extracted_records = []
    for rec in raw_records:
        event_val = rec.get("event")
        # Filter records where event == 'flood' (case-insensitive)
        if not event_val or str(event_val).strip().lower() != "flood":
            continue

        # Extract location object or build one if missing
        raw_loc = rec.get("location")
        if isinstance(raw_loc, dict):
            location_data = dict(raw_loc)
        else:
            location_data = {}
            if "latitude" in rec and rec["latitude"] is not None:
                location_data["latitude"] = rec["latitude"]
            if "longitude" in rec and rec["longitude"] is not None:
                location_data["longitude"] = rec["longitude"]
            if "areaDesc" in rec and rec["areaDesc"] is not None:
                location_data["areaDesc"] = rec["areaDesc"]
            if "radius_km" in rec and rec["radius_km"] is not None:
                location_data["radius_km"] = rec["radius_km"]

        # Get coordinates for spatial match
        lon = location_data.get("longitude", rec.get("longitude"))
        lat = location_data.get("latitude", rec.get("latitude"))

        # Find planning area PLN_AREA_N
        pln_area_name = find_planning_area(gdf, lon, lat)
        location_data["PLN_AREA_N"] = pln_area_name

        # Construct filtered record with specified fields
        extracted_entry = {
            "datetime": rec.get("datetime"),
            "identifier": rec.get("identifier"),
            "event": rec.get("event"),
            "description": rec.get("description"),
            "severity": rec.get("severity"),
            "urgency": rec.get("urgency"),
            "instruction": rec.get("instruction"),
            "location": location_data
        }
        extracted_records.append(extracted_entry)

    print(f"Extracted {len(extracted_records)} flood records.")

    # Save to output file
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(extracted_records, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved enriched records to: {output_path}")
    return extracted_records


def main():
    default_in, default_geo, default_out = get_default_paths()

    parser = argparse.ArgumentParser(
        description="Extract flood records and enrich location with PLN_AREA_N from GeoJSON."
    )
    parser.add_argument(
        "-i", "--input",
        default=default_in,
        help=f"Path to input flood records JSON file (default: {default_in})"
    )
    parser.add_argument(
        "-g", "--geojson",
        default=default_geo,
        help=f"Path to enriched_planning_areas.geojson (default: {default_geo})"
    )
    parser.add_argument(
        "-o", "--output",
        default=default_out,
        help=f"Path to output JSON file (default: {default_out})"
    )

    args = parser.parse_args()

    if not args.input or not os.path.exists(args.input):
        print(f"Error: Input file not found. Checked: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not args.geojson or not os.path.exists(args.geojson):
        print(f"Error: GeoJSON file not found. Checked: {args.geojson}", file=sys.stderr)
        sys.exit(1)

    extract_and_enrich_flood_records(
        input_path=args.input,
        geojson_path=args.geojson,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
