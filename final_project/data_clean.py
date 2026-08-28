#!/usr/bin/env python3
"""
Data Cleaning Helper Script for Singapore Rainfall & Flood Datasets.

This script cleans `flood_rainfall_records.json` and `noflood_rainfall_records.json`
by removing records where the flood event description contains 'subsided' (case-insensitive).
"""

import os
import sys
import json
import argparse
from typing import Optional, Dict, Any, List, Set, Tuple


def find_file(candidate_names: List[str], base_dirs: List[str]) -> Optional[str]:
    """Find the first existing file among candidate names across search directories."""
    for base_dir in base_dirs:
        for name in candidate_names:
            path = os.path.join(base_dir, name)
            if os.path.exists(path):
                return os.path.abspath(path)
    return None


def get_default_paths() -> Tuple[Optional[str], Optional[str]]:
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

    return flood_path, noflood_path


def is_subsided_description(description: Optional[str]) -> bool:
    """Check if a description contains the keyword 'subsided' (case-insensitive)."""
    if not description or not isinstance(description, str):
        return False
    return "subsided" in description.lower()


def get_event_key(datetime_str: Optional[str], pln_area: Optional[str], lat: Optional[float] = None, lon: Optional[float] = None) -> Tuple[str, str, Optional[float], Optional[float]]:
    """Generate a normalized key to uniquely identify a flood event across files."""
    dt_key = str(datetime_str).strip() if datetime_str else ""
    pln_key = str(pln_area).strip().upper() if pln_area else ""
    lat_key = round(float(lat), 5) if lat is not None else None
    lon_key = round(float(lon), 5) if lon is not None else None
    return (dt_key, pln_key, lat_key, lon_key)


def clean_rainfall_data(
    flood_path: str,
    noflood_path: str,
    output_flood_path: Optional[str] = None,
    output_noflood_path: Optional[str] = None,
    dry_run: bool = False,
    create_backup: bool = False,
) -> Dict[str, Any]:
    """
    Remove records with 'subsided' in flood_event description from both flood
    and noflood rainfall JSON datasets.
    """
    output_flood_path = output_flood_path or flood_path
    output_noflood_path = output_noflood_path or noflood_path

    print("=" * 70)
    print("DATA CLEANING: Removing subsided flood event records")
    print("=" * 70)

    # 1. Load flood rainfall records
    if not os.path.exists(flood_path):
        raise FileNotFoundError(f"Flood rainfall file not found: {flood_path}")

    with open(flood_path, "r", encoding="utf-8") as f:
        flood_records: List[Dict[str, Any]] = json.load(f)

    # 2. Load noflood rainfall records
    if not os.path.exists(noflood_path):
        raise FileNotFoundError(f"No-flood rainfall file not found: {noflood_path}")

    with open(noflood_path, "r", encoding="utf-8") as f:
        noflood_records: List[Dict[str, Any]] = json.load(f)

    print(f"Loaded {len(flood_records)} flood records from: {flood_path}")
    print(f"Loaded {len(noflood_records)} no-flood records from: {noflood_path}\n")

    # 3. Identify subsided events in flood_rainfall_records.json
    subsided_event_keys: Set[Tuple[str, str, Optional[float], Optional[float]]] = set()
    subsided_datetimes: Set[str] = set()
    removed_flood_records: List[Dict[str, Any]] = []
    kept_flood_records: List[Dict[str, Any]] = []

    for idx, record in enumerate(flood_records):
        fe = record.get("flood_event", {})
        description = fe.get("description", "")
        dt = fe.get("datetime")
        pln = fe.get("PLN_AREA_N")
        lat = fe.get("latitude")
        lon = fe.get("longitude")

        if is_subsided_description(description):
            key = get_event_key(dt, pln, lat, lon)
            subsided_event_keys.add(key)
            if dt:
                subsided_datetimes.add(dt)
            removed_flood_records.append({
                "index": idx,
                "datetime": dt,
                "PLN_AREA_N": pln,
                "description": description
            })
        else:
            kept_flood_records.append(record)

    # 4. Filter noflood records matching subsided events
    removed_noflood_records: List[Dict[str, Any]] = []
    kept_noflood_records: List[Dict[str, Any]] = []

    for idx, record in enumerate(noflood_records):
        ref = record.get("flood_event_reference", {})
        dt = ref.get("datetime")
        pln = ref.get("flood_PLN_AREA_N") or ref.get("PLN_AREA_N")
        lat = ref.get("latitude")
        lon = ref.get("longitude")

        # Also check if description is present directly on record or ref
        direct_desc = record.get("description") or ref.get("description") or record.get("flood_event", {}).get("description")

        key = get_event_key(dt, pln, lat, lon)
        is_subsided = (
            key in subsided_event_keys
            or (dt is not None and dt in subsided_datetimes)
            or is_subsided_description(direct_desc)
        )

        if is_subsided:
            removed_noflood_records.append({
                "index": idx,
                "datetime": dt,
                "flood_PLN_AREA_N": pln
            })
        else:
            kept_noflood_records.append(record)

    # 5. Display summary of removed records
    print("-" * 70)
    print(f"Identified {len(removed_flood_records)} subsided flood event records to remove:")
    print("-" * 70)
    for item in removed_flood_records:
        print(f"  - [Index {item['index']}] {item['datetime']} | {item['PLN_AREA_N']}")
        print(f"    Description: {item['description']}")
    print("-" * 70)

    print(f"\nSummary:")
    print(f"  Flood records:    {len(flood_records)} -> {len(kept_flood_records)} ({len(removed_flood_records)} removed)")
    print(f"  No-flood records: {len(noflood_records)} -> {len(kept_noflood_records)} ({len(removed_noflood_records)} removed)")

    # 6. Save or dry run
    if dry_run:
        print("\n[DRY RUN] No files were modified.")
    else:
        if create_backup:
            bak_flood = flood_path + ".bak"
            bak_noflood = noflood_path + ".bak"
            with open(bak_flood, "w", encoding="utf-8") as f:
                json.dump(flood_records, f, indent=2)
            with open(bak_noflood, "w", encoding="utf-8") as f:
                json.dump(noflood_records, f, indent=2)
            print(f"\nBackups created:\n  - {bak_flood}\n  - {bak_noflood}")

        with open(output_flood_path, "w", encoding="utf-8") as f:
            json.dump(kept_flood_records, f, indent=2)
        print(f"\nSaved cleaned flood records to: {output_flood_path}")

        with open(output_noflood_path, "w", encoding="utf-8") as f:
            json.dump(kept_noflood_records, f, indent=2)
        print(f"Saved cleaned no-flood records to: {output_noflood_path}")

    return {
        "flood_before": len(flood_records),
        "flood_after": len(kept_flood_records),
        "noflood_before": len(noflood_records),
        "noflood_after": len(kept_noflood_records),
        "removed_flood_count": len(removed_flood_records),
        "removed_noflood_count": len(removed_noflood_records),
    }


def main():
    default_flood, default_noflood = get_default_paths()

    parser = argparse.ArgumentParser(
        description="Clean flood and no-flood rainfall records by removing events with 'subsided' in the description."
    )
    parser.add_argument(
        "--flood-records",
        type=str,
        default=default_flood,
        help=f"Path to flood_rainfall_records.json (default: {default_flood})",
    )
    parser.add_argument(
        "--noflood-records",
        type=str,
        default=default_noflood,
        help=f"Path to noflood_rainfall_records.json (default: {default_noflood})",
    )
    parser.add_argument(
        "--output-flood",
        type=str,
        default=None,
        help="Path for cleaned flood output file (defaults to overwrite input file)",
    )
    parser.add_argument(
        "--output-noflood",
        type=str,
        default=None,
        help="Path for cleaned no-flood output file (defaults to overwrite input file)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the cleaning process without writing any files",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Create .bak backup files before overwriting (default: True)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_false",
        dest="backup",
        help="Do not create backup files before overwriting",
    )

    args = parser.parse_args()

    clean_rainfall_data(
        flood_path=args.flood_records,
        noflood_path=args.noflood_records,
        output_flood_path=args.output_flood,
        output_noflood_path=args.output_noflood,
        dry_run=args.dry_run,
        create_backup=args.backup and not args.dry_run,
    )


if __name__ == "__main__":
    main()
