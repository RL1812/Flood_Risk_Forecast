#!/usr/bin/env python3
"""
Test XGBoost Flood Prediction Model using sample rows from training_dataset.csv.

Supports:
1. Balanced XGBoost Ensemble or standalone XGBClassifier models.
2. Probability Calibrator (Isotonic Regression or Platt scaling).
3. Tuned decision threshold from configuration.
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Any
from xgboost import XGBClassifier
from scipy.special import expit

# Ensure final_project directory is on sys.path to unpickle BalancedXGBEnsemble & FocalLossObjective
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from xgboost_training import BalancedXGBEnsemble, FocalLossObjective
    # Register in __main__ to allow seamless unpickling
    setattr(sys.modules["__main__"],
            "BalancedXGBEnsemble", BalancedXGBEnsemble)
    setattr(sys.modules["__main__"], "FocalLossObjective", FocalLossObjective)
except ImportError:
    pass


def find_file(candidate_names: List[str], base_dirs: List[str]) -> Optional[str]:
    """Find the first existing file among candidate names across search directories."""
    for base_dir in base_dirs:
        for name in candidate_names:
            path = os.path.join(base_dir, name)
            if os.path.exists(path):
                return os.path.abspath(path)
    return None


def get_default_paths() -> Tuple[str, str, Optional[str], str]:
    """Determine default file paths based on script location and current working directory."""
    cwd = os.getcwd()
    search_dirs = [script_dir, cwd, os.path.join(
        cwd, "final_project"), os.path.join(script_dir, "final_project")]

    seen = set()
    unique_search_dirs = []
    for d in search_dirs:
        if d not in seen and os.path.isdir(d):
            seen.add(d)
            unique_search_dirs.append(d)

    data_path = find_dataset_file(["training_dataset.csv"], unique_search_dirs)
    if not data_path:
        data_path = os.path.join(script_dir, "training_dataset.csv")

    model_path = find_dataset_file([
        "xgboost_flood_ensemble.pkl",
        "xgboost_flood_model.pkl",
        "xgboost_flood_model.json"
    ], unique_search_dirs)
    if not model_path:
        model_path = os.path.join(script_dir, "xgboost_flood_ensemble.pkl")

    calibrator_path = find_dataset_file(
        ["xgboost_flood_calibrator.pkl"], unique_search_dirs)
    config_path = find_dataset_file(
        ["xgboost_flood_model_config.json"], unique_search_dirs)
    if not config_path:
        config_path = os.path.join(
            script_dir, "xgboost_flood_model_config.json")

    return data_path, model_path, calibrator_path, config_path


def find_dataset_file(candidate_names: List[str], base_dirs: List[str]) -> Optional[str]:
    return find_file(candidate_names, base_dirs)


def load_model_object(model_path: str) -> Any:
    """Load model from either native XGBoost JSON format or joblib pickle format."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_path.endswith(".json"):
        model = XGBClassifier()
        model.load_model(model_path)
    elif model_path.endswith(".pkl") or model_path.endswith(".joblib"):
        model = joblib.load(model_path)
    else:
        try:
            model = joblib.load(model_path)
        except Exception:
            model = XGBClassifier()
            model.load_model(model_path)

    return model


def load_calibrator_object(calibrator_path: Optional[str]) -> Optional[Any]:
    if calibrator_path and os.path.exists(calibrator_path):
        try:
            return joblib.load(calibrator_path)
        except Exception:
            return None
    return None


def load_config_data(config_path: str) -> dict:
    """Load model metadata and tuned threshold if available."""
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def test_model_on_row(
    data_path: str,
    model_path: str,
    calibrator_path: Optional[str] = None,
    config_path: Optional[str] = None,
    row_index: int = 0,
    threshold_override: Optional[float] = None,
) -> dict:
    """
    Test the trained model / ensemble against a specific row in the dataset.
    """
    print("=" * 75)
    print("TESTING ADVANCED XGBOOST FLOOD PREDICTION PIPELINE")
    print("=" * 75)
    print(f"Dataset path:    {data_path}")
    print(f"Model path:      {model_path}")
    if calibrator_path:
        print(f"Calibrator path: {calibrator_path}")
    print(f"Testing row:     Index {row_index}\n")

    # 1. Load dataset
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    if row_index < 0 or row_index >= len(df):
        raise IndexError(
            f"Row index {row_index} out of bounds (0 to {len(df)-1})")

    row = df.iloc[row_index]

    # 2. Extract features and target
    target_col = "target" if "target" in df.columns else df.columns[-1]
    actual_target = int(row[target_col])

    metadata_cols = {"station_id", "pln_area", target_col}
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    sample_features = pd.DataFrame([row[feature_cols].to_dict()])

    # 3. Load Model, Calibrator, and Config
    model = load_model_object(model_path)
    calibrator = load_calibrator_object(calibrator_path)
    config = load_config_data(config_path) if config_path else {}

    # 4. Model Prediction & Calibration
    raw_probabilities = model.predict_proba(sample_features)[0]
    raw_p_flood = float(raw_probabilities[1])

    if calibrator is not None:
        if hasattr(calibrator, "predict"):  # IsotonicRegression
            cal_p_flood = float(np.clip(calibrator.predict(
                np.array([raw_p_flood])), 0.0, 1.0)[0])
        elif hasattr(calibrator, "predict_proba"):  # Platt scaling LogisticRegression
            eps = 1e-12
            logit = np.log(
                raw_p_flood / (1.0 - raw_p_flood + eps)).reshape(-1, 1)
            cal_p_flood = float(calibrator.predict_proba(logit)[0, 1])
        else:
            cal_p_flood = raw_p_flood
    else:
        cal_p_flood = raw_p_flood

    # Tiered Decision Thresholds from configuration
    watch_th = float(config.get("threshold_watch", config.get(
        "tiered_thresholds", {}).get("watch", 0.030)))
    warning_th = float(config.get("threshold_warning", config.get(
        "tiered_thresholds", {}).get("warning", 0.120)))

    if threshold_override is not None:
        threshold = threshold_override
        threshold_src = f"manual override ({threshold:.4f})"
    elif "optimal_threshold" in config:
        threshold = float(config["optimal_threshold"])
        threshold_src = f"tuned optimal from config ({threshold:.4f})"
    else:
        threshold = watch_th
        threshold_src = f"default watch threshold ({watch_th:.4f})"

    # Check Temporal Resistance Check (if all past readings < 0.2mm, consider not flood)
    rain_max = float(row.get("rain_max_5m", 0.0)) if pd.notnull(
        row.get("rain_max_5m")) else 0.0
    temporal_resistance_triggered = (rain_max < 0.2)

    if temporal_resistance_triggered:
        cal_p_flood = 0.0
        alert_level = "NORMAL / NO ALERT"
        alert_code = 0
        alert_desc = "[TIER 0 - NORMAL] Temporal Resistance Check: Peak rainfall < 0.2mm across 105-min window. Insufficient rain for flood."
    elif cal_p_flood >= warning_th:
        alert_level = "FLOOD WARNING"
        alert_code = 2
        alert_desc = "[ALERT TIER 2 - WARNING] High confidence flood risk. Immediate tactical response: Deploy flood barriers & trigger traffic diversions."
    elif cal_p_flood >= watch_th:
        alert_level = "FLOOD WATCH"
        alert_code = 1
        alert_desc = "[ALERT TIER 1 - WATCH] Elevated flood likelihood (high recall). Heightened awareness: Standby response teams & monitor drainage sensors."
    else:
        alert_level = "NORMAL / NO ALERT"
        alert_code = 0
        alert_desc = "[TIER 0 - NORMAL] Routine baseline conditions. Continue routine sensor monitoring."

    binary_prediction = 1 if cal_p_flood >= threshold else 0

    # 5. Display detailed results
    print("-" * 75)
    print("INPUT SAMPLE DETAILS:")
    print("-" * 75)
    if "station_id" in row:
        print(f"  Station ID:        {row['station_id']}")
    if "pln_area" in row:
        print(f"  Planning Area:     {row['pln_area']}")
    print(
        f"  Coordinates:       (Lat: {row.get('latitude', 'N/A')}, Lon: {row.get('longitude', 'N/A')})")
    print(
        f"  Elevation Stats:   Mean={row.get('elev_mean', 'N/A')}m, Min={row.get('elev_min', 'N/A')}m, Std={row.get('elev_std', 'N/A')}m")
    print(f"  Rainfall Readings: 15m Sum={row.get('rain_sum_15m', 'N/A')}mm, 30m Sum={row.get('rain_sum_30m', 'N/A')}mm, 90m Sum={row.get('rain_sum_90m', 'N/A')}mm, 5m Max={row.get('rain_max_5m', 'N/A')}mm")

    print("\n" + "-" * 75)
    print("MODEL INFERENCE & TIERED OPERATIONAL ALERT:")
    print("-" * 75)
    label_map = {0: "No Flood (0)", 1: "Flood Warning (1)"}
    print(f"  Architecture:            {type(model).__name__}")
    print(
        f"  Raw P(Flood):            {raw_p_flood:.4f} ({raw_p_flood*100:.2f}%)")
    print(
        f"  Calibrated P(Flood):     {cal_p_flood:.4f} ({cal_p_flood*100:.2f}%)")
    print(
        f"  Tier Thresholds:         Watch >= {watch_th:.3f} | Warning >= {warning_th:.3f}")
    print(f"  Operational Tier:        {alert_level}")
    print(f"  Recommended Action:      {alert_desc}")
    print(
        f"  Ground Truth Target:     {label_map.get(actual_target, actual_target)}")
    print(
        f"  Binary Prediction:       {label_map.get(binary_prediction, binary_prediction)} (at threshold {threshold:.4f})")

    match = (binary_prediction == actual_target)
    result_status = "CORRECT [PASS]" if match else "INCORRECT [FAIL]"
    print(f"  Prediction Status:       {result_status}")
    print("-" * 75 + "\n")

    return {
        "row_index": row_index,
        "actual_target": actual_target,
        "predicted_target": binary_prediction,
        "alert_level": alert_level,
        "alert_code": alert_code,
        "raw_prob_flood": raw_p_flood,
        "calibrated_prob_flood": cal_p_flood,
        "threshold": threshold,
        "watch_threshold": watch_th,
        "warning_threshold": warning_th,
        "match": match,
    }


def main():
    default_data, default_model, default_calibrator, default_config = get_default_paths()

    parser = argparse.ArgumentParser(
        description="Test the trained XGBoost flood model / ensemble on sample rows from training_dataset.csv."
    )
    parser.add_argument(
        "-d", "--data-path",
        type=str,
        default=default_data,
        help=f"Path to training_dataset.csv (default: {default_data})"
    )
    parser.add_argument(
        "-m", "--model-path",
        type=str,
        default=default_model,
        help=f"Path to trained model or ensemble pickle (default: {default_model})"
    )
    parser.add_argument(
        "--calibrator-path",
        type=str,
        default=default_calibrator,
        help=f"Path to calibrator pickle (default: {default_calibrator})"
    )
    parser.add_argument(
        "-c", "--config-path",
        type=str,
        default=default_config,
        help=f"Path to model config JSON with tuned threshold (default: {default_config})"
    )
    parser.add_argument(
        "-r", "--row-index",
        type=int,
        default=0,
        help="Row index from dataset to test (default: 0 - the first row)"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=None,
        help="Manual threshold override (defaults to tuned threshold from config or 0.5)"
    )

    args = parser.parse_args()

    test_model_on_row(
        data_path=args.data_path,
        model_path=args.model_path,
        calibrator_path=args.calibrator_path,
        config_path=args.config_path,
        row_index=args.row_index,
        threshold_override=args.threshold,
    )


if __name__ == "__main__":
    main()
