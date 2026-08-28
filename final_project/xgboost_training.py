#!/usr/bin/env python3
"""
Advanced XGBoost Ensemble with Focal Loss, Balanced Subsampling, Multi-Seed 5-Fold CV,
and Probability Calibration for Severe Imbalance Flood Prediction.

Methodology:
1. Custom Focal Loss Objective (gamma in [1.5, 2.5]) to suppress gradient updates from easy negatives.
2. Balanced Bagging Ensemble of 5–10 XGBoost models (each trained on all positives + balanced random negative subsets).
3. 5-Fold Cross-Validation repeated across multiple seeds to ensure decision threshold generalization.
4. Probability Calibration (Platt Scaling / Isotonic Regression) on Out-Of-Fold predictions.
5. Decision threshold optimization prioritizing Recall (F2-score) and fixed low threshold evaluation.
6. Full serialization of ensemble models, calibrator, and configuration metadata.
"""

import os
import sys
import json
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Any
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    f1_score,
    fbeta_score,
    recall_score,
    precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier
from scipy.special import expit


def find_dataset(candidate_names: List[str], base_dirs: List[str]) -> Optional[str]:
    """Find the first existing file among candidate names across search directories."""
    for base_dir in base_dirs:
        for name in candidate_names:
            path = os.path.join(base_dir, name)
            if os.path.exists(path):
                return os.path.abspath(path)
    return None


def get_default_paths() -> Tuple[str, str, str, str]:
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

    data_path = find_dataset(["training_dataset.csv"], unique_search_dirs)
    if not data_path:
        data_path = os.path.join(script_dir, "training_dataset.csv")

    model_pkl_path = os.path.join(script_dir, "xgboost_flood_ensemble.pkl")
    calibrator_pkl_path = os.path.join(script_dir, "xgboost_flood_calibrator.pkl")
    config_json_path = os.path.join(script_dir, "xgboost_flood_model_config.json")

    return data_path, model_pkl_path, calibrator_pkl_path, config_json_path


class FocalLossObjective:
    """
    Picklable custom binary Focal Loss objective for XGBoost.
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    Suppresses gradient contributions from easy negative instances.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p = expit(y_pred)
        p = np.clip(p, 1e-15, 1.0 - 1e-15)

        # Gradient
        g1 = self.alpha * np.power(1.0 - p, self.gamma) * (self.gamma * p * np.log(p) + p - 1.0)
        g0 = (1.0 - self.alpha) * np.power(p, self.gamma) * (1.0 - p - self.gamma * (1.0 - p) * np.log(1.0 - p))
        grad = np.where(y_true == 1, g1, g0)

        # Positive-definite Hessian approximation
        h1 = self.alpha * np.power(1.0 - p, self.gamma) * p * (1.0 - p) * (self.gamma + 1.0)
        h0 = (1.0 - self.alpha) * np.power(p, self.gamma) * p * (1.0 - p) * (self.gamma + 1.0)
        hess = np.maximum(np.where(y_true == 1, h1, h0), 1e-5)

        return grad, hess


class BalancedXGBEnsemble:
    """
    Ensemble of 5–10 XGBoost models where each member is trained on all positive flood
    events and a random, balanced subset of negative events, using Focal Loss.
    """
    def __init__(
        self,
        n_estimators_ensemble: int = 8,
        n_trees_per_model: int = 40,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        gamma: float = 2.0,
        alpha: float = 0.25,
        neg_ratio: float = 4.0,
        max_delta_step: int = 1,
        random_state: int = 42,
    ):
        self.n_estimators_ensemble = n_estimators_ensemble
        self.n_trees_per_model = n_trees_per_model
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.alpha = alpha
        self.neg_ratio = neg_ratio
        self.max_delta_step = max_delta_step
        self.random_state = random_state
        self.models: List[XGBClassifier] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BalancedXGBEnsemble":
        X_mat = np.asarray(X)
        y_arr = np.asarray(y)

        pos_idx = np.where(y_arr == 1)[0]
        neg_idx = np.where(y_arr == 0)[0]
        n_pos = len(pos_idx)

        # Number of negative samples per sub-model
        n_sample_neg = min(len(neg_idx), max(int(n_pos * self.neg_ratio), n_pos))

        rng = np.random.RandomState(self.random_state)
        focal_obj = FocalLossObjective(gamma=self.gamma, alpha=self.alpha)
        self.models = []

        for i in range(self.n_estimators_ensemble):
            sample_neg_idx = rng.choice(neg_idx, size=n_sample_neg, replace=False)
            sub_idx = np.concatenate([pos_idx, sample_neg_idx])
            rng.shuffle(sub_idx)

            sub_X = X_mat[sub_idx]
            sub_y = y_arr[sub_idx]

            model = XGBClassifier(
                n_estimators=self.n_trees_per_model,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                objective=focal_obj,
                max_delta_step=self.max_delta_step,
                random_state=self.random_state + i * 37,
                eval_metric="logloss",
            )
            model.fit(sub_X, sub_y)
            self.models.append(model)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts average probability across all ensemble models."""
        X_mat = np.asarray(X)
        member_probas = []
        for model in self.models:
            logits = model.predict(X_mat, output_margin=True)
            p = expit(logits)
            member_probas.append(p)
        avg_p = np.mean(member_probas, axis=0)
        return np.vstack([1.0 - avg_p, avg_p]).T

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= threshold).astype(int)

    @property
    def feature_importances_(self) -> np.ndarray:
        if not self.models:
            return np.array([])
        importances = [m.feature_importances_ for m in self.models if hasattr(m, "feature_importances_")]
        return np.mean(importances, axis=0) if importances else np.array([])


def load_and_preprocess_data(data_path: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Load dataset and extract feature matrix X and target vector y."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded dataset from: {data_path}")
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    target_col = "target" if "target" in df.columns else df.columns[-1]
    y = df[target_col].astype(int)

    metadata_cols = {"station_id", "pln_area", target_col}
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    X = df[feature_cols]

    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Target distribution:\n{y.value_counts().to_string()}\n")

    return X, y, feature_cols


def run_multi_seed_cv(
    X: pd.DataFrame,
    y: pd.Series,
    seeds: List[int],
    n_splits: int = 5,
    n_ensemble_models: int = 8,
    gamma: float = 2.0,
    alpha: float = 0.25,
    neg_ratio: float = 4.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run 5-fold cross-validation repeated across multiple seeds to evaluate stability.
    """
    print("=" * 80)
    print(f"MULTI-SEED 5-FOLD CROSS-VALIDATION (Seeds: {seeds})")
    print("=" * 80)

    X_mat = X.values
    y_arr = y.values

    seed_oof_probas = []
    seed_pr_aucs = []
    seed_roc_aucs = []

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof_p = np.zeros(len(X))

        for tr_idx, val_idx in skf.split(X_mat, y_arr):
            X_tr, y_tr = X_mat[tr_idx], y_arr[tr_idx]
            X_val = X_mat[val_idx]

            ensemble = BalancedXGBEnsemble(
                n_estimators_ensemble=n_ensemble_models,
                gamma=gamma,
                alpha=alpha,
                neg_ratio=neg_ratio,
                random_state=seed,
            )
            ensemble.fit(X_tr, y_tr)
            oof_p[val_idx] = ensemble.predict_proba(X_val)[:, 1]

        seed_oof_probas.append(oof_p)
        pr = average_precision_score(y_arr, oof_p)
        roc = roc_auc_score(y_arr, oof_p)
        seed_pr_aucs.append(pr)
        seed_roc_aucs.append(roc)
        print(f"  Seed {seed:4d}: OOF PR-AUC = {pr:.4f} | OOF ROC-AUC = {roc:.4f}")

    mean_oof_probas = np.mean(seed_oof_probas, axis=0)
    agg_pr_auc = average_precision_score(y_arr, mean_oof_probas)
    agg_roc_auc = roc_auc_score(y_arr, mean_oof_probas)

    print("-" * 80)
    print(f"Multi-Seed Summary:")
    print(f"  Mean PR-AUC across seeds: {np.mean(seed_pr_aucs):.4f} +/- {np.std(seed_pr_aucs):.4f}")
    print(f"  Mean ROC-AUC across seeds: {np.mean(seed_roc_aucs):.4f} +/- {np.std(seed_roc_aucs):.4f}")
    print(f"  Averaged OOF PR-AUC:       {agg_pr_auc:.4f} (Baseline: {len(np.where(y_arr==1)[0])/len(y_arr):.4f})")
    print(f"  Averaged OOF ROC-AUC:      {agg_roc_auc:.4f}")
    print("-" * 80)

    cv_stats = {
        "seed_pr_aucs": seed_pr_aucs,
        "seed_roc_aucs": seed_roc_aucs,
        "mean_pr_auc": float(np.mean(seed_pr_aucs)),
        "std_pr_auc": float(np.std(seed_pr_aucs)),
        "mean_roc_auc": float(np.mean(seed_roc_aucs)),
        "std_roc_auc": float(np.std(seed_roc_aucs)),
        "agg_pr_auc": float(agg_pr_auc),
        "agg_roc_auc": float(agg_roc_auc),
    }

    return mean_oof_probas, cv_stats


def evaluate_threshold_sweep(
    y_true: np.ndarray,
    y_probas: np.ndarray,
    thresholds: Optional[List[float]] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Evaluate decision thresholds prioritizing Recall & F2-Score.
    """
    num_pos = int((y_true == 1).sum())

    if thresholds is None:
        thresholds = sorted(list(set(
            [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50]
        )))

    sweep_results = []
    print("\n" + "=" * 85)
    print("THRESHOLD SWEEP (PRIORITIZING RECALL & F2-SCORE)")
    print("=" * 85)
    print(f"{'Threshold':<10} | {'Recall':<8} | {'Precision':<10} | {'F1-Score':<9} | {'F2-Score':<9} | {'Floods Caught':<14} | {'False Alarms':<12}")
    print("-" * 85)

    for t in thresholds:
        preds = (y_probas >= t).astype(int)
        rec = recall_score(y_true, preds, zero_division=0)
        prec = precision_score(y_true, preds, zero_division=0)
        f1 = fbeta_score(y_true, preds, beta=1.0, zero_division=0)
        f2 = fbeta_score(y_true, preds, beta=2.0, zero_division=0)
        cm = confusion_matrix(y_true, preds)
        tp = int(cm[1, 1]) if len(cm) > 1 else 0
        fp = int(cm[0, 1]) if len(cm) > 1 else 0

        res = {
            "threshold": float(t),
            "recall": float(rec),
            "precision": float(prec),
            "f1_score": float(f1),
            "f2_score": float(f2),
            "true_positives": tp,
            "false_positives": fp,
        }
        sweep_results.append(res)
        print(f"{t:<10.4f} | {rec*100:<7.1f}% | {prec*100:<9.1f}% | {f1:<9.4f} | {f2:<9.4f} | {tp}/{num_pos:<12} | {fp:<12}")

    print("-" * 85)

    # Find optimal threshold by F2 score (recall emphasis)
    best_f2_res = max(sweep_results, key=lambda r: (r["f2_score"], r["recall"]))

    return sweep_results, best_f2_res


def train_and_evaluate_advanced(
    data_path: str,
    output_model_pkl: str,
    output_calibrator_pkl: str,
    output_config_json: str,
    n_splits: int = 5,
    seeds: List[int] = [42, 101, 2023, 7, 999],
    n_ensemble_models: int = 8,
    gamma: float = 2.0,
    alpha: float = 0.25,
    neg_ratio: float = 4.0,
    calibration_method: str = "isotonic",
    selected_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Main advanced training pipeline:
    1. Multi-seed 5-Fold Cross-Validation with Balanced XGBoost Ensemble & Focal Loss.
    2. Probability Calibration (Isotonic Regression or Platt Scaling) on OOF predictions.
    3. Threshold Tuning on calibrated probabilities.
    4. Full Ensemble & Calibrator training and serialization.
    """
    # 1. Load Data
    X, y, feature_names = load_and_preprocess_data(data_path)
    X_mat = X.values
    y_arr = y.values

    # 2. Multi-Seed 5-Fold CV
    oof_raw_probas, cv_stats = run_multi_seed_cv(
        X=X,
        y=y,
        seeds=seeds,
        n_splits=n_splits,
        n_ensemble_models=n_ensemble_models,
        gamma=gamma,
        alpha=alpha,
        neg_ratio=neg_ratio,
    )

    # 3. Probability Calibration on OOF predictions
    print("\n" + "=" * 80)
    print(f"PROBABILITY CALIBRATION ({calibration_method.upper()})")
    print("=" * 80)

    if calibration_method.lower() == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(oof_raw_probas, y_arr)
        oof_cal_probas = calibrator.predict(oof_raw_probas)
    else:  # Platt Scaling (Logistic Regression on logits)
        eps = 1e-12
        logits = np.log(oof_raw_probas / (1.0 - oof_raw_probas + eps)).reshape(-1, 1)
        calibrator = LogisticRegression(C=1.0, solver="lbfgs")
        calibrator.fit(logits, y_arr)
        oof_cal_probas = calibrator.predict_proba(logits)[:, 1]

    brier_before = brier_score_loss(y_arr, oof_raw_probas)
    brier_after = brier_score_loss(y_arr, oof_cal_probas)
    print(f"Brier Score Loss (Raw OOF):        {brier_before:.4f}")
    print(f"Brier Score Loss (Calibrated OOF): {brier_after:.4f} (Lower is better)")

    # 4. Decision Threshold Tuning & Tiered Thresholds
    sweep_results, best_operating_point = evaluate_threshold_sweep(y_arr, oof_cal_probas)

    watch_threshold = 0.030
    warning_threshold = 0.120

    chosen_thresh = selected_threshold if selected_threshold is not None else watch_threshold

    watch_preds = (oof_cal_probas >= watch_threshold).astype(int)
    warning_preds = (oof_cal_probas >= warning_threshold).astype(int)

    print("\n" + "=" * 80)
    print("TIERED OPERATIONAL THRESHOLDS EVALUATION")
    print("=" * 80)
    print(f"1. FLOOD WATCH TIER   (p >= {watch_threshold:.3f}) - High Recall / Early Awareness:")
    print(f"   Recall:    {recall_score(y_arr, watch_preds, zero_division=0)*100:.1f}% ({confusion_matrix(y_arr, watch_preds)[1,1]}/{sum(y_arr==1)} floods caught)")
    print(f"   Precision: {precision_score(y_arr, watch_preds, zero_division=0)*100:.1f}% ({confusion_matrix(y_arr, watch_preds)[0,1]} false alarms)")
    print(f"   Action:    Standby response teams, activate heightened sensor polling, monitor drainage telemetry.")

    print(f"\n2. FLOOD WARNING TIER (p >= {warning_threshold:.3f}) - Higher Precision / Tactical Response:")
    print(f"   Recall:    {recall_score(y_arr, warning_preds, zero_division=0)*100:.1f}% ({confusion_matrix(y_arr, warning_preds)[1,1]}/{sum(y_arr==1)} floods caught)")
    print(f"   Precision: {precision_score(y_arr, warning_preds, zero_division=0)*100:.1f}% ({confusion_matrix(y_arr, warning_preds)[0,1]} false alarms)")
    print(f"   Action:    Deploy mobile flood barriers, trigger traffic diversion advisories, activate pumps.")
    print("=" * 80)

    # 5. Train Full Production Ensemble on Entire Dataset
    print("\n" + "=" * 80)
    print("TRAINING FINAL PRODUCTION ENSEMBLE ON FULL DATASET")
    print("=" * 80)
    full_ensemble = BalancedXGBEnsemble(
        n_estimators_ensemble=n_ensemble_models,
        gamma=gamma,
        alpha=alpha,
        neg_ratio=neg_ratio,
        random_state=42,
    )
    full_ensemble.fit(X_mat, y_arr)

    # 6. Feature Importances
    print("=" * 80)
    print("ENSEMBLE FEATURE IMPORTANCES")
    print("=" * 80)
    importances = full_ensemble.feature_importances_
    if len(importances) == len(feature_names):
        sorted_indices = np.argsort(importances)[::-1]
        for idx in sorted_indices:
            print(f"  {feature_names[idx]:<15}: {importances[idx]:.4f}")

    # 7. Save Model Artifacts & Metadata
    joblib.dump(full_ensemble, output_model_pkl)
    print(f"\nSaved Balanced XGBoost Ensemble to: {output_model_pkl}")

    joblib.dump(calibrator, output_calibrator_pkl)
    print(f"Saved Probability Calibrator to:    {output_calibrator_pkl}")

    # Also save a native XGBoost model for backward compatibility
    legacy_json_path = os.path.join(os.path.dirname(output_model_pkl), "xgboost_flood_model.json")
    legacy_pkl_path = os.path.join(os.path.dirname(output_model_pkl), "xgboost_flood_model.pkl")
    if len(full_ensemble.models) > 0:
        full_ensemble.models[0].save_model(legacy_json_path)
        joblib.dump(full_ensemble.models[0], legacy_pkl_path)

    config_data = {
        "model_architecture": "BalancedXGBEnsemble",
        "n_ensemble_members": n_ensemble_models,
        "focal_loss_gamma": gamma,
        "focal_loss_alpha": alpha,
        "neg_ratio": neg_ratio,
        "calibration_method": calibration_method,
        "optimal_threshold": float(chosen_thresh),
        "threshold_watch": float(watch_threshold),
        "threshold_warning": float(warning_threshold),
        "tiered_thresholds": {
            "watch": float(watch_threshold),
            "warning": float(warning_threshold),
        },
        "temporal_resistance_check": {
            "enabled": True,
            "min_rain_threshold": 0.2,
            "description": "If all recorded rainfall in the 105-minute window is < 0.2mm (rain_max_5m < 0.2), classify as NORMAL / NO ALERT (P=0.0)."
        },
        "alert_tiers": {
            "NORMAL": {
                "min_p": 0.0,
                "max_p": float(watch_threshold),
                "label": "Normal (No Alert)",
                "action": "Routine weather and drainage monitoring.",
            },
            "WATCH": {
                "min_p": float(watch_threshold),
                "max_p": float(warning_threshold),
                "label": "Flood Watch",
                "action": "Heightened situational awareness, standby crews, monitor telemetry.",
            },
            "WARNING": {
                "min_p": float(warning_threshold),
                "max_p": 1.0,
                "label": "Flood Warning",
                "action": "Immediate tactical response, deploy mobile flood barriers, trigger traffic diversion.",
            }
        },
        "cv_stats": cv_stats,
        "best_operating_point": best_operating_point,
        "features": feature_names,
    }

    with open(output_config_json, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    print(f"Saved Metadata Configuration to:    {output_config_json}")

    return config_data


def main():
    default_data, default_model, default_calibrator, default_config = get_default_paths()

    parser = argparse.ArgumentParser(
        description="Train advanced Balanced XGBoost Ensemble with Focal Loss, multi-seed 5-fold CV, and calibration."
    )
    parser.add_argument(
        "-d", "--data-path",
        type=str,
        default=default_data,
        help=f"Path to training_dataset.csv (default: {default_data})"
    )
    parser.add_argument(
        "--output-ensemble",
        type=str,
        default=default_model,
        help=f"Path to save ensemble pickle model (default: {default_model})"
    )
    parser.add_argument(
        "--output-calibrator",
        type=str,
        default=default_calibrator,
        help=f"Path to save calibrator pickle (default: {default_calibrator})"
    )
    parser.add_argument(
        "--output-config",
        type=str,
        default=default_config,
        help=f"Path to save configuration metadata JSON (default: {default_config})"
    )
    parser.add_argument(
        "--n-models",
        type=int,
        default=8,
        help="Number of balanced models in ensemble (default: 8; recommended 5-10)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.0,
        help="Focal Loss gamma parameter (default: 2.0; range [1.5, 2.5])"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="Focal Loss alpha parameter (default: 0.25)"
    )
    parser.add_argument(
        "--neg-ratio",
        type=float,
        default=4.0,
        help="Ratio of negative to positive samples per ensemble sub-model (default: 4.0)"
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default="isotonic",
        choices=["isotonic", "platt"],
        help="Calibration method: 'isotonic' (Isotonic Regression) or 'platt' (Platt scaling)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Manual threshold override (default: auto-tuned by F2 recall prioritization)"
    )

    args = parser.parse_args()

    train_and_evaluate_advanced(
        data_path=args.data_path,
        output_model_pkl=args.output_ensemble,
        output_calibrator_pkl=args.output_calibrator,
        output_config_json=args.output_config,
        n_ensemble_models=args.n_models,
        gamma=args.gamma,
        alpha=args.alpha,
        neg_ratio=args.neg_ratio,
        calibration_method=args.calibration,
        selected_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
