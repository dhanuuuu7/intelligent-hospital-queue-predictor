"""
Step 10.12: Automated Model Quality Gate Validation Engine
Evaluates candidate model artifacts against strict operational performance thresholds.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

from models.data_loader import load_and_validate_dataset, TARGET_COLUMN
from models.feature_engineering import add_engineered_features, ALL_EXPANDED_PREDICTIVE_FEATURES
from models.train_fe import get_chronological_splits_fe
from models.train import calculate_metrics

QUALITY_GATE_THRESHOLDS = {
    "max_val_mae": 12.0,      # MAE must be <= 12.0 min
    "max_val_rmse": 55.0,     # RMSE must be <= 55.0 min
    "min_val_r2": 0.10,       # Validation R2 must be >= 0.10
    "max_test_mae": 11.5,     # Test MAE must be <= 11.5 min
    "max_test_rmse": 50.0,    # Test RMSE must be <= 50.0 min
    "min_test_r2": 0.15,      # Test R2 must be >= 0.15
}


def evaluate_model_quality_gate(model_path: str = None) -> Tuple[bool, Dict[str, Any]]:
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "artifacts", "final_waiting_time_model.joblib")
        if not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(__file__), "artifacts", "best_waiting_time_model.joblib")

    if not os.path.exists(model_path):
        return False, {"error": f"Model artifact not found at {model_path}"}

    print("\n" + "=" * 70)
    print("STEP 10.12 — AUTOMATED MODEL QUALITY GATE VALIDATION")
    print("=" * 70)

    # 1. Artifact Loadability Check
    try:
        pipeline = joblib.load(model_path)
        load_success = True
    except Exception as e:
        return False, {"error": f"Failed to load model artifact: {e}"}

    # 2. Evaluate Performance Metrics on Chronological Splits
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "hospital_encounters.csv")
    raw_df = load_and_validate_dataset(data_path)
    eng_df = add_engineered_features(raw_df)
    X_train, y_train, X_val, y_val, X_test, y_test = get_chronological_splits_fe(eng_df)

    val_preds = pipeline.predict(X_val)
    test_preds = pipeline.predict(X_test)

    val_metrics = calculate_metrics(y_val.values, val_preds)
    test_metrics = calculate_metrics(y_test.values, test_preds)

    checks = {
        "val_mae_pass": val_metrics["MAE"] <= QUALITY_GATE_THRESHOLDS["max_val_mae"],
        "val_rmse_pass": val_metrics["RMSE"] <= QUALITY_GATE_THRESHOLDS["max_val_rmse"],
        "val_r2_pass": val_metrics["R2"] >= QUALITY_GATE_THRESHOLDS["min_val_r2"],
        "test_mae_pass": test_metrics["MAE"] <= QUALITY_GATE_THRESHOLDS["max_test_mae"],
        "test_rmse_pass": test_metrics["RMSE"] <= QUALITY_GATE_THRESHOLDS["max_test_rmse"],
        "test_r2_pass": test_metrics["R2"] >= QUALITY_GATE_THRESHOLDS["min_test_r2"],
    }

    all_pass = all(checks.values()) and load_success
    status_str = "PASS" if all_pass else "FAIL"

    report = {
        "status": status_str,
        "model_artifact": os.path.basename(model_path),
        "thresholds": QUALITY_GATE_THRESHOLDS,
        "evaluated_val_metrics": {
            "val_mae": round(val_metrics["MAE"], 4),
            "val_rmse": round(val_metrics["RMSE"], 4),
            "val_r2": round(val_metrics["R2"], 4)
        },
        "evaluated_test_metrics": {
            "test_mae": round(test_metrics["MAE"], 4),
            "test_rmse": round(test_metrics["RMSE"], 4),
            "test_r2": round(test_metrics["R2"], 4)
        },
        "quality_checks": checks
    }

    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    report_path = os.path.join(artifacts_dir, "model_quality_gate_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"Quality Gate Status: {status_str}")
    print(f"Val MAE : {val_metrics['MAE']:.4f} (Max: {QUALITY_GATE_THRESHOLDS['max_val_mae']}) -> {'PASS' if checks['val_mae_pass'] else 'FAIL'}")
    print(f"Test MAE: {test_metrics['MAE']:.4f} (Max: {QUALITY_GATE_THRESHOLDS['max_test_mae']}) -> {'PASS' if checks['test_mae_pass'] else 'FAIL'}")
    print(f"Test RMSE: {test_metrics['RMSE']:.4f} (Max: {QUALITY_GATE_THRESHOLDS['max_test_rmse']}) -> {'PASS' if checks['test_rmse_pass'] else 'FAIL'}")
    print(f"Test R2 : {test_metrics['R2']:.4f} (Min: {QUALITY_GATE_THRESHOLDS['min_test_r2']}) -> {'PASS' if checks['test_r2_pass'] else 'FAIL'}")
    print(f"Report saved to: {report_path}")

    return all_pass, report


if __name__ == "__main__":
    evaluate_model_quality_gate()
