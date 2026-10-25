"""
Phase 13.1: Autonomous Self-Healing MLOps & Automated Retraining Engine
Triggered when feature drift (PSI > 0.25) or continuous error (MAE > 12.5m) is detected.
"""

import os
import json
import joblib
import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any

from models.data_loader import load_and_validate_dataset, TARGET_COLUMN
from models.feature_engineering import add_engineered_features
from models.train_fe import get_chronological_splits_fe, train_feature_engineered_rf
from models.model_quality_gate import evaluate_model_quality_gate


def trigger_autonomous_retraining(data_path: str = None) -> Dict[str, Any]:
    """
    Executes end-to-end automated retraining, quality gate validation, and zero-downtime hot swap.
    """
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "hospital_encounters.csv")

    print("\n" + "=" * 75)
    print("PHASE 13.1 — AUTONOMOUS MLOPS SELF-HEALING MODEL RETRAINING")
    print("=" * 75)

    # 1. Load dataset & engineered features
    raw_df = load_and_validate_dataset(data_path)
    eng_df = add_engineered_features(raw_df)
    X_train, y_train, X_val, y_val, X_test, y_test = get_chronological_splits_fe(eng_df)

    # 2. Retrain tuned Random Forest pipeline
    print("\n[MLOps Auto-Retrain] Training updated Random Forest model pipeline...")
    rf_pipeline, val_metrics = train_feature_engineered_rf(X_train, y_train, X_val, y_val)

    # 3. Evaluate against Automated Quality Gate
    test_preds = rf_pipeline.predict(X_test)
    test_mae = float(np.mean(np.abs(y_test.values - test_preds)))
    test_rmse = float(np.sqrt(np.mean((y_test.values - test_preds) ** 2)))
    
    ss_tot = np.sum((y_test.values - np.mean(y_test.values)) ** 2)
    ss_res = np.sum((y_test.values - test_preds) ** 2)
    test_r2 = float(1.0 - (ss_res / ss_tot))

    gate_result = evaluate_model_quality_gate(
        val_mae=val_metrics["val_mae"],
        test_mae=test_mae,
        test_rmse=test_rmse,
        test_r2=test_r2
    )

    if gate_result["passed"]:
        # 4. Export updated model artifact & model registry entry
        artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        
        model_path = os.path.join(artifacts_dir, "final_waiting_time_model.joblib")
        joblib.dump(rf_pipeline, model_path)

        registry_path = os.path.join(artifacts_dir, "model_registry.json")
        new_version = f"1.1.{int(datetime.datetime.utcnow().timestamp()) % 1000}"

        registry_data = {
            "active_model_version": new_version,
            "algorithm": "Tuned Random Forest Regressor (Self-Healed)",
            "updated_at": datetime.datetime.utcnow().isoformat(),
            "val_mae": val_metrics["val_mae"],
            "test_mae": round(test_mae, 4),
            "test_rmse": round(test_rmse, 4),
            "test_r2": round(test_r2, 4),
            "quality_gate_status": "PASS"
        }

        with open(registry_path, "w") as f:
            json.dump(registry_data, f, indent=4)

        print(f"[MLOps Auto-Retrain] Hot-swap artifact deployed successfully: Version {new_version}")

        return {
            "status": "SUCCESS",
            "action": "HOT_SWAP_DEPLOYED",
            "new_version": new_version,
            "quality_gate": gate_result,
            "test_mae": round(test_mae, 4),
            "test_r2": round(test_r2, 4)
        }
    else:
        print("[MLOps Auto-Retrain] Quality gate rejected retrained candidate. Keeping previous model.")
        return {
            "status": "REJECTED",
            "action": "REVERTED_TO_STABLE_MODEL",
            "quality_gate": gate_result
        }


if __name__ == "__main__":
    trigger_autonomous_retraining()
