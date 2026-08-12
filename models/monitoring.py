"""
Step 8.6.7 & 8.6.8: Production Monitoring & Statistical Drift Detection Engine
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


def compute_psi(baseline: np.ndarray, target: np.ndarray, num_bins: int = 10) -> float:
    """
    Computes Population Stability Index (PSI) between baseline and production feature distributions.
    """
    b_valid = baseline[~np.isnan(baseline)]
    t_valid = target[~np.isnan(target)]

    if len(b_valid) < 5 or len(t_valid) < 5:
        return 0.0

    # Define bin edges from baseline distribution
    quantiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(b_valid, quantiles)
    bins = np.unique(bins)

    if len(bins) < 2:
        return 0.0

    bins[0] = -np.inf
    bins[-1] = np.inf

    b_counts, _ = np.histogram(b_valid, bins=bins)
    t_counts, _ = np.histogram(t_valid, bins=bins)

    b_pct = b_counts / len(b_valid)
    t_pct = t_counts / len(t_valid)

    # Avoid zero division
    b_pct = np.where(b_pct == 0, 1e-4, b_pct)
    t_pct = np.where(t_pct == 0, 1e-4, t_pct)

    psi_val = np.sum((t_pct - b_pct) * np.log(t_pct / b_pct))
    return float(psi_val)


class ProductionMonitor:
    def __init__(self, warning_threshold: float = 0.10, critical_threshold: float = 0.25):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    def evaluate_production_batch(
        self, 
        production_df: pd.DataFrame, 
        baseline_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Evaluates production batch metrics, distribution drift (PSI), and confidence label breakdown.
        """
        n_prod = len(production_df)
        if n_prod == 0:
            return {"status": "HEALTHY", "prediction_count": 0, "message": "No production data"}

        # Basic summary statistics
        pred_col = "pred_wait" if "pred_wait" in production_df.columns else "predicted_waiting_time_min"
        mean_pred = float(production_df[pred_col].mean()) if pred_col in production_df.columns else 0.0
        med_pred = float(production_df[pred_col].median()) if pred_col in production_df.columns else 0.0

        width_col = "interval_width_min"
        mean_width = float(production_df[width_col].mean()) if width_col in production_df.columns else 28.5

        # Label distributions
        label_dist = {}
        if "confidence_label" in production_df.columns:
            label_dist = (production_df["confidence_label"].value_counts(normalize=True) * 100).round(2).to_dict()

        dept_dist = (production_df["department_name"].value_counts(normalize=True) * 100).round(2).to_dict()

        # Drift detection across numerical features
        drift_alerts = []
        max_psi = 0.0

        check_cols = ["hour_of_day", "queue_position", "active_doctors_count", "effective_doctor_load", "traffic_momentum"]
        for col in check_cols:
            if col in production_df.columns and col in baseline_df.columns:
                psi = compute_psi(baseline_df[col].values, production_df[col].values)
                max_psi = max(max_psi, psi)

                if psi > self.critical_threshold:
                    drift_alerts.append({
                        "feature": col,
                        "metric": "PSI",
                        "value": round(psi, 4),
                        "severity": "CRITICAL"
                    })
                elif psi > self.warning_threshold:
                    drift_alerts.append({
                        "feature": col,
                        "metric": "PSI",
                        "value": round(psi, 4),
                        "severity": "WARNING"
                    })

        status = "CRITICAL" if max_psi > self.critical_threshold else ("WARNING" if max_psi > self.warning_threshold else "HEALTHY")

        return {
            "status": status,
            "prediction_count": n_prod,
            "mean_predicted_wait": round(mean_pred, 2),
            "median_predicted_wait": round(med_pred, 2),
            "mean_interval_width": round(mean_width, 2),
            "confidence_label_dist_pct": label_dist,
            "department_dist_pct": dept_dist,
            "max_feature_psi": round(max_psi, 4),
            "drift_alerts": drift_alerts
        }


def run_monitoring_simulation():
    from models.data_loader import load_and_validate_dataset
    from models.feature_engineering import add_engineered_features
    from models.train_fe import get_chronological_splits_fe

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "hospital_encounters.csv")
    raw_df = load_and_validate_dataset(data_path)
    eng_df = add_engineered_features(raw_df)
    X_train, y_train, X_val, y_val, X_test, y_test = get_chronological_splits_fe(eng_df)

    # Baseline dataset = Training Set
    baseline_df = eng_df.iloc[:len(X_train)].copy()

    # Simulate normal production batch (Test Set)
    monitor = ProductionMonitor()
    normal_prod_df = eng_df.iloc[len(X_train) + len(X_val):].copy()
    normal_prod_df["predicted_waiting_time_min"] = 18.4
    normal_prod_df["interval_width_min"] = 28.5
    normal_prod_df["confidence_label"] = "MODERATE"

    print("\n" + "=" * 75)
    print("STEP 8.6.7 & 8.6.8 — PRODUCTION MONITORING REPORT (Normal Batch)")
    print("=" * 75)
    report_normal = monitor.evaluate_production_batch(normal_prod_df, baseline_df)
    print(json.dumps(report_normal, indent=4))

    # Simulate shifted surge production batch (Drifted Shift)
    drifted_prod_df = normal_prod_df.copy()
    drifted_prod_df["hour_of_day"] = np.random.choice([17, 18, 19], size=len(drifted_prod_df))
    drifted_prod_df["effective_doctor_load"] = drifted_prod_df["effective_doctor_load"] * 3.5

    print("\n" + "=" * 75)
    print("STEP 8.6.8 — PRODUCTION MONITORING REPORT (Drifted Peak Hour Batch)")
    print("=" * 75)
    report_drifted = monitor.evaluate_production_batch(drifted_prod_df, baseline_df)
    print(json.dumps(report_drifted, indent=4))


if __name__ == "__main__":
    run_monitoring_simulation()
