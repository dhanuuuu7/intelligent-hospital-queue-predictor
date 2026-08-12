"""
Step 8.6.12 Testing: Production Monitoring & Statistical Drift Detection Tests
"""

import pytest
import numpy as np
import pandas as pd

from models.monitoring import ProductionMonitor, compute_psi


def test_compute_psi_identical_distributions():
    base = np.random.normal(10, 2, 1000)
    target = np.random.normal(10, 2, 1000)
    psi = compute_psi(base, target)
    assert psi < 0.10, f"PSI should be near zero for identical distributions, got {psi}"


def test_compute_psi_drifted_distributions():
    base = np.random.normal(10, 2, 1000)
    target = np.random.normal(25, 2, 1000)  # Significant mean shift
    psi = compute_psi(base, target)
    assert psi > 0.25, f"PSI should be large for shifted distributions, got {psi}"


def test_production_monitor_evaluation():
    base_df = pd.DataFrame({
        "hour_of_day": np.random.choice(range(8, 17), 500),
        "queue_position": np.random.randint(1, 15, 500),
        "active_doctors_count": np.random.randint(2, 6, 500),
        "effective_doctor_load": np.random.uniform(0.2, 2.0, 500),
        "traffic_momentum": np.random.uniform(0.1, 0.8, 500),
        "department_name": np.random.choice(["General Medicine", "Pediatrics"], 500)
    })

    # Normal production batch
    prod_df = base_df.copy()
    prod_df["predicted_waiting_time_min"] = 18.0
    prod_df["interval_width_min"] = 24.0
    prod_df["confidence_label"] = "MODERATE"

    monitor = ProductionMonitor(warning_threshold=0.10, critical_threshold=0.25)
    report = monitor.evaluate_production_batch(prod_df, base_df)

    assert report["status"] == "HEALTHY"
    assert report["prediction_count"] == 500
    assert report["mean_predicted_wait"] == 18.0
    assert len(report["drift_alerts"]) == 0

    # Drifted production batch
    drift_df = prod_df.copy()
    drift_df["hour_of_day"] = 23  # Extreme shift
    drift_df["effective_doctor_load"] = 8.0  # Extreme load shift

    drift_report = monitor.evaluate_production_batch(drift_df, base_df)
    assert drift_report["status"] in ["WARNING", "CRITICAL"]
    assert len(drift_report["drift_alerts"]) > 0
