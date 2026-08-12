"""
Step 8.5 Testing: Prediction Intervals & Uncertainty Estimation Unit & Integration Tests
"""

import os
import pytest
import json
import numpy as np
import pandas as pd

from models.prediction_intervals import PredictionIntervalCalibrator, calculate_interval_metrics
from models.predict import WaitingTimePredictor, predict_waiting_time


@pytest.fixture
def calibrator_instance():
    # Fit calibrator on synthetic validation errors
    val_y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    val_preds = np.array([12.0, 18.0, 35.0, 38.0, 60.0])
    
    cal = PredictionIntervalCalibrator(metadata_path=None)
    cal.fit(val_y, val_preds)
    return cal


def test_calibrator_fits_on_validation_only(calibrator_instance):
    assert calibrator_instance.is_calibrated
    assert "0.8" in calibrator_instance.residual_quantiles
    assert "0.9" in calibrator_instance.residual_quantiles
    assert "0.95" in calibrator_instance.residual_quantiles


def test_interval_non_negativity_and_bounds(calibrator_instance):
    # Test zero/small point prediction edge case
    res_zero = calibrator_instance.predict_interval(point_prediction=0.0, confidence_level=0.90)
    assert res_zero["lower_bound_min"] >= 0.0, "Lower bound must never be negative!"
    assert res_zero["lower_bound_min"] <= res_zero["predicted_waiting_time_min"]
    assert res_zero["predicted_waiting_time_min"] <= res_zero["upper_bound_min"]
    assert res_zero["lower_bound_min"] <= res_zero["upper_bound_min"]

    # Test normal point prediction
    res_normal = calibrator_instance.predict_interval(point_prediction=25.0, confidence_level=0.90)
    assert res_normal["lower_bound_min"] >= 0.0
    assert res_normal["lower_bound_min"] <= 25.0
    assert 25.0 <= res_normal["upper_bound_min"]
    assert res_normal["interval_width_min"] == round(res_normal["upper_bound_min"] - res_normal["lower_bound_min"], 1)
    assert res_normal["confidence_label"] in ["HIGH", "MODERATE", "LOW"]


def test_metadata_persistence_and_loading(tmp_path):
    meta_file = os.path.join(tmp_path, "test_metadata.json")
    cal = PredictionIntervalCalibrator(metadata_path=meta_file)
    cal.fit(np.array([10, 20, 30]), np.array([11, 19, 32]))

    assert os.path.exists(meta_file)
    
    # Reload in new calibrator instance
    cal2 = PredictionIntervalCalibrator(metadata_path=meta_file)
    assert cal2.is_calibrated
    assert "0.9" in cal2.residual_quantiles


def test_predictor_returns_prediction_interval():
    predictor = WaitingTimePredictor()
    sample = {
        "department_name": "General Medicine",
        "day_of_week": 1,
        "hour_of_day": 10,
        "is_weekend_or_holiday": 0,
        "season_indicator": "Winter",
        "queue_position": 5,
        "scheduled_patients_ahead": 3,
        "walkin_patients_ahead": 2,
        "patient_arrival_type": "Scheduled",
        "patient_case_complexity": "Medium",
        "active_doctors_count": 4,
        "doctors_on_break_count": 0,
        "triage_staff_count": 2,
        "shift_handover_flag": 0,
        "emergency_cases_active": 0,
        "arrival_rate_past_30min": 10.0,
        "recent_throughput_past_hour": 15,
        "avg_waiting_time_last_30min": 12.0,
        "dept_service_time_variability": 3.0,
        "hospital_occupancy_rate": 0.65
    }

    res = predictor.predict_encounter_with_interval(sample, confidence_level=0.90)
    assert "predicted_waiting_time_min" in res
    assert "lower_bound_min" in res
    assert "upper_bound_min" in res
    assert "interval_width_min" in res
    assert "confidence_level" in res
    assert "confidence_label" in res
    assert res["lower_bound_min"] >= 0.0
    assert res["lower_bound_min"] <= res["predicted_waiting_time_min"] <= res["upper_bound_min"]
