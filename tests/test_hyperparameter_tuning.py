"""
Step 8.4.9: Unit & Integration Tests for Hyperparameter Tuning Pipeline & Final Artifacts
"""

import os
import pytest
import joblib
import pandas as pd
import numpy as np

from models.data_loader import load_and_validate_dataset, TARGET_COLUMN
from models.feature_engineering import add_engineered_features, ALL_EXPANDED_PREDICTIVE_FEATURES


@pytest.fixture
def artifacts_dir():
    return os.path.join(os.path.dirname(__file__), "..", "models", "artifacts")


def test_tuning_results_files_exist(artifacts_dir):
    xgb_csv = os.path.join(artifacts_dir, "xgboost_tuning_results.csv")
    rf_csv = os.path.join(artifacts_dir, "random_forest_tuning_results.csv")

    assert os.path.exists(xgb_csv), "xgboost_tuning_results.csv is missing!"
    assert os.path.exists(rf_csv), "random_forest_tuning_results.csv is missing!"

    xgb_df = pd.read_csv(xgb_csv)
    rf_df = pd.read_csv(rf_csv)

    assert len(xgb_df) > 0
    assert len(rf_df) > 0

    # Ensure validation metrics exist in results
    for col in ["Val_MAE", "Val_RMSE", "Val_R2", "Train_MAE"]:
        assert col in xgb_df.columns
        assert col in rf_df.columns

    # Verify no test metrics were recorded in tuning logs
    for col in ["Test_MAE", "Test_RMSE", "Test_R2"]:
        assert col not in xgb_df.columns, f"Tuning log should not contain test metric {col}"
        assert col not in rf_df.columns, f"Tuning log should not contain test metric {col}"


def test_final_model_artifact_load_and_predict(artifacts_dir):
    final_model_path = os.path.join(artifacts_dir, "final_waiting_time_model.joblib")
    assert os.path.exists(final_model_path), "final_waiting_time_model.joblib missing!"

    pipeline = joblib.load(final_model_path)

    # Test prediction on a sample 24-feature dataframe
    sample_df = pd.DataFrame([{
        "department_name": "General Medicine",
        "day_of_week": 1,
        "hour_of_day": 10,
        "is_weekend_or_holiday": 0,
        "season_indicator": "Winter",
        "queue_position": 8,
        "scheduled_patients_ahead": 5,
        "walkin_patients_ahead": 3,
        "patient_arrival_type": "Scheduled",
        "patient_case_complexity": "Medium",
        "active_doctors_count": 4,
        "doctors_on_break_count": 0,
        "triage_staff_count": 2,
        "shift_handover_flag": 0,
        "emergency_cases_active": 0,
        "arrival_rate_past_30min": 12.0,
        "recent_throughput_past_hour": 18,
        "avg_waiting_time_last_30min": 14.5,
        "dept_service_time_variability": 3.0,
        "hospital_occupancy_rate": 0.70,
        "effective_doctor_load": 2.0,
        "traffic_momentum": 0.63,
        "emergency_pressure": 0.0,
        "scheduled_ratio_ahead": 0.55
    }])

    preds = pipeline.predict(sample_df)
    assert len(preds) == 1
    assert isinstance(preds[0], (float, np.floating))
    assert not np.isnan(preds[0])
    assert not np.isinf(preds[0])
    assert preds[0] >= 0.0
