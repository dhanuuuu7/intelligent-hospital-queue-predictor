"""
Step 8.6.12 Testing: Model Explainability, Feature Attribution & High-Error Analysis Tests
"""

import os
import pytest
import pandas as pd
import numpy as np

from models.explainability import ModelExplainer
from models.explanation_text import generate_human_explanation


@pytest.fixture
def explainer_instance():
    return ModelExplainer()


def test_explainer_loads_model_and_maps_features(explainer_instance):
    assert explainer_instance.pipeline is not None
    assert len(explainer_instance.all_feature_names) == 33  # 13 cat one-hot + 20 num
    assert "effective_doctor_load" in explainer_instance.all_feature_names
    assert "traffic_momentum" in explainer_instance.all_feature_names


def test_global_feature_importance_generation(explainer_instance):
    fi_df = explainer_instance.get_global_feature_importance()
    assert isinstance(fi_df, pd.DataFrame)
    assert len(fi_df) == 33
    assert "Feature" in fi_df.columns
    assert "Importance" in fi_df.columns
    assert abs(fi_df["Importance"].sum() - 1.0) < 1e-3


def test_single_prediction_explanation(explainer_instance):
    sample_patient = {
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
        "hospital_occupancy_rate": 0.70
    }

    res = explainer_instance.explain_single_prediction(sample_patient)
    assert "predicted_waiting_time_min" in res
    assert "explanation" in res
    exp = res["explanation"]
    assert "summary" in exp
    assert "increasing_factors" in exp
    assert "decreasing_factors" in exp
    assert "patient_bullets" in exp
    assert isinstance(exp["increasing_factors"], list)
    assert isinstance(exp["decreasing_factors"], list)


def test_department_analysis_and_artifacts(explainer_instance):
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "hospital_encounters.csv")
    from models.data_loader import load_and_validate_dataset
    from models.feature_engineering import add_engineered_features
    
    df = load_and_validate_dataset(data_path)
    eng_df = add_engineered_features(df)

    dept_df, error_df = explainer_instance.analyze_department_performance(eng_df.head(500))
    assert len(dept_df) == 4
    assert len(error_df) == 4
    
    artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "models", "artifacts")
    assert os.path.exists(os.path.join(artifacts_dir, "department_model_analysis.csv"))
    assert os.path.exists(os.path.join(artifacts_dir, "department_error_analysis.csv"))
