"""
Unit tests for Step 5.5 Prediction Interface
"""

import os
import pytest
from models.predict import predict_waiting_time, WaitingTimePredictor


def test_prediction_function():
    sample_patient = {
        "department_name": "General Medicine",
        "day_of_week": 1,
        "hour_of_day": 10,
        "is_weekend_or_holiday": 0,
        "season_indicator": "Winter",
        "queue_position": 5,
        "scheduled_patients_ahead": 3,
        "walkin_patients_ahead": 2,
        "patient_arrival_type": "Scheduled",
        "patient_case_complexity": "Low",
        "active_doctors_count": 4,
        "doctors_on_break_count": 0,
        "triage_staff_count": 2,
        "shift_handover_flag": 0,
        "emergency_cases_active": 0,
        "arrival_rate_past_30min": 10.0,
        "recent_throughput_past_hour": 20,
        "avg_waiting_time_last_30min": 12.0,
        "dept_service_time_variability": 3.0,
        "hospital_occupancy_rate": 0.65
    }

    predicted_min = predict_waiting_time(sample_patient)
    assert isinstance(predicted_min, float)
    assert predicted_min >= 0.0
    assert predicted_min < 300.0  # Realistic bound
