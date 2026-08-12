"""
Step 9.16: End-to-End (E2E) Pipeline Integration Tests
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_full_end_to_end_prediction_flow(client):
    payload = {
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

    # 1. POST /predict request
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}: {response.text}"

    data = response.json()

    # 2. Check point prediction
    assert "predicted_waiting_time_min" in data
    assert isinstance(data["predicted_waiting_time_min"], (float, int))
    assert data["predicted_waiting_time_min"] >= 0.0

    # 3. Check prediction interval
    assert "lower_bound_min" in data
    assert "upper_bound_min" in data
    assert "interval_width_min" in data
    assert "confidence_level" in data
    assert "confidence_label" in data

    assert data["lower_bound_min"] >= 0.0
    assert data["lower_bound_min"] <= data["predicted_waiting_time_min"]
    assert data["predicted_waiting_time_min"] <= data["upper_bound_min"]
    assert data["confidence_level"] == 0.90
    assert data["confidence_label"] in ["HIGH", "MODERATE", "LOW"]

    # 4. Check explanation
    assert "explanation" in data
    exp = data["explanation"]
    assert "summary" in exp
    assert "increasing_factors" in exp
    assert "decreasing_factors" in exp

    # 5. Check metadata
    assert data["model_version"] == "1.0.0"
    assert data["department_name"] == "General Medicine"
    assert data["queue_position"] == 8
