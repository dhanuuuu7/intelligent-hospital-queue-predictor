"""
Step 10.13: Production Deployment Smoke Tests
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings


@pytest.fixture
def client():
    return TestClient(app)


def test_production_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["version"] == settings.MODEL_VERSION


def test_production_readiness_endpoint(client):
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True


def test_production_prediction_endpoint_full_contract(client):
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

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Verify contracts
    assert data["predicted_waiting_time_min"] >= 0.0
    assert data["lower_bound_min"] <= data["predicted_waiting_time_min"] <= data["upper_bound_min"]
    assert data["confidence_label"] in ["HIGH", "MODERATE", "LOW"]
    assert "explanation" in data
    assert data["explanation"] is not None
    assert data["model_version"] == settings.MODEL_VERSION
