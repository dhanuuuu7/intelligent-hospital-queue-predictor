"""
Unit and Integration Tests for FastAPI Endpoints
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()


def test_health_check_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert "model_name" in data


def test_predict_endpoint():
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

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_waiting_time_min" in data
        assert isinstance(data["predicted_waiting_time_min"], float)
        assert data["predicted_waiting_time_min"] >= 0.0
        assert data["department_name"] == "General Medicine"


def test_queue_status_endpoint():
    with TestClient(app) as client:
        response = client.get("/queue-status?department_name=Pediatrics")
        assert response.status_code == 200
        data = response.json()
        assert data["department_name"] == "Pediatrics"
        assert "estimated_waiting_time_min" in data
        assert data["crowd_level"] in ["LOW", "MEDIUM", "HIGH", "VERY HIGH"]
