"""
Phase 12 Automated Testing Suite: Database, WebSockets, Analytics Forecasting, ML Feedback Loop, and Security Audit
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.db.database import init_db, SessionLocal
from app.db.models import QueueTicketDB, PredictionFeedbackDB, AuditLogDB
from app.feedback_loop import record_prediction_outcome, evaluate_continuous_feedback_metrics
from app.analytics import detect_department_bottlenecks, forecast_future_queue, generate_ai_staff_recommendation


@pytest.fixture(scope="module")
def client():
    init_db()
    return TestClient(app)


def test_database_initialization_and_orm():
    db = SessionLocal()
    try:
        # Insert test audit entry
        audit = AuditLogDB(user_email="admin@hospital.com", action="TEST_INIT", details="Initializing DB test")
        db.add(audit)
        db.commit()

        queried = db.query(AuditLogDB).filter(AuditLogDB.user_email == "admin@hospital.com").first()
        assert queried is not None
        assert queried.action == "TEST_INIT"
    finally:
        db.close()


def test_analytics_bottleneck_detection_endpoint(client):
    response = client.get("/analytics/bottlenecks")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 4
    assert "is_bottleneck" in data[0]
    assert "risk_level" in data[0]


def test_analytics_future_queue_forecast_endpoint(client):
    response = client.get("/analytics/forecast/Pediatrics")
    assert response.status_code == 200
    data = response.json()
    assert data["department"] == "Pediatrics"
    assert "current" in data
    assert "forecast_15m" in data
    assert "forecast_30m" in data
    assert "forecast_60m" in data
    assert "trend_direction" in data


def test_ai_staff_recommendation_endpoint(client):
    response = client.get("/analytics/recommendation/Cardiology")
    assert response.status_code == 200
    data = response.json()
    assert data["department"] == "Cardiology"
    assert "action_type" in data
    assert "expected_impact" in data
    assert "reasoning" in data


def test_ml_feedback_loop_recording_and_metrics(client):
    # Record test feedback
    rec_res = client.post("/feedback/record?ticket_code=CARD-999&department_name=Cardiology&predicted_wait_min=30.0&actual_wait_min=35.0")
    assert rec_res.status_code == 200
    rec_data = rec_res.json()
    assert rec_data["error_min"] == 5.0

    # Query metrics
    metrics_res = client.get("/feedback/metrics?department_name=Cardiology")
    assert metrics_res.status_code == 200
    m_data = metrics_res.json()
    assert m_data["evaluations_count"] >= 1
    assert "continuous_mae_min" in m_data
    assert "drift_status" in m_data


def test_audit_logs_endpoint(client):
    response = client.get("/audit/logs?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
