"""
Phase 13 Automated Testing Suite: Self-Healing MLOps Retraining, SMS Alerts, Regional Fleet Rerouting, and Executive Reports
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.notifications import send_queue_position_sms
from app.fleet import get_regional_network_status, recommend_patient_rerouting
from app.reports import generate_executive_daily_summary


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_patient_sms_alert_endpoint(client):
    response = client.post("/notifications/send-sms?op_number=OP202600123&patient_name=Dhanunjay&department_name=Pediatrics&queue_position=3&room_number=204")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "DELIVERED"
    assert "Pediatrics" in data["message"]
    assert "Position #3" in data["message"]


def test_notification_logs_endpoint(client):
    response = client.get("/notifications/logs")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1


def test_regional_fleet_network_status_endpoint(client):
    response = client.get("/fleet/network-status")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 3
    assert data[0]["facility_id"] == "HOSP-01"


def test_patient_rerouting_recommendation_endpoint(client):
    response = client.get("/fleet/reroute/Cardiology")
    assert response.status_code == 200
    data = response.json()
    assert "reroute_recommended" in data
    assert "primary_facility" in data


def test_executive_summary_report_endpoint(client):
    response = client.get("/reports/executive-summary")
    assert response.status_code == 200
    data = response.json()
    assert "report_title" in data
    assert "summary" in data
    assert "department_breakdown" in data


def test_staff_registration_and_list_endpoints(client):
    # Register new staff member
    reg_res = client.post("/auth/register-staff", json={"email": "newdoctor@hospital.com", "password": "docpass123", "name": "Dr. Sarah (Neurology)"})
    assert reg_res.status_code == 200
    reg_data = reg_res.json()
    assert reg_data["email"] == "newdoctor@hospital.com"
    assert reg_data["status"] == "REGISTERED_SUCCESSFULLY"

    # Verify listing
    list_res = client.get("/staff/list-registered")
    assert list_res.status_code == 200
    staff_list = list_res.json()
    emails = [s["email"] for s in staff_list]
    assert "newdoctor@hospital.com" in emails


def test_admin_whole_data_overview_endpoint(client):
    response = client.get("/admin/all-data")
    assert response.status_code == 200
    data = response.json()
    assert data["total_dataset_encounters"] == 24002
    assert "registered_staff_members" in data
    assert "live_department_states" in data
    assert "continuous_feedback_summary" in data

