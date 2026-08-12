"""
Step 11.1 & 11.3 Testing: Hospital Operational State Engine & What-If Simulator Tests
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.hospital_state import hospital_state


@pytest.fixture
def client():
    return TestClient(app)


def test_hospital_state_initialization():
    state = hospital_state.get_department_state("General Medicine")
    assert state["active_doctors"] == 4
    assert state["waiting_patients"] >= 0
    assert len(state["queue_list"]) >= 0


def test_get_hospital_state_endpoint(client):
    response = client.get("/hospital/state/Pediatrics")
    assert response.status_code == 200
    data = response.json()
    assert data["department"] == "Pediatrics"
    assert "active_doctors" in data
    assert "waiting_patients" in data


def test_update_staff_department_state_endpoint(client):
    response = client.post("/staff/department-state?department_name=Pediatrics&active_doctors=5")
    assert response.status_code == 200
    data = response.json()
    assert data["active_doctors"] == 5

    # Verify state persistence
    state = hospital_state.get_department_state("Pediatrics")
    assert state["active_doctors"] == 5


def test_patient_checkin_endpoint(client):
    response = client.post("/patient/checkin?department_name=Orthopedics&arrival_type=Walk-in")
    assert response.status_code == 200
    data = response.json()
    assert "ticket_id" in data
    assert data["department"] == "Orthopedics"
    assert data["queue_position"] >= 1
    assert "prediction" in data


def test_staff_what_if_simulation_endpoint(client):
    response = client.post("/staff/what-if?department_name=Cardiology&doctor_delta=1&emergency_delta=0")
    assert response.status_code == 200
    data = response.json()
    assert data["department"] == "Cardiology"
    assert "current_predicted_wait_min" in data
    assert "simulated_predicted_wait_min" in data
    assert "impact_summary" in data


def test_staff_login_authentication_success(client):
    response = client.post("/auth/login", json={"email": "staff@hospital.com", "password": "staff123"})
    assert response.status_code == 200
    data = response.json()
    assert data["role"] == "staff"
    assert "access_token" in data


def test_staff_login_authentication_failure(client):
    response = client.post("/auth/login", json={"email": "hacker@otherdomain.com", "password": "wrong"})
    assert response.status_code == 401


def test_patient_op_verification_success(client):
    response = client.post("/auth/patient-verify", json={"op_number": "OP202600123", "patient_name": "Dhanunjay"})
    assert response.status_code == 200
    data = response.json()
    assert data["role"] == "patient"
    assert data["op_number"] == "OP202600123"

