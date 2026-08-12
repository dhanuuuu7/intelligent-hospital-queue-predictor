"""
Step 11.1: Hospital Operational State Engine & Single Source of Truth
Manages live department staffing, waiting queues, emergency preemption, and queue state.
"""

import threading
from typing import Dict, Any, List, Optional
from models.predict import WaitingTimePredictor

class HospitalStateManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._predictor = None
        
        # Initial live hospital operational state
        self._state: Dict[str, Dict[str, Any]] = {
            "General Medicine": {
                "department": "General Medicine",
                "active_doctors": 4,
                "doctors_on_break": 0,
                "waiting_patients": 8,
                "emergency_cases": 0,
                "currently_consulting": 4,
                "average_consultation_time": 10.0,
                "queue_list": [
                    {"ticket_id": f"GM-{101+i}", "arrival_type": "Scheduled" if i % 2 == 0 else "Walk-in", "queue_position": i+1}
                    for i in range(8)
                ]
            },
            "Pediatrics": {
                "department": "Pediatrics",
                "active_doctors": 3,
                "doctors_on_break": 1,
                "waiting_patients": 17,
                "emergency_cases": 1,
                "currently_consulting": 3,
                "average_consultation_time": 12.0,
                "queue_list": [
                    {"ticket_id": f"PED-{101+i}", "arrival_type": "Scheduled" if i % 3 != 0 else "Walk-in", "queue_position": i+1}
                    for i in range(17)
                ]
            },
            "Orthopedics": {
                "department": "Orthopedics",
                "active_doctors": 2,
                "doctors_on_break": 0,
                "waiting_patients": 31,
                "emergency_cases": 2,
                "currently_consulting": 2,
                "average_consultation_time": 16.0,
                "queue_list": [
                    {"ticket_id": f"ORTH-{101+i}", "arrival_type": "Walk-in" if i % 2 == 0 else "Scheduled", "queue_position": i+1}
                    for i in range(31)
                ]
            },
            "Cardiology": {
                "department": "Cardiology",
                "active_doctors": 2,
                "doctors_on_break": 0,
                "waiting_patients": 42,
                "emergency_cases": 3,
                "currently_consulting": 2,
                "average_consultation_time": 22.0,
                "queue_list": [
                    {"ticket_id": f"CARD-{101+i}", "arrival_type": "Emergency" if i < 3 else "Scheduled", "queue_position": i+1}
                    for i in range(42)
                ]
            }
        }

    def _get_predictor(self):
        if self._predictor is None:
            try:
                self._predictor = WaitingTimePredictor()
            except Exception:
                self._predictor = None
        return self._predictor

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return self._state

    def get_department_state(self, department_name: str) -> Dict[str, Any]:
        with self._lock:
            if department_name not in self._state:
                raise ValueError(f"Unknown department: {department_name}")
            return dict(self._state[department_name])

    def update_department_state(
        self, 
        department_name: str, 
        active_doctors: Optional[int] = None,
        doctors_on_break: Optional[int] = None,
        emergency_cases: Optional[int] = None
    ) -> Dict[str, Any]:
        with self._lock:
            if department_name not in self._state:
                raise ValueError(f"Unknown department: {department_name}")

            dept = self._state[department_name]
            if active_doctors is not None:
                dept["active_doctors"] = max(1, active_doctors)
            if doctors_on_break is not None:
                dept["doctors_on_break"] = max(0, doctors_on_break)
            if emergency_cases is not None:
                dept["emergency_cases"] = max(0, emergency_cases)

            return dict(dept)

    def checkin_patient(self, department_name: str, arrival_type: str = "Scheduled") -> Dict[str, Any]:
        with self._lock:
            if department_name not in self._state:
                raise ValueError(f"Unknown department: {department_name}")

            dept = self._state[department_name]
            dept["waiting_patients"] += 1
            new_position = dept["waiting_patients"]

            prefix_map = {
                "General Medicine": "GM",
                "Pediatrics": "PED",
                "Orthopedics": "ORTH",
                "Cardiology": "CARD"
            }
            prefix = prefix_map.get(department_name, "GEN")
            ticket_id = f"{prefix}-{100 + new_position}"

            patient_entry = {
                "ticket_id": ticket_id,
                "arrival_type": arrival_type,
                "queue_position": new_position
            }
            dept["queue_list"].append(patient_entry)

            return {
                "ticket_id": ticket_id,
                "department": department_name,
                "assigned_queue_position": new_position,
                "arrival_type": arrival_type,
                "active_doctors": dept["active_doctors"],
                "waiting_patients": dept["waiting_patients"]
            }

    def call_next_patient(self, department_name: str) -> Dict[str, Any]:
        with self._lock:
            if department_name not in self._state:
                raise ValueError(f"Unknown department: {department_name}")

            dept = self._state[department_name]
            if dept["waiting_patients"] > 0:
                dept["waiting_patients"] -= 1
                if dept["queue_list"]:
                    dept["queue_list"].pop(0)

            return dict(dept)

    def simulate_what_if(self, department_name: str, doctor_delta: int = 0, emergency_delta: int = 0) -> Dict[str, Any]:
        """
        Simulates what happens to department waiting time if doctors or emergency cases change.
        """
        dept = self.get_department_state(department_name)
        curr_docs = dept["active_doctors"]
        curr_waiting = dept["waiting_patients"]
        curr_emerg = dept["emergency_cases"]

        sim_docs = max(1, curr_docs + doctor_delta)
        sim_emerg = max(0, curr_emerg + emergency_delta)

        predictor = self._get_predictor()
        
        base_payload = {
            "department_name": department_name,
            "day_of_week": 1,
            "hour_of_day": 11,
            "is_weekend_or_holiday": 0,
            "season_indicator": "Winter",
            "queue_position": curr_waiting,
            "scheduled_patients_ahead": max(0, curr_waiting - 2),
            "walkin_patients_ahead": 2,
            "patient_arrival_type": "Scheduled",
            "patient_case_complexity": "Medium",
            "active_doctors_count": curr_docs,
            "doctors_on_break_count": dept["doctors_on_break"],
            "triage_staff_count": 2,
            "shift_handover_flag": 0,
            "emergency_cases_active": curr_emerg,
            "arrival_rate_past_30min": 10.0,
            "recent_throughput_past_hour": 15,
            "avg_waiting_time_last_30min": 15.0,
            "dept_service_time_variability": 3.5,
            "hospital_occupancy_rate": 0.68
        }

        sim_payload = dict(base_payload)
        sim_payload["active_doctors_count"] = sim_docs
        sim_payload["emergency_cases_active"] = sim_emerg

        if predictor:
            curr_wait = predictor.predict_encounter(base_payload)
            sim_wait = predictor.predict_encounter(sim_payload)
        else:
            avg_consult = dept["average_consultation_time"]
            curr_wait = round((curr_waiting * avg_consult) / curr_docs, 1)
            sim_wait = round((curr_waiting * avg_consult) / sim_docs + (sim_emerg * 10), 1)

        diff = round(sim_wait - curr_wait, 1)

        return {
            "department": department_name,
            "current_doctors": curr_docs,
            "simulated_doctors": sim_docs,
            "current_emergency": curr_emerg,
            "simulated_emergency": sim_emerg,
            "current_predicted_wait_min": round(curr_wait, 1),
            "simulated_predicted_wait_min": round(sim_wait, 1),
            "wait_time_delta_min": diff,
            "impact_summary": f"Adding {doctor_delta} doctor(s) reduces wait by {abs(diff)} minutes." if diff < 0 else f"Wait increases by {diff} minutes."
        }

# Global singleton
hospital_state = HospitalStateManager()
