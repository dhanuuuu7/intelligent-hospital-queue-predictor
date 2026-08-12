"""
Priority Queue & Patient Encounter Logger for SimPy Engine
"""

import math
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
from simulator.config import PRIORITY_LEVELS, DEPARTMENT_PROFILES, SHIFT_HANDOVER_HOURS


class PatientEncounterRecord:
    def __init__(self, patient_id: str, department: str, arrival_dt: datetime):
        self.patient_id = patient_id
        self.department_name = department
        self.arrival_timestamp = arrival_dt
        self.consultation_start_dt = None
        self.actual_waiting_time_min = None
        
        # Features captured at T_arrival
        self.day_of_week = arrival_dt.weekday()
        self.hour_of_day = arrival_dt.hour
        self.is_weekend_or_holiday = 1 if arrival_dt.weekday() >= 5 else 0
        self.season_indicator = self._get_season(arrival_dt)
        
        # Queue state at arrival
        self.queue_position = 1
        self.scheduled_patients_ahead = 0
        self.walkin_patients_ahead = 0
        self.total_patients_waiting_ahead = 0
        self.patient_arrival_type = "Walk-in"
        self.patient_case_complexity = "Medium"
        
        # Resources & Capacity at arrival
        self.active_doctors_count = 1
        self.doctors_on_break_count = 0
        self.triage_staff_count = 2
        self.shift_handover_flag = 0
        self.emergency_cases_active = 0
        
        # Rolling operational signals at arrival
        self.arrival_rate_past_30min = 0.0
        self.recent_throughput_past_hour = 0
        self.avg_waiting_time_last_30min = 0.0
        self.dept_service_time_variability = 0.0
        self.hospital_occupancy_rate = 0.65

    def _get_season(self, dt: datetime) -> str:
        month = dt.month
        if month in [6, 7, 8, 9]:
            return "Monsoon"
        elif month in [11, 12, 1, 2]:
            return "Winter"
        else:
            return "Summer"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "department_name": self.department_name,
            "arrival_timestamp": self.arrival_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "day_of_week": self.day_of_week,
            "hour_of_day": self.hour_of_day,
            "is_weekend_or_holiday": self.is_weekend_or_holiday,
            "season_indicator": self.season_indicator,
            "queue_position": self.queue_position,
            "scheduled_patients_ahead": self.scheduled_patients_ahead,
            "walkin_patients_ahead": self.walkin_patients_ahead,
            "total_patients_waiting_ahead": self.total_patients_waiting_ahead,
            "patient_arrival_type": self.patient_arrival_type,
            "patient_case_complexity": self.patient_case_complexity,
            "active_doctors_count": self.active_doctors_count,
            "doctors_on_break_count": self.doctors_on_break_count,
            "triage_staff_count": self.triage_staff_count,
            "shift_handover_flag": self.shift_handover_flag,
            "emergency_cases_active": self.emergency_cases_active,
            "arrival_rate_past_30min": round(self.arrival_rate_past_30min, 2),
            "recent_throughput_past_hour": self.recent_throughput_past_hour,
            "avg_waiting_time_last_30min": round(self.avg_waiting_time_last_30min, 2),
            "dept_service_time_variability": round(self.dept_service_time_variability, 2),
            "hospital_occupancy_rate": round(self.hospital_occupancy_rate, 2),
            "actual_waiting_time_min": round(self.actual_waiting_time_min, 2) if self.actual_waiting_time_min is not None else None
        }


class QueueTracker:
    def __init__(self, department: str):
        self.department = department
        self.completed_encounters: List[PatientEncounterRecord] = []
        self.waiting_encounters: List[PatientEncounterRecord] = []
        self.recent_arrivals_30min: List[datetime] = []
        self.recent_completions_60min: List[datetime] = []
        self.recent_wait_times_30min: List[float] = []

    def log_arrival(self, record: PatientEncounterRecord, current_dt: datetime, active_doctors: int, doctors_on_break: int):
        # Clean rolling buffers
        cutoff_30min = current_dt - timedelta(minutes=30)
        cutoff_60min = current_dt - timedelta(minutes=60)
        
        self.recent_arrivals_30min = [t for t in self.recent_arrivals_30min if t >= cutoff_30min]
        self.recent_completions_60min = [t for t in self.recent_completions_60min if t >= cutoff_60min]
        
        # Count queue ahead
        sched_ahead = sum(1 for p in self.waiting_encounters if p.patient_arrival_type == "Scheduled")
        walkin_ahead = sum(1 for p in self.waiting_encounters if p.patient_arrival_type == "Walk-in")
        emerg_ahead = sum(1 for p in self.waiting_encounters if p.patient_arrival_type == "Emergency")
        
        record.scheduled_patients_ahead = sched_ahead
        record.walkin_patients_ahead = walkin_ahead
        record.total_patients_waiting_ahead = len(self.waiting_encounters)
        record.queue_position = len(self.waiting_encounters) + 1
        
        record.active_doctors_count = active_doctors
        record.doctors_on_break_count = doctors_on_break
        record.shift_handover_flag = 1 if current_dt.hour in SHIFT_HANDOVER_HOURS else 0
        record.emergency_cases_active = emerg_ahead
        
        # Operational metrics
        record.arrival_rate_past_30min = len(self.recent_arrivals_30min)
        record.recent_throughput_past_hour = len(self.recent_completions_60min)
        
        if self.recent_wait_times_30min:
            record.avg_waiting_time_last_30min = float(np.mean(self.recent_wait_times_30min[-10:]))
        else:
            record.avg_waiting_time_last_30min = 15.0
            
        record.dept_service_time_variability = DEPARTMENT_PROFILES[self.department]["consult_time_std_dev"]
        
        # Add to state
        self.recent_arrivals_30min.append(current_dt)
        self.waiting_encounters.append(record)

    def log_consultation_start(self, record: PatientEncounterRecord, current_dt: datetime):
        if record in self.waiting_encounters:
            self.waiting_encounters.remove(record)
            
        record.consultation_start_dt = current_dt
        wait_duration = (current_dt - record.arrival_timestamp).total_seconds() / 60.0
        record.actual_waiting_time_min = max(0.0, wait_duration)
        
        self.recent_completions_60min.append(current_dt)
        self.recent_wait_times_30min.append(record.actual_waiting_time_min)
        self.completed_encounters.append(record)
