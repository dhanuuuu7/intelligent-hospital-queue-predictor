"""
Pydantic Request & Response Data Schemas for Hospital Queue FastAPI Backend (Pydantic V2 Compliant)
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class EncounterPredictionRequest(BaseModel):
    department_name: str = Field(..., description="Hospital department", json_schema_extra={"example": "General Medicine"})
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)", json_schema_extra={"example": 1})
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of arrival (0-23)", json_schema_extra={"example": 10})
    is_weekend_or_holiday: int = Field(..., ge=0, le=1, description="1 if weekend/holiday, else 0", json_schema_extra={"example": 0})
    season_indicator: str = Field(..., description="Monsoon, Winter, or Summer", json_schema_extra={"example": "Winter"})
    
    queue_position: int = Field(..., ge=1, description="Patient line position at check-in", json_schema_extra={"example": 8})
    scheduled_patients_ahead: int = Field(..., ge=0, description="Scheduled appointment patients ahead", json_schema_extra={"example": 5})
    walkin_patients_ahead: int = Field(..., ge=0, description="Walk-in patients ahead", json_schema_extra={"example": 3})
    patient_arrival_type: str = Field(..., description="Scheduled, Walk-in, or Emergency", json_schema_extra={"example": "Scheduled"})
    patient_case_complexity: str = Field(..., description="Low, Medium, High, or Emergency", json_schema_extra={"example": "Medium"})
    
    active_doctors_count: int = Field(..., ge=1, description="Active consulting doctors in department", json_schema_extra={"example": 4})
    doctors_on_break_count: int = Field(..., ge=0, description="Doctors currently on rest break", json_schema_extra={"example": 0})
    triage_staff_count: int = Field(..., ge=1, description="Triage/registration desk capacity", json_schema_extra={"example": 2})
    shift_handover_flag: int = Field(..., ge=0, le=1, description="1 if within 30-min shift change", json_schema_extra={"example": 0})
    emergency_cases_active: int = Field(..., ge=0, description="Active emergency cases occupying doctors", json_schema_extra={"example": 0})
    
    arrival_rate_past_30min: float = Field(..., ge=0.0, description="Patients arrived in past 30 min", json_schema_extra={"example": 12.0})
    recent_throughput_past_hour: int = Field(..., ge=0, description="Patients completed in past 60 min", json_schema_extra={"example": 18})
    avg_waiting_time_last_30min: float = Field(..., ge=0.0, description="Rolling avg wait time of recently called patients", json_schema_extra={"example": 14.5})
    dept_service_time_variability: float = Field(..., ge=0.0, description="Department consult time std dev", json_schema_extra={"example": 3.0})
    hospital_occupancy_rate: float = Field(..., ge=0.0, le=1.0, description="Hospital occupancy index (0.0 to 1.0)", json_schema_extra={"example": 0.70})


class PredictionResponse(BaseModel):
    predicted_waiting_time_min: float = Field(..., description="Predicted waiting time in minutes", json_schema_extra={"example": 24.0})
    lower_bound_min: float = Field(default=18.0, description="Lower prediction interval bound in minutes", json_schema_extra={"example": 18.0})
    upper_bound_min: float = Field(default=34.0, description="Upper prediction interval bound in minutes", json_schema_extra={"example": 34.0})
    interval_width_min: float = Field(default=16.0, description="Interval width in minutes", json_schema_extra={"example": 16.0})
    confidence_level: float = Field(default=0.90, description="Target confidence level", json_schema_extra={"example": 0.90})
    confidence_label: str = Field(default="MODERATE", description="HIGH, MODERATE, or LOW uncertainty indicator", json_schema_extra={"example": "MODERATE"})
    explanation: Optional[Dict[str, Any]] = Field(default=None, description="Human-readable queue factors and operational summary")
    department_name: str = Field(..., json_schema_extra={"example": "General Medicine"})
    queue_position: int = Field(..., json_schema_extra={"example": 8})
    model_version: str = Field(default="1.0.0", json_schema_extra={"example": "1.0.0"})


class HealthCheckResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "ok"})
    model_loaded: bool = Field(..., json_schema_extra={"example": True})
    model_name: str = Field(..., json_schema_extra={"example": "Random Forest"})
    version: str = Field(default="1.0.0")


class QueueStatusResponse(BaseModel):
    department_name: str = Field(..., json_schema_extra={"example": "General Medicine"})
    estimated_waiting_time_min: float = Field(..., json_schema_extra={"example": 14.0})
    active_doctors_count: int = Field(..., json_schema_extra={"example": 4})
    crowd_level: str = Field(..., description="LOW, MEDIUM, HIGH, or VERY HIGH", json_schema_extra={"example": "MEDIUM"})
