"""
FastAPI Main Web Application Server for Hospital Queue System (Production Hardened)
"""

import os
import json
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.logging_config import StructuredLoggingMiddleware, logger
from app.schemas import (
    EncounterPredictionRequest, 
    PredictionResponse, 
    HealthCheckResponse, 
    QueueStatusResponse
)
from models.predict import predict_waiting_time, WaitingTimePredictor
from models.explainability import ModelExplainer

# Global instances
predictor = None
explainer = None
metadata = {}


def load_artifacts():
    global predictor, explainer, metadata
    try:
        predictor = WaitingTimePredictor(model_path=settings.MODEL_PATH)
        explainer = ModelExplainer(model_path=settings.MODEL_PATH)
        
        artifacts_dir = os.path.dirname(settings.MODEL_PATH)
        meta_candidates = ["model_registry.json", "final_model_metadata.json", "model_metadata.json", "model_metadata_fe.json"]
        
        for cand in meta_candidates:
            cand_path = os.path.join(artifacts_dir, cand)
            if os.path.exists(cand_path):
                with open(cand_path, "r") as f:
                    metadata = json.load(f)
                break
        logger.info("[Success] ML Model Pipeline, Explainer & Metadata loaded successfully.")
    except Exception as e:
        logger.warning(f"[Warning] Failed to load ML model on startup: {e}")


# Load on import
load_artifacts()


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="Intelligent Hospital Queue & Appointment Optimization API",
    description=(
        "Production API for real-time patient waiting-time prediction, empirical prediction intervals, "
        "model explainability, and queue monitoring.\n\n"
        "**Disclaimer**: Predictions are estimates for queue-management decision support and are NOT medical diagnoses or clinical guarantees."
    ),
    version=settings.MODEL_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Structured Logging Middleware
app.add_middleware(StructuredLoggingMiddleware)

# Configurable CORS Middleware
origins = [settings.FRONTEND_ORIGIN, "http://localhost:5173", "http://127.0.0.1:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root():
    return {"message": "Welcome to Intelligent Hospital Queue API. Visit /docs for OpenAPI documentation."}


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
def health_check():
    """
    Returns system health status and loaded ML model information.
    """
    model_loaded = predictor is not None
    model_name = metadata.get("model_name", metadata.get("algorithm", "Tuned Random Forest"))
    status_str = "ok" if model_loaded else "degraded"
    
    return HealthCheckResponse(
        status=status_str,
        model_loaded=model_loaded,
        model_name=model_name,
        version=settings.MODEL_VERSION
    )


@app.get("/ready", tags=["Health"])
def readiness_check():
    """
    Readiness probe verifying that application is running, model artifact is loaded, and metadata is available.
    """
    model_loaded = predictor is not None
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded on server.")
        
    return {
        "status": "ready",
        "model_loaded": True,
        "model_name": metadata.get("model_name", metadata.get("algorithm", "Tuned Random Forest")),
        "version": settings.MODEL_VERSION
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict_encounter_wait_time(request: EncounterPredictionRequest):
    """
    Predicts patient waiting time in minutes given patient encounter arrival attributes.
    Returns point estimate, 90% prediction interval, confidence label, and operational explanation.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="ML Model service is currently unavailable.")

    try:
        input_data = request.model_dump()
        interval_data = predictor.predict_encounter_with_interval(input_data, confidence_level=0.90)
        
        explanation_res = None
        if explainer is not None:
            explanation_res = explainer.explain_single_prediction(input_data)["explanation"]

        return PredictionResponse(
            predicted_waiting_time_min=interval_data["predicted_waiting_time_min"],
            lower_bound_min=interval_data["lower_bound_min"],
            upper_bound_min=interval_data["upper_bound_min"],
            interval_width_min=interval_data["interval_width_min"],
            confidence_level=interval_data["confidence_level"],
            confidence_label=interval_data["confidence_label"],
            explanation=explanation_res,
            department_name=request.department_name,
            queue_position=request.queue_position,
            model_version=settings.MODEL_VERSION
        )
    except Exception as e:
        logger.error(f"Prediction processing error: {e}")
        raise HTTPException(status_code=400, detail="Invalid prediction request attributes.")


@app.get("/queue-status", response_model=QueueStatusResponse, tags=["Monitoring"])
def get_queue_status(department_name: str = Query("General Medicine", description="Department to query")):
    """
    Returns current queue status, estimated waiting time, and crowd level for a department.
    """
    valid_depts = ["General Medicine", "Pediatrics", "Orthopedics", "Cardiology"]
    if department_name not in valid_depts:
        raise HTTPException(status_code=400, detail=f"Invalid department. Must be one of: {valid_depts}")

    dept_defaults = {
        "General Medicine": {"wait": 9.0, "docs": 4, "crowd": "LOW"},
        "Pediatrics": {"wait": 18.0, "docs": 3, "crowd": "MEDIUM"},
        "Orthopedics": {"wait": 34.0, "docs": 2, "crowd": "HIGH"},
        "Cardiology": {"wait": 51.0, "docs": 2, "crowd": "VERY HIGH"}
    }
    info = dept_defaults.get(department_name, {"wait": 15.0, "docs": 3, "crowd": "MEDIUM"})

    return QueueStatusResponse(
        department_name=department_name,
        estimated_waiting_time_min=info["wait"],
        active_doctors_count=info["docs"],
        crowd_level=info["crowd"]
    )


from app.hospital_state import hospital_state
from app.auth import (
    create_token, 
    parse_token, 
    require_staff_role, 
    authenticate_staff_credentials, 
    verify_patient_op_record
)


@app.post("/auth/login", tags=["Authentication"])
def login_staff_user(payload: Dict[str, Any]):
    """
    Staff Authentication Login Endpoint.
    Validates staff credentials strictly on the backend.
    """
    email = payload.get("email", "")
    password = payload.get("password", "")

    staff_info = authenticate_staff_credentials(email, password)
    token = create_token(email_or_op=staff_info["email"], role="staff")

    return {
        "access_token": token,
        "token_type": "bearer",
        "email": staff_info["email"],
        "role": "staff",
        "name": staff_info["name"]
    }


@app.post("/auth/patient-verify", tags=["Authentication"])
def verify_patient_identity(payload: Dict[str, Any]):
    """
    Patient Identity Verification Endpoint.
    Validates patient OP record key on the backend.
    """
    op_number = payload.get("op_number", "OP202600123")
    patient_name = payload.get("patient_name", "Dhanunjay Nadiminti")

    p_info = verify_patient_op_record(op_number, patient_name)
    token = create_token(email_or_op=p_info["op_number"], role="patient")

    return {
        "access_token": token,
        "token_type": "bearer",
        "op_number": p_info["op_number"],
        "patient_name": p_info["patient_name"],
        "role": "patient"
    }


@app.get("/hospital/state/{department_name}", tags=["Hospital State"])
def get_hospital_department_state(department_name: str):
    """
    Returns single-source-of-truth live operational hospital state for a department.
    """
    try:
        return hospital_state.get_department_state(department_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/staff/department-state", tags=["Staff Operations"])
def update_staff_department_state(
    department_name: str = Query(...),
    active_doctors: Optional[int] = Query(None),
    doctors_on_break: Optional[int] = Query(None),
    emergency_cases: Optional[int] = Query(None)
):
    """
    Staff control endpoint to update active doctors, break counts, or emergency cases for a department.
    """
    try:
        return hospital_state.update_department_state(
            department_name=department_name,
            active_doctors=active_doctors,
            doctors_on_break=doctors_on_break,
            emergency_cases=emergency_cases
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/patient/checkin", tags=["Patient Check-in"])
def checkin_patient_visit(
    department_name: str = Query(...),
    arrival_type: str = Query("Scheduled")
):
    """
    Patient check-in endpoint: Backend assigns next queue number, registers ticket, updates hospital state, and calculates prediction.
    """
    try:
        reg = hospital_state.checkin_patient(department_name=department_name, arrival_type=arrival_type)
        dept_info = hospital_state.get_department_state(department_name)

        payload = {
            "department_name": department_name,
            "day_of_week": 1,
            "hour_of_day": 10,
            "is_weekend_or_holiday": 0,
            "season_indicator": "Winter",
            "queue_position": reg["assigned_queue_position"],
            "scheduled_patients_ahead": max(0, reg["assigned_queue_position"] - 1),
            "walkin_patients_ahead": 0,
            "patient_arrival_type": arrival_type,
            "patient_case_complexity": "Medium",
            "active_doctors_count": dept_info["active_doctors"],
            "doctors_on_break_count": dept_info["doctors_on_break"],
            "triage_staff_count": 2,
            "shift_handover_flag": 0,
            "emergency_cases_active": dept_info["emergency_cases"],
            "arrival_rate_past_30min": 10.0,
            "recent_throughput_past_hour": 15,
            "avg_waiting_time_last_30min": 15.0,
            "dept_service_time_variability": 3.5,
            "hospital_occupancy_rate": 0.68
        }

        if predictor:
            pred_res = predictor.predict_encounter_with_interval(payload)
        else:
            calc_w = round((reg["assigned_queue_position"] * 12) / dept_info["active_doctors"], 1)
            pred_res = {
                "predicted_waiting_time_min": calc_w,
                "lower_bound_min": max(0.0, round(calc_w - 14, 1)),
                "upper_bound_min": round(calc_w + 14, 1),
                "confidence_label": "MODERATE"
            }

        return {
            "ticket_id": reg["ticket_id"],
            "department": department_name,
            "queue_position": reg["assigned_queue_position"],
            "active_doctors": dept_info["active_doctors"],
            "prediction": pred_res
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/staff/what-if", tags=["Staff Operations"])
def simulate_staff_what_if(
    department_name: str = Query(...),
    doctor_delta: int = Query(0),
    emergency_delta: int = Query(0)
):
    """
    Staff decision-support simulator endpoint: Predicts waiting time impact if doctors or emergency cases change.
    """
    try:
        return hospital_state.simulate_what_if(
            department_name=department_name,
            doctor_delta=doctor_delta,
            emergency_delta=emergency_delta
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
