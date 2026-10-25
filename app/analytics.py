"""
Phase 12.4: Operations Analytics, Bottleneck Detection & Future Queue Forecasting Engine
"""

import math
from typing import Dict, Any, List
from app.hospital_state import hospital_state


def detect_department_bottlenecks() -> List[Dict[str, Any]]:
    """
    Analyzes queue arrival rates vs clearance throughput across all departments to flag operational bottlenecks.
    """
    states = hospital_state.get_all_states()
    bottlenecks = []

    for dept_name, state in states.items():
        waiting = state["waiting_patients"]
        doctors = state["active_doctors"]
        emergencies = state["emergency_cases"]
        avg_consult = state["average_consultation_time"]

        throughput_per_hr = (doctors * 60.0) / avg_consult
        estimated_wait = (waiting * avg_consult) / max(1, doctors)

        is_bottleneck = (waiting >= 15 and doctors <= 3) or emergencies >= 2 or estimated_wait > 30.0
        risk_level = "CRITICAL" if estimated_wait > 35.0 else "WARNING" if is_bottleneck else "NORMAL"

        bottlenecks.append({
            "department": dept_name,
            "waiting_patients": waiting,
            "active_doctors": doctors,
            "emergency_cases": emergencies,
            "estimated_wait_min": round(estimated_wait, 1),
            "clearance_throughput_per_hr": round(throughput_per_hr, 1),
            "risk_level": risk_level,
            "is_bottleneck": is_bottleneck
        })

    return bottlenecks


def forecast_future_queue(department_name: str) -> Dict[str, Any]:
    """
    Projects expected queue depth and waiting times for T+15m, T+30m, and T+60m into the future.
    """
    dept = hospital_state.get_department_state(department_name)
    curr_waiting = dept["waiting_patients"]
    docs = dept["active_doctors"]
    avg_consult = dept["average_consultation_time"]

    # Arrival & clearance rate assumptions based on department characteristics
    arrival_rate_per_min = 0.5 if department_name == "General Medicine" else 0.4
    clearance_rate_per_min = docs / avg_consult

    net_growth_per_min = arrival_rate_per_min - clearance_rate_per_min

    q_15 = max(0, round(curr_waiting + (net_growth_per_min * 15)))
    q_30 = max(0, round(curr_waiting + (net_growth_per_min * 30)))
    q_60 = max(0, round(curr_waiting + (net_growth_per_min * 60)))

    w_curr = round((curr_waiting * avg_consult) / max(1, docs), 1)
    w_15 = round((q_15 * avg_consult) / max(1, docs), 1)
    w_30 = round((q_30 * avg_consult) / max(1, docs), 1)
    w_60 = round((q_60 * avg_consult) / max(1, docs), 1)

    return {
        "department": department_name,
        "current": {"queue_depth": curr_waiting, "predicted_wait_min": w_curr},
        "forecast_15m": {"queue_depth": q_15, "predicted_wait_min": w_15},
        "forecast_30m": {"queue_depth": q_30, "predicted_wait_min": w_30},
        "forecast_60m": {"queue_depth": q_60, "predicted_wait_min": w_60},
        "trend_direction": "UPWARD" if net_growth_per_min > 0 else "DOWNWARD" if net_growth_per_min < 0 else "STABLE"
    }


def generate_ai_staff_recommendation(department_name: str) -> Dict[str, Any]:
    """
    Generates intelligent, actionable staffing recommendations to optimize department flow.
    """
    dept = hospital_state.get_department_state(department_name)
    curr_waiting = dept["waiting_patients"]
    curr_docs = dept["active_doctors"]
    emergencies = dept["emergency_cases"]
    avg_consult = dept["average_consultation_time"]

    curr_wait = round((curr_waiting * avg_consult) / max(1, curr_docs), 1)
    
    if curr_wait > 25.0:
        recommended_docs = curr_docs + 1
        rec_wait = round((curr_waiting * avg_consult) / recommended_docs, 1)
        wait_reduction = round(curr_wait - rec_wait, 1)

        return {
            "department": department_name,
            "action_type": "ACTIVATE_DOCTOR",
            "urgency": "HIGH",
            "current_wait_min": curr_wait,
            "recommended_doctors": recommended_docs,
            "projected_wait_min": rec_wait,
            "expected_impact": f"Activate 1 additional doctor in {department_name} to reduce wait time from {curr_wait}m → {rec_wait}m (↓ {wait_reduction} min reduction).",
            "reasoning": f"Queue depth ({curr_waiting}) exceeds clearance capacity with {curr_docs} active doctor(s) and {emergencies} active emergency cases."
        }
    else:
        return {
            "department": department_name,
            "action_type": "MAINTAIN_STAFFING",
            "urgency": "LOW",
            "current_wait_min": curr_wait,
            "recommended_doctors": curr_docs,
            "projected_wait_min": curr_wait,
            "expected_impact": f"Maintain current staffing of {curr_docs} doctor(s) in {department_name}.",
            "reasoning": f"Current queue flow ({curr_waiting} patients) is well within optimal clearance thresholds."
        }
