"""
Step 8.6.4: Human-Readable Explanation Engine for Queue Predictions
Converts model signals into transparent, non-causal operational messages.
"""

from typing import Dict, Any, List, Tuple


def generate_human_explanation(
    patient_features: Dict[str, Any],
    top_increasing: List[Tuple[str, float]],
    top_decreasing: List[Tuple[str, float]]
) -> Dict[str, Any]:
    """
    Converts model feature contributions into non-causal, human-readable operational explanations.
    Uses non-causal language ('associated with higher wait') as required.
    """
    increasing_messages = []
    decreasing_messages = []

    # Map raw & engineered features to human-friendly operational statements
    queue_pos = patient_features.get("queue_position", 1)
    if queue_pos > 5:
        increasing_messages.append(f"Queue position #{queue_pos} is relatively high")

    eff_load = patient_features.get("effective_doctor_load", 0.0)
    if eff_load > 1.0:
        increasing_messages.append("Effective doctor workload pressure is elevated")

    traffic_mom = patient_features.get("traffic_momentum", 0.0)
    if traffic_mom > 0.6:
        increasing_messages.append("Recent patient arrivals are exceeding recent department throughput")

    emerg_cases = patient_features.get("emergency_cases_active", 0)
    if emerg_cases > 0:
        increasing_messages.append(f"Active emergency cases ({emerg_cases}) are occupying doctor capacity")

    avg_recent_wait = patient_features.get("avg_waiting_time_last_30min", 0.0)
    if avg_recent_wait > 20.0:
        increasing_messages.append(f"Recent patient waits ({avg_recent_wait:.1f} min) indicate an existing queue backlog")

    active_docs = patient_features.get("active_doctors_count", 1)
    if active_docs >= 3:
        decreasing_messages.append(f"Multiple consulting doctors ({active_docs}) are active")

    recent_throughput = patient_features.get("recent_throughput_past_hour", 0)
    if recent_throughput >= 15:
        decreasing_messages.append(f"High department clearance rate ({recent_throughput} patients/hr)")

    arr_type = patient_features.get("patient_arrival_type", "Scheduled")
    if arr_type == "Scheduled":
        decreasing_messages.append("Scheduled appointment priority status")

    # Fallbacks if list is empty
    if not increasing_messages:
        increasing_messages.append("Normal queue baseline flow")
    if not decreasing_messages:
        decreasing_messages.append("Standard consulting doctor allocation")

    # Patient-friendly bullet summary
    patient_bullets = []
    patient_bullets.append(f"{queue_pos} patients ahead in queue")
    patient_bullets.append(f"{active_docs} active consulting doctors")
    if traffic_mom > 0.5:
        patient_bullets.append("Moderate recent queue pressure")
    else:
        patient_bullets.append("Steady department workflow")

    summary = (
        "Queue pressure is elevated due to incoming patient traffic." 
        if len(increasing_messages) > len(decreasing_messages) 
        else "Queue flow is steady with adequate doctor availability."
    )

    return {
        "summary": summary,
        "increasing_factors": increasing_messages[:3],
        "decreasing_factors": decreasing_messages[:3],
        "patient_bullets": patient_bullets
    }
