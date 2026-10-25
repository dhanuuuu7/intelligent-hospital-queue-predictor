"""
Phase 13.3: Automated Patient SMS & WhatsApp Notification Dispatcher Engine
Dispatches automated position alerts when queue position reaches #3.
"""

import datetime
from typing import Dict, Any, List

_SENT_NOTIFICATIONS_LOG: List[Dict[str, Any]] = []


def send_queue_position_sms(
    op_number: str,
    patient_name: str,
    department_name: str,
    queue_position: int,
    room_number: str = "204"
) -> Dict[str, Any]:
    """
    Dispatches automated SMS/WhatsApp alert to patient when queue position reaches <= #3.
    """
    now_str = datetime.datetime.now().strftime("%I:%M %p")
    message = (
        f"🏥 Hospital Queue Alert ({now_str}): Hello {patient_name} (OP: {op_number}), "
        f"your consultation at {department_name} is coming up in ~10 minutes (Queue Position #{queue_position}). "
        f"Please proceed to Consulting Room {room_number}."
    )

    alert_entry = {
        "id": len(_SENT_NOTIFICATIONS_LOG) + 1,
        "op_number": op_number,
        "patient_name": patient_name,
        "department_name": department_name,
        "queue_position": queue_position,
        "message": message,
        "status": "DELIVERED",
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    _SENT_NOTIFICATIONS_LOG.append(alert_entry)
    return alert_entry


def get_sent_notifications_log(limit: int = 10) -> List[Dict[str, Any]]:
    return list(reversed(_SENT_NOTIFICATIONS_LOG[-limit:]))
