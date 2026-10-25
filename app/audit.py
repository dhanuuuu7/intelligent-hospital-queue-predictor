"""
Phase 12.5: Non-Repudiable Audit Logging & Security Event Tracking Engine
"""

import json
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from app.db.models import AuditLogDB


def log_security_event(
    db: Session,
    user_email: str,
    action: str,
    details: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Logs administrative, staffing, or security mutation events for compliance auditing.
    """
    if isinstance(details, (dict, list)):
        details_json = json.dumps(details)
    elif details is not None:
        details_json = str(details)
    else:
        details_json = None
    
    audit_entry = AuditLogDB(
        user_email=user_email,
        action=action,
        details=details_json
    )
    db.add(audit_entry)
    db.commit()
    db.refresh(audit_entry)

    return {
        "id": audit_entry.id,
        "user_email": audit_entry.user_email,
        "action": audit_entry.action,
        "timestamp": audit_entry.timestamp.isoformat()
    }


def parse_details_safely(raw_details: Optional[str]) -> Any:
    if not raw_details:
        return {}
    try:
        return json.loads(raw_details)
    except Exception:
        return {"info": raw_details}


def get_recent_audit_logs(db: Session, limit: int = 20) -> list:
    """
    Retrieves recent security audit logs for staff monitoring.
    """
    logs = db.query(AuditLogDB).order_by(AuditLogDB.timestamp.desc()).limit(limit).all()
    return [
        {
            "id": l.id,
            "user_email": l.user_email,
            "action": l.action,
            "details": parse_details_safely(l.details),
            "timestamp": l.timestamp.isoformat()
        }
        for l in logs
    ]
