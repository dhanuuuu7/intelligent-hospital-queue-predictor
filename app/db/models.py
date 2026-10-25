"""
Phase 12.1: Relational ORM Models for Hospital Operations Platform
"""

import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from app.db.database import Base


class UserDB(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(128), unique=True, index=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    role = Column(String(32), nullable=False, default="patient")
    name = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class PatientDB(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    op_number = Column(String(64), unique=True, index=True, nullable=False)
    name = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class DepartmentDB(Base):
    __tablename__ = "departments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(64), unique=True, index=True, nullable=False)
    active_doctors = Column(Integer, default=3)
    doctors_on_break = Column(Integer, default=0)
    waiting_patients = Column(Integer, default=0)
    emergency_cases = Column(Integer, default=0)
    average_consultation_time = Column(Float, default=12.0)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


class QueueTicketDB(Base):
    __tablename__ = "queue_tickets"

    id = Column(Integer, primary_key=True, index=True)
    ticket_code = Column(String(64), unique=True, index=True, nullable=False)
    op_number = Column(String(64), nullable=False, index=True)
    department_name = Column(String(64), nullable=False, index=True)
    arrival_type = Column(String(32), nullable=False, default="Scheduled")
    queue_position = Column(Integer, nullable=False)
    status = Column(String(32), nullable=False, default="WAITING")  # WAITING, CALLED, IN_CONSULTATION, COMPLETED, CANCELLED
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class PredictionDB(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    ticket_code = Column(String(64), nullable=False, index=True)
    department_name = Column(String(64), nullable=False, index=True)
    predicted_wait_min = Column(Float, nullable=False)
    lower_bound_min = Column(Float, nullable=False)
    upper_bound_min = Column(Float, nullable=False)
    confidence_label = Column(String(32), nullable=False, default="MODERATE")
    model_version = Column(String(32), nullable=False, default="1.0.0")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class PredictionFeedbackDB(Base):
    __tablename__ = "prediction_feedback"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, nullable=True)
    ticket_code = Column(String(64), nullable=False, index=True)
    department_name = Column(String(64), nullable=False, index=True)
    predicted_wait_min = Column(Float, nullable=False)
    actual_wait_min = Column(Float, nullable=False)
    error_min = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class AuditLogDB(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String(128), nullable=False, index=True)
    action = Column(String(128), nullable=False)
    details = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
