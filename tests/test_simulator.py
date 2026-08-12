"""
Unit and Validation Tests for Hospital Queue Simulator & Dataset Integrity
"""

import os
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def dataset_df():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "hospital_encounters.csv")
    assert os.path.exists(data_path), f"Dataset file not found at {data_path}"
    df = pd.read_csv(data_path)
    return df


def test_dataset_column_schema(dataset_df):
    expected_columns = [
        "patient_id", "department_name", "arrival_timestamp", "day_of_week",
        "hour_of_day", "is_weekend_or_holiday", "season_indicator",
        "queue_position", "scheduled_patients_ahead", "walkin_patients_ahead",
        "total_patients_waiting_ahead", "patient_arrival_type",
        "patient_case_complexity", "active_doctors_count", "doctors_on_break_count",
        "triage_staff_count", "shift_handover_flag", "emergency_cases_active",
        "arrival_rate_past_30min", "recent_throughput_past_hour",
        "avg_waiting_time_last_30min", "dept_service_time_variability",
        "hospital_occupancy_rate", "actual_waiting_time_min"
    ]
    assert list(dataset_df.columns) == expected_columns, "Dataset columns do not match expected schema!"


def test_no_missing_values(dataset_df):
    assert dataset_df.isnull().sum().sum() == 0, "Dataset contains null/missing values!"


def test_non_negative_target(dataset_df):
    assert (dataset_df["actual_waiting_time_min"] >= 0.0).all(), "Found negative waiting times!"


def test_queue_position_consistency(dataset_df):
    derived_total = dataset_df["scheduled_patients_ahead"] + dataset_df["walkin_patients_ahead"]
    assert (dataset_df["total_patients_waiting_ahead"] >= derived_total).all()


def test_doctor_break_wait_time_impact(dataset_df):
    # During doctor lunch break (hour 13), wait times should be higher than steady afternoon (hour 14)
    lunch_break_wait = dataset_df[dataset_df["hour_of_day"] == 13]["actual_waiting_time_min"].mean()
    steady_afternoon_wait = dataset_df[dataset_df["hour_of_day"] == 14]["actual_waiting_time_min"].mean()
    print(f"\nLunch Break (1 PM) Avg Wait: {lunch_break_wait:.2f} min | Post-Lunch (2 PM) Avg Wait: {steady_afternoon_wait:.2f} min")
    assert lunch_break_wait > steady_afternoon_wait, "Doctor break hour should increase average wait time!"
