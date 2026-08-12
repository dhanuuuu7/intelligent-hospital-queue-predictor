"""
Step 5.1: Dataset Validation & Chronological Data Loader Module
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any

# Excluded columns per design rules
EXCLUDED_FEATURES = ["patient_id", "arrival_timestamp", "total_patients_waiting_ahead"]

# Target column
TARGET_COLUMN = "actual_waiting_time_min"

# Explicit Categorical & Numerical feature specifications
CATEGORICAL_FEATURES = [
    "department_name",
    "season_indicator",
    "patient_arrival_type",
    "patient_case_complexity"
]

NUMERICAL_FEATURES = [
    "day_of_week",
    "hour_of_day",
    "is_weekend_or_holiday",
    "queue_position",
    "scheduled_patients_ahead",
    "walkin_patients_ahead",
    "active_doctors_count",
    "doctors_on_break_count",
    "triage_staff_count",
    "shift_handover_flag",
    "emergency_cases_active",
    "arrival_rate_past_30min",
    "recent_throughput_past_hour",
    "avg_waiting_time_last_30min",
    "dept_service_time_variability",
    "hospital_occupancy_rate"
]

ALL_PREDICTIVE_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


def load_and_validate_dataset(csv_path: str) -> pd.DataFrame:
    """
    Loads dataset, executes strict data validation, verifies chronological ordering,
    and returns sanitized DataFrame.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at path: {csv_path}")

    df = pd.read_csv(csv_path)

    # 1. Column presence check
    expected_all_cols = EXCLUDED_FEATURES + ALL_PREDICTIVE_FEATURES + [TARGET_COLUMN]
    missing_cols = [c for c in expected_all_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    # 2. Missing values check
    null_counts = df.isnull().sum().sum()
    if null_counts > 0:
        raise ValueError(f"Dataset validation failed: Found {null_counts} missing/null values!")

    # 3. Duplicate rows check
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        raise ValueError(f"Dataset validation failed: Found {duplicate_count} duplicate rows!")

    # 4. Target non-negative check
    if (df[TARGET_COLUMN] < 0).any():
        raise ValueError("Dataset validation failed: Target variable contains negative values!")

    # 5. Chronological sorting verification
    df["arrival_timestamp"] = pd.to_datetime(df["arrival_timestamp"])
    if not df["arrival_timestamp"].is_monotonic_increasing:
        df = df.sort_values("arrival_timestamp").reset_index(drop=True)
        print("[Info] Dataset was not chronologically sorted. Explicitly sorted by arrival_timestamp.")

    print(f"[Success] Dataset Validation Passed: {len(df):,} valid patient encounter records.")
    return df


def get_chronological_splits(
    df: pd.DataFrame, 
    train_ratio: float = 0.70, 
    val_ratio: float = 0.15, 
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits dataset chronologically into 70% Train, 15% Validation, and 15% Test sets.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0!"

    # Separate feature matrix X and target vector y
    X = df[ALL_PREDICTIVE_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    total_rows = len(df)
    train_end_idx = int(total_rows * train_ratio)
    val_end_idx = int(total_rows * (train_ratio + val_ratio))

    X_train, y_train = X.iloc[:train_end_idx], y.iloc[:train_end_idx]
    X_val, y_val = X.iloc[train_end_idx:val_end_idx], y.iloc[train_end_idx:val_end_idx]
    X_test, y_test = X.iloc[val_end_idx:], y.iloc[val_end_idx:]

    print("\n--- Chronological Split Summary ---")
    print(f"Total Records   : {total_rows:,}")
    print(f"Train Set (70%) : {len(X_train):,} rows (Indices 0 to {train_end_idx - 1})")
    print(f"Val Set   (15%) : {len(X_val):,} rows (Indices {train_end_idx} to {val_end_idx - 1})")
    print(f"Test Set  (15%) : {len(X_test):,} rows (Indices {val_end_idx} to {total_rows - 1})")

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    data_file = os.path.join(os.path.dirname(__file__), "..", "data", "hospital_encounters.csv")
    df = load_and_validate_dataset(data_file)
    X_train, y_train, X_val, y_val, X_test, y_test = get_chronological_splits(df)
