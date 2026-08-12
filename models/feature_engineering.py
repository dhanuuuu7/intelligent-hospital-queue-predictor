"""
Step 8.2: Domain-Driven Feature Engineering for Queue Dynamics
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, List
from models.data_loader import load_and_validate_dataset, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN

ENGINEERED_NUMERICAL_FEATURES = [
    "effective_doctor_load",
    "traffic_momentum",
    "emergency_pressure",
    "scheduled_ratio_ahead"
]

ALL_EXPANDED_NUMERICAL_FEATURES = NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES
ALL_EXPANDED_PREDICTIVE_FEATURES = CATEGORICAL_FEATURES + ALL_EXPANDED_NUMERICAL_FEATURES


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs 4 non-linear queue ratio features with zero-division safeguards:
    1. effective_doctor_load = queue_position / active_doctors_count
    2. traffic_momentum      = arrival_rate_past_30min / (recent_throughput_past_hour + 1)
    3. emergency_pressure    = emergency_cases_active / active_doctors_count
    4. scheduled_ratio_ahead = scheduled_patients_ahead / (scheduled_patients_ahead + walkin_patients_ahead + 1)
    """
    df_engineered = df.copy()

    # Safeguard active_doctors_count >= 1
    safe_active_doctors = np.maximum(1, df_engineered["active_doctors_count"].values)

    # 1. Effective Doctor Load Ratio
    df_engineered["effective_doctor_load"] = df_engineered["queue_position"].values / safe_active_doctors

    # 2. Traffic Momentum Ratio (Arrival Rate vs Recent Throughput)
    df_engineered["traffic_momentum"] = (
        df_engineered["arrival_rate_past_30min"].values / (df_engineered["recent_throughput_past_hour"].values + 1.0)
    )

    # 3. Emergency Pressure Ratio
    df_engineered["emergency_pressure"] = df_engineered["emergency_cases_active"].values / safe_active_doctors

    # 4. Scheduled Ratio Ahead
    total_ahead = df_engineered["scheduled_patients_ahead"].values + df_engineered["walkin_patients_ahead"].values
    df_engineered["scheduled_ratio_ahead"] = df_engineered["scheduled_patients_ahead"].values / (total_ahead + 1.0)

    return df_engineered


def run_feature_engineering_inspection():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "hospital_encounters.csv")
    raw_df = load_and_validate_dataset(data_path)

    print("\n" + "=" * 65)
    print("PHASE 8.2 — DOMAIN FEATURE ENGINEERING INSPECTION")
    print("=" * 65)

    engineered_df = add_engineered_features(raw_df)

    print(f"\nOriginal Feature Count  : {len(CATEGORICAL_FEATURES) + len(NUMERICAL_FEATURES)}")
    print(f"Engineered Feature Count: {len(ENGINEERED_NUMERICAL_FEATURES)}")
    print(f"Total Expanded Features : {len(ALL_EXPANDED_PREDICTIVE_FEATURES)}")

    print("\n--- Summary Statistics of New Engineered Features ---")
    stats_df = engineered_df[ENGINEERED_NUMERICAL_FEATURES].describe(percentiles=[0.25, 0.5, 0.75, 0.95]).round(4)
    print(stats_df.to_string())

    print("\n--- First 5 Rows of Engineered Features ---")
    sample_df = engineered_df[["department_name", "queue_position", "active_doctors_count"] + ENGINEERED_NUMERICAL_FEATURES].head(5)
    print(sample_df.to_string(index=False))

    return engineered_df


if __name__ == "__main__":
    run_feature_engineering_inspection()
