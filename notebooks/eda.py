"""
Exploratory Data Analysis (EDA) & Data Profiling Script for Hospital Encounters Dataset
"""

import os
import pandas as pd
import numpy as np


def run_eda():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "hospital_encounters.csv")
    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    print("=" * 60)
    print("HOSPITAL QUEUE DATASET - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    print(f"\nTotal Records: {len(df):,}")
    print(f"Total Features: {df.shape[1]}")

    # 1. Target Variable Profile
    print("\n--- Target Variable Profile ('actual_waiting_time_min') ---")
    target_stats = df["actual_waiting_time_min"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95])
    print(target_stats.to_string())

    # 2. Department Analysis
    print("\n--- Waiting Time Statistics by Department ---")
    dept_stats = df.groupby("department_name")["actual_waiting_time_min"].agg(
        Count="count", Mean="mean", Std="std", Median="median", IQR=lambda x: x.quantile(0.75) - x.quantile(0.25)
    ).round(2)
    print(dept_stats.to_string())

    # 3. Priority Tier Analysis
    print("\n--- Waiting Time Statistics by Patient Arrival Type ---")
    type_stats = df.groupby("patient_arrival_type")["actual_waiting_time_min"].agg(
        Count="count", Mean="mean", Std="std", Median="median", Max="max"
    ).round(2)
    print(type_stats.to_string())

    # 4. Correlation Analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()["actual_waiting_time_min"].drop("actual_waiting_time_min").sort_values(ascending=False)
    
    print("\n--- Top Positive Correlations with Waiting Time ---")
    print(correlations.head(10).round(3).to_string())

    print("\n--- Top Negative Correlations with Waiting Time ---")
    print(correlations.tail(5).round(3).to_string())


if __name__ == "__main__":
    run_eda()
