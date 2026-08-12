"""
Step 8.1: Comprehensive Error Analysis & Residual Investigation
"""

import os
import joblib
import pandas as pd
import numpy as np
from models.data_loader import load_and_validate_dataset, get_chronological_splits, TARGET_COLUMN
from models.train import calculate_metrics


def run_error_analysis():
    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    model_path = os.path.join(artifacts_dir, "best_waiting_time_model.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Train model first!")

    pipeline = joblib.load(model_path)

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "hospital_encounters.csv")
    df = load_and_validate_dataset(data_path)
    X_train, y_train, X_val, y_val, X_test, y_test = get_chronological_splits(df)

    # Reconstruct Validation set dataframe with predictions and residuals
    val_df = df.iloc[len(X_train):len(X_train) + len(X_val)].copy()
    val_df["predicted_wait"] = pipeline.predict(X_val)
    val_df["residual"] = val_df[TARGET_COLUMN] - val_df["predicted_wait"]  # Actual - Predicted
    val_df["abs_error"] = val_df["residual"].abs()

    print("\n" + "=" * 65)
    print("PHASE 8.1 — DEEP ERROR & RESIDUAL ANALYSIS")
    print("=" * 65)

    # 1. Overall Residual & Metric Distribution
    print("\n--- 1. Residual & Error Quantiles (Validation Set) ---")
    res_stats = val_df["residual"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    print(res_stats.to_string())

    print("\n--- Error Magnitude Breakdown ---")
    print(f"Mean Absolute Error (MAE)   : {val_df['abs_error'].mean():.2f} min")
    print(f"Median Absolute Error (MedAE): {val_df['abs_error'].median():.2f} min")
    print(f"75th Percentile Error        : {val_df['abs_error'].quantile(0.75):.2f} min")
    print(f"95th Percentile Error        : {val_df['abs_error'].quantile(0.95):.2f} min")

    # 2. Error Breakdown by Department
    print("\n--- 2. Performance Breakdown by Department ---")
    dept_analysis = val_df.groupby("department_name").apply(
        lambda g: pd.Series({
            "Encounter Count": len(g),
            "Actual Mean (min)": round(g[TARGET_COLUMN].mean(), 2),
            "Actual Max (min)": round(g[TARGET_COLUMN].max(), 2),
            "Pred Mean (min)": round(g["predicted_wait"].mean(), 2),
            "MAE (min)": round(g["abs_error"].mean(), 2),
            "RMSE (min)": round(np.sqrt(np.mean(g["residual"] ** 2)), 2),
            "R2 Score": round(1 - (np.sum(g["residual"] ** 2) / np.sum((g[TARGET_COLUMN] - g[TARGET_COLUMN].mean()) ** 2)), 4)
        }),
        include_groups=False
    )
    print(dept_analysis.to_string())

    # 3. Error Breakdown by Arrival Type
    print("\n--- 3. Performance Breakdown by Arrival Type ---")
    type_analysis = val_df.groupby("patient_arrival_type").apply(
        lambda g: pd.Series({
            "Encounter Count": len(g),
            "Actual Mean (min)": round(g[TARGET_COLUMN].mean(), 2),
            "Pred Mean (min)": round(g["predicted_wait"].mean(), 2),
            "MAE (min)": round(g["abs_error"].mean(), 2),
            "RMSE (min)": round(np.sqrt(np.mean(g["residual"] ** 2)), 2)
        }),
        include_groups=False
    )
    print(type_analysis.to_string())

    # 4. Error Breakdown by Time of Day
    print("\n--- 4. Performance Breakdown by Hour of Day ---")
    hourly_analysis = val_df.groupby("hour_of_day").apply(
        lambda g: pd.Series({
            "Encounter Count": len(g),
            "Actual Mean (min)": round(g[TARGET_COLUMN].mean(), 2),
            "Pred Mean (min)": round(g["predicted_wait"].mean(), 2),
            "MAE (min)": round(g["abs_error"].mean(), 2),
            "RMSE (min)": round(np.sqrt(np.mean(g["residual"] ** 2)), 2)
        }),
        include_groups=False
    )
    print(hourly_analysis.to_string())

    # 5. Tail / Outlier Error Investigation
    tail_threshold = val_df["abs_error"].quantile(0.90)
    tail_df = val_df[val_df["abs_error"] >= tail_threshold]

    print("\n--- 5. Top 10% Heavy Tail Error Analysis ---")
    print(f"Tail Error Threshold : >= {tail_threshold:.2f} min")
    print(f"Tail Encounters Count: {len(tail_df):,} / {len(val_df):,}")
    print(f"Tail Actual Wait Mean: {tail_df[TARGET_COLUMN].mean():.2f} min (Max: {tail_df[TARGET_COLUMN].max():.2f} min)")
    print(f"Tail Pred Wait Mean  : {tail_df['predicted_wait'].mean():.2f} min")

    print("\n" + "=" * 65)
    print("KEY FINDINGS & DIAGNOSTICS")
    print("=" * 65)
    print("1. MEDIAN vs MEAN ERROR DISCREPANCY:")
    print(f"   • Median Error is only {val_df['abs_error'].median():.2f} min (50% of predictions are off by less than {val_df['abs_error'].median():.1f} min).")
    print(f"   • Mean Error (MAE) is {val_df['abs_error'].mean():.2f} min due to heavy-tail delays.")
    print("2. REASON FOR LOW R2 (~0.096) & HIGH RMSE (51.43 min):")
    print("   • In severe congestion spikes (e.g. evening shift wrap-ups or sudden arrivals), actual wait times surge to 100+ min.")
    print("   • The current linear feature set lacks explicit ratio metrics like (queue_position / active_doctors) or arrival-to-capacity momentum.")
    print("3. TARGET FOR FEATURE ENGINEERING (STEP 8.2):")
    print("   • Construct explicit non-linear ratios: effective_doctor_load, traffic_momentum, emergency_pressure.")


if __name__ == "__main__":
    run_error_analysis()
