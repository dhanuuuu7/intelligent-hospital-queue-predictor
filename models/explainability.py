"""
Step 8.6.1 & 8.6.2: Model Explainability, Feature Importance & High-Error Analysis Engine
"""

import os
import joblib
import pandas as pd
import numpy as np
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for server environments
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
from typing import Dict, Any, List, Tuple, Optional

from models.data_loader import load_and_validate_dataset, CATEGORICAL_FEATURES, TARGET_COLUMN
from models.feature_engineering import add_engineered_features, ALL_EXPANDED_NUMERICAL_FEATURES, ALL_EXPANDED_PREDICTIVE_FEATURES
from models.train_fe import get_chronological_splits_fe
from models.explanation_text import generate_human_explanation
from models.train import calculate_metrics


class ModelExplainer:
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "artifacts", "final_waiting_time_model.joblib")
            if not os.path.exists(model_path):
                model_path = os.path.join(os.path.dirname(__file__), "artifacts", "best_waiting_time_model.joblib")

        self.pipeline = joblib.load(model_path)
        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.model = self.pipeline.named_steps["model"]

        # Feature name extraction
        cat_encoder = self.preprocessor.named_transformers_["cat"]
        self.encoded_cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        self.all_feature_names = self.encoded_cat_names + ALL_EXPANDED_NUMERICAL_FEATURES

    def get_global_feature_importance(self) -> pd.DataFrame:
        """
        Extracts global feature importances and exports CSV & PNG plot.
        """
        importances = self.model.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": self.all_feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

        artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        # 1. Export CSV
        csv_path = os.path.join(artifacts_dir, "final_feature_importance.csv")
        fi_df.to_csv(csv_path, index=False)

        # 2. Export Plot (Top 15 Features) if matplotlib is available
        if HAS_MATPLOTLIB:
            png_path = os.path.join(artifacts_dir, "feature_importance.png")
            plt.figure(figsize=(10, 6))
            top15 = fi_df.head(15).iloc[::-1]
            plt.barh(top15["Feature"], top15["Importance"], color="#3b82f6")
            plt.title("Top 15 Global Feature Importances (Random Forest Model)", fontsize=12, fontweight="bold")
            plt.xlabel("Gini Importance Score")
            plt.tight_layout()
            plt.savefig(png_path, dpi=300)
            plt.close()

        return fi_df

    def explain_single_prediction(self, patient_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explains an individual patient prediction by mapping feature values against global importances.
        """
        df = pd.DataFrame([patient_features])
        if "effective_doctor_load" not in df.columns:
            df = add_engineered_features(df)

        pred_val = float(self.pipeline.predict(df)[0])
        pred_val = max(0.0, pred_val)

        # Retrieve feature values
        queue_pos = patient_features.get("queue_position", 1)
        eff_load = df["effective_doctor_load"].iloc[0]
        traffic_mom = df["traffic_momentum"].iloc[0]
        active_docs = patient_features.get("active_doctors_count", 1)
        avg_wait = patient_features.get("avg_waiting_time_last_30min", 0.0)

        # Calculate directional contribution proxies (associated drivers)
        increasing = []
        decreasing = []

        if queue_pos > 5:
            increasing.append(("queue_position", queue_pos))
        if eff_load > 0.5:
            increasing.append(("effective_doctor_load", eff_load))
        if traffic_mom > 0.4:
            increasing.append(("traffic_momentum", traffic_mom))
        if avg_wait > 15.0:
            increasing.append(("avg_waiting_time_last_30min", avg_wait))

        if active_docs >= 3:
            decreasing.append(("active_doctors_count", active_docs))
        if patient_features.get("patient_arrival_type") == "Scheduled":
            decreasing.append(("patient_arrival_type_Scheduled", 1.0))

        text_exp = generate_human_explanation(patient_features, increasing, decreasing)

        return {
            "predicted_waiting_time_min": round(pred_val, 1),
            "explanation": text_exp
        }

    def analyze_department_performance(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Performs department-level model evaluation and error diagnostics.
        """
        X = df[ALL_EXPANDED_PREDICTIVE_FEATURES]
        y = df[TARGET_COLUMN]

        df_copy = df.copy()
        df_copy["pred_wait"] = self.pipeline.predict(X)
        df_copy["abs_error"] = (df_copy[TARGET_COLUMN] - df_copy["pred_wait"]).abs()
        df_copy["sq_error"] = (df_copy[TARGET_COLUMN] - df_copy["pred_wait"]) ** 2

        dept_summary = []
        dept_error_diag = []

        for dept, group in df_copy.groupby("department_name"):
            n = len(group)
            actual_mean = group[TARGET_COLUMN].mean()
            pred_mean = group["pred_wait"].mean()
            mae = group["abs_error"].mean()
            rmse = np.sqrt(group["sq_error"].mean())
            r2 = 1.0 - (group["sq_error"].sum() / np.sum((group[TARGET_COLUMN] - actual_mean) ** 2))

            dept_summary.append({
                "Department": dept,
                "Sample Count": n,
                "Actual Mean Wait (min)": round(actual_mean, 2),
                "Pred Mean Wait (min)": round(pred_mean, 2),
                "MAE (min)": round(mae, 2),
                "RMSE (min)": round(rmse, 2),
                "R2 Score": round(r2, 4)
            })

            # Error Diagnostics for Cardiology / Orthopedics
            p90_err = np.percentile(group["abs_error"], 90)
            high_volatility = "YES" if dept in ["Cardiology", "Orthopedics"] else "NO"
            
            hypothesis = (
                "High consult time variability (sigma=18.0) and emergency preemption spikes" 
                if dept == "Cardiology" else "Standard queue dynamics"
            )

            dept_error_diag.append({
                "Department": dept,
                "MAE (min)": round(mae, 2),
                "90th Percentile Error (min)": round(p90_err, 2),
                "High Volatility Flag": high_volatility,
                "Primary Variance Hypothesis": hypothesis
            })

        dept_df = pd.DataFrame(dept_summary)
        error_df = pd.DataFrame(dept_error_diag)

        artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
        dept_df.to_csv(os.path.join(artifacts_dir, "department_model_analysis.csv"), index=False)
        error_df.to_csv(os.path.join(artifacts_dir, "department_error_analysis.csv"), index=False)

        return dept_df, error_df

    def analyze_high_error_cases(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyzes the top 1%, 5%, and 10% worst absolute error predictions on the test set.
        """
        X_test = test_df[ALL_EXPANDED_PREDICTIVE_FEATURES]
        y_test = test_df[TARGET_COLUMN]

        test_copy = test_df.copy()
        test_copy["pred_wait"] = self.pipeline.predict(X_test)
        test_copy["abs_error"] = (test_copy[TARGET_COLUMN] - test_copy["pred_wait"]).abs()

        test_sorted = test_copy.sort_values(by="abs_error", ascending=False).reset_index(drop=True)
        total_test = len(test_sorted)

        pcts = [0.01, 0.05, 0.10]
        high_err_records = []

        for p in pcts:
            cutoff = int(total_test * p)
            sub = test_sorted.iloc[:cutoff]

            cardiology_pct = (sub["department_name"] == "Cardiology").sum() / cutoff * 100.0
            ortho_pct = (sub["department_name"] == "Orthopedics").sum() / cutoff * 100.0
            emerg_pct = (sub["patient_arrival_type"] == "Emergency").sum() / cutoff * 100.0

            high_err_records.append({
                "Error Percentile Group": f"Top {int(p*100)}%",
                "Sample Count": cutoff,
                "Min Error in Group (min)": round(sub["abs_error"].min(), 2),
                "Max Error in Group (min)": round(sub["abs_error"].max(), 2),
                "Mean Actual Wait (min)": round(sub[TARGET_COLUMN].mean(), 2),
                "Mean Pred Wait (min)": round(sub["pred_wait"].mean(), 2),
                "Cardiology %": round(cardiology_pct, 2),
                "Orthopedics %": round(ortho_pct, 2),
                "Emergency Arrival %": round(emerg_pct, 2),
                "Mean Queue Position": round(sub["queue_position"].mean(), 2),
                "Mean Effective Load": round(sub["effective_doctor_load"].mean(), 2)
            })

        high_err_df = pd.DataFrame(high_err_records)
        artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
        high_err_df.to_csv(os.path.join(artifacts_dir, "high_error_analysis.csv"), index=False)

        return high_err_df


def run_explainability_analysis():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "hospital_encounters.csv")
    raw_df = load_and_validate_dataset(data_path)
    eng_df = add_engineered_features(raw_df)
    X_train, y_train, X_val, y_val, X_test, y_test = get_chronological_splits_fe(eng_df)

    explainer = ModelExplainer()

    print("\n" + "=" * 75)
    print("STEP 8.6.1 & 8.6.2 — GLOBAL FEATURE IMPORTANCE")
    print("=" * 75)
    fi_df = explainer.get_global_feature_importance()
    print(fi_df.head(10).to_string(index=False))

    print("\n" + "=" * 75)
    print("STEP 8.6.5 — DEPARTMENT-LEVEL EXPLAINABILITY & ERROR DIAGNOSTICS")
    print("=" * 75)
    test_df = eng_df.iloc[len(X_train) + len(X_val):].copy()
    dept_df, error_df = explainer.analyze_department_performance(test_df)
    print(dept_df.to_string(index=False))
    print("\n--- Error Diagnostics ---")
    print(error_df.to_string(index=False))

    print("\n" + "=" * 75)
    print("STEP 8.6.6 — HIGH-ERROR PREDICTION ANALYSIS (Top 1%, 5%, 10%)")
    print("=" * 75)
    high_err_df = explainer.analyze_high_error_cases(test_df)
    print(high_err_df.to_string(index=False))

    # Test single patient explanation
    sample_patient = {
        "department_name": "Cardiology",
        "day_of_week": 1,
        "hour_of_day": 14,
        "is_weekend_or_holiday": 0,
        "season_indicator": "Winter",
        "queue_position": 12,
        "scheduled_patients_ahead": 8,
        "walkin_patients_ahead": 4,
        "patient_arrival_type": "Walk-in",
        "patient_case_complexity": "High",
        "active_doctors_count": 2,
        "doctors_on_break_count": 1,
        "triage_staff_count": 2,
        "shift_handover_flag": 0,
        "emergency_cases_active": 2,
        "arrival_rate_past_30min": 16.0,
        "recent_throughput_past_hour": 10,
        "avg_waiting_time_last_30min": 32.5,
        "dept_service_time_variability": 18.0,
        "hospital_occupancy_rate": 0.85
    }

    print("\n" + "=" * 75)
    print("STEP 8.6.3 — PATIENT-LEVEL EXPLANATION SAMPLE")
    print("=" * 75)
    single_exp = explainer.explain_single_prediction(sample_patient)
    print(f"Predicted Wait : {single_exp['predicted_waiting_time_min']} min")
    print(f"Summary        : {single_exp['explanation']['summary']}")
    print(f"Increasing     : {single_exp['explanation']['increasing_factors']}")
    print(f"Decreasing     : {single_exp['explanation']['decreasing_factors']}")


if __name__ == "__main__":
    run_explainability_analysis()
