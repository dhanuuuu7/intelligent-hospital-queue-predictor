"""
Step 5.5: Production Prediction Interface for Hospital Queue Waiting Time
"""

import os
import joblib
import pandas as pd
from typing import Dict, Any, Union


from models.feature_engineering import add_engineered_features
from models.prediction_intervals import PredictionIntervalCalibrator


class WaitingTimePredictor:
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "artifacts", "best_waiting_time_model.joblib")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model artifact not found at {model_path}. Train model first!")

        self.pipeline = joblib.load(model_path)
        self.calibrator = PredictionIntervalCalibrator()

    def predict_encounter(self, encounter_data: Dict[str, Any]) -> float:
        """
        Predicts patient waiting time in minutes for a single patient arrival state.
        """
        res = self.predict_encounter_with_interval(encounter_data)
        return res["predicted_waiting_time_min"]

    def predict_encounter_with_interval(
        self, 
        encounter_data: Dict[str, Any], 
        confidence_level: float = 0.90
    ) -> Dict[str, Any]:
        """
        Predicts point estimate and uncertainty interval bounds for a single patient encounter.
        """
        df = pd.DataFrame([encounter_data])
        if "effective_doctor_load" not in df.columns:
            df = add_engineered_features(df)

        raw_pred = float(self.pipeline.predict(df)[0])
        point_pred = max(0.0, raw_pred)

        interval_res = self.calibrator.predict_interval(
            point_prediction=point_pred,
            confidence_level=confidence_level,
            department_name=encounter_data.get("department_name")
        )
        return interval_res


# Global singleton instance for easy import
_predictor_instance = None


def predict_waiting_time(encounter_data: Dict[str, Any]) -> float:
    """
    Standalone reusable prediction function.
    """
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = WaitingTimePredictor()
    return _predictor_instance.predict_encounter(encounter_data)


if __name__ == "__main__":
    # Test prediction on a sample patient arrival state
    sample_patient = {
        "department_name": "General Medicine",
        "day_of_week": 1,  # Tuesday
        "hour_of_day": 10, # 10:00 AM
        "is_weekend_or_holiday": 0,
        "season_indicator": "Winter",
        "queue_position": 8,
        "scheduled_patients_ahead": 5,
        "walkin_patients_ahead": 3,
        "patient_arrival_type": "Scheduled",
        "patient_case_complexity": "Medium",
        "active_doctors_count": 4,
        "doctors_on_break_count": 0,
        "triage_staff_count": 2,
        "shift_handover_flag": 0,
        "emergency_cases_active": 0,
        "arrival_rate_past_30min": 12.0,
        "recent_throughput_past_hour": 18,
        "avg_waiting_time_last_30min": 14.5,
        "dept_service_time_variability": 3.0,
        "hospital_occupancy_rate": 0.70
    }

    predicted_wait = predict_waiting_time(sample_patient)
    print("\n--- SAMPLE PRODUCTION INFERENCE ---")
    print(f"Department         : {sample_patient['department_name']}")
    print(f"Arrival Time       : Tuesday 10:00 AM")
    print(f"Queue Position     : #{sample_patient['queue_position']}")
    print(f"Active Doctors     : {sample_patient['active_doctors_count']}")
    print(f"Predicted Wait Time: {predicted_wait} minutes")
