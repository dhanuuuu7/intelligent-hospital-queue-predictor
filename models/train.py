"""
Step 5.3: Model Training, Validation Comparison, Best Model Selection & Persistence
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from models.data_loader import load_and_validate_dataset, get_chronological_splits, ALL_PREDICTIVE_FEATURES, TARGET_COLUMN
from models.preprocessing import create_full_pipeline


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates Regression Evaluation Metrics:
    - MAE  : Mean Absolute Error (minutes)
    - RMSE : Root Mean Squared Error (minutes)
    - R2   : Coefficient of Determination
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4)
    }


def train_and_evaluate():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "hospital_encounters.csv")
    df = load_and_validate_dataset(data_path)
    X_train, y_train, X_val, y_val, X_test, y_test = get_chronological_splits(df)

    # 1. Candidate Models Definition
    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    }

    if HAS_XGBOOST:
        candidate_models["XGBoost"] = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)

    print("\n" + "=" * 65)
    print("TRAINING & EVALUATING BASELINE ML MODELS")
    print("=" * 65)

    results = []
    trained_pipelines = {}

    for model_name, estimator in candidate_models.items():
        print(f"\nTraining {model_name}...")
        pipeline = create_full_pipeline(model=estimator, scale_numeric=True)
        
        # Train strictly on Training Set
        pipeline.fit(X_train, y_train)
        trained_pipelines[model_name] = pipeline

        # Predict on Train & Validation sets
        train_preds = pipeline.predict(X_train)
        val_preds = pipeline.predict(X_val)

        train_metrics = calculate_metrics(y_train, train_preds)
        val_metrics = calculate_metrics(y_val, val_preds)

        results.append({
            "Model": model_name,
            "Train MAE (min)": train_metrics["MAE"],
            "Val MAE (min)": val_metrics["MAE"],
            "Val RMSE (min)": val_metrics["RMSE"],
            "Val R2": val_metrics["R2"]
        })

    results_df = pd.DataFrame(results).sort_values(by="Val MAE (min)").reset_index(drop=True)

    print("\n--- MODEL COMPARISON TABLE (Validation Set) ---")
    print(results_df.to_string(index=False))

    # 2. Select Best Model Based on Lowest Validation MAE
    best_model_name = results_df.iloc[0]["Model"]
    best_val_mae = results_df.iloc[0]["Val MAE (min)"]
    best_pipeline = trained_pipelines[best_model_name]

    print("\n" + "=" * 65)
    print(f"BEST MODEL SELECTED: {best_model_name} (Val MAE: {best_val_mae:.2f} min)")
    print("=" * 65)

    # 3. Final Verification on Untouched Test Set
    test_preds = best_pipeline.predict(X_test)
    test_metrics = calculate_metrics(y_test, test_preds)

    print("\n--- UNTOUCHED TEST SET PERFORMANCE ---")
    print(f"Test MAE  : {test_metrics['MAE']} minutes")
    print(f"Test RMSE : {test_metrics['RMSE']} minutes")
    print(f"Test R2   : {test_metrics['R2']}")

    # 4. Save Artifacts (Model + Metadata)
    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    model_artifact_path = os.path.join(artifacts_dir, "best_waiting_time_model.joblib")
    metadata_artifact_path = os.path.join(artifacts_dir, "model_metadata.json")

    joblib.dump(best_pipeline, model_artifact_path)
    print(f"\nModel Pipeline saved to: {model_artifact_path}")

    metadata = {
        "model_name": best_model_name,
        "trained_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_dataset_records": len(df),
        "train_records": len(X_train),
        "val_records": len(X_val),
        "test_records": len(X_test),
        "input_features": ALL_PREDICTIVE_FEATURES,
        "target_variable": TARGET_COLUMN,
        "validation_metrics": {
            "MAE": results_df.iloc[0]["Val MAE (min)"],
            "RMSE": results_df.iloc[0]["Val RMSE (min)"],
            "R2": results_df.iloc[0]["Val R2"]
        },
        "test_metrics": test_metrics
    }

    with open(metadata_artifact_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to: {metadata_artifact_path}")
    return results_df, test_metrics


if __name__ == "__main__":
    train_and_evaluate()
