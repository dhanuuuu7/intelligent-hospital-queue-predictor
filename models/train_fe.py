"""
Step 8.3: Retrain & Benchmark Models with Feature Engineering
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

from models.data_loader import load_and_validate_dataset, CATEGORICAL_FEATURES, TARGET_COLUMN
from models.feature_engineering import (
    add_engineered_features, 
    ENGINEERED_NUMERICAL_FEATURES, 
    ALL_EXPANDED_NUMERICAL_FEATURES, 
    ALL_EXPANDED_PREDICTIVE_FEATURES
)
from models.preprocessing import create_full_pipeline
from models.train import calculate_metrics

# Baseline metrics from Phase 5 for direct benchmark comparison
BASELINE_METRICS = {
    "Random Forest": {"Val MAE": 11.8236, "Val R2": 0.0965},
    "XGBoost": {"Val MAE": 11.8257, "Val R2": 0.0312},
    "Gradient Boosting": {"Val MAE": 11.9945, "Val R2": -0.1267},
    "Linear Regression": {"Val MAE": 12.7968, "Val R2": 0.0534},
    "Decision Tree": {"Val MAE": 12.9164, "Val R2": -0.5066}
}


def get_chronological_splits_fe(
    df: pd.DataFrame, 
    train_ratio: float = 0.70, 
    val_ratio: float = 0.15, 
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits feature-engineered dataset chronologically into 70% Train, 15% Validation, and 15% Test.
    """
    X = df[ALL_EXPANDED_PREDICTIVE_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    total_rows = len(df)
    train_end_idx = int(total_rows * train_ratio)
    val_end_idx = int(total_rows * (train_ratio + val_ratio))

    X_train = X.iloc[:train_end_idx]
    y_train = y.iloc[:train_end_idx]
    X_val = X.iloc[train_end_idx:val_end_idx]
    y_val = y.iloc[train_end_idx:val_end_idx]
    X_test = X.iloc[val_end_idx:]
    y_test = y.iloc[val_end_idx:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def run_benchmark_step_8_3():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "hospital_encounters.csv")
    raw_df = load_and_validate_dataset(data_path)
    
    # 8.3.1 Prepare Feature-Engineered Dataset
    eng_df = add_engineered_features(raw_df)
    X_train_fe, y_train, X_val_fe, y_val, X_test_fe, y_test = get_chronological_splits_fe(eng_df)

    print("\n" + "=" * 70)
    print("STEP 8.3 — RETRAIN & BENCHMARK MODELS WITH FEATURE ENGINEERING")
    print("=" * 70)
    print(f"Features: 20 Raw Features + 4 Engineered Features = 24 Predictive Features")
    print(f"Train: {len(X_train_fe):,} | Val: {len(X_val_fe):,} | Test: {len(X_test_fe):,}")

    # 8.3.2 Retrain Candidate Models (Same Hyperparameters as Baseline)
    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    }

    if HAS_XGBOOST:
        candidate_models["XGBoost"] = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)

    results = []
    trained_pipelines = {}

    # 8.3.3 Evaluate on Validation Set
    for model_name, estimator in candidate_models.items():
        pipeline = create_full_pipeline(
            model=estimator,
            categorical_cols=CATEGORICAL_FEATURES,
            numerical_cols=ALL_EXPANDED_NUMERICAL_FEATURES,
            scale_numeric=True
        )
        
        pipeline.fit(X_train_fe, y_train)
        trained_pipelines[model_name] = pipeline

        train_preds = pipeline.predict(X_train_fe)
        val_preds = pipeline.predict(X_val_fe)

        train_metrics = calculate_metrics(y_train, train_preds)
        val_metrics = calculate_metrics(y_val, val_preds)

        base_val_mae = BASELINE_METRICS.get(model_name, {}).get("Val MAE", 0.0)
        base_val_r2 = BASELINE_METRICS.get(model_name, {}).get("Val R2", 0.0)

        mae_diff = base_val_mae - val_metrics["MAE"]
        pct_improvement = (mae_diff / base_val_mae) * 100.0 if base_val_mae > 0 else 0.0

        results.append({
            "Model": model_name,
            "Baseline Val MAE": base_val_mae,
            "FE Val MAE": val_metrics["MAE"],
            "MAE Improvement": round(mae_diff, 4),
            "% Improvement": round(pct_improvement, 2),
            "Baseline R2": base_val_r2,
            "FE R2": val_metrics["R2"],
            "FE Val RMSE": val_metrics["RMSE"]
        })

    results_df = pd.DataFrame(results).sort_values(by="FE Val MAE").reset_index(drop=True)

    print("\n--- 8.3.3 MODEL COMPARISON TABLE (Validation Set) ---")
    print(results_df.to_string(index=False))

    # 8.3.4 Model Selection (Based on Validation MAE & RMSE)
    best_fe_model_name = results_df.iloc[0]["Model"]
    best_fe_val_mae = results_df.iloc[0]["FE Val MAE"]
    best_fe_pipeline = trained_pipelines[best_fe_model_name]
    baseline_rf_val_mae = BASELINE_METRICS["Random Forest"]["Val MAE"]

    mae_change = baseline_rf_val_mae - best_fe_val_mae
    pct_change = (mae_change / baseline_rf_val_mae) * 100.0

    print("\n" + "=" * 70)
    print(f"BEST FEATURE-ENGINEERED MODEL: {best_fe_model_name}")
    print(f"   Validation MAE: {best_fe_val_mae:.4f} min (Baseline RF: {baseline_rf_val_mae:.4f} min)")
    print(f"   MAE Change    : {mae_change:+.4f} min ({pct_change:+.2f}%)")
    print("=" * 70)

    is_improved = best_fe_val_mae < baseline_rf_val_mae

    if is_improved:
        print(f"[Success] Feature engineering IMPROVED model performance by {pct_change:.2f}%!")
    else:
        print(f"[Notice] Feature engineering did NOT improve validation MAE ({best_fe_val_mae:.2f} vs {baseline_rf_val_mae:.2f} min).")

    # 8.3.5 Feature Importance Ranking
    print("\n--- 8.3.5 Feature Importance Ranking (Top 10) ---")
    preprocessor = best_fe_pipeline.named_steps["preprocessor"]
    model = best_fe_pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        cat_encoder = preprocessor.named_transformers_["cat"]
        encoded_cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        all_feature_names = encoded_cat_names + ALL_EXPANDED_NUMERICAL_FEATURES

        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": all_feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

        print(importance_df.head(10).round(4).to_string(index=False))

        print("\n--- Ranking of 4 Engineered Features ---")
        fe_ranks = importance_df[importance_df["Feature"].isin(ENGINEERED_NUMERICAL_FEATURES)].copy()
        fe_ranks["Rank"] = fe_ranks.index + 1
        print(fe_ranks.to_string(index=False))

    # 8.3.6 Department-Level Evaluation
    val_df_eng = eng_df.iloc[len(X_train_fe):len(X_train_fe) + len(X_val_fe)].copy()
    val_df_eng["pred_wait"] = best_fe_pipeline.predict(X_val_fe)
    val_df_eng["abs_err"] = (val_df_eng[TARGET_COLUMN] - val_df_eng["pred_wait"]).abs()
    val_df_eng["sq_err"] = (val_df_eng[TARGET_COLUMN] - val_df_eng["pred_wait"]) ** 2

    print("\n--- 8.3.6 Department-Level Performance (Validation Set) ---")
    dept_eval = val_df_eng.groupby("department_name").apply(
        lambda g: pd.Series({
            "Count": len(g),
            "Actual Mean": round(g[TARGET_COLUMN].mean(), 2),
            "Pred Mean": round(g["pred_wait"].mean(), 2),
            "MAE": round(g["abs_err"].mean(), 2),
            "RMSE": round(np.sqrt(g["sq_err"].mean()), 2),
            "R2": round(1 - (g["sq_err"].sum() / np.sum((g[TARGET_COLUMN] - g[TARGET_COLUMN].mean()) ** 2)), 4)
        }),
        include_groups=False
    )
    print(dept_eval.to_string())

    # 8.3.7 Arrival-Type Evaluation
    print("\n--- 8.3.7 Arrival-Type Performance (Validation Set) ---")
    type_eval = val_df_eng.groupby("patient_arrival_type").apply(
        lambda g: pd.Series({
            "Count": len(g),
            "Actual Mean": round(g[TARGET_COLUMN].mean(), 2),
            "Pred Mean": round(g["pred_wait"].mean(), 2),
            "MAE": round(g["abs_err"].mean(), 2),
            "RMSE": round(np.sqrt(g["sq_err"].mean()), 2)
        }),
        include_groups=False
    )
    print(type_eval.to_string())

    # 8.3.8 Untouched Test Set Evaluation
    test_preds = best_fe_pipeline.predict(X_test_fe)
    test_abs_err = np.abs(y_test.values - test_preds)
    test_metrics = calculate_metrics(y_test.values, test_preds)

    med_ae = np.median(test_abs_err)
    p75_ae = np.percentile(test_abs_err, 75)
    p90_ae = np.percentile(test_abs_err, 90)

    print("\n--- 8.3.8 UNTOUCHED TEST SET EVALUATION ---")
    print(f"Test MAE                 : {test_metrics['MAE']:.4f} min")
    print(f"Test RMSE                : {test_metrics['RMSE']:.4f} min")
    print(f"Test R2                  : {test_metrics['R2']:.4f}")
    print(f"Median Absolute Error    : {med_ae:.4f} min")
    print(f"75th Percentile Error    : {p75_ae:.4f} min")
    print(f"90th Percentile Error    : {p90_ae:.4f} min")

    # 8.3.9 Model Artifact Persistence
    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    fe_model_path = os.path.join(artifacts_dir, "best_waiting_time_model_fe.joblib")
    fe_metadata_path = os.path.join(artifacts_dir, "model_metadata_fe.json")

    joblib.dump(best_fe_pipeline, fe_model_path)
    print(f"\nSaved FE Model Pipeline to: {fe_model_path}")

    metadata_fe = {
        "model_name": best_fe_model_name,
        "feature_version": "2.0_engineered",
        "total_features_count": len(ALL_EXPANDED_PREDICTIVE_FEATURES),
        "input_features": ALL_EXPANDED_PREDICTIVE_FEATURES,
        "engineered_features": ENGINEERED_NUMERICAL_FEATURES,
        "target_variable": TARGET_COLUMN,
        "trained_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "validation_metrics": {
            "MAE": results_df.iloc[0]["FE Val MAE"],
            "RMSE": results_df.iloc[0]["FE Val RMSE"],
            "R2": results_df.iloc[0]["FE R2"],
            "Baseline_RF_Val_MAE": baseline_rf_val_mae,
            "MAE_Improvement_vs_Baseline": round(mae_change, 4),
            "Pct_Improvement_vs_Baseline": round(pct_change, 2)
        },
        "test_metrics": {
            "Test_MAE": test_metrics["MAE"],
            "Test_RMSE": test_metrics["RMSE"],
            "Test_R2": test_metrics["R2"],
            "Median_AE": round(float(med_ae), 4),
            "P75_AE": round(float(p75_ae), 4),
            "P90_AE": round(float(p90_ae), 4)
        }
    }

    with open(fe_metadata_path, "w") as f:
        json.dump(metadata_fe, f, indent=4)

    print(f"Saved FE Metadata to: {fe_metadata_path}")

    # Conclusion statement
    if is_improved:
        conclusion = f"Feature engineering improved validation MAE from {baseline_rf_val_mae:.2f} to {best_fe_val_mae:.2f} minutes ({pct_change:.2f}% improvement), therefore the feature-engineered model is preferred."
    else:
        conclusion = f"Feature engineering did not improve validation MAE ({best_fe_val_mae:.2f} vs {baseline_rf_val_mae:.2f} minutes), therefore the baseline model is currently preferred."

    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print(conclusion)
    print("=" * 70)

    return results_df, metadata_fe


def train_feature_engineered_rf(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[Any, Dict[str, float]]:
    """
    Fits and returns feature-engineered Random Forest pipeline and validation metrics.
    """
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    pipeline = create_full_pipeline(
        model=rf_model,
        categorical_cols=CATEGORICAL_FEATURES,
        numerical_cols=ALL_EXPANDED_NUMERICAL_FEATURES,
        scale_numeric=True
    )
    pipeline.fit(X_train, y_train)
    val_preds = pipeline.predict(X_val)
    metrics = calculate_metrics(y_val, val_preds)
    return pipeline, {"val_mae": metrics["MAE"], "val_rmse": metrics["RMSE"], "val_r2": metrics["R2"]}


if __name__ == "__main__":
    run_benchmark_step_8_3()
