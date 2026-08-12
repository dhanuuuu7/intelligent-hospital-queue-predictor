"""
Step 8.4: Hyperparameter Tuning Orchestrator, Stability Analysis & Final Artifact Persistence
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone

from models.data_loader import load_and_validate_dataset, CATEGORICAL_FEATURES, TARGET_COLUMN
from models.feature_engineering import add_engineered_features, ALL_EXPANDED_NUMERICAL_FEATURES, ALL_EXPANDED_PREDICTIVE_FEATURES
from models.train_fe import get_chronological_splits_fe
from models.preprocessing import create_full_pipeline
from models.train import calculate_metrics
from models.tune_xgboost import run_xgboost_tuning
from models.tune_random_forest import run_random_forest_tuning


def run_full_step_8_4():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "hospital_encounters.csv")
    raw_df = load_and_validate_dataset(data_path)
    eng_df = add_engineered_features(raw_df)
    X_train, y_train, X_val, y_val, X_test, y_test = get_chronological_splits_fe(eng_df)

    # 1. Run XGBoost & Random Forest Tuning
    xgb_df, xgb_params, xgb_pipeline = run_xgboost_tuning()
    rf_df, rf_params, rf_pipeline = run_random_forest_tuning()

    print("\n" + "=" * 75)
    print("STEP 8.4.3 — FINAL TUNING COMPARISON TABLE (Validation Set)")
    print("=" * 75)

    best_xgb_val_mae = xgb_df.iloc[0]["Val_MAE"]
    best_xgb_val_rmse = xgb_df.iloc[0]["Val_RMSE"]
    best_xgb_val_r2 = xgb_df.iloc[0]["Val_R2"]
    best_xgb_gen_gap = xgb_df.iloc[0]["Gen_Gap"]

    best_rf_val_mae = rf_df.iloc[0]["Val_MAE"]
    best_rf_val_rmse = rf_df.iloc[0]["Val_RMSE"]
    best_rf_val_r2 = rf_df.iloc[0]["Val_R2"]
    best_rf_gen_gap = rf_df.iloc[0]["Gen_Gap"]

    comparison_rows = [
        {"Model": "Baseline Random Forest", "Validation MAE": 11.8236, "Validation RMSE": 51.4318, "Validation R2": 0.0965, "Gen Gap": 4.6769},
        {"Model": "Baseline XGBoost", "Validation MAE": 11.8257, "Validation RMSE": 53.2574, "Validation R2": 0.0312, "Gen Gap": 4.7064},
        {"Model": "FE Random Forest", "Validation MAE": 11.7083, "Validation RMSE": 50.3472, "Validation R2": 0.1342, "Gen Gap": 4.5616},
        {"Model": "FE XGBoost", "Validation MAE": 11.3824, "Validation RMSE": 53.9014, "Validation R2": 0.0076, "Gen Gap": 4.2631},
        {"Model": "Tuned Random Forest", "Validation MAE": best_rf_val_mae, "Validation RMSE": best_rf_val_rmse, "Validation R2": best_rf_val_r2, "Gen Gap": best_rf_gen_gap},
        {"Model": "Tuned XGBoost", "Validation MAE": best_xgb_val_mae, "Validation RMSE": best_xgb_val_rmse, "Validation R2": best_xgb_val_r2, "Gen Gap": best_xgb_gen_gap}
    ]

    comp_df = pd.DataFrame(comparison_rows).sort_values(by="Validation MAE").reset_index(drop=True)
    print(comp_df.to_string(index=False))

    # 8.4.4 Overfitting Analysis
    print("\n--- 8.4.4 Overfitting & Generalization Gap Analysis ---")
    print(f"Tuned XGBoost Gen Gap : {best_xgb_gen_gap:+.4f} min (Val MAE - Train MAE)")
    print(f"Tuned RF Gen Gap      : {best_rf_gen_gap:+.4f} min (Val MAE - Train MAE)")

    # 8.4.5 Time-Aware Stability Check (3-Window TimeSeriesSplit on Train+Val)
    print("\n--- 8.4.5 Time-Aware Stability Check (TimeSeriesSplit 3 Windows) ---")
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)

    tscv = TimeSeriesSplit(n_splits=3)
    xgb_cv_maes = []
    rf_cv_maes = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_val), 1):
        X_tr_w, y_tr_w = X_train_val.iloc[train_idx], y_train_val.iloc[train_idx]
        X_va_w, y_va_w = X_train_val.iloc[val_idx], y_train_val.iloc[val_idx]

        xgb_p = clone(xgb_pipeline)
        xgb_p.fit(X_tr_w, y_tr_w)
        xgb_mae = mean_absolute_error(y_va_w, xgb_p.predict(X_va_w))
        xgb_cv_maes.append(xgb_mae)

        rf_p = clone(rf_pipeline)
        rf_p.fit(X_tr_w, y_tr_w)
        rf_mae = mean_absolute_error(y_va_w, rf_p.predict(X_va_w))
        rf_cv_maes.append(rf_mae)

    print(f"Tuned XGBoost CV MAEs : {[round(m, 2) for m in xgb_cv_maes]} | Mean: {np.mean(xgb_cv_maes):.2f} ± {np.std(xgb_cv_maes):.2f} min")
    print(f"Tuned RF CV MAEs      : {[round(m, 2) for m in rf_cv_maes]} | Mean: {np.mean(rf_cv_maes):.2f} ± {np.std(rf_cv_maes):.2f} min")

    # 8.4.6 Final Model Selection based strictly on Validation & Stability
    if best_xgb_val_mae <= best_rf_val_mae:
        final_model_name = "Tuned XGBoost"
        final_pipeline = xgb_pipeline
        final_params = xgb_params
        final_val_mae = best_xgb_val_mae
    else:
        final_model_name = "Tuned Random Forest"
        final_pipeline = rf_pipeline
        final_params = rf_params
        final_val_mae = best_rf_val_mae

    baseline_fe_mae = 11.3824
    mae_diff = baseline_fe_mae - final_val_mae
    pct_imp = (mae_diff / baseline_fe_mae) * 100.0

    print("\n" + "=" * 75)
    print(f"FINAL SELECTED MODEL: {final_model_name}")
    print(f"   Validation MAE: {final_val_mae:.4f} min (FE Baseline: {baseline_fe_mae:.4f} min)")
    print(f"   Tuning Gain   : {mae_diff:+.4f} min ({pct_imp:+.2f}%)")
    print("=" * 75)

    # 8.4.7 Final Test Set Evaluation (UNTOUCHED TEST SET)
    test_preds = final_pipeline.predict(X_test)
    test_abs_err = np.abs(y_test.values - test_preds)
    test_metrics = calculate_metrics(y_test.values, test_preds)

    med_ae = np.median(test_abs_err)
    p75_ae = np.percentile(test_abs_err, 75)
    p90_ae = np.percentile(test_abs_err, 90)

    print("\n--- 8.4.7 UNTOUCHED TEST SET EVALUATION ---")
    print(f"Test MAE                 : {test_metrics['MAE']:.4f} min")
    print(f"Test RMSE                : {test_metrics['RMSE']:.4f} min")
    print(f"Test R2                  : {test_metrics['R2']:.4f}")
    print(f"Median Absolute Error    : {med_ae:.4f} min")
    print(f"75th Percentile Error    : {p75_ae:.4f} min")
    print(f"90th Percentile Error    : {p90_ae:.4f} min")

    # 8.4.8 Save Final Artifacts
    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    final_model_path = os.path.join(artifacts_dir, "final_waiting_time_model.joblib")
    final_metadata_path = os.path.join(artifacts_dir, "final_model_metadata.json")
    final_importance_path = os.path.join(artifacts_dir, "final_feature_importance.csv")

    joblib.dump(final_pipeline, final_model_path)
    joblib.dump(final_pipeline, os.path.join(artifacts_dir, "best_waiting_time_model.joblib"))

    # Feature Importance Export
    preprocessor = final_pipeline.named_steps["preprocessor"]
    model = final_pipeline.named_steps["model"]
    cat_encoder = preprocessor.named_transformers_["cat"]
    encoded_cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_feature_names = encoded_cat_names + ALL_EXPANDED_NUMERICAL_FEATURES

    if hasattr(model, "feature_importances_"):
        fi_df = pd.DataFrame({
            "Feature": all_feature_names,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
        fi_df.to_csv(final_importance_path, index=False)

    metadata_final = {
        "model_name": final_model_name,
        "feature_version": "2.0_engineered_and_tuned",
        "total_features": len(ALL_EXPANDED_PREDICTIVE_FEATURES),
        "input_features": ALL_EXPANDED_PREDICTIVE_FEATURES,
        "hyperparameters": final_params,
        "trained_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "validation_metrics": {
            "MAE": final_val_mae,
            "RMSE": comp_df.iloc[0]["Validation RMSE"],
            "R2": comp_df.iloc[0]["Validation R2"]
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

    with open(final_metadata_path, "w") as f:
        json.dump(metadata_final, f, indent=4)

    print(f"\nSaved Final Model Artifact to: {final_model_path}")
    print(f"Saved Final Metadata to: {final_metadata_path}")
    print(f"Saved Feature Importance to: {final_importance_path}")

    # Conclusion summary
    if abs(pct_imp) < 1.0:
        imp_desc = "modest/negligible"
    elif pct_imp >= 1.0:
        imp_desc = "substantial"
    else:
        imp_desc = "negative"

    conclusion = f"Hyperparameter tuning changed validation MAE from 11.38 to {final_val_mae:.2f} minutes, representing a {pct_imp:+.2f}% ({imp_desc}) change. The final selected model is {final_model_name}."
    print("\n" + "=" * 75)
    print("ENGINEERING CONCLUSION:")
    print(conclusion)
    print("=" * 75)

    return comp_df, metadata_final


if __name__ == "__main__":
    run_full_step_8_4()
