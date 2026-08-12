"""
Step 8.4.2: Fast Controlled Staged Hyperparameter Tuning for Random Forest Regressor
"""

import os
import itertools
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

from sklearn.ensemble import RandomForestRegressor
from models.data_loader import load_and_validate_dataset, CATEGORICAL_FEATURES, TARGET_COLUMN
from models.feature_engineering import add_engineered_features, ALL_EXPANDED_NUMERICAL_FEATURES
from models.train_fe import get_chronological_splits_fe
from models.preprocessing import create_full_pipeline
from models.train import calculate_metrics


def run_random_forest_tuning() -> Tuple[pd.DataFrame, Dict[str, Any], object]:
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "hospital_encounters.csv")
    raw_df = load_and_validate_dataset(data_path)
    eng_df = add_engineered_features(raw_df)
    X_train, y_train, X_val, y_val, X_test, y_test = get_chronological_splits_fe(eng_df)

    print("\n" + "=" * 70)
    print("STEP 8.4.2 — CONTROLLED HYPERPARAMETER TUNING: RANDOM FOREST REGRESSOR")
    print("=" * 70)

    tuning_records = []

    # Fast staged grid: 6 Stage A models + 6 Stage B models = 12 total models
    stage_a_grid = {
        "n_estimators": [150, 300],
        "max_depth": [10, 15, 20]
    }

    best_stage_a_mae = float("inf")
    best_stage_a_params = {}

    keys_a = list(stage_a_grid.keys())
    for vals_a in itertools.product(*stage_a_grid.values()):
        params_a = dict(zip(keys_a, vals_a))
        
        model = RandomForestRegressor(
            n_estimators=params_a["n_estimators"],
            max_depth=params_a["max_depth"],
            random_state=42,
            n_jobs=-1
        )

        pipeline = create_full_pipeline(
            model=model,
            categorical_cols=CATEGORICAL_FEATURES,
            numerical_cols=ALL_EXPANDED_NUMERICAL_FEATURES,
            scale_numeric=True
        )

        pipeline.fit(X_train, y_train)
        train_preds = pipeline.predict(X_train)
        val_preds = pipeline.predict(X_val)

        train_m = calculate_metrics(y_train.values, train_preds)
        val_m = calculate_metrics(y_val.values, val_preds)

        record = {
            "Stage": "Stage_A",
            **params_a,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "Train_MAE": train_m["MAE"],
            "Val_MAE": val_m["MAE"],
            "Val_RMSE": val_m["RMSE"],
            "Val_R2": val_m["R2"],
            "Gen_Gap": round(val_m["MAE"] - train_m["MAE"], 4)
        }
        tuning_records.append(record)

        if val_m["MAE"] < best_stage_a_mae:
            best_stage_a_mae = val_m["MAE"]
            best_stage_a_params = params_a

    # Stage B: Regularization (max_features, min_samples_leaf)
    stage_b_grid = {
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", 0.7, 1.0]
    }

    best_val_mae = float("inf")
    best_overall_params = {}
    best_pipeline = None

    keys_b = list(stage_b_grid.keys())
    for vals_b in itertools.product(*stage_b_grid.values()):
        params_b = dict(zip(keys_b, vals_b))
        combined_params = {**best_stage_a_params, **params_b, "min_samples_split": 2}

        model = RandomForestRegressor(
            n_estimators=combined_params["n_estimators"],
            max_depth=combined_params["max_depth"],
            min_samples_split=combined_params["min_samples_split"],
            min_samples_leaf=combined_params["min_samples_leaf"],
            max_features=combined_params["max_features"],
            random_state=42,
            n_jobs=-1
        )

        pipeline = create_full_pipeline(
            model=model,
            categorical_cols=CATEGORICAL_FEATURES,
            numerical_cols=ALL_EXPANDED_NUMERICAL_FEATURES,
            scale_numeric=True
        )

        pipeline.fit(X_train, y_train)
        train_preds = pipeline.predict(X_train)
        val_preds = pipeline.predict(X_val)

        train_m = calculate_metrics(y_train.values, train_preds)
        val_m = calculate_metrics(y_val.values, val_preds)

        record = {
            "Stage": "Stage_B",
            **combined_params,
            "Train_MAE": train_m["MAE"],
            "Val_MAE": val_m["MAE"],
            "Val_RMSE": val_m["RMSE"],
            "Val_R2": val_m["R2"],
            "Gen_Gap": round(val_m["MAE"] - train_m["MAE"], 4)
        }
        tuning_records.append(record)

        if val_m["MAE"] < best_val_mae:
            best_val_mae = val_m["MAE"]
            best_overall_params = combined_params
            best_pipeline = pipeline

    results_df = pd.DataFrame(tuning_records).sort_values(by="Val_MAE").reset_index(drop=True)

    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    csv_path = os.path.join(artifacts_dir, "random_forest_tuning_results.csv")
    results_df.to_csv(csv_path, index=False)

    print(f"\n[Success] Random Forest Tuning Complete! Tested {len(results_df)} parameter combinations.")
    print(f"Best RF Params : {best_overall_params}")
    print(f"Best RF Val MAE: {best_val_mae:.4f} min")
    print(f"Saved results to: {csv_path}")

    return results_df, best_overall_params, best_pipeline


if __name__ == "__main__":
    run_random_forest_tuning()
