"""
Step 5.4: Detailed Model Evaluation & Feature Importance Analysis
"""

import os
import joblib
import pandas as pd
import numpy as np
from models.data_loader import load_and_validate_dataset, get_chronological_splits, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN
from models.train import calculate_metrics


def evaluate_model_details():
    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    model_path = os.path.join(artifacts_dir, "best_waiting_time_model.joblib")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model artifact not found at {model_path}. Run models/train.py first!")

    pipeline = joblib.load(model_path)
    
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "hospital_encounters.csv")
    df = load_and_validate_dataset(data_path)
    X_train, y_train, X_val, y_val, X_test, y_test = get_chronological_splits(df)

    # Reconstruct test dataframe with predictions
    test_df = df.iloc[len(X_train) + len(X_val):].copy()
    test_df["predicted_waiting_time_min"] = pipeline.predict(X_test)
    test_df["absolute_error_min"] = (test_df["predicted_waiting_time_min"] - test_df[TARGET_COLUMN]).abs()

    print("\n" + "=" * 65)
    print("DETAILED MODEL EVALUATION & ERROR BREAKDOWN")
    print("=" * 65)

    # 1. Performance by Department
    print("\n--- Performance by Department ---")
    dept_metrics = test_df.groupby("department_name").apply(
        lambda g: pd.Series({
            "Count": len(g),
            "Actual Avg Wait": round(g[TARGET_COLUMN].mean(), 2),
            "Pred Avg Wait": round(g["predicted_waiting_time_min"].mean(), 2),
            "MAE (min)": round(g["absolute_error_min"].mean(), 2)
        }),
        include_groups=False
    )
    print(dept_metrics.to_string())

    # 2. Performance by Arrival Type
    print("\n--- Performance by Arrival Type ---")
    type_metrics = test_df.groupby("patient_arrival_type").apply(
        lambda g: pd.Series({
            "Count": len(g),
            "Actual Avg Wait": round(g[TARGET_COLUMN].mean(), 2),
            "Pred Avg Wait": round(g["predicted_waiting_time_min"].mean(), 2),
            "MAE (min)": round(g["absolute_error_min"].mean(), 2)
        }),
        include_groups=False
    )
    print(type_metrics.to_string())

    # 3. Feature Importance Analysis (Tree-based model inside pipeline)
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        cat_encoder = preprocessor.named_transformers_["cat"]
        encoded_cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        all_feature_names = encoded_cat_names + NUMERICAL_FEATURES

        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            "Feature": all_feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

        print("\n--- Top 10 Most Important Predictive Features ---")
        print(feature_importance_df.head(10).round(4).to_string(index=False))


if __name__ == "__main__":
    evaluate_model_details()
