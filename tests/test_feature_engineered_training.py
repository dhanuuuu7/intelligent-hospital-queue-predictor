"""
Step 8.3.10: Unit and Validation Tests for Feature-Engineered Model Pipeline & Artifacts
"""

import os
import pytest
import joblib
import pandas as pd
import numpy as np

from models.data_loader import load_and_validate_dataset, TARGET_COLUMN
from models.feature_engineering import add_engineered_features, ALL_EXPANDED_PREDICTIVE_FEATURES, ENGINEERED_NUMERICAL_FEATURES
from models.train_fe import get_chronological_splits_fe


@pytest.fixture
def fe_dataset_and_splits():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "hospital_encounters.csv")
    raw_df = load_and_validate_dataset(data_path)
    eng_df = add_engineered_features(raw_df)
    splits = get_chronological_splits_fe(eng_df)
    return eng_df, splits


def test_24_predictive_features_exist(fe_dataset_and_splits):
    eng_df, (X_train, y_train, X_val, y_val, X_test, y_test) = fe_dataset_and_splits
    assert len(ALL_EXPANDED_PREDICTIVE_FEATURES) == 24, "Expected 24 predictive features!"
    for col in ALL_EXPANDED_PREDICTIVE_FEATURES:
        assert col in X_train.columns, f"Feature {col} missing in X_train!"


def test_engineered_features_no_nans(fe_dataset_and_splits):
    eng_df, (X_train, y_train, X_val, y_val, X_test, y_test) = fe_dataset_and_splits
    for col in ENGINEERED_NUMERICAL_FEATURES:
        assert not X_train[col].isnull().any(), f"NaN found in X_train feature {col}"
        assert not X_val[col].isnull().any(), f"NaN found in X_val feature {col}"
        assert not X_test[col].isnull().any(), f"NaN found in X_test feature {col}"


def test_no_infinite_values(fe_dataset_and_splits):
    eng_df, (X_train, y_train, X_val, y_val, X_test, y_test) = fe_dataset_and_splits
    for col in ENGINEERED_NUMERICAL_FEATURES:
        assert not np.isinf(X_train[col]).any(), f"Inf found in X_train feature {col}"
        assert not np.isinf(X_val[col]).any(), f"Inf found in X_val feature {col}"
        assert not np.isinf(X_test[col]).any(), f"Inf found in X_test feature {col}"


def test_chronological_split_ordering(fe_dataset_and_splits):
    eng_df, (X_train, y_train, X_val, y_val, X_test, y_test) = fe_dataset_and_splits
    n = len(eng_df)
    assert len(X_train) == int(n * 0.70)
    assert len(X_val) == int(n * 0.85) - int(n * 0.70)
    assert len(X_test) == n - int(n * 0.85)


def test_target_excluded_from_X(fe_dataset_and_splits):
    eng_df, (X_train, y_train, X_val, y_val, X_test, y_test) = fe_dataset_and_splits
    assert TARGET_COLUMN not in X_train.columns, "Target column must not be in X_train!"
    assert TARGET_COLUMN not in X_val.columns, "Target column must not be in X_val!"
    assert TARGET_COLUMN not in X_test.columns, "Target column must not be in X_test!"


def test_fe_model_artifact_load_and_predict(fe_dataset_and_splits):
    eng_df, (X_train, y_train, X_val, y_val, X_test, y_test) = fe_dataset_and_splits
    artifact_path = os.path.join(os.path.dirname(__file__), "..", "models", "artifacts", "best_waiting_time_model_fe.joblib")
    
    assert os.path.exists(artifact_path), "FE model artifact missing!"
    pipeline = joblib.load(artifact_path)

    preds = pipeline.predict(X_val.head(10))
    assert len(preds) == 10
    assert not np.isnan(preds).any(), "Predictions contain NaN values!"
    assert not np.isinf(preds).any(), "Predictions contain Inf values!"
