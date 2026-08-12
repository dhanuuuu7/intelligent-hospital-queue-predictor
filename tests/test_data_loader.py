"""
Unit tests for Step 5.1 Data Loader and Chronological Splitter
"""

import os
import pytest
import pandas as pd
from models.data_loader import load_and_validate_dataset, get_chronological_splits, ALL_PREDICTIVE_FEATURES, TARGET_COLUMN


@pytest.fixture
def dataset_path():
    return os.path.join(os.path.dirname(__file__), "..", "data", "hospital_encounters.csv")


def test_load_and_validate(dataset_path):
    df = load_and_validate_dataset(dataset_path)
    assert len(df) > 20000
    assert TARGET_COLUMN in df.columns
    assert (df[TARGET_COLUMN] >= 0.0).all()


def test_chronological_split(dataset_path):
    df = load_and_validate_dataset(dataset_path)
    X_train, y_train, X_val, y_val, X_test, y_test = get_chronological_splits(df)

    # Check row count proportions
    n = len(df)
    assert len(X_train) == int(n * 0.70)
    assert len(X_val) == int(n * 0.85) - int(n * 0.70)
    assert len(X_test) == n - int(n * 0.85)

    # Verify target vectors match feature matrix rows
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)

    # Verify predictive features exclude non-predictive metadata
    assert "patient_id" not in X_train.columns
    assert "arrival_timestamp" not in X_train.columns
    assert "total_patients_waiting_ahead" not in X_train.columns
    assert list(X_train.columns) == ALL_PREDICTIVE_FEATURES
