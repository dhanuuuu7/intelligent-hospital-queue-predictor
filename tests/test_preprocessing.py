"""
Unit tests for Step 5.2 Feature Preprocessing and ColumnTransformer
"""

import os
import pytest
import numpy as np
from models.data_loader import load_and_validate_dataset, get_chronological_splits
from models.preprocessing import build_preprocessor


@pytest.fixture
def dataset_splits():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "hospital_encounters.csv")
    df = load_and_validate_dataset(data_path)
    return get_chronological_splits(df)


def test_build_preprocessor(dataset_splits):
    X_train, y_train, X_val, y_val, X_test, y_test = dataset_splits
    preprocessor = build_preprocessor(scale_numeric=True)

    # Fit on training data ONLY
    X_train_trans = preprocessor.fit_transform(X_train)
    X_val_trans = preprocessor.transform(X_val)

    # Verify transformed dimensions
    assert X_train_trans.shape[0] == len(X_train)
    assert X_val_trans.shape[0] == len(X_val)
    assert X_train_trans.shape[1] == 29, f"Expected 29 transformed features, got {X_train_trans.shape[1]}"
    assert not np.isnan(X_train_trans).any(), "Transformed training matrix contains NaNs!"
    assert not np.isnan(X_val_trans).any(), "Transformed validation matrix contains NaNs!"
