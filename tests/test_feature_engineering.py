"""
Unit tests for Step 8.2 Feature Engineering
"""

import os
import pytest
import pandas as pd
import numpy as np
from models.data_loader import load_and_validate_dataset
from models.feature_engineering import add_engineered_features, ENGINEERED_NUMERICAL_FEATURES, ALL_EXPANDED_PREDICTIVE_FEATURES


@pytest.fixture
def raw_df():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "hospital_encounters.csv")
    return load_and_validate_dataset(data_path)


def test_add_engineered_features(raw_df):
    eng_df = add_engineered_features(raw_df)

    # Check all engineered columns exist
    for col in ENGINEERED_NUMERICAL_FEATURES:
        assert col in eng_df.columns, f"Engineered feature {col} missing!"
        assert not eng_df[col].isnull().any(), f"Engineered feature {col} contains NaNs!"
        assert not np.isinf(eng_df[col]).any(), f"Engineered feature {col} contains Inf!"

    # Verify math calculations on sample row
    sample = eng_df.iloc[0]
    expected_load = sample["queue_position"] / max(1, sample["active_doctors_count"])
    assert abs(sample["effective_doctor_load"] - expected_load) < 1e-5
