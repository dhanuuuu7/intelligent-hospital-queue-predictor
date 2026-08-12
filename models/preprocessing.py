"""
Step 5.2 & 8.2: Feature Preprocessing & Unified Pipeline Builder
"""

from typing import List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from models.data_loader import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def build_preprocessor(
    categorical_cols: Optional[List[str]] = None,
    numerical_cols: Optional[List[str]] = None,
    scale_numeric: bool = True
) -> ColumnTransformer:
    """
    Constructs a Scikit-Learn ColumnTransformer for feature preprocessing.
    Supports both baseline features and expanded engineered features.
    """
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_FEATURES
    if numerical_cols is None:
        numerical_cols = NUMERICAL_FEATURES

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore", 
        sparse_output=False
    )

    if scale_numeric:
        numerical_transformer = StandardScaler()
    else:
        numerical_transformer = "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numerical_transformer, numerical_cols)
        ],
        remainder="drop"
    )

    return preprocessor


def create_full_pipeline(
    model: object, 
    categorical_cols: Optional[List[str]] = None,
    numerical_cols: Optional[List[str]] = None,
    scale_numeric: bool = True
) -> Pipeline:
    """
    Wraps preprocessor and model estimator into a single Scikit-Learn Pipeline.
    """
    preprocessor = build_preprocessor(
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        scale_numeric=scale_numeric
    )
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    return pipeline
