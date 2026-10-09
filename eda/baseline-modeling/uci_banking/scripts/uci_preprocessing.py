"""Prepare the UCI Bank Marketing data for a pre-call baseline model."""

from typing import NamedTuple
import math

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


class UCIPreprocessedData(NamedTuple):
    X_train: object
    X_val: object
    X_test: object
    y_train: object
    y_val: object
    y_test: object
    X_train_processed: object
    X_val_processed: object
    X_test_processed: object
    preprocessor: ColumnTransformer
    feature_names: object


def load_and_preprocess_uci(
    seed: int = 42,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
) -> UCIPreprocessedData:
    """Fetch UCI dataset 222, split it, and fit preprocessing on training rows.

    Split sizes are fractions of the full dataset and must be positive and sum to 1.
    Source unknown categories remain "unknown". The pdays=-1 sentinel becomes
    missing and is filled with the training median after splitting. The fitted
    preprocessor is then applied to validation and test rows.
    """
    sizes = (train_size, val_size, test_size)
    if not all(math.isfinite(size) and 0 < size < 1 for size in sizes) or not math.isclose(
        sum(sizes), 1.0, rel_tol=0, abs_tol=1e-9
    ):
        raise ValueError("train_size, val_size, and test_size must be positive fractions summing to 1")

    bank_marketing = fetch_ucirepo(id=222)
    X = bank_marketing.data.features.copy().rename(columns={"day_of_week": "day"})
    y = bank_marketing.data.targets["y"].copy()

    assert X.index.equals(y.index)
    assert set(y.dropna().unique()) == {"yes", "no"}
    assert y.notna().all()

    # The UCI API may represent source "unknown" categories as missing values.
    unknown_columns = ["job", "education", "contact", "poutcome"]
    X[unknown_columns] = X[unknown_columns].fillna("unknown")
    assert not X.isna().any().any(), "Review unexpected missing values before modeling"

    # Duration is known only after a call and would leak the outcome for pre-call prediction.
    X = X.drop(columns="duration")
    X["previously_contacted"] = (X["pdays"] != -1).astype(int)
    X["pdays"] = X["pdays"].replace(-1, np.nan)
    y = y.map({"no": 0, "yes": 1}).astype(int)

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=val_size + test_size, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout, y_holdout, test_size=test_size / (val_size + test_size),
        stratify=y_holdout, random_state=seed
    )

    categorical_columns = X_train.select_dtypes(include=["object", "str", "category"]).columns.tolist()
    numeric_columns = X_train.columns.difference(categorical_columns, sort=False).tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                numeric_columns,
            ),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
        ],
        remainder="drop",
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    return UCIPreprocessedData(
        X_train, X_val, X_test, y_train, y_val, y_test,
        X_train_processed, X_val_processed, X_test_processed,
        preprocessor, feature_names,
    )
