"""UCI Bank Marketing preprocessing with training-only SMOTENC oversampling."""

from typing import NamedTuple

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .uci_preprocessing import UCIPreprocessedData, load_and_preprocess_uci


class UCISMOTENCData(NamedTuple):
    base: UCIPreprocessedData
    X_train_smotenc: object
    y_train_smotenc: object
    smotenc_ratio: float
    n_synthetic: int


def _smotenc_input_transformer(X):
    """Impute and scale numeric fields while preserving categorical values."""
    categorical = X.select_dtypes(include=["object", "str", "category"]).columns.tolist()
    numeric = X.columns.difference(categorical, sort=False).tolist()
    transformer = ColumnTransformer(
        [
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            ("categorical", "passthrough", categorical),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    return transformer, categorical


def make_smotenc_model_pipeline(estimator, preprocessor, X_reference, sampling_strategy, seed=42):
    """Build a model pipeline that fits SMOTENC within each training fold."""
    input_transformer, categorical = _smotenc_input_transformer(X_reference)
    return make_imb_pipeline(
        input_transformer,
        SMOTENC(
            categorical_features=categorical,
            sampling_strategy=sampling_strategy,
            random_state=seed,
        ),
        clone(preprocessor),
        clone(estimator),
    )


def load_and_preprocess_uci_smotenc(
    n_synthetic: int,
    seed: int = 42,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
) -> UCISMOTENCData:
    """Generate exactly ``n_synthetic`` training rows with categorical-aware SMOTENC.

    Validation and test sets retain their original class balance. The ratio
    returned here is used for fold-specific sampling during model evaluation.
    """
    if isinstance(n_synthetic, bool) or not isinstance(n_synthetic, int) or n_synthetic <= 0:
        raise ValueError("n_synthetic must be a positive integer")

    base = load_and_preprocess_uci(
        seed=seed, train_size=train_size, val_size=val_size, test_size=test_size
    )
    counts = base.y_train.value_counts()
    minority_count = int(counts.loc[1])
    majority_count = int(counts.loc[0])
    target_minority_count = minority_count + n_synthetic
    if target_minority_count > majority_count:
        raise ValueError(
            f"n_synthetic cannot exceed {majority_count - minority_count} "
            "when oversampling only the minority class"
        )

    input_transformer, categorical = _smotenc_input_transformer(base.X_train)
    X_ready = input_transformer.fit_transform(base.X_train)
    sampler = SMOTENC(
        categorical_features=categorical,
        sampling_strategy={1: target_minority_count},
        random_state=seed,
    )
    X_resampled, y_resampled = sampler.fit_resample(X_ready, base.y_train)
    X_train_smotenc = clone(base.preprocessor).fit_transform(X_resampled)
    return UCISMOTENCData(
        base=base,
        X_train_smotenc=X_train_smotenc,
        y_train_smotenc=y_resampled,
        smotenc_ratio=target_minority_count / majority_count,
        n_synthetic=n_synthetic,
    )
