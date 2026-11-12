"""Privacy, distinguishability, and augmentation checks for one synthetic set.

All comparisons use cleaned tables. ``source_row`` and ``is_synthetic`` are metadata,
not model features. The synthetic table contains only positive-class rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from agentic_gan.schemas import CheckResult, QualityThresholds
from agentic_gan.tools.baselines import pr_auc_score

# The hard privacy checks this module emits; agents use the exact names, not substrings.
PRIVACY_CHECKS = {"exact_train_copies", "synthetic_duplicates", "dcr_ratio"}


def _features(
    real_train: pd.DataFrame, real_val: pd.DataFrame, synthetic: pd.DataFrame, target: str
) -> list[str]:
    columns = [c for c in real_train if c not in {target, "source_row", "is_synthetic"}]
    if not columns or target not in real_train or target not in real_val or target not in synthetic:
        raise ValueError("All tables need a target and at least one feature")
    if any(
        set(columns) != set(frame.columns) - {target, "source_row", "is_synthetic"}
        for frame in (real_val, synthetic)
    ):
        raise ValueError("Feature columns differ across real and synthetic tables")
    if any(frame.empty for frame in (real_train, real_val, synthetic)):
        raise ValueError("Real and synthetic tables must be nonempty")
    return columns


def _preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = [c for c in frame if pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in frame if c not in numeric]
    return ColumnTransformer(
        [
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")
                ),
                categorical,
            ),
        ]
    )


def exact_copy_checks(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str],
    target: str,
    thresholds: QualityThresholds,
) -> list[CheckResult]:
    """Count synthetic rows copied from training data and repeated synthetic rows."""
    keys = [*columns, target]
    matched = synthetic[keys].merge(
        real_train[keys].drop_duplicates(), on=keys, how="left", indicator=True
    )
    copies = int(matched["_merge"].eq("both").sum())
    duplicates = int(synthetic.duplicated(subset=keys).sum())
    limit = thresholds.exact_duplicates_max
    return [
        CheckResult(
            name="exact_train_copies", value=copies, threshold=limit, passed=copies <= limit
        ),
        CheckResult(
            name="synthetic_duplicates",
            value=duplicates,
            threshold=limit,
            passed=duplicates <= limit,
        ),
    ]


def closest_real_check(
    real_train: pd.DataFrame,
    real_val: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str],
    target: str,
    positive_class: int | str,
    thresholds: QualityThresholds,
) -> tuple[CheckResult, dict[str, float]]:
    """Compare minority-row nearest-neighbor distance medians in training-fitted space."""
    train = real_train.loc[real_train[target].eq(positive_class), columns]
    val = real_val.loc[real_val[target].eq(positive_class), columns]
    if train.empty or val.empty:
        raise ValueError("Both real splits need positive-class rows for privacy comparison")
    transform = _preprocessor(train).fit(train)
    neighbors = NearestNeighbors(n_neighbors=1, algorithm="brute").fit(transform.transform(train))
    val_median = float(np.median(neighbors.kneighbors(transform.transform(val))[0]))
    synthetic_distances = neighbors.kneighbors(transform.transform(synthetic[columns]))[0]
    synthetic_median = float(np.median(synthetic_distances))
    if val_median == 0:
        raise ValueError("Real validation median distance is zero; DCR ratio is undefined")
    ratio = synthetic_median / val_median
    return (
        CheckResult(
            name="dcr_ratio",
            value=ratio,
            threshold=thresholds.dcr_ratio_min,
            passed=ratio >= thresholds.dcr_ratio_min,
        ),
        {"synthetic_median": synthetic_median, "validation_median": val_median},
    )


def _discriminator(real: pd.DataFrame, other: pd.DataFrame, seed: int) -> tuple[float, list[str]]:
    size = min(len(real), len(other))
    if size < 4:
        raise ValueError("Discriminator needs at least four rows per group")
    real = real.sample(n=size, random_state=seed)
    other = other.sample(n=size, random_state=seed)
    features = pd.concat([real, other], ignore_index=True)
    labels = np.r_[np.zeros(size, dtype=int), np.ones(size, dtype=int)]
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, stratify=labels, random_state=seed
    )
    model = make_pipeline(
        _preprocessor(x_train),
        RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=seed),
    )
    model.fit(x_train, y_train)
    auc = float(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]))
    importance = permutation_importance(
        model, x_test, y_test, scoring="roc_auc", n_repeats=3, random_state=seed
    )
    revealing = [
        x_test.columns[i]
        for i in np.argsort(-importance.importances_mean)[:5]
        if importance.importances_mean[i] > 0
    ]
    return auc, revealing


def distinguishability_check(
    real_train: pd.DataFrame,
    real_val: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str],
    target: str,
    positive_class: int | str,
    seed: int = 42,
) -> tuple[CheckResult, dict[str, float | list[str]]]:
    """ROC-AUC and original columns exposing synthetic rows; diagnostic only."""
    train = real_train.loc[real_train[target].eq(positive_class), columns]
    val = real_val.loc[real_val[target].eq(positive_class), columns]
    auc, revealing = _discriminator(train, synthetic[columns], seed)
    real_auc, _ = _discriminator(train, val, seed)
    return (
        CheckResult(
            name="real_vs_synthetic_auc", value=auc, threshold=0.5, passed=auc <= 0.5, hard=False
        ),
        {"real_vs_real_auc": real_auc, "revealing_columns": revealing},
    )


def _best_f1(y: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    if not len(thresholds):
        return 0.0
    scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(precision[:-1]),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    return float(scores.max())


def utility_checks(
    real_train: pd.DataFrame,
    real_val: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str],
    target: str,
    positive_class: int | str,
    seed: int = 42,
    hard: bool = False,
) -> tuple[list[CheckResult], dict[str, float]]:
    """Compare real-only with real-plus-synthetic on the same real validation rows.

    The two gains are always computed and reported. They decide the verdict only when ``hard`` is
    set (``QualityThresholds.utility_gain_hard``): whether augmentation helps is judged by the
    training and evaluation agents on real validation data, not by the gate.
    """
    y_train = real_train[target].eq(positive_class).astype(int).to_numpy()
    y_val = real_val[target].eq(positive_class).astype(int).to_numpy()
    if len(np.unique(y_train)) != 2 or len(np.unique(y_val)) != 2:
        raise ValueError("Both real splits need both target classes for utility")
    transform = _preprocessor(real_train[columns]).fit(real_train[columns])
    x_train = transform.transform(real_train[columns])
    x_val = transform.transform(real_val[columns])
    x_augmented = transform.transform(synthetic[columns])
    probabilities = []
    for x, y in (
        (x_train, y_train),
        (
            sparse.vstack([x_train, x_augmented])
            if sparse.issparse(x_train)
            else np.vstack([x_train, x_augmented]),
            np.r_[y_train, np.ones(len(synthetic), dtype=int)],
        ),
    ):
        model = LogisticRegression(max_iter=1000, random_state=seed)
        model.fit(x, y)
        probabilities.append(model.predict_proba(x_val)[:, 1])
    real_pr, augmented_pr = (pr_auc_score(y_val, p) for p in probabilities)
    real_f1, augmented_f1 = (_best_f1(y_val, p) for p in probabilities)
    checks = [
        CheckResult(
            name="augmentation_pr_auc_gain",
            value=augmented_pr - real_pr,
            threshold=0,
            passed=augmented_pr >= real_pr,
            hard=hard,
        ),
        CheckResult(
            name="augmentation_best_f1_gain",
            value=augmented_f1 - real_f1,
            threshold=0,
            passed=augmented_f1 >= real_f1,
            hard=hard,
        ),
    ]
    return checks, {
        "real_only_pr_auc": real_pr,
        "augmented_pr_auc": augmented_pr,
        "real_only_best_f1": real_f1,
        "augmented_best_f1": augmented_f1,
    }


def quality_privacy_checks(
    real_train: pd.DataFrame,
    real_val: pd.DataFrame,
    synthetic: pd.DataFrame,
    target: str,
    positive_class: int | str = 1,
    *,
    thresholds: QualityThresholds | None = None,
    seed: int = 42,
) -> tuple[list[CheckResult], dict[str, float | list[str]]]:
    """Return gate-ready checks and diagnostic measurements for one synthetic set."""
    thresholds = thresholds or QualityThresholds()
    columns = _features(real_train, real_val, synthetic, target)
    if not synthetic[target].eq(positive_class).all():
        raise ValueError("Synthetic rows must all have the positive target")
    exact = exact_copy_checks(real_train, synthetic, columns, target, thresholds)
    distance, distance_details = closest_real_check(
        real_train, real_val, synthetic, columns, target, positive_class, thresholds
    )
    discriminator, discriminator_details = distinguishability_check(
        real_train, real_val, synthetic, columns, target, positive_class, seed
    )
    utility, utility_details = utility_checks(
        real_train,
        real_val,
        synthetic,
        columns,
        target,
        positive_class,
        seed,
        hard=thresholds.utility_gain_hard,
    )
    return [*exact, distance, discriminator, *utility], {
        **distance_details,
        **discriminator_details,
        **utility_details,
    }
