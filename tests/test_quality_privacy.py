"""Checks for the privacy and utility half of the quality gate."""

import numpy as np
import pandas as pd
import pytest

from agentic_gan.schemas import CheckResult, QualityThresholds
from agentic_gan.tools.quality_privacy import (
    closest_real_check,
    distinguishability_check,
    exact_copy_checks,
    quality_privacy_checks,
    utility_checks,
)


def test_exact_copies_and_synthetic_duplicates_ignore_metadata():
    train = pd.DataFrame({"source_row": [1, 2], "x": [1, 2], "y": [1, 1]})
    synthetic = pd.DataFrame({"x": [1, 3, 3], "y": [1, 1, 1], "is_synthetic": [1, 1, 1]})
    checks = exact_copy_checks(train, synthetic, ["x"], "y", QualityThresholds())
    assert all(isinstance(check, CheckResult) for check in checks)
    assert [(check.name, check.value, check.passed) for check in checks] == [
        ("exact_train_copies", 1, False),
        ("synthetic_duplicates", 1, False),
    ]


def test_nearest_real_ratio_uses_unseen_real_minority_as_reference():
    train = pd.DataFrame({"x": [0.0, 10.0, 100.0], "y": [1, 1, 0]})
    val = pd.DataFrame({"x": [1.0, 9.0, 99.0], "y": [1, 1, 0]})
    near = pd.DataFrame({"x": [0.1, 0.2], "y": [1, 1]})
    far = pd.DataFrame({"x": [2.0, 8.0], "y": [1, 1]})
    near_check, details = closest_real_check(train, val, near, ["x"], "y", 1, QualityThresholds())
    far_check, _ = closest_real_check(train, val, far, ["x"], "y", 1, QualityThresholds())
    assert near_check.value == pytest.approx(0.15)
    assert details["validation_median"] > details["synthetic_median"]
    assert not near_check.passed
    assert far_check.value == pytest.approx(2.0)
    assert far_check.passed


def _tables():
    rng = np.random.default_rng(11)
    train_x = rng.normal(size=80)
    val_x = rng.normal(size=40)
    train = pd.DataFrame(
        {
            "source_row": range(80),
            "x": train_x,
            "group": np.where(train_x > 0, "right", "left"),
            "y": (train_x > 0.4).astype(int),
        }
    )
    val = pd.DataFrame(
        {
            "source_row": range(80, 120),
            "x": val_x,
            "group": np.where(val_x > 0, "right", "left"),
            "y": (val_x > 0.4).astype(int),
        }
    )
    synthetic = pd.DataFrame(
        {
            "x": np.linspace(0.5, 2.0, 20),
            "group": ["right"] * 20,
            "y": [1] * 20,
            "is_synthetic": [1] * 20,
        }
    )
    return train, val, synthetic


def test_utility_compares_two_arms_on_real_validation():
    train, val, synthetic = _tables()
    checks, metrics = utility_checks(train, val, synthetic, ["x", "group"], "y", 1)
    assert all(isinstance(check, CheckResult) for check in checks)
    assert checks[0].value == pytest.approx(
        metrics["augmented_pr_auc"] - metrics["real_only_pr_auc"]
    )
    assert checks[1].value == pytest.approx(
        metrics["augmented_best_f1"] - metrics["real_only_best_f1"]
    )
    assert all(0 <= value <= 1 for value in metrics.values())


def test_utility_handles_sparse_categorical_features():
    train = pd.DataFrame(
        {"category": [f"c{i % 12}" for i in range(48)], "y": [i % 2 for i in range(48)]}
    )
    val = pd.DataFrame(
        {"category": [f"c{i % 12}" for i in range(12)], "y": [i % 2 for i in range(12)]}
    )
    synthetic = pd.DataFrame({"category": ["c1", "c3"], "y": [1, 1]})
    checks, _ = utility_checks(train, val, synthetic, ["category"], "y", 1)
    assert len(checks) == 2


def test_discriminator_reports_column_that_separates_synthetic_rows():
    real = pd.DataFrame({"signal": ["real"] * 40, "x": range(40), "y": [1] * 40})
    val = pd.DataFrame({"signal": ["real"] * 40, "x": range(40), "y": [1] * 40})
    synthetic = pd.DataFrame({"signal": ["synthetic"] * 40, "x": range(40), "y": [1] * 40})
    check, details = distinguishability_check(real, val, synthetic, ["signal", "x"], "y", 1)
    assert check.value > 0.9
    assert "signal" in details["revealing_columns"]
    assert not check.hard


def test_full_quality_privacy_result_includes_diagnostic_columns():
    train, val, synthetic = _tables()
    checks, diagnostics = quality_privacy_checks(train, val, synthetic, "y")
    assert {check.name for check in checks} == {
        "exact_train_copies",
        "synthetic_duplicates",
        "dcr_ratio",
        "real_vs_synthetic_auc",
        "augmentation_pr_auc_gain",
        "augmentation_best_f1_gain",
    }
    assert all(isinstance(check, CheckResult) for check in checks)
    assert next(check for check in checks if check.name == "real_vs_synthetic_auc").hard is False
    assert 0 <= diagnostics["real_vs_real_auc"] <= 1
    assert set(diagnostics["revealing_columns"]) <= {"x", "group"}


def test_utility_gains_are_soft_unless_the_thresholds_make_them_hard():
    train, val, synthetic = _tables()
    gains = ["augmentation_pr_auc_gain", "augmentation_best_f1_gain"]

    def hardness(thresholds):
        checks, _ = quality_privacy_checks(train, val, synthetic, "y", thresholds=thresholds)
        return [check.hard for check in checks if check.name in gains]

    assert hardness(QualityThresholds()) == [False, False]
    assert hardness(QualityThresholds(utility_gain_hard=True)) == [True, True]


def test_rejects_nonpositive_synthetic_targets():
    train, val, synthetic = _tables()
    synthetic.loc[0, "y"] = 0
    with pytest.raises(ValueError, match="positive target"):
        quality_privacy_checks(train, val, synthetic, "y")
