from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from agentic_gan.tools import baselines as b
from agentic_gan.utils.datasets import DATASETS


class RecordingPreprocessor(b.SharedPreprocessor):
    fits = []

    def fit(self, X, y):
        super().fit(X, y)
        self.fits.append((tuple(X.index), self.fitted_.fill_values["A1"]))
        return self


@pytest.mark.parametrize("stacking", [False, True])
def test_preprocessing_uses_only_fold_training_rows(monkeypatch, stacking):
    X = pd.DataFrame({"A1": [np.nan, 1, 2, 3, 4, 5, 6, 7, 8, 90, 100, 200]}, index=range(12))
    y = pd.Series([0, 1] * 6, index=X.index)
    hold = pd.DataFrame({"A1": [99999.0, np.nan, -99999.0, 0.0]}, index=range(100, 104))
    hold_y = pd.Series([0, 1, 0, 1], index=hold.index)
    model = DummyClassifier(strategy="prior")
    if stacking:
        model = StackingClassifier(
            estimators=[("dummy", model)], final_estimator=LogisticRegression(), cv=2
        )
    monkeypatch.setattr(b, "build_models", lambda seed: {"model": model})
    config = replace(DATASETS["polish"], discrete=())
    prep = RecordingPreprocessor(config)
    RecordingPreprocessor.fits = []
    result = b.run_models(X, y, hold, hold_y, hold, hold_y, preprocessor=prep, cv_folds=3)
    assert result.status.tolist() == ["success"]
    assert not hasattr(prep, "fitted_")
    fits = RecordingPreprocessor.fits
    expected = [
        tuple(X.iloc[train].index)
        for train, _ in StratifiedKFold(3, shuffle=True, random_state=42).split(X, y)
    ]
    assert all(ids in [f[0] for f in fits] for ids in expected)
    for ids, median in fits:
        assert set(ids) <= set(X.index)  # No validation/test rows fit preprocessing.
        assert median == X.loc[list(ids), "A1"].median()
    assert any(len(ids) == len(X) for ids, _ in fits)
    if stacking:
        assert any(len(ids) < 8 for ids, _ in fits)  # Inner folds refit as well.
    else:
        assert len(fits) == 4  # Three CV fits plus final training fit.


def test_both_arms_load_raw_without_cleaned_files(tmp_path):
    source = tmp_path / "data/raw/default.csv"
    source.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "LIMIT_BAL": [1.0, 2.0, np.nan, 4.0] * 30,
            "SEX": [1, 2] * 60,
            "y": [0, 1] * 60,
        }
    ).to_csv(source, index=False)
    original, _, original_metadata = b.prepare_inputs("default", "original", tmp_path, 42)
    cleaned, transformer, cleaned_metadata = b.prepare_inputs("default", "cleaned", tmp_path, 42)
    assert not (tmp_path / "data/interim").exists()
    assert original_metadata["input_kind"] == cleaned_metadata["input_kind"] == "raw"
    for a, c in zip(original, cleaned, strict=True):
        assert a.equals(c)
    assert cleaned[0].LIMIT_BAL.isna().any()
    assert not hasattr(transformer, "fitted_")


def test_bank_original_arm_handles_normalized_duration_exclusion(tmp_path):
    source = tmp_path / "data/raw/bank_marketing.csv"
    source.parent.mkdir(parents=True)
    n = 60
    pd.DataFrame(
        {
            "job": ["admin.", "technician"] * 30,
            "marital": ["single", "married"] * 30,
            "education": ["tertiary", "secondary"] * 30,
            "default": ["no"] * n,
            "housing": ["yes", "no"] * 30,
            "loan": ["no"] * n,
            "contact": ["cellular", "telephone"] * 30,
            "month": ["may"] * n,
            "day": np.tile(np.arange(1, 31), 2),
            "duration": np.arange(n),
            "campaign": np.ones(n),
            "pdays": np.where(np.arange(n) % 2, 7, -1),
            "previous": np.ones(n),
            "poutcome": ["success", "unknown"] * 30,
            "y": ["no", "yes"] * 30,
        }
    ).to_csv(source, index=False)

    inputs, _, _ = b.prepare_inputs("bank_marketing", "original", tmp_path, 42)

    assert "duration" not in inputs[0].columns
