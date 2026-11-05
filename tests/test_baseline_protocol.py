import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

from agentic_gan.agents.training import run_training_arms
from agentic_gan.tools import baselines as b


class ScoreClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.asarray(X)[:, 0]
        return np.column_stack([1 - p, p])


def test_fixed_suite():
    models = b.build_models()
    assert set(models) == {
        "Gradient boosting",
        "LightGBM",
        "CatBoost",
        "Stacking ensemble",
    }
    assert len(models["Stacking ensemble"].estimators) == 2


@pytest.mark.parametrize(
    "probabilities",
    [
        [0.05, 0.10, 0.20, 0.30, 0.40, 0.45],
        [0.2] * 6,
        [0, 0, 0.2, 0.2, 1, 1],
    ],
)
def test_threshold_matches_exhaustive_search(probabilities):
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array(probabilities)
    candidates = np.unique(p)
    expected = max(candidates, key=lambda t: (f1_score(y, p >= t), t))
    assert b.select_f1_threshold(y, p) == expected


@pytest.mark.parametrize("probabilities", [[np.nan, 0.2], [0.2, np.inf], [-0.1, 0.2], [1.2, 0.2]])
def test_invalid_probabilities_rejected(probabilities):
    with pytest.raises(ValueError, match="Probabilities"):
        b.select_f1_threshold([0, 1], probabilities)


def test_threshold_uses_validation_only_and_frozen_override(monkeypatch):
    monkeypatch.setattr(b, "build_models", lambda seed: {"score": ScoreClassifier()})
    train = np.array([[0.1], [0.2]] * 6)
    y_train = np.array([0, 1] * 6)
    val = np.array([[0.05], [0.2], [0.3], [0.4]])
    y_val = np.array([0, 0, 1, 1])
    test = np.array([[0.8], [0.9], [0.1], [0.2]])
    y_test = np.array([0, 0, 1, 1])
    first = b.run_models(train, y_train, val, y_val, test, y_test, cv_folds=0)
    changed = b.run_models(train, y_train, val, y_val, test, 1 - y_test, cv_folds=0)
    assert first.threshold.iloc[0] == changed.threshold.iloc[0] == 0.3
    assert first.val_f1.iloc[0] > first.val_f1_at_0_5.iloc[0]
    monkeypatch.setattr(b, "select_f1_threshold", lambda *args: pytest.fail("Retuned an arm"))
    frozen = b.run_models(
        train, y_train, val, 1 - y_val, test, y_test, cv_folds=0, thresholds={"score": 0.3}
    )
    assert frozen.threshold.iloc[0] == 0.3


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "build_models", lambda seed: {"dummy": DummyClassifier()})
    table = pd.DataFrame({"model": ["dummy"], "threshold": [0.2], "status": ["success"]})
    metadata = {
        "source_sha256": "source",
        "split_hashes": {"train": "ids"},
        "config": {"name": "test"},
        "versions": b._versions(),
    }
    path = tmp_path / "protocol.json"
    y = np.array([0, 1, 0, 1])
    protocol = b.freeze_protocol(
        path, table, metadata, dataset="test", seed=42, cv_folds=0, y_val=y, y_test=y
    )
    return path, metadata, protocol, y


def test_protocol_roundtrip_and_rejects_drift(frozen):
    path, metadata, protocol, y = frozen
    assert b.load_protocol(path, cv_folds=0, metadata=metadata, y_val=y) == protocol
    for changes in (
        {"seed": 43},
        {"cv_folds": 3},
        {"y_val": 1 - y},
        {"metadata": {**metadata, "source_sha256": "changed"}},
    ):
        with pytest.raises(ValueError, match="mismatch|do not match"):
            b.load_protocol(path, **{"cv_folds": 0, **changes})
    protocol["thresholds"]["dummy"] = 0.8
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="checksum"):
        b.load_protocol(path, cv_folds=0)


def test_training_agent_reuses_thresholds_on_every_arm(frozen, monkeypatch):
    path, _, protocol, y = frozen
    X = np.array([[0.1], [0.2]] * 4)
    y_train = np.array([0, 1] * 4)
    monkeypatch.setattr(b, "select_f1_threshold", lambda *args: pytest.fail("Retuned an arm"))
    result = run_training_arms(
        {"original": (X, y_train), "cleaned": (X, y_train), "gan": (X, y_train)},
        X[:4],
        y,
        X[:4],
        y,
        protocol_path=path,
    )
    assert set(result.threshold) == {0.2}
    assert set(result.protocol_id) == {protocol["protocol_id"]}


def test_cleaned_cannot_silently_calibrate(tmp_path):
    source = tmp_path / "data/raw/default.csv"
    source.parent.mkdir(parents=True)
    pd.DataFrame({"LIMIT_BAL": list(range(100)), "y": [0, 1] * 50}).to_csv(source, index=False)
    with pytest.raises(ValueError, match="Run --arm original"):
        b.run_dataset("default", root=tmp_path, cv_folds=0)
