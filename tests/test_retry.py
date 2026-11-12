"""Pure unit tests for issue #123 retry knobs (step 2). No GAN, no run folder."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentic_gan.agents import model_selection as selection
from agentic_gan.agents.model_selection import (
    CTGAN_DEFAULTS,
    RETRY_OPTIONS,
    RetryLimits,
    RetrySettings,
    _normalize,
    _retry_limits,
    _rule_hash,
    allowed_rake_columns,
)
from agentic_gan.schemas import RunConfig
from agentic_gan.tools.ctgan_sampling import recipe_for
from agentic_gan.utils.datasets import DATASETS


def _config(tmp_path, **overrides):
    source = tmp_path / "source.csv"
    source.write_text("age,y\n1,1\n")
    fields = {
        "dataset_name": "bank_marketing",
        "dataset_path": source,
        "target": "y",
        "positive_class": "yes",
        "gan_bounds": {"epochs": (300, 600), "pac": (5, 10)},
    }
    fields.update(overrides)
    return RunConfig(**fields)


def _legal_answer(**overrides):
    fields = {
        "epochs": 400,
        "pac": 10,
        "discriminator_steps": 1,
        "batch_size": 500,
        "sampling_mode": "unconditional",
        "rule_filter": True,
        "rake_columns": ["job", "month"],
        "reason": "fix job distance",
    }
    fields.update(overrides)
    return RetrySettings(**fields)


def test_retry_settings_rejects_unknown_fields_and_bad_knobs():
    _legal_answer()
    with pytest.raises(ValidationError):
        _legal_answer(seed=1)
    with pytest.raises(ValidationError):
        _legal_answer(sampling_mode="mixed")
    with pytest.raises(ValidationError):
        _legal_answer(rule_filter="yes")
    with pytest.raises(ValidationError):
        _legal_answer(rule_filter=1)
    with pytest.raises(ValidationError):
        _legal_answer(reason="")


def test_rule_hash_moves_with_retry_options_and_gan_bounds(tmp_path, monkeypatch):
    base = _config(tmp_path)
    same = _config(tmp_path)
    assert _rule_hash(base) == _rule_hash(same)
    assert _rule_hash(base) != _rule_hash(
        _config(tmp_path, gan_bounds={"epochs": (300, 700), "pac": (5, 10)})
    )
    before = _rule_hash(base)
    monkeypatch.setattr(
        selection,
        "RETRY_OPTIONS",
        {"sampling_mode": ("conditional",), "rule_filter": (False, True)},
    )
    assert _rule_hash(base) != before


def test_allowed_rake_columns_includes_bank_day_of_month_and_excludes_target():
    discrete = list(DATASETS["bank_marketing"].discrete)
    recipe = recipe_for("bank_marketing")
    table = [
        "age",
        "job",
        "marital",
        "month",
        "poutcome",
        "day_of_month",
        "y",
    ]
    allowed = allowed_rake_columns(discrete, recipe.rake_columns, table)
    assert "day_of_month" in allowed
    assert "y" not in allowed
    assert "job" in allowed
    assert allowed == [name for name in table if name in set(allowed)]


def test_retry_limits_privacy_caps_epochs_high_at_current():
    limits = _retry_limits(
        CTGAN_DEFAULTS,
        {"epochs": (100, 600), "pac": (5, 10)},
        privacy=True,
        rake_allowed=("job",),
    )
    assert isinstance(limits, RetryLimits)
    assert limits.numeric["epochs"][1] == CTGAN_DEFAULTS["epochs"]
    assert limits.numeric["epochs"][0] == 100
    assert limits.options == {
        "sampling_mode": ("conditional", "unconditional"),
        "rule_filter": (False, True),
    }
    assert limits.rake_allowed == ("job",)


def test_retry_limits_never_returns_none_when_options_exist():
    limits = _retry_limits(CTGAN_DEFAULTS, {}, privacy=False)
    assert limits is not None
    assert all(low == high for low, high in limits.numeric.values())


def test_normalize_accepts_a_legal_answer():
    limits = RetryLimits(
        numeric={
            "epochs": (300, 600),
            "pac": (5, 10),
            "discriminator_steps": (1, 5),
            "batch_size": (250, 1000),
        },
        options=dict(RETRY_OPTIONS),
        rake_allowed=("job", "month", "poutcome"),
    )
    out = _normalize(_legal_answer(), limits)
    assert out is not None
    assert out["epochs"] == 400
    assert out["sampling_mode"] == "unconditional"
    assert out["rule_filter"] is True
    assert out["rake_columns"] == ["job", "month"]


def test_normalize_rejects_rake_outside_allowed_and_duplicates():
    limits = RetryLimits(
        numeric={name: (v, v) for name, v in CTGAN_DEFAULTS.items()},
        options=dict(RETRY_OPTIONS),
        rake_allowed=("job", "month"),
    )
    assert _normalize(_legal_answer(rake_columns=["job", "age"]), limits) is None
    assert _normalize(_legal_answer(rake_columns=["job", "job"]), limits) is None


def test_normalize_snaps_batch_500_pac_8_to_496():
    limits = RetryLimits(
        numeric={
            "epochs": (300, 600),
            "pac": (5, 12),
            "discriminator_steps": (1, 5),
            "batch_size": (480, 520),
        },
        options=dict(RETRY_OPTIONS),
        rake_allowed=("job",),
    )
    out = _normalize(_legal_answer(pac=8, batch_size=500, rake_columns=["job"]), limits)
    assert out is not None
    assert out["pac"] == 8
    assert out["batch_size"] == 496


def test_normalize_returns_none_when_no_legal_batch_multiple_exists():
    limits = RetryLimits(
        numeric={
            "epochs": (300, 600),
            "pac": (5, 12),
            "discriminator_steps": (1, 5),
            "batch_size": (500, 503),
        },
        options=dict(RETRY_OPTIONS),
        rake_allowed=("job",),
    )
    assert _normalize(_legal_answer(pac=8, batch_size=500, rake_columns=["job"]), limits) is None
