"""Unit tests for issue #123 retry knobs and attempt history (steps 2–3)."""

from __future__ import annotations

from pathlib import Path

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
    attempt_history,
    settings_key,
)
from agentic_gan.orchestrator.decisions import log_decision, read_decisions
from agentic_gan.orchestrator.manifest import create_run
from agentic_gan.schemas import (
    ArmFiles,
    CheckResult,
    LevelReport,
    ModelSelectionInput,
    PreprocessOutput,
    QualityReport,
    RunConfig,
    Status,
    Verdict,
)
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


# --- Step 3: attempt history -------------------------------------------------------------------

BANK_V2 = {
    "epochs": 300,
    "batch_size": 500,
    "pac": 5,
    "discriminator_steps": 3,
    "sampling_mode": "unconditional",
    "rule_filter": True,
    "rake_columns": ["job", "month", "poutcome", "marital", "day_of_month"],
}


def _fail_report(*names):
    checks = [
        CheckResult(name=name, value=1.0, threshold=0.1, passed=False) for name in names
    ]
    return QualityReport(
        status=Status.SUCCESS,
        verdict=Verdict.FAIL,
        levels=[LevelReport(fill_level=0.25, passed=False, checks=checks)],
        report_path=Path("quality.json"),
    )


def _history_request(tmp_path, config, attempt, previous, run_id="hist"):
    run = create_run(config, run_id=run_id, runs_dir=tmp_path / "runs")
    train_path = tmp_path / f"{run_id}_train.csv"
    train_path.write_text("y\n0\n")
    unread = tmp_path / f"{run_id}_unread.csv"
    unread.write_text("x\n")
    profile = tmp_path / f"{run_id}_profile.json"
    profile.write_text("{}\n")
    files = ArmFiles(
        train_path=train_path.resolve(),
        val_path=unread.resolve(),
        test_path=unread.resolve(),
    )
    preprocess = PreprocessOutput(
        status=Status.SUCCESS,
        raw=files,
        baseline=files,
        profile_path=profile.resolve(),
        treatment_log_path=profile.resolve(),
        transformer_path=profile.resolve(),
        transformer_fit_rows=1,
        baseline_train_positives=1,
        baseline_train_negatives=1,
    )
    request = ModelSelectionInput(
        config=config,
        preprocess=preprocess,
        attempt=attempt,
        previous_report=previous,
    )
    return request, run


def _log_hyperparameters(run, attempt, settings):
    log_decision(
        "model_selection",
        "hyperparameters",
        {"attempt": attempt, **settings, "seed": 42, "generator": "ctgan"},
        "SET",
        "CTGAN settings are inside gan_bounds.",
        run=run,
    )


def _save_quality_report(run, attempt, *failed_names):
    report = _fail_report(*failed_names)
    path = run.subdir("quality") / f"quality_report_attempt_{attempt}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(), encoding="utf-8")
    return report


def test_attempt_history_from_logged_rows_and_saved_report(tmp_path):
    config = _config(tmp_path)
    attempt2 = {**BANK_V2, "epochs": 400, "rake_columns": ["job", "month"]}
    previous = _fail_report("column_distance:job")
    request, run = _history_request(tmp_path, config, attempt=3, previous=previous)
    _log_hyperparameters(run, 1, BANK_V2)
    _log_hyperparameters(run, 2, attempt2)
    _save_quality_report(run, 1, "column_distance:job")
    before = len(read_decisions(run))
    history = attempt_history(request, run)
    assert len(read_decisions(run)) == before
    assert [record.attempt for record in history] == [1, 2]
    assert history[0].settings["pac"] == 5
    assert history[0].settings["discriminator_steps"] == 3
    assert history[0].failed_checks == ["column_distance:job"]
    assert history[1].settings["epochs"] == 400
    assert history[1].failed_checks == ["column_distance:job"]


def test_attempt_history_missing_report_gives_empty_failed_checks(tmp_path):
    config = _config(tmp_path)
    previous = _fail_report("mean_column_distance")
    request, run = _history_request(tmp_path, config, attempt=2, previous=previous)
    _log_hyperparameters(run, 1, BANK_V2)
    history = attempt_history(request, run)
    assert len(history) == 1
    # File missing for attempt 1, but previous_report wins for attempt == request.attempt - 1.
    assert history[0].failed_checks == ["mean_column_distance"]


def test_attempt_history_missing_report_without_previous_preference(tmp_path):
    config = _config(tmp_path)
    previous = _fail_report("column_distance:job")
    request, run = _history_request(tmp_path, config, attempt=3, previous=previous)
    _log_hyperparameters(run, 1, BANK_V2)
    _log_hyperparameters(run, 2, {**BANK_V2, "epochs": 400})
    # Only attempt 2 has a saved report; attempt 1 file is missing → [].
    _save_quality_report(run, 2, "column_distance:job")
    history = attempt_history(request, run)
    assert history[0].failed_checks == []
    assert history[1].failed_checks == ["column_distance:job"]


def test_attempt_history_missing_attempt_1_falls_back_to_bank_recipe(tmp_path):
    config = _config(tmp_path)
    previous = _fail_report("column_distance")
    request, run = _history_request(tmp_path, config, attempt=2, previous=previous)
    # No hyperparameters row for attempt 1.
    history = attempt_history(request, run)
    assert len(history) == 1
    assert history[0].attempt == 1
    assert history[0].settings["pac"] == 5
    assert history[0].settings["discriminator_steps"] == 3
    assert history[0].settings["pac"] != CTGAN_DEFAULTS["pac"]
    assert history[0].settings["discriminator_steps"] != CTGAN_DEFAULTS["discriminator_steps"]
    assert history[0].settings["sampling_mode"] == "unconditional"
    assert history[0].settings["rake_columns"] == BANK_V2["rake_columns"]


def test_settings_key_ignores_rake_columns_order():
    left = {**BANK_V2, "rake_columns": ["job", "month", "poutcome"]}
    right = {**BANK_V2, "rake_columns": ["poutcome", "job", "month"]}
    assert settings_key(left) == settings_key(right)


def test_attempt_history_excludes_attempts_at_or_above_current(tmp_path):
    config = _config(tmp_path)
    previous = _fail_report("column_distance")
    request, run = _history_request(tmp_path, config, attempt=2, previous=previous)
    _log_hyperparameters(run, 1, BANK_V2)
    _log_hyperparameters(run, 2, {**BANK_V2, "epochs": 400})
    _log_hyperparameters(run, 3, {**BANK_V2, "epochs": 500})
    history = attempt_history(request, run)
    assert [record.attempt for record in history] == [1]
    assert all(record.attempt < request.attempt for record in history)
