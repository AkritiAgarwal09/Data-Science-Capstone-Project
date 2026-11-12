"""Unit tests for issue #123 retry knobs, history, and ladder (steps 2–4)."""

from __future__ import annotations

import copy
import random
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_gan.agents import model_selection as selection
from agentic_gan.agents.model_selection import (
    CTGAN_DEFAULTS,
    RETRY_OPTIONS,
    AttemptRecord,
    RetryLimits,
    RetrySettings,
    _ladder_retry,
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
from agentic_gan.tools.quality_privacy import PRIVACY_CHECKS
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


# --- Step 4: deterministic ladder --------------------------------------------------------------

DEFAULT_BOUNDS = {
    "epochs": (300, 600),
    "pac": (5, 10),
    "discriminator_steps": (1, 5),
    "batch_size": (250, 1000),
}
DISCRETE = ["job", "month", "poutcome", "marital", "education"]
ALLOWED_RAKE = ["job", "month", "poutcome", "marital", "day_of_month", "education"]


def _ladder_report(*hard_names, soft_distances=None, hard_soft_columns=()):
    """Build a FAIL report; soft column distances are hard unless listed."""
    checks = [
        CheckResult(name=name, value=1.0, threshold=0.1, passed=False, hard=True)
        for name in hard_names
    ]
    for col, distance in (soft_distances or {}).items():
        checks.append(
            CheckResult(
                name=f"column_distance:{col}",
                value=float(distance),
                threshold=0.25,
                passed=False,
                hard=col in hard_soft_columns,
            )
        )
    if not any(check.hard and not check.passed for check in checks):
        # LevelReport requires passed to match hard checks; keep a FAIL verdict.
        checks.insert(
            0,
            CheckResult(name="worst_column_distance", value=1.0, threshold=0.25, passed=False),
        )
    return QualityReport(
        status=Status.SUCCESS,
        verdict=Verdict.FAIL,
        levels=[LevelReport(fill_level=0.25, passed=False, checks=checks)],
        report_path=Path("quality.json"),
    )


def _ladder_request(tmp_path, report, gan_bounds=None, attempt=2):
    config = _config(tmp_path, gan_bounds=dict(gan_bounds or DEFAULT_BOUNDS))
    request, _run = _history_request(tmp_path, config, attempt=attempt, previous=report)
    return request


def _base_settings(**overrides):
    fields = {
        "epochs": 300,
        "batch_size": 500,
        "pac": 10,
        "discriminator_steps": 1,
        "sampling_mode": "conditional",
        "rule_filter": False,
        "rake_columns": [],
    }
    fields.update(overrides)
    return fields


def test_ladder_row1_worst_column_distance(tmp_path):
    report = _ladder_report(
        "worst_column_distance",
        soft_distances={"job": 0.40, "month": 0.30, "poutcome": 0.20, "age": 0.50},
    )
    request = _ladder_request(tmp_path, report)
    previous = _base_settings(rake_columns=["marital"])
    result = _ladder_retry(previous, [], request, DISCRETE, ALLOWED_RAKE)
    assert result.rung == 1
    assert result.exhausted is False
    assert result.settings["sampling_mode"] == "unconditional"
    # age is numeric → not in rake; job then month, poutcome as next two; keep marital.
    assert result.settings["rake_columns"][:3] == ["job", "month", "poutcome"]
    assert "marital" in result.settings["rake_columns"]


def test_ladder_row1_column_distance_job(tmp_path):
    report = _ladder_report(
        soft_distances={"job": 0.335, "education": 0.28, "month": 0.27, "poutcome": 0.10},
        hard_soft_columns=("job",),
    )
    request = _ladder_request(tmp_path, report)
    previous = _base_settings()
    result = _ladder_retry(previous, [], request, DISCRETE, ALLOWED_RAKE)
    assert result.rung == 1
    assert result.trigger == "column_distance:job"
    assert result.settings["sampling_mode"] == "unconditional"
    assert "job" in result.settings["rake_columns"]
    # next two worst categoricals after job: education, month
    assert result.settings["rake_columns"][:3] == ["job", "education", "month"]


def test_ladder_numeric_column_distance_does_not_select_row1(tmp_path):
    # Only numeric column_distance failures (not in discrete) — must not pick row 1.
    report = QualityReport(
        status=Status.SUCCESS,
        verdict=Verdict.FAIL,
        levels=[
            LevelReport(
                fill_level=0.25,
                passed=False,
                checks=[
                    CheckResult(
                        name="column_distance:balance",
                        value=0.40,
                        threshold=0.25,
                        passed=False,
                        hard=True,
                    ),
                    CheckResult(
                        name="column_distance:age",
                        value=0.35,
                        threshold=0.25,
                        passed=False,
                        hard=False,
                    ),
                ],
            )
        ],
        report_path=Path("quality.json"),
    )
    request = _ladder_request(tmp_path, report)
    previous = _base_settings(epochs=300)
    result = _ladder_retry(previous, [], request, DISCRETE, ALLOWED_RAKE)
    assert result.rung != 1
    assert result.rung == 5
    assert result.settings["epochs"] == 400


def test_ladder_row2_rule_only(tmp_path):
    report = _ladder_report("rule:pdays_poutcome")
    request = _ladder_request(tmp_path, report)
    previous = _base_settings(rule_filter=False)
    result = _ladder_retry(previous, [], request, DISCRETE, ALLOWED_RAKE)
    assert result.rung == 2
    assert result.trigger == "rule:pdays_poutcome"
    assert result.settings["rule_filter"] is True
    assert result.settings["epochs"] == previous["epochs"]


def test_ladder_row3_mean_column_distance(tmp_path):
    report = _ladder_report("mean_column_distance")
    request = _ladder_request(tmp_path, report)
    previous = _base_settings(discriminator_steps=1, pac=10, epochs=300)
    result = _ladder_retry(previous, [], request, DISCRETE, ALLOWED_RAKE)
    assert result.rung == 3
    assert result.settings["discriminator_steps"] == 3
    assert result.settings["pac"] == 5
    assert result.settings["epochs"] == 400


def test_ladder_row4_privacy(tmp_path):
    report = _ladder_report("dcr_ratio")
    request = _ladder_request(tmp_path, report)
    previous = _base_settings(epochs=400, batch_size=500, pac=10)
    result = _ladder_retry(previous, [], request, DISCRETE, ALLOWED_RAKE)
    assert result.rung == 4
    assert result.settings["epochs"] == 300
    assert result.settings["batch_size"] == 1000


def test_ladder_row5_catch_all(tmp_path):
    report = _ladder_report("schema_match")
    request = _ladder_request(tmp_path, report)
    previous = _base_settings(epochs=300)
    result = _ladder_retry(previous, [], request, DISCRETE, ALLOWED_RAKE)
    assert result.rung == 5
    assert result.trigger == "none"
    assert result.settings["epochs"] == 400


def test_ladder_row1_preferred_over_row3_unless_in_history(tmp_path):
    report = _ladder_report(
        "worst_column_distance",
        "mean_column_distance",
        soft_distances={"job": 0.4, "month": 0.3, "poutcome": 0.2},
    )
    request = _ladder_request(tmp_path, report)
    previous = _base_settings(sampling_mode="conditional", rake_columns=[])
    first = _ladder_retry(previous, [], request, DISCRETE, ALLOWED_RAKE)
    assert first.rung == 1
    history = [AttemptRecord(attempt=1, settings=first.settings, failed_checks=[])]
    second = _ladder_retry(previous, history, request, DISCRETE, ALLOWED_RAKE)
    assert second.rung == 3


@pytest.mark.parametrize("privacy_check", sorted(PRIVACY_CHECKS))
def test_ladder_privacy_plus_fidelity_never_raises_epochs(tmp_path, privacy_check):
    report = _ladder_report(privacy_check, "mean_column_distance")
    request = _ladder_request(tmp_path, report)
    previous = _base_settings(epochs=400, discriminator_steps=1, pac=10)
    result = _ladder_retry(previous, [], request, DISCRETE, ALLOWED_RAKE)
    # Row 3 matches first in table order, but privacy invariant caps epochs.
    assert result.settings["epochs"] <= previous["epochs"]


def test_ladder_caps_steps_epochs_batch_and_epochs_floor(tmp_path):
    report = _ladder_report("mean_column_distance")
    request = _ladder_request(tmp_path, report)
    # steps 4 -> 5 not 6
    result = _ladder_retry(
        _base_settings(discriminator_steps=4, epochs=300, pac=10),
        [],
        request,
        DISCRETE,
        ALLOWED_RAKE,
    )
    assert result.settings["discriminator_steps"] == 5
    # epochs 550 -> 600
    result = _ladder_retry(
        _base_settings(discriminator_steps=1, epochs=550, pac=10),
        [],
        request,
        DISCRETE,
        ALLOWED_RAKE,
    )
    assert result.settings["epochs"] == 600
    # batch 800 -> 1000 on privacy; epochs floor 300
    privacy = _ladder_report("exact_train_copies")
    request_p, _ = _history_request(
        tmp_path,
        _config(tmp_path, gan_bounds=dict(DEFAULT_BOUNDS)),
        attempt=2,
        previous=privacy,
        run_id="caps",
    )
    result = _ladder_retry(
        _base_settings(epochs=300, batch_size=800, pac=10),
        [],
        request_p,
        DISCRETE,
        ALLOWED_RAKE,
    )
    assert result.rung == 4
    assert result.settings["epochs"] == 300
    assert result.settings["batch_size"] == 1000


def test_ladder_exhausted_when_every_candidate_in_history(tmp_path):
    report = _ladder_report("rule:pdays_poutcome")
    request = _ladder_request(tmp_path, report)
    previous = _base_settings(rule_filter=False, epochs=300)
    # Candidates: row2 (rule_filter True) and row5 (epochs 400).
    row2 = {**previous, "rule_filter": True}
    row5 = {**previous, "epochs": 400}
    history = [
        AttemptRecord(attempt=1, settings=row2, failed_checks=[]),
        AttemptRecord(attempt=0, settings=row5, failed_checks=[]),
    ]
    result = _ladder_retry(previous, history, request, DISCRETE, ALLOWED_RAKE)
    assert result.exhausted is True
    assert result.settings == previous
    assert "every rung is used up" in result.rationale


def test_ladder_property_inside_bounds_and_legal(tmp_path):
    rng = random.Random(0)
    report = _ladder_report(
        "worst_column_distance",
        "mean_column_distance",
        "rule:x",
        "dcr_ratio",
        soft_distances={"job": 0.4, "month": 0.3, "poutcome": 0.2},
    )
    request = _ladder_request(tmp_path, report)
    for _ in range(40):
        previous = _base_settings(
            epochs=rng.choice([300, 400, 500, 550, 600]),
            batch_size=rng.choice([250, 500, 750, 1000]),
            pac=rng.choice([5, 10]),
            discriminator_steps=rng.choice([1, 2, 3, 4, 5]),
            sampling_mode=rng.choice(["conditional", "unconditional"]),
            rule_filter=rng.choice([False, True]),
            rake_columns=rng.sample(ALLOWED_RAKE, k=rng.randint(0, 3)),
        )
        if previous["batch_size"] % previous["pac"] != 0:
            previous["batch_size"] = (previous["batch_size"] // previous["pac"]) * previous["pac"]
        before = copy.deepcopy(previous)
        result = _ladder_retry(previous, [], request, DISCRETE, ALLOWED_RAKE)
        assert previous == before
        settings = result.settings
        for name, (low, high) in DEFAULT_BOUNDS.items():
            assert low <= settings[name] <= high
        assert settings["batch_size"] % settings["pac"] == 0
        assert settings["sampling_mode"] in RETRY_OPTIONS["sampling_mode"]
        assert settings["rule_filter"] in RETRY_OPTIONS["rule_filter"]
        assert set(settings["rake_columns"]) <= set(ALLOWED_RAKE)


def test_ladder_purity_and_no_input_mutation(tmp_path):
    report = _ladder_report(
        "column_distance:job",
        soft_distances={"job": 0.3, "month": 0.2, "poutcome": 0.1},
    )
    # Build checks with hard column_distance:job for a clear trigger.
    report = QualityReport(
        status=Status.SUCCESS,
        verdict=Verdict.FAIL,
        levels=[
            LevelReport(
                fill_level=0.25,
                passed=False,
                checks=[
                    CheckResult(
                        name="column_distance:job",
                        value=0.3,
                        threshold=0.25,
                        passed=False,
                        hard=True,
                    ),
                    CheckResult(
                        name="column_distance:month",
                        value=0.2,
                        threshold=0.25,
                        passed=False,
                        hard=False,
                    ),
                    CheckResult(
                        name="column_distance:poutcome",
                        value=0.1,
                        threshold=0.25,
                        passed=False,
                        hard=False,
                    ),
                ],
            )
        ],
        report_path=Path("quality.json"),
    )
    request = _ladder_request(tmp_path, report)
    previous = _base_settings(rake_columns=["marital"])
    history = [AttemptRecord(attempt=1, settings=_base_settings(epochs=999), failed_checks=[])]
    snap_previous = copy.deepcopy(previous)
    snap_history = copy.deepcopy(history)
    first = _ladder_retry(previous, history, request, DISCRETE, ALLOWED_RAKE)
    second = _ladder_retry(previous, history, request, DISCRETE, ALLOWED_RAKE)
    assert first == second
    assert previous == snap_previous
    assert history == snap_history

