"""Unit tests for scripts/audit_retry.py (issue #123 step 9)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from agentic_gan.orchestrator.decisions import log_decision
from agentic_gan.orchestrator.manifest import create_run
from agentic_gan.schemas import RunConfig

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_retry.py"


def _load_audit():
    spec = importlib.util.spec_from_file_location("audit_retry", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit_retry = _load_audit()
audit = audit_retry.audit
format_table = audit_retry.format_table

SEVEN_A = {
    "epochs": 300,
    "batch_size": 500,
    "pac": 5,
    "discriminator_steps": 3,
    "sampling_mode": "conditional",
    "rule_filter": True,
    "rake_columns": ["job", "month"],
}
SEVEN_B = {
    **SEVEN_A,
    "sampling_mode": "unconditional",
    "rake_columns": ["job", "month", "poutcome"],
}
SEVEN_C = {
    **SEVEN_B,
    "epochs": 400,
    "pac": 10,
    "discriminator_steps": 5,
}


def _config(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("age,y\n1,yes\n")
    return RunConfig(
        dataset_name="bank_marketing",
        dataset_path=source,
        target="y",
        positive_class="yes",
        gan_bounds={"epochs": (300, 600), "pac": (5, 10)},
    )


def _run(tmp_path, run_id="audit"):
    return create_run(_config(tmp_path), run_id=run_id, runs_dir=tmp_path / "runs")


def _hyperparameters(run, attempt, settings):
    log_decision(
        "model_selection",
        "hyperparameters",
        {"attempt": attempt, **settings, "seed": 42, "generator": "ctgan"},
        "SET",
        "CTGAN settings are inside gan_bounds.",
        run=run,
    )


def _retry(run, attempt, previous, applied, *, source="ladder", trigger="worst_column_distance"):
    log_decision(
        "model_selection",
        "hyperparameter_retry",
        {
            "attempt": attempt,
            "source": source,
            "trigger": trigger,
            "previous_settings": previous,
            "proposed": None,
            "applied": applied,
            "failed_checks": ["worst_column_distance"],
            "rung": 1,
        },
        "ADJUSTED",
        f"ladder rung on {trigger}",
        run=run,
    )


def test_good_run_has_no_violations(tmp_path):
    run = _run(tmp_path, "good")
    _hyperparameters(run, 1, SEVEN_A)
    _retry(run, 2, SEVEN_A, SEVEN_B, source="ladder")
    _hyperparameters(run, 2, SEVEN_B)
    _retry(run, 3, SEVEN_B, SEVEN_C, source="llm", trigger="mean_column_distance")
    _hyperparameters(run, 3, SEVEN_C)
    assert audit(run) == []
    table = format_table(run)
    assert "ladder" in table
    assert "sampling_mode" in table or "rake_columns" in table


def test_retry_unchanged_without_exhaustion_is_a_violation(tmp_path):
    run = _run(tmp_path, "unchanged")
    _hyperparameters(run, 1, SEVEN_A)
    _retry(run, 2, SEVEN_A, SEVEN_B)
    _hyperparameters(run, 2, SEVEN_B)
    log_decision(
        "model_selection",
        "retry_unchanged",
        {
            "attempt": 3,
            "source": "ladder",
            "trigger": "none",
            "previous_settings": SEVEN_B,
            "applied": SEVEN_B,
            "rung": 5,
        },
        "UNCHANGED",
        "kept the previous settings for an unclear reason",
        run=run,
    )
    violations = audit(run)
    assert any("retry_unchanged" in message for message in violations)


def test_retry_unchanged_when_exhausted_is_allowed(tmp_path):
    run = _run(tmp_path, "exhausted")
    _hyperparameters(run, 1, SEVEN_A)
    _retry(run, 2, SEVEN_A, SEVEN_B)
    _hyperparameters(run, 2, SEVEN_B)
    log_decision(
        "model_selection",
        "retry_unchanged",
        {
            "attempt": 3,
            "source": "ladder",
            "trigger": "none",
            "previous_settings": SEVEN_B,
            "applied": SEVEN_B,
            "exhausted": True,
            "rung": 5,
        },
        "UNCHANGED",
        "every rung is used up",
        run=run,
    )
    # Attempt 3 reached via retry_unchanged, but no hyperparameters for attempt 3 → violation.
    # Log hyperparameters equal to attempt 2 would also violate distinctness; skip attempt-3 hp.
    # For this case, do not mark attempt 3 as "reached" via hyperparameters — only via retry.
    # The audit treats retry_unchanged attempt 3 as reached_three, so add distinct hp.
    _hyperparameters(run, 3, SEVEN_C)
    assert audit(run) == []


def test_unsupported_adjustment_is_a_violation(tmp_path):
    run = _run(tmp_path, "unsupported")
    _hyperparameters(run, 1, SEVEN_A)
    _retry(run, 2, SEVEN_A, SEVEN_B)
    _hyperparameters(run, 2, SEVEN_B)
    log_decision(
        "model_selection",
        "unsupported_adjustment",
        {"attempt": 2, "adjustment": "change_sampling"},
        "IGNORED",
        "not a tunable knob",
        run=run,
    )
    violations = audit(run)
    assert any("unsupported_adjustment" in message for message in violations)


def test_skipped_attempt_3_is_a_violation(tmp_path):
    run = _run(tmp_path, "skipped3")
    _hyperparameters(run, 1, SEVEN_A)
    _retry(run, 2, SEVEN_A, SEVEN_B)
    _hyperparameters(run, 2, SEVEN_B)
    _hyperparameters(run, 3, SEVEN_C)
    log_decision(
        "model_selection",
        "fallback",
        {"attempt": 3, "primary": "ctgan", "fallback": "wgan_gp"},
        "SKIPPED",
        "no implemented fallback generator",
        run=run,
    )
    violations = audit(run)
    assert any("fallback SKIPPED" in message for message in violations)


def test_attempt_3_matching_earlier_settings_is_a_violation(tmp_path):
    run = _run(tmp_path, "same3")
    _hyperparameters(run, 1, SEVEN_A)
    _retry(run, 2, SEVEN_A, SEVEN_B)
    _hyperparameters(run, 2, SEVEN_B)
    _hyperparameters(run, 3, SEVEN_B)  # same as attempt 2
    violations = audit(run)
    assert any("attempt 3 hyperparameters match attempt 2" in message for message in violations)


def test_stage_error_failed_is_a_violation(tmp_path):
    run = _run(tmp_path, "stagefail")
    _hyperparameters(run, 1, SEVEN_A)
    _retry(run, 2, SEVEN_A, SEVEN_B)
    _hyperparameters(run, 2, SEVEN_B)
    log_decision(
        "orchestrator",
        "stage_error",
        {"stage": "model_selection", "attempt": 2},
        "FAILED",
        "boom",
        run=run,
    )
    violations = audit(run)
    assert any("stage_error FAILED" in message for message in violations)


def test_cli_exits_one_on_violations(tmp_path):
    run = _run(tmp_path, "cli")
    # No retry row → violation.
    _hyperparameters(run, 1, SEVEN_A)
    assert audit_retry.main([str(run.root)]) == 1


def test_cli_exits_zero_on_clean_run(tmp_path):
    run = _run(tmp_path, "cli-ok")
    _hyperparameters(run, 1, SEVEN_A)
    _retry(run, 2, SEVEN_A, SEVEN_B)
    _hyperparameters(run, 2, SEVEN_B)
    assert audit_retry.main([str(run.root)]) == 0
