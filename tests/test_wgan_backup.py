"""WGAN-GP backup on attempt 3, and WGAN-GP as the primary generator for all-numeric tables."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from agentic_gan.agents.model_selection import (
    WGAN_DEFAULTS,
    _rule_hash,
    run_model_selection,
)
from agentic_gan.llm import RuleOverrideError
from agentic_gan.orchestrator.decisions import read_decisions
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
    fill_rows,
)
from agentic_gan.utils.synthetic import IS_SYNTHETIC

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _labels(rows: int, positives: int) -> list[int]:
    return [1] * positives + [0] * (rows - positives)


def _config(tmp_path: Path, **overrides) -> RunConfig:
    source = tmp_path / "source.csv"
    source.write_text("age,y\n1,1\n")
    fields = {
        "dataset_name": "bank_marketing",
        "dataset_path": source,
        "target": "y",
        "positive_class": "yes",
        "fill_levels": [0.25, 0.5],
    }
    fields.update(overrides)
    return RunConfig(**fields)


def _request(
    tmp_path: Path,
    config: RunConfig,
    frame: pd.DataFrame,
    positives: int,
    negatives: int,
    *,
    attempt: int = 1,
    previous=None,
    run_id: str = "sel",
):
    run = create_run(config, run_id=run_id, runs_dir=tmp_path / "runs")
    train_path = tmp_path / f"{run_id}_train.csv"
    frame.to_csv(train_path, index=False)
    unread = tmp_path / f"{run_id}_unread.csv"
    unread.write_text("not-a-training-table\n")
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
        transformer_fit_rows=max(len(frame), 1),
        baseline_train_positives=positives,
        baseline_train_negatives=negatives,
    )
    return ModelSelectionInput(config=config, preprocess=preprocess, attempt=attempt,
                               previous_report=previous), run


def _bank_frame(rows: int = 2000, positives: int = 500) -> pd.DataFrame:
    return pd.DataFrame({
        "source_row": range(rows),
        "age": range(rows),
        "job": ["admin"] * rows,
        "month": ["may", "jun"] * (rows // 2) + ["may"] * (rows % 2),
        "poutcome": ["unknown"] * rows,
        "marital": ["married"] * rows,
        "day_of_month": [(i % 28) + 1 for i in range(rows)],
        "y": _labels(rows, positives),
    })


def _polish_all_numeric_frame(rows: int = 2000, positives: int = 600) -> pd.DataFrame:
    return pd.DataFrame({
        "A1": range(rows),
        "A2": range(rows),
        "cohort": ["3year"] * rows,
        "bankrupt": _labels(rows, positives),
    })


def _fail_report(*names):
    checks = [CheckResult(name=name, value=1.0, threshold=0.1, passed=False) for name in names]
    return QualityReport(
        status=Status.SUCCESS,
        verdict=Verdict.FAIL,
        levels=[LevelReport(fill_level=0.25, passed=False, checks=checks)],
        report_path=Path("quality.json"),
    )


def _logged(run, rule: str) -> str:
    matched = read_decisions(run).query("rule == @rule")
    assert len(matched) >= 1, f"rule {rule!r} not found in decisions"
    return matched.outcome.iloc[-1]


# ---------------------------------------------------------------------------
# WGAN-GP stub
# ---------------------------------------------------------------------------


class _WganStub:
    def __init__(self):
        self.trains: list = []
        self.samples: list = []

    def train(self, train_df, target_col, positive=1, discrete_columns=(), **kwargs):
        self.trains.append((train_df.copy(), kwargs))
        log_path = kwargs.get("log_path")
        if log_path is not None:
            pd.DataFrame({
                "Epoch": [1], "Generator Loss": [0.1],
                "Discriminator Loss": [0.2], "Gradient Penalty": [0.5],
            }).to_csv(log_path, index=False)
        feature_cols = [c for c in train_df.columns if c != target_col]
        return SimpleNamespace(
            target_col=target_col, positive=positive, columns=feature_cols,
            numeric_ranges={}, integer_valued=set(), discrete_columns=list(discrete_columns),
            transformer=None,
        )

    def save(self, model, path):
        Path(path).write_bytes(b"wgangp")

    def sample(self, model, n, *, seed, row_filter=None, drop_duplicates_of=None, **kwargs):
        self.samples.append(n)
        frame = pd.DataFrame({name: [1] * n for name in model.columns})
        frame[model.target_col] = model.positive
        frame[IS_SYNTHETIC] = 1
        return frame


class _CtganStub:
    def __init__(self):
        self.trains: list = []
        self.samples: list = []

    def train(self, train_df, target_col, **kwargs):
        self.trains.append((train_df.copy(), kwargs))
        log_path = kwargs.get("log_path")
        if log_path is not None:
            Path(log_path).write_text("Generator Loss,Discriminator Loss\n0.1,0.2\n")
        return SimpleNamespace(target_col=target_col)

    def save(self, model, path):
        Path(path).write_bytes(b"ctgan")

    def sample(self, model, n, target_value=1, seed=None, drop_duplicates_of=None, **kwargs):
        self.samples.append(n)
        frame = pd.DataFrame({name: [0] * n for name in drop_duplicates_of.columns})
        frame[drop_duplicates_of.columns[-1]] = target_value
        return frame


@pytest.fixture
def stubs(monkeypatch):
    wgan = _WganStub()
    ctgan = _CtganStub()
    monkeypatch.setattr("agentic_gan.tools.wgan_gp.train_wgan_gp", wgan.train)
    monkeypatch.setattr("agentic_gan.tools.wgan_gp.save_model", wgan.save)
    monkeypatch.setattr("agentic_gan.tools.wgan_gp.sample_wgan_gp", wgan.sample)
    monkeypatch.setattr("agentic_gan.tools.ctgan.train_ctgan", ctgan.train)
    monkeypatch.setattr("agentic_gan.tools.ctgan.save_model", ctgan.save)
    monkeypatch.setattr("agentic_gan.tools.ctgan.sample_synthetic", ctgan.sample)
    return wgan, ctgan


# ---------------------------------------------------------------------------
# Backup tests: CTGAN attempt 3 falls through to WGAN-GP
# ---------------------------------------------------------------------------


def test_wgan_backup_attempt_3_returns_success_and_writes_files(tmp_path, stubs):
    wgan, ctgan = stubs
    frame = _bank_frame()
    request, run = _request(
        tmp_path, _config(tmp_path), frame, 500, 1500,
        attempt=3, previous=_fail_report("column_distance"),
    )
    output = run_model_selection(request, run)

    assert output.status is Status.SUCCESS
    assert output.generator == "wgan_gp"
    assert output.synthetic_sets, "synthetic_sets must be non-empty"

    # checkpoint and loss files exist
    assert output.checkpoint_path is not None and Path(output.checkpoint_path).exists()
    assert output.loss_history_path is not None and Path(output.loss_history_path).exists()

    # fill files exist on disk
    for item in output.synthetic_sets:
        assert Path(item.path).exists(), f"fill file missing: {item.path}"

    # row counts match expectations
    expected = [fill_rows(level, 500, 1500) for level in request.config.fill_levels]
    assert [item.rows for item in output.synthetic_sets] == expected

    assert wgan.trains, "WGAN-GP stub must be called"
    assert ctgan.trains == [], "CTGAN must not be called on backup path"


def test_wgan_backup_attempt_3_logs_hyperparameters_with_wgan_gp_generator(tmp_path, stubs):
    frame = _bank_frame()
    request, run = _request(
        tmp_path, _config(tmp_path), frame, 500, 1500,
        attempt=3, previous=_fail_report("column_distance"),
    )
    run_model_selection(request, run)

    decisions = read_decisions(run)
    hp_rows = decisions.query("rule == 'hyperparameters'")
    assert not hp_rows.empty, "hyperparameters must be logged"
    last_hp = hp_rows.iloc[-1]
    assert last_hp.inputs["generator"] == "wgan_gp"
    for key, val in WGAN_DEFAULTS.items():
        assert last_hp.inputs[key] == val

    # no SKIPPED fallback row (backup succeeded)
    skipped_rows = decisions.query("rule == 'backup_failed'")
    assert skipped_rows.empty, "backup_failed must not be logged on success"


def test_wgan_backup_attempt_3_fallback_to_ctgan_on_runtime_error(tmp_path, stubs, monkeypatch):
    wgan, ctgan = stubs

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr("agentic_gan.tools.wgan_gp.train_wgan_gp", _raise)

    frame = _bank_frame()
    request, run = _request(
        tmp_path, _config(tmp_path), frame, 500, 1500,
        attempt=3, previous=_fail_report("column_distance"),
    )
    output = run_model_selection(request, run)

    assert output.status is Status.SUCCESS
    assert output.generator == "ctgan"
    assert ctgan.trains, "CTGAN must be trained after WGAN-GP failure"
    assert _logged(run, "backup_failed") == "FALLBACK_CTGAN"
    # partial WGAN files must be cleaned up
    folder = run.subdir("synthetic")
    assert not (folder / "wgan_gp_attempt_3.pkl").exists()
    assert not (folder / "wgan_gp_loss_attempt_3.csv").exists()


def test_wgan_backup_import_error_treated_same_as_runtime_error(tmp_path, stubs, monkeypatch):
    _, ctgan = stubs

    def _raise(*args, **kwargs):
        raise ImportError("torch not available")

    monkeypatch.setattr("agentic_gan.tools.wgan_gp.train_wgan_gp", _raise)

    frame = _bank_frame()
    request, run = _request(
        tmp_path, _config(tmp_path), frame, 500, 1500,
        attempt=3, previous=_fail_report("column_distance"),
    )
    output = run_model_selection(request, run)

    assert output.status is Status.SUCCESS
    assert output.generator == "ctgan"
    assert ctgan.trains
    assert _logged(run, "backup_failed") == "FALLBACK_CTGAN"


def test_wgan_backup_rule_override_error_propagates(tmp_path, stubs, monkeypatch):
    # RuleOverrideError must NOT be swallowed by the backup exception handler.
    def _raise(*args, **kwargs):
        raise RuleOverrideError("the LLM changed the generator")

    monkeypatch.setattr("agentic_gan.tools.wgan_gp.train_wgan_gp", _raise)

    frame = _bank_frame()
    request, run = _request(
        tmp_path, _config(tmp_path), frame, 500, 1500,
        attempt=3, previous=_fail_report("column_distance"),
    )
    with pytest.raises(RuleOverrideError):
        run_model_selection(request, run)


def test_wgan_backup_minority_below_500_logs_backup_skipped_and_trains_ctgan(tmp_path, stubs):
    wgan, ctgan = stubs
    # 299 positives < minority_min=500 → minority_gate is FORBIDDEN_CLASS_ONLY → not eligible
    frame = _bank_frame(rows=2000, positives=299)
    # Rebalance to make size_gate pass (rows=2000 >= 2000) but minority<500
    request, run = _request(
        tmp_path, _config(tmp_path), frame, 299, 1701,
        attempt=3, previous=_fail_report("column_distance"),
    )
    output = run_model_selection(request, run)

    assert output.status is Status.SUCCESS
    assert output.generator == "ctgan"
    assert ctgan.trains, "CTGAN must be trained when backup is not eligible"
    assert wgan.trains == [], "WGAN-GP must not be called when backup_skipped"
    assert _logged(run, "backup_skipped") == "NOT_ELIGIBLE"


# ---------------------------------------------------------------------------
# Primary WGAN-GP route (all-numeric tables)
# ---------------------------------------------------------------------------


def test_all_numeric_polish_selects_wgan_gp_as_primary_on_attempt_1(tmp_path, stubs):
    wgan, ctgan = stubs
    rows, positives = 2000, 600
    frame = _polish_all_numeric_frame(rows, positives)
    config = _config(tmp_path, dataset_name="polish", target="bankrupt", positive_class=1)
    request, run = _request(tmp_path, config, frame, positives, rows - positives)

    output = run_model_selection(request, run)

    assert output.status is Status.SUCCESS
    assert output.generator == "wgan_gp"
    assert wgan.trains, "WGAN-GP stub must be called on attempt 1"
    assert ctgan.trains == [], "CTGAN must not be called"


def test_all_numeric_wgan_gp_primary_does_not_log_retry_not_applicable_on_attempt_1(
    tmp_path, stubs
):
    rows, positives = 2000, 600
    frame = _polish_all_numeric_frame(rows, positives)
    config = _config(tmp_path, dataset_name="polish", target="bankrupt", positive_class=1)
    request, run = _request(tmp_path, config, frame, positives, rows - positives)
    run_model_selection(request, run)

    decisions = read_decisions(run)
    retry_rows = decisions.query("rule == 'retry_not_applicable'")
    assert retry_rows.empty, "retry_not_applicable must not be logged on attempt 1"


def test_all_numeric_wgan_gp_primary_logs_retry_not_applicable_on_attempt_2(tmp_path, stubs):
    rows, positives = 2000, 600
    frame = _polish_all_numeric_frame(rows, positives)
    config = _config(tmp_path, dataset_name="polish", target="bankrupt", positive_class=1)
    request, run = _request(
        tmp_path, config, frame, positives, rows - positives,
        attempt=2, previous=_fail_report("column_distance"),
    )
    run_model_selection(request, run)
    assert _logged(run, "retry_not_applicable") == "NOT_APPLIED"


# ---------------------------------------------------------------------------
# torch isolation: module-level import must not load torch
# ---------------------------------------------------------------------------


def test_importing_model_selection_does_not_import_torch():
    result = subprocess.run(
        [
            sys.executable, "-c",
            (
                "import agentic_gan.agents.model_selection; import sys; "
                "assert 'torch' not in sys.modules, "
                "'torch was imported at model_selection module level'"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# _rule_hash: WGAN_DEFAULTS must be part of the fingerprint
# ---------------------------------------------------------------------------


def test_changing_wgan_defaults_changes_rule_hash(tmp_path, monkeypatch):
    from agentic_gan.agents import model_selection as ms

    config = _config(tmp_path)
    original = _rule_hash(config)
    monkeypatch.setattr(ms, "WGAN_DEFAULTS", {**ms.WGAN_DEFAULTS, "epochs": 999})
    assert _rule_hash(config) != original, "WGAN_DEFAULTS must be part of the rule hash"
