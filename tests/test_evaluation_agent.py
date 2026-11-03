"""Offline reporting tests, using real training output and scripted narratives."""

import json
from dataclasses import replace

import pandas as pd
import pytest
from test_training_agent import make_experiment, tiny_suite

from agentic_gan.agents import evaluation
from agentic_gan.agents.training import run_training
from agentic_gan.llm import FakeLLM, RuleOverrideError
from agentic_gan.orchestrator.decisions import read_decisions
from agentic_gan.schemas import EvaluationInput, Stage, Status
from agentic_gan.tools import baselines


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(baselines, "build_models", tiny_suite)
    request, run, *_ = make_experiment(tmp_path)
    result = run_training(request, run)
    treatments = run.subdir("profile") / "treatment_log.csv"
    treatments.write_text("column,treatment\nLIMIT_BAL,median\n")
    incoming = EvaluationInput(
        config=request.config,
        training=result,
        treatment_log_path=treatments,
        decision_log_path=run.decisions_path,
        manifest_path=run.manifest_path,
        state_path=run.state_path,
    )
    return incoming, run


def answer(**changes):
    return {**evaluation.FALLBACK.model_dump(), **changes}


def test_verified_narrative_tools_and_registered_artifacts(experiment):
    request, run = experiment
    fake = FakeLLM([answer(summary="The observed results use 24 test rows.")])
    result = evaluation.run_evaluation(request, run, client=fake)
    assert result.status is Status.SUCCESS and result.numbers_verified
    assert len(fake.calls) == 1
    report = result.report_path.read_text()
    assert "The observed results" in report
    assert "## Negative or inconclusive results" in report
    assert "## Who benefits" in report
    assert "Small test positive counts" in report
    assert evaluation.evaluation_tools.verify_report_numbers(
        report, result.metrics_table_paths
    ).verified
    tables = {p.stem: p for p in result.metrics_table_paths}
    comparison = pd.read_csv(tables["comparisons"])
    assert {"original", "cleaned", "smotenc"} <= set(comparison.reference)
    segments = pd.read_csv(tables["segments"])
    assert not segments.empty and segments.small.any()
    assert run.load_manifest().register(run, Stage.EVALUATION, result).verify(run) == []


def test_made_up_number_rejected_then_regenerated_once(experiment):
    request, run = experiment
    fake = FakeLLM(
        [
            answer(summary="There were 987654321 test observations."),
            answer(summary="Only recorded evidence is reported."),
        ]
    )
    result = evaluation.run_evaluation(request, run, client=fake)
    assert result.status is Status.SUCCESS and len(fake.calls) == 2
    assert "987654321" not in result.report_path.read_text()
    assert "Only recorded evidence" in result.report_path.read_text()
    decisions = read_decisions(run)
    assert "REJECTED" in decisions.outcome.tolist()
    assert "LLM_FALLBACK" not in decisions.outcome.tolist()


def test_repeated_fabrication_uses_verified_plain_template(experiment):
    request, run = experiment
    fake = FakeLLM([answer(improvements="Performance improved by 654321%.")] * 2)
    result = evaluation.run_evaluation(request, run, client=fake)
    assert result.numbers_verified and len(fake.calls) == 2
    assert evaluation.FALLBACK.summary in result.report_path.read_text()
    assert "654321" not in result.report_path.read_text()
    assert "LLM_FALLBACK" in read_decisions(run).outcome.tolist()


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("offline"), {"summary": "missing sections"}, {**answer(), "threshold": 1}],
)
def test_call_or_schema_failure_uses_same_retry_budget(experiment, failure):
    request, run = experiment
    fake = FakeLLM([failure, failure])
    result = evaluation.run_evaluation(request, run, client=fake)
    assert result.status is Status.SUCCESS and len(fake.calls) == 2
    assert "LLM_FALLBACK" in read_decisions(run).outcome.tolist()


def test_unavailable_gan_and_unset_volume_are_explicit(experiment):
    request, run = experiment
    run.load_state().with_gan(False, "Quality gate did not pass").save(run.state_path)
    result = evaluation.run_evaluation(request, run, client=FakeLLM([answer()]))
    report = result.report_path.read_text()
    assert "Unavailable" in report and "volume rule did not run" in report
    comparison = pd.read_csv(next(p for p in result.metrics_table_paths if p.stem == "comparisons"))
    assert "ctgan" not in comparison.arm.tolist()
    evidence = json.loads(
        next(p for p in result.metrics_table_paths if p.stem == "evidence").read_text()
    )
    assert evidence["gan_unavailable_reason"] == "Quality gate did not pass"


def test_rule_mutation_is_rejected(experiment):
    request, run = experiment

    class MutatingLLM:
        def complete(self, *args):
            cfg = run.load_config()
            run.config_path.write_text(cfg.updated(min_rows=cfg.min_rows + 1).to_yaml())
            return answer()

    with pytest.raises(RuleOverrideError):
        evaluation.run_evaluation(request, run, client=MutatingLLM())
    assert not (run.subdir("report") / "final_report.md").exists()


def test_template_failure_returns_failed_not_unverified_success(experiment, monkeypatch):
    request, run = experiment
    original = evaluation.evaluation_tools.verify_report_numbers
    monkeypatch.setattr(
        evaluation.evaluation_tools,
        "verify_report_numbers",
        lambda *a, **k: replace(original(*a, **k), verified=False),
    )
    fake = FakeLLM([answer()])
    result = evaluation.run_evaluation(request, run, client=fake)
    assert result.status is Status.FAILED and not result.numbers_verified
    assert not fake.calls and result.report_path is None


def test_final_report_is_checked_even_after_narrative_validation(experiment, monkeypatch):
    request, run = experiment
    original = evaluation.evaluation_tools.verify_report_numbers
    calls = []

    def checker(*args, **kwargs):
        calls.append(1)
        result = original(*args, **kwargs)
        # Initial template, runtime fallback, LLM answer, final rendered report.
        return replace(result, verified=False) if len(calls) == 4 else result

    monkeypatch.setattr(evaluation.evaluation_tools, "verify_report_numbers", checker)
    result = evaluation.run_evaluation(request, run, client=FakeLLM([answer()]))
    assert result.status is Status.FAILED and not result.numbers_verified
    assert not (run.subdir("report") / "final_report.md").exists()


def test_missing_segment_columns_are_an_explicit_limitation(experiment, monkeypatch):
    request, run = experiment
    monkeypatch.setattr(evaluation, "SEGMENT_COLUMNS", ())
    result = evaluation.run_evaluation(request, run, client=FakeLLM([answer()]))
    assert result.status is Status.SUCCESS
    assert "who benefits by group is unknown" in result.report_path.read_text()
    segments = pd.read_csv(next(p for p in result.metrics_table_paths if p.stem == "segments"))
    assert segments.empty


def test_result_artifacts_cannot_change_during_narration(experiment):
    request, run = experiment

    class MutatingLLM:
        def complete(self, *args):
            path = run.subdir("report") / "evidence.json"
            data = json.loads(path.read_text())
            data["invented"] = 987654321
            path.write_text(json.dumps(data))
            return answer(summary="The test set has 987654321 rows.")

    with pytest.raises(RuleOverrideError):
        evaluation.run_evaluation(request, run, client=MutatingLLM())
    assert not (run.subdir("report") / "final_report.md").exists()
