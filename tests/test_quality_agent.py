"""Quality-agent reports, its pass, retry, and exhaustion routes, and the LLM diagnosis."""

import json
import os
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

import agentic_gan.agents.quality as quality_module
import agentic_gan.orchestrator.graph as graph_module
from agentic_gan.llm import FakeLLM, RuleOverrideError
from agentic_gan.orchestrator.decisions import read_decisions
from agentic_gan.orchestrator.manifest import create_run
from agentic_gan.orchestrator.stubs import default_agents
from agentic_gan.schemas import (
    ArmFiles,
    CheckResult,
    GeneratorOutput,
    PreprocessOutput,
    QualityDiagnosis,
    QualityInput,
    QualityReport,
    RunConfig,
    Stage,
    Status,
    SyntheticSet,
    Verdict,
)


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path.resolve()


class PreprocessFixture:
    def __call__(self, request, run):
        data = run.subdir("data")
        train = pd.DataFrame({"x": range(40), "y": [0, 1] * 20})
        val = pd.DataFrame({"x": range(16), "y": [0, 1] * 8})
        train_path = data / "baseline_train.csv"
        val_path = data / "baseline_val.csv"
        train.to_csv(train_path, index=False)
        val.to_csv(val_path, index=False)
        files = ArmFiles(
            train_path=train_path.resolve(),
            val_path=val_path.resolve(),
            test_path=val_path.resolve(),
        )
        profile = run.subdir("profile")
        return PreprocessOutput(
            status=Status.SUCCESS,
            raw=files,
            baseline=files,
            profile_path=_write(profile / "profile.json", "{}\n"),
            treatment_log_path=_write(profile / "treatment.json", "{}\n"),
            transformer_path=_write(profile / "transformer.txt", "fixture\n"),
            transformer_fit_rows=40,
            baseline_train_positives=20,
            baseline_train_negatives=20,
        )


class GeneratorFixture:
    def __init__(self):
        self.calls = 0
        self.previous_verdicts = []
        self.previous_diagnoses = []

    def __call__(self, request, run):
        self.calls += 1
        previous = request.previous_report
        self.previous_verdicts.append(None if previous is None else previous.verdict)
        self.previous_diagnoses.append(None if previous is None else previous.diagnosis)
        folder = run.subdir("synthetic")
        sets = []
        for level in request.config.fill_levels:
            path = folder / f"attempt_{self.calls}_fill_{str(level).replace('.', 'p')}.csv"
            pd.DataFrame(
                {"x": range(6), "y": [1] * 6, "level": [level] * 6, "attempt": [self.calls] * 6}
            ).to_csv(path, index=False)
            sets.append(SyntheticSet(fill_level=level, rows=6, path=path.resolve()))
        return GeneratorOutput(
            status=Status.SUCCESS,
            generator="fixture",
            rationale="Test generator",
            checkpoint_path=_write(folder / f"checkpoint_{self.calls}.txt", "fixture\n"),
            loss_history_path=_write(folder / f"loss_{self.calls}.csv", "loss\n0\n"),
            synthetic_sets=sets,
        )


def _config(tmp_path):
    source = _write(tmp_path / "source.csv", "x,y\n0,0\n1,1\n")
    return RunConfig(
        dataset_name="quality_fixture",
        dataset_path=source,
        target="y",
        positive_class="yes",
        fill_levels=[0.1, 0.25, 0.5],
    )


# A valid LLM answer for the failing fixture checks below ("fidelity" on column "x").
FIXTURE_DIAGNOSIS = {
    "failed_checks": ["fidelity"],
    "columns": ["x"],
    "likely_cause": "Fidelity fails on x; the classifier AUC is 0.7.",
    "adjustments": ["more_epochs"],
}


@pytest.mark.parametrize(
    ("scenario", "attempts", "available", "outcomes"),
    [
        ("pass", 1, True, ["ACCEPT"]),
        ("retry", 2, True, ["RETRY_MODEL_SELECTION", "ACCEPT"]),
        ("fail", 3, False, ["RETRY_MODEL_SELECTION", "RETRY_MODEL_SELECTION", "NOT_AVAILABLE"]),
    ],
)
def test_quality_agent_in_graph(tmp_path, monkeypatch, scenario, attempts, available, outcomes):
    def fake_fidelity(real, synthetic, **kwargs):
        passed = scenario == "pass" or (scenario == "retry" and synthetic["attempt"].iloc[0] > 1)
        passed = passed and synthetic["level"].iloc[0] == 0.25
        return SimpleNamespace(
            checks=[
                CheckResult(name="fidelity", value=0 if passed else 1, threshold=0.5, passed=passed)
            ],
            columns=pd.DataFrame({"distance": [0.1]}, index=["x"]),
        )

    def fake_privacy(train, val, synthetic, target, positive, **kwargs):
        return (
            [
                CheckResult(name="privacy", value=0, threshold=0, passed=True),
                CheckResult(
                    name="real_vs_synthetic_auc", value=0.7, threshold=0.5, passed=False, hard=False
                ),
            ],
            {
                "validation_median": 1.0,
                "synthetic_median": 1.1,
                "real_only_pr_auc": 0.5,
                "augmented_pr_auc": 0.6,
                "real_only_best_f1": 0.5,
                "augmented_best_f1": 0.6,
                "revealing_columns": ["x"],
                "real_vs_real_auc": 0.5,
            },
        )

    monkeypatch.setattr(quality_module, "fidelity_checks", fake_fidelity)
    monkeypatch.setattr(quality_module, "quality_privacy_checks", fake_privacy)
    agents = default_agents()
    agents[Stage.PREPROCESSING] = PreprocessFixture()
    generator = GeneratorFixture()
    agents[Stage.MODEL_SELECTION] = generator
    llm = FakeLLM([FIXTURE_DIAGNOSIS] * 3)
    # Explicit fixtures keep this routing test independent of the real split/repair stages.
    agents[Stage.QUALITY_CHECK] = partial(quality_module.run_quality, client=llm)
    state, run = graph_module.run_pipeline(
        _config(tmp_path), runs_dir=tmp_path / "runs", agents=agents
    )
    assert state.is_complete
    assert state.gan_available is available
    assert state.attempts(Stage.QUALITY_CHECK) == attempts
    assert state.attempts(Stage.MODEL_SELECTION) == attempts
    assert generator.previous_verdicts == [None] + [Verdict.FAIL] * (attempts - 1)
    diagnosis = QualityDiagnosis(**FIXTURE_DIAGNOSIS)
    assert len(llm.calls) == attempts - int(available)
    assert generator.previous_diagnoses == [None] + [diagnosis] * (attempts - 1)
    decisions = read_decisions(run)
    assert decisions.query("rule == 'quality_gate'").outcome.tolist() == outcomes
    assert decisions.query("rule == 'quality_verdict'").outcome.tolist() == [
        *["FAIL"] * (attempts - 1),
        "PASS" if available else "FAIL",
    ]
    assert len(decisions.query("rule == 'fill_level'")) == 3 * attempts
    assert run.load_manifest().verify(run) == []
    report = QualityReport.model_validate_json(
        (run.subdir("quality") / f"quality_report_attempt_{attempts}.json").read_text()
    )
    assert report.verdict is (Verdict.PASS if available else Verdict.FAIL)
    assert report.diagnosis == (None if available else diagnosis)
    assert len(report.levels) == 3
    assert sum(level.passed for level in report.levels) == int(available)
    plots = list(run.subdir("quality").glob("*.png"))
    assert len(plots) == 3 * attempts
    assert plots[0].read_bytes().startswith(b"\x89PNG")
    gan_models = [
        entry
        for entry in run.load_manifest().produced_by(Stage.TRAINING)
        if entry.path.name.startswith("gan_")
    ]
    assert bool(gan_models) is available
    if not available:
        assert "NOT_AVAILABLE" in (run.subdir("report") / "final_report.md").read_text()


# --- the LLM diagnosis, on the CTGAN v1 results for Bank -----------------------------------------
# Column distances, checks and classifier AUC for CTGAN v1 on Bank at 25% fill (see docs/notes/
# ctgan_bank_v1_quality.md). The real tools give the same failures on the committed v1 sets in
# data/interim/bank_marketing. The detective model scores AUC 0.86; two real samples score 0.52.

V1_DISTANCES = {
    "age": 0.0733, "job": 0.2792, "marital": 0.1236, "education": 0.0971, "default": 0.0122,
    "balance": 0.0895, "housing": 0.1129, "loan": 0.0122, "contact": 0.0378,
    "day_of_month": 0.2299, "month": 0.174, "campaign": 0.0877, "pdays": 0.1332,
    "previous": 0.0559, "poutcome": 0.1187, "previously_contacted": 0.0559,
}  # fmt: skip
V1_DRIVERS = ["job", "month", "day_of_month"]
V1_ANSWER = {
    "failed_checks": ["mean_column_distance", "worst_column_distance", "real_vs_synthetic_auc"],
    "columns": ["job", "day_of_month", "month"],
    "likely_cause": (
        "CTGAN did not learn job, day_of_month and month: a classifier tells real from "
        "synthetic rows with AUC 0.86, where two real samples score 0.52."
    ),
    "adjustments": ["more_epochs", "rebalance_categories"],
}


def _v1_fidelity(real, synthetic, **kwargs):
    def check(name, value, limit, hard=True):
        return CheckResult(
            name=name, value=value, threshold=limit, passed=value <= limit, hard=hard
        )

    checks = [
        check("mean_column_distance", 0.106, 0.10),
        check("worst_column_distance", 0.279, 0.25),
        check("spearman_diff", 0.066, 0.10),
        check("cramers_v_diff", 0.046, 0.10),
        check("rule:pdays_poutcome", 0.016, 0.0),
        check("rule:previously_contacted_pdays", 0.0, 0.0),
        check("rule:previous_pdays", 0.0, 0.0),
        check("rule:pdays_range", 0.0, 0.0),
        check("label_balance", 0.0, 0.01),
        check("schema_match", 0.0, 0.0),
        check("column_distance:job", 0.279, 0.25, hard=False),
    ]
    columns = pd.DataFrame({"distance": V1_DISTANCES.values()}, index=V1_DISTANCES.keys())
    return SimpleNamespace(checks=checks, columns=columns)


def _v1_privacy(train, val, synthetic, target, positive, **kwargs):
    checks = [
        CheckResult(name="exact_train_copies", value=0, threshold=0, passed=True),
        CheckResult(name="synthetic_duplicates", value=0, threshold=0, passed=True),
        CheckResult(name="dcr_ratio", value=1.106, threshold=1.0, passed=True),
        CheckResult(
            name="real_vs_synthetic_auc", value=0.8626, threshold=0.5, passed=False, hard=False
        ),
        CheckResult(name="augmentation_pr_auc_gain", value=-0.0515, threshold=0, passed=False),
        CheckResult(name="augmentation_best_f1_gain", value=-0.0515, threshold=0, passed=False),
    ]
    details = {
        "validation_median": 1.5627,
        "synthetic_median": 1.7278,
        "real_only_pr_auc": 0.4048,
        "augmented_pr_auc": 0.3533,
        "real_only_best_f1": 0.4477,
        "augmented_best_f1": 0.3962,
        "revealing_columns": ["job", "day_of_month", "month", "pdays", "balance"],
        "real_vs_real_auc": 0.5177,
    }
    return checks, details


@pytest.fixture
def v1(tmp_path, monkeypatch):
    """A run and quality request whose checks reproduce the CTGAN v1 failure."""
    monkeypatch.setattr(quality_module, "fidelity_checks", _v1_fidelity)
    monkeypatch.setattr(quality_module, "quality_privacy_checks", _v1_privacy)
    config = _config(tmp_path)
    run = create_run(config, runs_dir=tmp_path / "runs")
    generator = GeneratorFixture()(SimpleNamespace(config=config, previous_report=None), run)
    request = QualityInput(
        config=config, preprocess=PreprocessFixture()(None, run), generator=generator
    )
    return request, run


def test_diagnosis_of_v1_is_stored_and_the_llm_sees_the_results(v1):
    request, run = v1
    llm = FakeLLM([V1_ANSWER])
    report = quality_module.run_quality(request, run, client=llm)

    assert report.verdict is Verdict.FAIL
    assert report.diagnosis == QualityDiagnosis(**V1_ANSWER)
    stored = QualityReport.model_validate_json(report.report_path.read_text())
    assert stored.diagnosis == report.diagnosis

    ((_, context, output_model),) = llm.calls
    assert output_model is QualityDiagnosis
    facts = json.loads(context)
    assert facts["allowed_adjustments"] == [
        "more_epochs", "change_pac", "rebalance_categories", "change_sampling",
    ]  # fmt: skip
    assert [level["fill_level"] for level in facts["levels"]] == [0.1, 0.25, 0.5]
    level = facts["levels"][1]
    failed = [check["name"] for check in level["failed_hard_checks"]]
    assert failed == [
        "mean_column_distance",
        "worst_column_distance",
        "rule:pdays_poutcome",
        "augmentation_pr_auc_gain",
        "augmentation_best_f1_gain",
    ]
    assert list(level["largest_column_distances"])[:3] == ["job", "day_of_month", "month"]
    assert level["revealing_columns"][:3] == ["job", "day_of_month", "month"]
    assert (level["real_vs_synthetic_auc"], level["real_vs_real_auc"]) == (0.863, 0.518)

    decisions = read_decisions(run)
    assert decisions.query("rule == 'quality_verdict'").outcome.tolist() == ["FAIL"]
    (step,) = decisions.query("rule == 'llm_step'").itertuples()
    assert step.agent == "quality_check"
    assert step.outcome["columns"] == V1_ANSWER["columns"]


def test_diagnosis_of_v1_falls_back_to_the_facts_when_the_llm_is_down(v1):
    request, run = v1
    report = quality_module.run_quality(
        request, run, client=FakeLLM([RuntimeError("offline"), RuntimeError("offline")])
    )

    assert report.verdict is Verdict.FAIL
    diagnosis = report.diagnosis
    assert set(V1_DRIVERS) <= set(diagnosis.columns)
    assert {"mean_column_distance", "real_vs_synthetic_auc"} <= set(diagnosis.failed_checks)
    assert "AUC 0.86" in diagnosis.likely_cause
    assert diagnosis.adjustments == ["more_epochs", "change_sampling"]
    assert read_decisions(run).query("rule == 'llm_step'").outcome.tolist() == ["LLM_FALLBACK"]


@pytest.mark.parametrize(
    "change",
    [
        {"columns": ["age_group"]},  # a column that is not in the results
        {"failed_checks": ["made_up_check"]},  # a check that did not fail
        {"adjustments": ["replace_threshold"]},  # not on the allowed list
        {"verdict": "PASS"},  # a field the diagnosis does not have
    ],
)
def test_a_diagnosis_that_does_not_match_the_results_is_replaced_and_logged(v1, change):
    request, run = v1
    answer = {**V1_ANSWER, **change}
    report = quality_module.run_quality(request, run, client=FakeLLM([answer, answer]))

    assert report.verdict is Verdict.FAIL
    assert set(V1_DRIVERS) <= set(report.diagnosis.columns)
    assert "age_group" not in report.diagnosis.columns
    assert "made_up_check" not in report.diagnosis.failed_checks
    assert report.diagnosis.adjustments == ["more_epochs", "change_sampling"]
    assert read_decisions(run).query("rule == 'llm_step'").outcome.iloc[-1] == "LLM_FALLBACK"


def test_changing_a_threshold_during_the_llm_call_is_rejected(v1):
    request, run = v1

    class ThresholdEditingLLM:
        def complete(self, system_prompt, context, output_model):
            text = run.config_path.read_text()
            run.config_path.write_text(
                text.replace("mean_column_distance_max: 0.1", "mean_column_distance_max: 0.9")
            )
            return V1_ANSWER

    with pytest.raises(RuleOverrideError):
        quality_module.run_quality(request, run, client=ThresholdEditingLLM())
    assert "RULE_OVERRIDE_REJECTED" in read_decisions(run).outcome.tolist()


@pytest.mark.parametrize(
    "change",
    [{"failed_checks": []}, {"likely_cause": "  "}, {"adjustments": []}],
)
def test_a_diagnosis_needs_failed_checks_a_cause_and_an_adjustment(change):
    with pytest.raises(ValidationError):
        QualityDiagnosis(**{**V1_ANSWER, **change})


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("RUN_LIVE_LLM"), reason="set RUN_LIVE_LLM=1 to call Vertex")
def test_live_gemini_diagnoses_the_v1_failure(v1):
    """One real Vertex call; run with RUN_LIVE_LLM=1 and -m live."""
    request, run = v1
    report = quality_module.run_quality(request, run)

    step = read_decisions(run).query("rule == 'llm_step'").outcome.iloc[0]
    assert step != "LLM_FALLBACK", "the Vertex call failed; see the rationale in decisions.jsonl"
    text = " ".join([*report.diagnosis.columns, report.diagnosis.likely_cause])
    assert all(column in text for column in V1_DRIVERS)
    assert "0.86" in report.diagnosis.likely_cause
