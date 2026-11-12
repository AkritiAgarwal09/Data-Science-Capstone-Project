"""Quality-agent reports, its pass, retry, and exhaustion routes, and the LLM diagnosis."""

import json
import logging
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
    QualityThresholds,
    RunConfig,
    Stage,
    Status,
    SyntheticSet,
    Verdict,
)
from agentic_gan.tools.quality_fidelity import RULES, schema_thresholds


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path.resolve()


class PreprocessFixture:
    def __init__(self, val=None):
        # The default validation split has 8 positive rows, too few to calibrate the limits on.
        self.val = val

    def __call__(self, request, run):
        data = run.subdir("data")
        train = pd.DataFrame({"x": range(40), "y": [0, 1] * 20})
        val = self.val if self.val is not None else pd.DataFrame({"x": range(16), "y": [0, 1] * 8})
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
    llm = FakeLLM([FIXTURE_DIAGNOSIS] * (attempts - int(available)))
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


# --- calibrated limits and soft utility gains (issue 101) -----------------------------------------

DETAILS = {
    "validation_median": 1.0,
    "synthetic_median": 1.1,
    "real_only_pr_auc": 0.5,
    "augmented_pr_auc": 0.48,
    "real_only_best_f1": 0.5,
    "augmented_best_f1": 0.47,
    "revealing_columns": ["x"],
    "real_vs_real_auc": 0.5,
}
OFFLINE = [RuntimeError("offline")] * 4  # the LLM is down, so a FAIL gets the facts-only diagnosis


def _gate_fidelity(passed):
    def fidelity(real, synthetic, **kwargs):
        value = 0.01 if passed else 0.2
        check = CheckResult(name="mean_column_distance", value=value, threshold=0.07, passed=passed)
        return SimpleNamespace(
            checks=[check], columns=pd.DataFrame({"distance": [value]}, index=["x"])
        )

    return fidelity


def _privacy_with_negative_gains(train, val, synthetic, target, positive, *, thresholds, **kwargs):
    """Healthy privacy, a classifier that tells the sets apart, and augmentation that hurts."""
    hard = thresholds.utility_gain_hard
    checks = [
        CheckResult(name="exact_train_copies", value=0, threshold=0, passed=True),
        CheckResult(
            name="real_vs_synthetic_auc", value=0.7, threshold=0.5, passed=False, hard=False
        ),
        CheckResult(
            name="augmentation_pr_auc_gain", value=-0.02, threshold=0, passed=False, hard=hard
        ),
        CheckResult(
            name="augmentation_best_f1_gain", value=-0.03, threshold=0, passed=False, hard=hard
        ),
    ]
    return checks, DETAILS


def _quality_request(tmp_path, config=None, val=None):
    config = config or _config(tmp_path)
    run = create_run(config, runs_dir=tmp_path / "runs")
    generator = GeneratorFixture()(SimpleNamespace(config=config, previous_report=None), run)
    return QualityInput(
        config=config, preprocess=PreprocessFixture(val)(None, run), generator=generator
    ), run


def _val_with_20_positives():
    """Real validation positives sit 4 to the right of the training ones: a KS distance of 0.1."""
    return pd.DataFrame({"x": [*range(5, 45, 2), *range(0, 40, 2)], "y": [1] * 20 + [0] * 20})


def _record_limits(monkeypatch):
    """Run the real fidelity tool, and return the list of limits it is called with."""
    seen = []
    real_fidelity = quality_module.fidelity_checks

    def fidelity(real, synthetic, **kwargs):
        seen.append(kwargs["thresholds"])
        return real_fidelity(real, synthetic, **kwargs)

    monkeypatch.setattr(quality_module, "fidelity_checks", fidelity)
    monkeypatch.setattr(quality_module, "quality_privacy_checks", _privacy_with_negative_gains)
    return seen


CALIBRATED = {  # 3 x the real-vs-real values 0.1, 0.1, 0, 0; the floors lift label_balance
    "mean_column_distance": 0.3,
    "worst_column_distance": 0.3,
    "label_balance": 0.01,
    "schema_match": 0.0,
}


def test_limits_are_calibrated_from_the_real_vs_real_baseline(tmp_path, monkeypatch):
    seen = _record_limits(monkeypatch)
    baselines = []
    real_baseline = quality_module.real_vs_real

    def baseline(*args, **kwargs):
        baselines.append(1)
        return real_baseline(*args, **kwargs)

    monkeypatch.setattr(quality_module, "real_vs_real", baseline)
    request, run = _quality_request(tmp_path, val=_val_with_20_positives())
    answer = {
        "failed_checks": ["mean_column_distance"],
        "columns": ["x"],
        "likely_cause": "x is off.",
        "adjustments": ["more_epochs"],
    }
    llm = FakeLLM([answer, answer])

    quality_module.run_quality(request, run, client=llm)

    assert seen == [CALIBRATED] * 3  # one call per fill level, all judged by the same limits
    saved = json.loads((run.subdir("quality") / "calibrated_thresholds.json").read_text())
    assert saved["source"] == "calibrated" and saved["margin"] == 3.0
    assert saved["baseline"]["mean_column_distance"] == pytest.approx(0.1)
    assert saved["limits"] == CALIBRATED
    (step,) = read_decisions(run).query("rule == 'thresholds_calibrated'").itertuples()
    assert step.outcome == "CALIBRATED" and step.inputs["limits"] == CALIBRATED
    ((_, context, _),) = llm.calls
    limits = {"source": "calibrated", "margin": 3.0, "values": CALIBRATED}
    assert json.loads(context)["limits"] == limits  # the diagnosis LLM sees the limits

    quality_module.run_quality(request, run, client=llm)  # a retry: same limits, no new baseline
    assert len(baselines) == 1
    assert seen[3:] == [CALIBRATED] * 3


def test_a_larger_margin_widens_every_calibrated_limit(tmp_path, monkeypatch):
    seen = _record_limits(monkeypatch)
    config = _config(tmp_path)
    config = config.updated(quality_gate=config.quality_gate.updated(margin=4.0))
    request, run = _quality_request(tmp_path, config, val=_val_with_20_positives())

    quality_module.run_quality(request, run, client=FakeLLM(OFFLINE))

    assert seen[0]["mean_column_distance"] == 0.4
    assert seen[0]["worst_column_distance"] == 0.4


def test_calibration_off_uses_the_fixed_limits(tmp_path, monkeypatch):
    seen = _record_limits(monkeypatch)

    def no_baseline(*args, **kwargs):
        raise AssertionError("no baseline is computed when calibrate is off")

    monkeypatch.setattr(quality_module, "real_vs_real", no_baseline)
    config = _config(tmp_path)
    gate = config.quality_gate.updated(calibrate=False)
    request, run = _quality_request(
        tmp_path, config.updated(quality_gate=gate), val=_val_with_20_positives()
    )

    quality_module.run_quality(request, run, client=FakeLLM(OFFLINE))

    assert seen == [schema_thresholds(gate, [])] * 3
    assert not (run.subdir("quality") / "calibrated_thresholds.json").exists()
    (step,) = read_decisions(run).query("rule == 'thresholds_calibrated'").itertuples()
    assert step.outcome == "FIXED"


def test_too_few_validation_positives_fall_back_to_the_fixed_limits_with_a_warning(
    tmp_path, monkeypatch, caplog
):
    seen = _record_limits(monkeypatch)
    request, run = _quality_request(tmp_path)  # 8 positive validation rows

    with caplog.at_level(logging.WARNING, logger=quality_module.__name__):
        quality_module.run_quality(request, run, client=FakeLLM(OFFLINE))

    assert seen == [schema_thresholds(QualityThresholds(), [])] * 3
    assert "Only 8 positive validation rows" in caplog.text
    (step,) = read_decisions(run).query("rule == 'thresholds_calibrated'").itertuples()
    assert step.outcome == "FIXED" and step.inputs["validation_positives"] == 8


@pytest.mark.parametrize("hard", [False, True])
def test_utility_gains_are_reported_and_decide_only_when_configured(tmp_path, monkeypatch, hard):
    monkeypatch.setattr(quality_module, "fidelity_checks", _gate_fidelity(passed=True))
    monkeypatch.setattr(quality_module, "quality_privacy_checks", _privacy_with_negative_gains)
    config = _config(tmp_path)
    config = config.updated(quality_gate=config.quality_gate.updated(utility_gain_hard=hard))
    request, run = _quality_request(tmp_path, config)

    report = quality_module.run_quality(request, run, client=FakeLLM(OFFLINE))

    gains = [
        check
        for level in report.levels
        for check in level.checks
        if check.name.startswith("augmentation_")
    ]
    assert len(gains) == 6  # two per fill level, all negative
    assert all(check.value < 0 and not check.passed for check in gains)
    assert all(check.hard is hard for check in gains)
    assert report.verdict is (Verdict.FAIL if hard else Verdict.PASS)


def test_a_set_that_fails_a_fidelity_check_fails_even_with_soft_utility_gains(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(quality_module, "fidelity_checks", _gate_fidelity(passed=False))
    monkeypatch.setattr(quality_module, "quality_privacy_checks", _privacy_with_negative_gains)
    request, run = _quality_request(tmp_path)

    report = quality_module.run_quality(request, run, client=FakeLLM(OFFLINE))

    assert report.verdict is Verdict.FAIL
    assert not any(level.passed for level in report.levels)
    hard_failures = {
        check.name
        for level in report.levels
        for check in level.checks
        if check.hard and not check.passed
    }
    assert hard_failures == {"mean_column_distance"}


@pytest.mark.parametrize(
    "change", [{"margin": 4.0}, {"calibrate": False}, {"utility_gain_hard": True}]
)
def test_the_calibration_settings_are_locked_by_both_rule_hashes(tmp_path, change):
    config = _config(tmp_path)
    changed = config.updated(quality_gate=config.quality_gate.updated(**change))

    assert quality_module._rule_hash(changed) != quality_module._rule_hash(config)
    assert graph_module.rule_hash(changed) != graph_module.rule_hash(config)


BANK = Path(__file__).resolve().parents[1] / "data" / "interim" / "bank_marketing"


@pytest.mark.parametrize(
    ("fill_file", "verdict"),
    [("bank_ctgan_fill25.csv", Verdict.FAIL), ("bank_ctgan_v2_fill25.csv", Verdict.PASS)],
)
def test_calibrated_gate_on_the_committed_bank_fills(tmp_path, fill_file, verdict):
    """The real agent and checks, no mocks: CTGAN v1 fails the gate, CTGAN v2 passes it."""
    config = RunConfig(
        dataset_name="bank_marketing",
        dataset_path=_write(tmp_path / "source.csv", "x,y\n0,0\n1,1\n"),
        target="y",
        positive_class="yes",
        fill_levels=[0.25],
        sanity_rules=[f"rule:{rule.name}" for rule in RULES["bank_marketing"]],
    )
    run = create_run(config, runs_dir=tmp_path / "runs")
    marker = _write(run.subdir("profile") / "marker.txt", "fixture\n")
    files = ArmFiles(
        train_path=BANK / "train_cleaned.csv",
        val_path=BANK / "validation_cleaned.csv",
        test_path=BANK / "test_cleaned.csv",
    )
    train = pd.read_csv(files.train_path)
    positives = int(train["y"].sum())
    preprocess = PreprocessOutput(
        status=Status.SUCCESS,
        raw=files,
        baseline=files,
        profile_path=marker,
        treatment_log_path=marker,
        transformer_path=marker,
        transformer_fit_rows=len(train),
        baseline_train_positives=positives,
        baseline_train_negatives=len(train) - positives,
    )
    fill = BANK / "synthetic" / fill_file
    generator = GeneratorOutput(
        status=Status.SUCCESS,
        generator="ctgan",
        rationale="committed fill",
        checkpoint_path=marker,
        loss_history_path=marker,
        synthetic_sets=[SyntheticSet(fill_level=0.25, rows=len(pd.read_csv(fill)), path=fill)],
    )

    report = quality_module.run_quality(
        QualityInput(config=config, preprocess=preprocess, generator=generator),
        run,
        client=FakeLLM(OFFLINE),
    )

    assert report.verdict is verdict
    (level,) = report.levels
    by_name = {check.name: check for check in level.checks}
    if verdict is Verdict.FAIL:
        job = by_name["column_distance:job"]  # v1 leaves job far from the real distribution
        assert job.value > job.threshold
        assert not by_name["worst_column_distance"].passed
    else:
        gains = [by_name["augmentation_pr_auc_gain"], by_name["augmentation_best_f1_gain"]]
        assert all(gain.value < 0 and not gain.hard for gain in gains)  # reported, never decisive
        assert all(check.passed for check in level.checks if check.hard)


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
