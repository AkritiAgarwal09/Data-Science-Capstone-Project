"""Quality-agent reports and its pass, retry, and exhaustion routes."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import agentic_gan.agents.quality as quality_module
import agentic_gan.orchestrator.graph as graph_module
from agentic_gan.orchestrator.decisions import read_decisions
from agentic_gan.orchestrator.stubs import default_agents
from agentic_gan.schemas import (
    ArmFiles,
    CheckResult,
    GeneratorOutput,
    PreprocessOutput,
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

    def __call__(self, request, run):
        self.calls += 1
        self.previous_verdicts.append(
            None if request.previous_report is None else request.previous_report.verdict
        )
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
    # Explicit fixtures keep this routing test independent of the real split/repair stages.
    agents[Stage.QUALITY_CHECK] = quality_module.run_quality
    state, run = graph_module.run_pipeline(
        _config(tmp_path), runs_dir=tmp_path / "runs", agents=agents
    )
    assert state.is_complete
    assert state.gan_available is available
    assert state.attempts(Stage.QUALITY_CHECK) == attempts
    assert state.attempts(Stage.MODEL_SELECTION) == attempts
    assert generator.previous_verdicts == [None] + [Verdict.FAIL] * (attempts - 1)
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
