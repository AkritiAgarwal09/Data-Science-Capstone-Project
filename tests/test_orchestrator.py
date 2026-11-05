"""End-to-end control flow with typed placeholder agents."""

from dataclasses import replace
from pathlib import Path

from agentic_gan.llm import FakeLLM
from agentic_gan.orchestrator import graph as graph_module
from agentic_gan.orchestrator.decisions import read_decisions
from agentic_gan.orchestrator.graph import build_graph, config_for_dataset, run_pipeline
from agentic_gan.orchestrator.stubs import StubAgent, default_agents
from agentic_gan.schemas import Arm, GeneratorOutput, RunConfig, Stage, Status
from agentic_gan.utils.datasets import DATASETS, load_configs


def config(tmp_path):
    dataset = tmp_path / "bank.csv"
    dataset.write_text("feature,y\n1,yes\n0,no\n")
    return RunConfig(
        dataset_name="bank_marketing", dataset_path=dataset, target="y", positive_class="yes"
    )


def test_graph_is_visualizable():
    diagram = build_graph().get_graph().draw_mermaid()
    assert "model_selection" in diagram
    assert "quality_check" in diagram
    assert "partial_report" in diagram


def test_dataset_configs_cover_all_four_sources():
    assert set(DATASETS) == {"bank_marketing", "default", "polish", "ieee_cis"}
    assert {config_for_dataset(name).target for name in DATASETS} == {"y", "bankrupt", "isFraud"}


def test_default_graph_uses_real_split_and_writes_failure_summary(tmp_path):
    state, run = run_pipeline(config(tmp_path), runs_dir=tmp_path / "runs")
    assert state.failed_stage is Stage.SPLIT
    assert state.stages[Stage.SPLIT].reason
    assert "split" in (run.subdir("report") / "narrative.md").read_text()


def test_pass_first_time(tmp_path, monkeypatch):
    run_config = config(tmp_path)
    monkeypatch.chdir(tmp_path)
    state, run = run_pipeline(run_config, runs_dir=Path("runs"), agents=default_agents())
    assert state.is_complete
    assert state.gan_available is True
    assert [state.attempts(stage) for stage in Stage] == [1] * 6
    assert run.load_manifest().verify(run) == []
    assert run.decisions_path.exists()
    assert (run.subdir("report") / "final_report.md").exists()


def test_quality_fail_then_pass(tmp_path):
    agents = default_agents()
    agents[Stage.QUALITY_CHECK] = StubAgent(Stage.QUALITY_CHECK, failures=1, failure_mode="quality")
    state, run = run_pipeline(config(tmp_path), runs_dir=tmp_path / "runs", agents=agents)
    assert state.is_complete
    assert state.gan_available is True
    assert state.attempts(Stage.MODEL_SELECTION) == 2
    assert state.attempts(Stage.QUALITY_CHECK) == 2
    assert read_decisions(run).query("rule == 'quality_gate'").outcome.tolist() == [
        "RETRY_MODEL_SELECTION",
        "ACCEPT",
    ]
    assert run.load_manifest().verify(run) == []


def test_quality_exhaustion_continues_without_gan(tmp_path):
    agents = default_agents()
    agents[Stage.QUALITY_CHECK] = StubAgent(
        Stage.QUALITY_CHECK, failures=-1, failure_mode="quality"
    )
    state, run = run_pipeline(
        config(tmp_path), runs_dir=tmp_path / "runs", agents=agents,
        llm=FakeLLM([RuntimeError("offline")] * 2),
    )
    assert state.is_complete
    assert state.gan_available is False
    assert state.gan_unavailable_reason
    assert state.attempts(Stage.MODEL_SELECTION) == 3
    assert state.attempts(Stage.QUALITY_CHECK) == 3
    assert state.status(Stage.TRAINING) is Status.SUCCESS
    assert read_decisions(run).query("rule == 'quality_gate'").outcome.iloc[-1] == "NOT_AVAILABLE"
    assert "NOT_AVAILABLE" in (run.subdir("report") / "final_report.md").read_text()
    assert "without accepted synthetic data" in (run.subdir("report") / "narrative.md").read_text()
    assert all(
        entry.agent is not Stage.TRAINING or entry.path.name != "gan.txt"
        for entry in run.load_manifest().entries
    )
    assert run.load_manifest().verify(run) == []


def test_stage_error_retries_once_then_stops(tmp_path):
    agents = default_agents()
    agents[Stage.PREPROCESSING] = StubAgent(Stage.PREPROCESSING, failures=-1)
    state, run = run_pipeline(config(tmp_path), runs_dir=tmp_path / "runs", agents=agents)
    assert state.failed_stage is Stage.PREPROCESSING
    assert state.attempts(Stage.PREPROCESSING) == 2
    assert state.attempts(Stage.MODEL_SELECTION) == 0
    assert (run.subdir("report") / "partial_report.md").exists()
    assert read_decisions(run).query("rule == 'stage_error'").outcome.tolist() == [
        "RETRY",
        "FAILED",
    ]
    assert run.load_manifest().verify(run) == []


def test_skipped_selection_skips_quality_and_continues(tmp_path):
    agents = default_agents()

    def skipped(request, run):
        return GeneratorOutput(
            status=Status.SKIPPED,
            reason="wgan_gp is not implemented",
            generator="wgan_gp",
            rationale="Documented fallback: ctgan.",
        )

    agents[Stage.MODEL_SELECTION] = skipped
    state, run = run_pipeline(config(tmp_path), runs_dir=tmp_path / "runs", agents=agents)
    assert state.is_complete
    assert state.gan_available is False
    assert state.gan_unavailable_reason
    assert state.status(Stage.MODEL_SELECTION) is Status.SKIPPED
    assert state.status(Stage.QUALITY_CHECK) is Status.SKIPPED
    assert state.attempts(Stage.QUALITY_CHECK) == 0
    assert state.attempts(Stage.MODEL_SELECTION) == 1
    assert state.status(Stage.TRAINING) is Status.SUCCESS
    assert read_decisions(run).query("rule == 'selection_skipped'").outcome.tolist() == ["SKIPPED"]
    assert run.load_manifest().verify(run) == []


def test_stage_error_then_pass(tmp_path):
    agents = default_agents()
    agents[Stage.TRAINING] = StubAgent(Stage.TRAINING, failures=1)
    state, run = run_pipeline(config(tmp_path), runs_dir=tmp_path / "runs", agents=agents)
    assert state.is_complete
    assert state.attempts(Stage.TRAINING) == 2
    assert any(
        entry.path.name.startswith(Arm.GAN.value)
        for entry in run.load_manifest().produced_by(Stage.TRAINING)
    )


def test_rule_override_is_rejected_and_logged(tmp_path):
    agents = default_agents()

    def override(request, run):
        run.config_path.write_text(
            run.config_path.read_text().replace(
                "mean_column_distance_max: 0.1", "mean_column_distance_max: 1.0"
            )
        )
        return StubAgent(Stage.PREPROCESSING)(request, run)

    agents[Stage.PREPROCESSING] = override
    state, run = run_pipeline(config(tmp_path), runs_dir=tmp_path / "runs", agents=agents)
    assert state.failed_stage is Stage.PREPROCESSING
    assert state.attempts(Stage.PREPROCESSING) == 1
    decisions = read_decisions(run)
    assert decisions.query("outcome == 'RULE_OVERRIDE_REJECTED'").shape[0] == 1
    assert decisions.query("rule == 'rule_lock'").outcome.tolist()[0] == "LOCKED"
    assert run.load_config().quality_gate.mean_column_distance_max == 0.1
    assert run.load_manifest().verify(run) == []
    assert (run.subdir("report") / "narrative.md").exists()


def test_dataset_rule_override_is_rejected_even_when_agent_raises(tmp_path, monkeypatch):
    datasets = load_configs()
    monkeypatch.setattr(graph_module, "load_configs", lambda: datasets)
    agents = default_agents()

    def override(request, run):
        current = datasets["bank_marketing"]
        datasets["bank_marketing"] = replace(current, include_duration=not current.include_duration)
        raise RuntimeError("agent failed after changing a dataset rule")

    agents[Stage.PREPROCESSING] = override
    state, run = run_pipeline(config(tmp_path), runs_dir=tmp_path / "runs", agents=agents)
    assert state.failed_stage is Stage.PREPROCESSING
    assert state.attempts(Stage.PREPROCESSING) == 1
    assert "RULE_OVERRIDE_REJECTED" in read_decisions(run).outcome.tolist()
    assert "preprocessing" in (run.subdir("report") / "narrative.md").read_text()


def test_narrative_uses_typed_fake_llm_and_retries_invalid_response(tmp_path):
    fake_llm = FakeLLM(
        [
            {"what_happened": "Incomplete"},
            {
                "what_happened": "The run completed.",
                "why": "Every stage succeeded.",
                "next_step": "Review the report.",
            },
        ]
    )

    state, run = run_pipeline(
        config(tmp_path), runs_dir=tmp_path / "runs", agents=default_agents(), llm=fake_llm
    )
    assert state.is_complete
    assert len(fake_llm.calls) == 2
    assert "The run completed." in (run.subdir("report") / "narrative.md").read_text()
    answer = read_decisions(run).query("rule == 'llm_step'").outcome.iloc[0]
    assert answer["what_happened"] == "The run completed."


def test_narrative_falls_back_after_llm_failure(tmp_path):
    failing_llm = FakeLLM([RuntimeError("provider unavailable")] * 2)

    state, run = run_pipeline(
        config(tmp_path), runs_dir=tmp_path / "runs", agents=default_agents(), llm=failing_llm
    )
    assert state.is_complete
    assert read_decisions(run).query("rule == 'llm_step'").outcome.tolist() == ["LLM_FALLBACK"]
