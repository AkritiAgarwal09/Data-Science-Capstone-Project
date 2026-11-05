"""The per-agent summary of answered and fallback LLM steps."""

import pytest

from agentic_gan.orchestrator.decisions import log_decision
from agentic_gan.orchestrator.manifest import create_run
from agentic_gan.schemas import RunConfig
from agentic_gan.tools.llm_health import first_fallback_reasons, llm_health


@pytest.fixture
def run(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("x,y\n0,0\n1,1\n")
    config = RunConfig(dataset_name="health", dataset_path=source, target="y", positive_class=1)
    return create_run(config, runs_dir=tmp_path / "runs")


def step(run, agent, outcome, why="Validated LLM answer within allowed options and bounds"):
    log_decision(agent, "llm_step", {"context": "..."}, outcome, why, run=run)


def test_each_agent_gets_answered_fallback_and_rejected_counts(run):
    step(run, "preprocessing", {"treatment": "median", "reason": "r"})
    step(run, "preprocessing", {"treatment": "mode", "reason": "r"})
    step(run, "training", "LLM_FALLBACK", "Timed out")
    step(run, "training", "LLM_FALLBACK", "Not JSON")
    step(run, "evaluation", "RULE_OVERRIDE_REJECTED", "Rules or thresholds changed")
    log_decision("training", "run_plan", {}, "x", "not an llm step", run=run)  # ignored

    table = llm_health(run)

    assert table.loc["preprocessing"].to_dict() == {
        "answered": 2,
        "fallback": 0,
        "rule_override_rejected": 0,
    }
    assert table.loc["training"].to_dict() == {
        "answered": 0,
        "fallback": 2,
        "rule_override_rejected": 0,
    }
    assert table.loc["evaluation"].rule_override_rejected == 1
    assert first_fallback_reasons(run) == {"training": "Timed out"}


def test_a_run_without_llm_steps_gives_an_empty_table(run):
    assert llm_health(run).empty
