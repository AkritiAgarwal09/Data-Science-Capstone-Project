"""Which LLM steps of a run the model answered, and which fell back to their defaults.

A healthy live run has no fallbacks. A run with unset or broken Vertex settings falls back on every
step and still completes, so this is the quick check that the LLM really took part.

    uv run python -m agentic_gan.tools.llm_health runs/<run_id>
"""

import argparse
from pathlib import Path

import pandas as pd

from agentic_gan.orchestrator.decisions import read_decisions
from agentic_gan.orchestrator.manifest import RunDir

RESULTS = ("answered", "fallback", "rule_override_rejected")


def _result(outcome) -> str:
    if outcome == "LLM_FALLBACK":
        return "fallback"
    if outcome == "RULE_OVERRIDE_REJECTED":
        return "rule_override_rejected"
    return "answered"


def llm_health(run: RunDir) -> pd.DataFrame:
    """One row per agent: how many of its LLM steps were answered, fell back or were rejected."""
    decisions = read_decisions(run)
    steps = decisions[decisions.rule == "llm_step"]
    counts = pd.crosstab(steps.agent, steps.outcome.map(_result))
    return counts.reindex(columns=list(RESULTS), fill_value=0).rename_axis("agent")


def first_fallback_reasons(run: RunDir) -> dict[str, str]:
    """The reason the first fallback of each agent gave, to see why the model did not answer."""
    decisions = read_decisions(run)
    fallbacks = decisions[(decisions.rule == "llm_step") & (decisions.outcome == "LLM_FALLBACK")]
    return fallbacks.groupby("agent").rationale.first().to_dict()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "run_dir", type=Path, help="a run folder, for example runs/bank_marketing_..."
    )
    run = RunDir(parser.parse_args().run_dir.resolve())
    table = llm_health(run)
    if table.empty:
        print("No LLM steps were logged in this run.")
        return 0
    print(table.to_string())
    for agent, reason in first_fallback_reasons(run).items():
        print(f"{agent}: first fallback because {reason[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
