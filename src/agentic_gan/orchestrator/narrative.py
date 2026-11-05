"""Explain a completed or failed run from its append-only decision log."""

import json

from pydantic import Field

from agentic_gan.llm import LLMStepOptions, RuleOverrideError, run_llm_step
from agentic_gan.schemas import Schema


class Narrative(Schema):
    what_happened: str = Field(min_length=1)
    why: str = Field(min_length=1)
    next_step: str = Field(min_length=1)


def write_narrative(run, state, *, expected_rule_hash, current_rule_hash, llm=None):
    decisions = [json.loads(line) for line in run.decisions_path.read_text().splitlines()]
    failed = state.failed_stage
    unavailable = state.gan_available is False
    fallback = Narrative(
        what_happened=(
            f"The run failed at {failed.value}."
            if failed
            else "The run completed without accepted synthetic data."
            if unavailable
            else "The run completed."
        ),
        why=(
            state.stages[failed].reason
            if failed
            else state.gan_unavailable_reason
            if unavailable
            else "All stages completed successfully."
        ),
        next_step="Fix the failed stage and rerun."
        if failed
        else "Review the saved report and artifacts.",
    )
    try:
        result = run_llm_step(
            "narrative",
            "Explain what happened, why, and what to do next in plain language. "
            "Use only the decision log; do not invent facts or change decisions.",
            json.dumps(decisions),
            Narrative,
            LLMStepOptions(
                fallback=fallback,
                expected_rule_hash=expected_rule_hash,
                current_rule_hash=current_rule_hash,
            ),
            run,
            client=llm,
        )
    except RuleOverrideError:
        result = fallback
    path = run.subdir("report") / "narrative.md"
    path.write_text(
        f"# Run summary\n\n## What happened\n\n{result.what_happened}\n\n"
        f"## Why\n\n{result.why}\n\n## What to do next\n\n{result.next_step}\n"
    )
    return path
