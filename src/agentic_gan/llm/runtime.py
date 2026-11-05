"""Validate bounded LLM answers and log every decision."""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from agentic_gan.orchestrator.decisions import log_decision
from agentic_gan.orchestrator.manifest import RunDir

from .vertex import VertexClient


class RuleOverrideError(ValueError):
    """The run's locked rules or thresholds changed."""


@dataclass(frozen=True)
class LLMStepOptions:
    fallback: BaseModel | Mapping[str, Any]
    expected_rule_hash: str
    current_rule_hash: Callable[[], str]
    allowed_options: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    numeric_bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)


class FakeLLM:
    """Scripted replies or exceptions; no credentials or network."""

    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []

    def complete(self, system_prompt, context, output_model):
        self.calls.append((system_prompt, context, output_model))
        answer = next(self.answers)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _validate(answer, output_model, options):
    raw = answer.model_dump() if isinstance(answer, BaseModel) else answer
    if not isinstance(raw, Mapping) or set(raw) - set(output_model.model_fields):
        raise ValueError("LLM returned unknown fields")
    result = output_model.model_validate(raw)
    for name, allowed in options.allowed_options.items():
        if name not in output_model.model_fields or getattr(result, name) not in allowed:
            raise ValueError(f"{name} is outside allowed options")
    for name, (low, high) in options.numeric_bounds.items():
        if name not in output_model.model_fields:
            raise ValueError(f"{name} is not an output field")
        value = getattr(result, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or not low <= value <= high
        ):
            raise ValueError(f"{name} is outside bounds [{low}, {high}]")
    return result


def run_llm_step(
    role: str,
    system_prompt: str,
    context: str,
    output_model: type[BaseModel],
    options: LLMStepOptions,
    run: RunDir,
    *,
    client=None,
) -> BaseModel:
    """Return a bounded Pydantic answer, or the logged deterministic fallback."""
    fallback = _validate(options.fallback, output_model, options)
    inputs = {
        "system_prompt": system_prompt,
        "context": context,
        "allowed_options": options.allowed_options,
        "numeric_bounds": options.numeric_bounds,
        "rule_hash": options.expected_rule_hash,
    }

    def check_rules():
        actual = options.current_rule_hash()
        if actual != options.expected_rule_hash:
            log_decision(
                role,
                "llm_step",
                {**inputs, "actual_rule_hash": actual},
                "RULE_OVERRIDE_REJECTED",
                "Rules or thresholds changed",
                run=run,
            )
            raise RuleOverrideError("rules or thresholds changed after lock")

    check_rules()
    last_error = None
    for _ in range(2):
        try:
            answer = (client or VertexClient()).complete(system_prompt, context, output_model)
            check_rules()
            result = _validate(answer, output_model, options)
            log_decision(
                role,
                "llm_step",
                inputs,
                result.model_dump(mode="json"),
                "Validated LLM answer within allowed options and bounds",
                run=run,
            )
            return result
        except RuleOverrideError:
            raise
        except Exception as exc:
            check_rules()
            last_error = exc
    log_decision(role, "llm_step", inputs, "LLM_FALLBACK", str(last_error), run=run)
    return fallback
