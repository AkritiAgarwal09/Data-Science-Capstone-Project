"""Choose a generator by rule, train CTGAN when that rule says so, and log every step."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import pandas as pd
from pydantic import Field, StrictBool

from agentic_gan.llm import LLMStepOptions, run_llm_step
from agentic_gan.orchestrator.decisions import log_decision, read_decisions
from agentic_gan.orchestrator.manifest import RunDir
from agentic_gan.schemas import (
    GeneratorOutput,
    ModelSelectionInput,
    QualityReport,
    RunConfig,
    Schema,
    Stage,
    Status,
    SyntheticSet,
    fill_rows,
)
from agentic_gan.tools.ctgan_sampling import recipe_for, sample_for_fill
from agentic_gan.tools.quality_fidelity import RULES
from agentic_gan.tools.quality_privacy import PRIVACY_CHECKS
from agentic_gan.tools.selection import (
    STEP_ORDER,
    ColumnSpec,
    DataCharacteristics,
    SelectionResult,
    select_generator,
    temporal_facts,
)
from agentic_gan.utils.datasets import DATASETS
from agentic_gan.utils.synthetic import IS_SYNTHETIC

# One retry step. CTGAN's loss moves slowly, and a smaller bump is not a new attempt.
EPOCH_STEP = 100
CTGAN_DEFAULTS = {
    "epochs": 300,
    "batch_size": 500,
    "pac": 10,
    "discriminator_steps": 1,
}
# The CTGAN settings a retry may move, each inside its gan_bounds range.
SETTING_NAMES = ("epochs", "pac", "discriminator_steps", "batch_size")
KNOB_NAMES = ("sampling_mode", "rule_filter", "rake_columns")
# Order used in AttemptRecord.settings and settings_key.
SEVEN_KNOBS = (
    "epochs",
    "batch_size",
    "pac",
    "discriminator_steps",
    "sampling_mode",
    "rule_filter",
    "rake_columns",
)
RETRY_OPTIONS = {
    "sampling_mode": ("conditional", "unconditional"),
    "rule_filter": (False, True),
}
FIDELITY_TOKENS = ("column_distance", "correlation", "ks", "js", "fidelity")
# A constant cohort is the horizon id, not a ratio the generator should learn.
HORIZON_LABEL = "cohort"

EXPLAIN_PROMPT = (
    "You explain, in plain language for a data scientist, why a fixed rule chose, or declined to "
    "choose, a synthetic-data generator for a tabular dataset. Use only the decision steps and "
    "routes_not_chosen in the context; each step lists its rule, inputs, outcome and rationale. "
    "Quote the specific numbers from those steps (row counts, limits, column counts). Set "
    "`generator` to exactly the value of `generator` in the context ('none' when no generator was "
    "chosen). You never choose, change or recommend a different generator, and you never change a "
    "rule, threshold or gate. Keep the explanation under 150 words."
)

RETRY_PROMPT = (
    "A CTGAN generator failed a synthetic-data quality gate. Propose the settings for the next "
    "attempt as the RetrySettings fields only. "
    "Knob guide: sampling_mode=\"unconditional\" fixed the technician skew in V2. rake_columns "
    "reweights the output to the real shares of the named columns; use it for the worst "
    "categorical columns. rule_filter drops rows that break dataset rules; use it when a rule:* "
    "check fails. discriminator_steps and pac strengthen the critic; use them for detectability "
    "and numeric drift (V2 used pac 5 and 3 steps). epochs gave marginal gains in our test, so "
    "try other knobs first and never raise it after a privacy failure. batch_size must be "
    "divisible by pac. "
    "Hard constraints: every numeric value must lie inside its inclusive range in limits; "
    "sampling_mode and rule_filter only from options; rake_columns only from "
    "allowed_rake_columns; you never change a rule, threshold, range, option list or the "
    "generator. "
    "Instructions: change at least one knob; never return a combination listed in history; "
    "answer only with the RetrySettings fields; reason must name the failed check or column "
    "that the change targets. "
    "Example (not data): job distance 0.335, all rule checks pass, so sampling_mode "
    "\"unconditional\" and rake_columns [\"job\"], other knobs unchanged, reason "
    "\"worst_column_distance on job is 0.335, a categorical skew, so unconditional sampling "
    "and raking on job\"."
)

# Explicit marker on the ladder fallback reason (finding F4). Prefer startswith(LADDER_PREFIX)
# over comparing free-text reasons: a real LLM answer could repeat the fallback string.
LADDER_PREFIX = "Ladder default: "
# Generators this agent can actually train. Until issue #124 lands, only CTGAN is here.
IMPLEMENTED_GENERATORS = ("ctgan",)


class SelectionExplanation(Schema):
    """The LLM's account of the rule's outcome. `generator` must echo the rule, never pick."""

    generator: str  # exactly the rule's generator, or "none"
    explanation: str = Field(min_length=1)


class RetrySettings(Schema):
    """CTGAN settings the LLM proposes for a retry; each is checked against gan_bounds."""

    epochs: int
    pac: int
    discriminator_steps: int
    batch_size: int
    sampling_mode: Literal["conditional", "unconditional"]
    rule_filter: StrictBool
    rake_columns: list[str]
    reason: str = Field(min_length=1)


@dataclass(frozen=True)
class RetryLimits:
    """Numeric ranges, categorical options, and the allowed rake set for one retry."""

    numeric: dict[str, tuple[int, int]]
    options: dict[str, tuple]
    rake_allowed: tuple[str, ...]


@dataclass(frozen=True)
class AttemptRecord:
    """Settings and failed hard checks from one earlier model-selection attempt."""

    attempt: int
    settings: dict
    failed_checks: list[str]


@dataclass(frozen=True)
class LadderResult:
    """One deterministic retry proposal from the quality-gate ladder."""

    settings: dict
    trigger: str
    rung: int
    exhausted: bool
    rationale: str


ROW3_CHECKS = frozenset(
    {
        "mean_column_distance",
        "spearman_diff",
        "cramers_v_diff",
        "real_vs_synthetic_auc",
    }
)
LADDER_EPOCHS_CAP = 600
LADDER_EPOCHS_FLOOR = 300
LADDER_STEPS_CAP = 5
LADDER_BATCH_CAP = 1000


def settings_key(settings: dict) -> tuple:
    """Comparable key: ints as ints, rake_columns as a sorted tuple (order-insensitive)."""
    return (
        int(settings["epochs"]),
        int(settings["batch_size"]),
        int(settings["pac"]),
        int(settings["discriminator_steps"]),
        settings["sampling_mode"],
        bool(settings["rule_filter"]),
        tuple(sorted(settings["rake_columns"])),
    )


def _seven_knobs_from_inputs(inputs: dict) -> dict:
    return {
        "epochs": int(inputs["epochs"]),
        "batch_size": int(inputs["batch_size"]),
        "pac": int(inputs["pac"]),
        "discriminator_steps": int(inputs["discriminator_steps"]),
        "sampling_mode": inputs["sampling_mode"],
        "rule_filter": bool(inputs["rule_filter"]),
        "rake_columns": list(inputs["rake_columns"]),
    }


def _recipe_attempt_settings(config: RunConfig) -> dict:
    """Attempt-1 settings when the hyperparameters log row is missing: recipe, clamped."""
    recipe = asdict(recipe_for(config.dataset_name))
    numeric = {name: int(recipe[name]) for name in SETTING_NAMES}
    clamped = _clamp(numeric, config.gan_bounds)
    if clamped is not None:
        numeric = clamped
    return {
        "epochs": numeric["epochs"],
        "batch_size": numeric["batch_size"],
        "pac": numeric["pac"],
        "discriminator_steps": numeric["discriminator_steps"],
        "sampling_mode": recipe["sampling_mode"],
        "rule_filter": bool(recipe["rule_filter"]),
        "rake_columns": list(recipe["rake_columns"]),
    }


def _failed_checks_from_report(report: QualityReport | None) -> list[str]:
    if report is None:
        return []
    return [
        check.name
        for level in report.levels
        for check in level.checks
        if check.hard and not check.passed
    ]


def _load_attempt_failed_checks(run: RunDir, attempt: int) -> list[str]:
    path = run.subdir("quality") / f"quality_report_attempt_{attempt}.json"
    if not path.exists():
        return []
    report = QualityReport.model_validate_json(path.read_text(encoding="utf-8"))
    return _failed_checks_from_report(report)


def attempt_history(request: ModelSelectionInput, run: RunDir) -> list[AttemptRecord]:
    """Earlier attempts' seven knobs and failed hard checks. No writes, no logging, no mutation."""
    decisions = read_decisions(run)
    by_attempt: dict[int, dict] = {}
    if not decisions.empty:
        rows = decisions[
            (decisions["agent"] == "model_selection") & (decisions["rule"] == "hyperparameters")
        ]
        for inputs in rows["inputs"]:
            attempt = int(inputs["attempt"])
            if attempt >= request.attempt:
                continue
            by_attempt[attempt] = _seven_knobs_from_inputs(inputs)

    if request.attempt > 1 and 1 not in by_attempt:
        by_attempt[1] = _recipe_attempt_settings(request.config)

    records: list[AttemptRecord] = []
    for attempt in sorted(by_attempt):
        failed = _load_attempt_failed_checks(run, attempt)
        if attempt == request.attempt - 1 and request.previous_report is not None:
            from_request = _failed_checks_from_report(request.previous_report)
            if from_request != failed:
                failed = from_request
        records.append(
            AttemptRecord(
                attempt=attempt,
                settings=by_attempt[attempt],
                failed_checks=failed,
            )
        )
    return records


def _rule_hash(config: RunConfig) -> str:
    """Fingerprint of what the LLM steps must not change."""
    params = inspect.signature(select_generator).parameters
    locked = {
        "min_rows": config.min_rows,
        "gan_bounds": config.gan_bounds,
        "step_order": STEP_ORDER,
        "ctgan_defaults": CTGAN_DEFAULTS,
        "epoch_step": EPOCH_STEP,
        "privacy_checks": sorted(PRIVACY_CHECKS),
        "minority_min": params["minority_min"].default,
        "min_steps": params["min_steps"].default,
        "retry_options": {
            "sampling_mode": list(RETRY_OPTIONS["sampling_mode"]),
            "rule_filter": list(RETRY_OPTIONS["rule_filter"]),
        },
    }
    return hashlib.sha256(json.dumps(locked, sort_keys=True).encode()).hexdigest()


def allowed_rake_columns(
    discrete_feature_columns: Sequence[str],
    recipe_rake_columns: Sequence[str],
    table_columns: Sequence[str],
) -> list[str]:
    """Discrete features (no target) ∪ recipe rake columns present in the table, in table order."""
    table = list(table_columns)
    present = set(table)
    wanted = {name for name in discrete_feature_columns if name in present} | {
        name for name in recipe_rake_columns if name in present
    }
    return [name for name in table if name in wanted]


def measure_training_frame(
    train: pd.DataFrame,
    dataset: str,
    target: str,
    positive: object,
    excluded_features: Sequence[str] = (),
) -> DataCharacteristics:
    """Measure a cleaned training table the way the agent does, with no run config.

    The EDA runners call this too, so a standalone CTGAN run records the same selection the
    agent would have made. Do not infer a sequence from a column name.
    """
    facts = temporal_facts(dataset, set(train.columns))
    pooled = _horizons_pooled(dataset, train)
    hidden = {"source_row"}
    if dataset == "polish" and HORIZON_LABEL in train.columns and not pooled:
        hidden.add(HORIZON_LABEL)
    discrete = set(DATASETS[dataset].discrete) if dataset in DATASETS else set()
    leakage_names = tuple(dict.fromkeys((*facts.leakage, *excluded_features)))
    columns = tuple(
        ColumnSpec(name, _kind(name, train[name], discrete), int(train[name].isna().sum()))
        for name in train.columns
        if name not in hidden
    )
    positives = int(train[target].eq(positive).sum())
    rows = len(train)
    return DataCharacteristics(
        dataset=dataset,
        rows=rows,
        target=target,
        minority_count=positives,
        minority_rate=(positives / rows) if rows else 0.0,
        columns=columns,
        tabular=True,
        entity_in_table=facts.entity_in_table,
        wide_steps=facts.wide_steps,
        repeated_observation=facts.repeated_observation,
        time_index_ready=facts.time_index_ready,
        leaky_columns=tuple(name for name in leakage_names if name in train.columns),
        horizons_pooled=pooled,
    )


def describe_characteristics(
    train: pd.DataFrame, request: ModelSelectionInput
) -> DataCharacteristics:
    """Measure the training table and check it against the baseline class counts."""
    target = request.config.target
    positive = _positive_class(request.config.positive_class, train[target])
    described = measure_training_frame(
        train,
        request.config.dataset_name,
        target,
        positive,
        request.config.excluded_features,
    )
    _check_class_counts(train, target, positive, described.minority_count, request)
    return described


def run_model_selection(
    request: ModelSelectionInput, run: RunDir, *, client=None
) -> GeneratorOutput:
    """Select from the baseline training split only. Validation and test are not read.

    `client` serves both LLM steps: the explanation on attempt 1 and the settings proposal on a
    retry. Tests pass a FakeLLM. With none, each step logs LLM_FALLBACK and the rule stands.
    """
    train = pd.read_csv(request.preprocess.baseline.train_path)
    if IS_SYNTHETIC in train.columns:
        raise ValueError(f"training data must not already contain {IS_SYNTHETIC!r}")
    described = describe_characteristics(train, request)
    model_frame = train.drop(columns=["source_row"], errors="ignore")
    if described.dataset == "polish" and not described.horizons_pooled:
        model_frame = model_frame.drop(columns=[HORIZON_LABEL], errors="ignore")
    result = select_generator(described, min_rows=request.config.min_rows)
    _log_steps(result, request.attempt, run)
    if request.attempt == 1:
        # The rule is the same on every attempt; explain it once. The answer lives in the
        # decision log only and never reaches GeneratorOutput.
        _log_skipped_routes(result, run)
        _explain(result, request, run, client)

    if result.status == "skipped" or not result.trainable:
        return _skipped(result)

    positives = request.preprocess.baseline_train_positives
    negatives = request.preprocess.baseline_train_negatives
    counts = [fill_rows(level, positives, negatives) for level in request.config.fill_levels]
    if any(count < 1 for count in counts):
        reason = "A fill level rounds to no rows, so nothing is generated."
        log_decision(
            "model_selection", "fill_rows",
            {"attempt": request.attempt, "counts": counts},
            "SKIPPED", reason, run=run,
        )
        return GeneratorOutput(status=Status.SKIPPED, reason=reason, rationale=reason)

    dataset = request.config.dataset_name
    target = request.config.target
    discrete = [name for name in _discrete_columns(model_frame, target, dataset) if name != target]
    recipe = recipe_for(dataset)
    allowed_rake = allowed_rake_columns(
        discrete, recipe.rake_columns, list(model_frame.columns)
    )

    settings = _clamp(CTGAN_DEFAULTS, request.config.gan_bounds)
    if settings is None:
        reason = "gan_bounds make batch_size not divisible by pac, so CTGAN is not trained."
        log_decision(
            "model_selection", "hyperparameter_bounds",
            {"attempt": request.attempt, "gan_bounds": _bounds_json(request.config.gan_bounds)},
            "SKIPPED", reason, run=run,
        )
        return GeneratorOutput(status=Status.SKIPPED, reason=reason, rationale=reason)

    if request.attempt >= 2:
        # Until issue #124 (WGAN-GP backup) lands, attempt 3 trains CTGAN with the next
        # ladder rung (same path as attempt 2), not a SKIPPED fallback generator.
        settings, exhausted = _propose_retry(
            request,
            run,
            client,
            discrete_columns=discrete,
            allowed_rake=allowed_rake,
        )
        if request.attempt >= 3 and exhausted:
            return _fallback_attempt(result, request.attempt, run, exhausted=True)
    return _train_ctgan(request, run, model_frame, settings, result)


def _horizons_pooled(dataset: str, train: pd.DataFrame) -> bool:
    if dataset != "polish" or HORIZON_LABEL not in train.columns:
        return False
    # A missing label is not a second horizon. Two observed cohort values are.
    return int(train[HORIZON_LABEL].dropna().nunique()) > 1


def _kind(name: str, series: pd.Series, discrete: set[str]) -> str:
    if name.endswith(("_missing", "_indicator")) and _binary_indicator(series):
        return "binary"
    if name in discrete:
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def _binary_indicator(series: pd.Series) -> bool:
    values = series.dropna()
    if values.empty:
        return False
    return set(pd.to_numeric(values, errors="coerce").dropna().unique()) <= {0, 1}


def _positive_class(configured: object, target: pd.Series) -> object:
    if target.eq(configured).any():
        return configured
    # Bank preprocessing maps the raw label "yes" to 1 in every cleaned split.
    if configured == "yes" and target.eq(1).any():
        return 1
    raise ValueError(f"positive class {configured!r} is absent from the training split")


def _check_class_counts(train, target, positive, positives, request: ModelSelectionInput) -> None:
    negatives = int(train[target].ne(positive).sum())
    if positives + negatives != len(train):
        raise ValueError("the training target is not binary after resolving the positive class")
    expected_pos = request.preprocess.baseline_train_positives
    expected_neg = request.preprocess.baseline_train_negatives
    if positives != expected_pos or negatives != expected_neg:
        raise ValueError(
            f"training class counts {(positives, negatives)} disagree with "
            f"baseline counts {(expected_pos, expected_neg)}"
        )


def _route_reasons(result: SelectionResult) -> dict[str, str]:
    """Why each unchosen generator lost. Reads the logged steps; does not re-run the rule."""
    by_rule = {step.rule: step for step in result.steps}
    stop_gate = next(
        (
            name
            for name in ("size_gate", "horizon_gate", "missing_gate", "leakage_gate")
            if by_rule.get(name) is not None and by_rule[name].outcome == "FAIL"
        ),
        None,
    )
    if stop_gate is None and result.status == "skipped":
        stop_gate = "dcgan_excluded"
    unchosen = [route for route in ("timegan", "wgan_gp", "ctgan") if route != result.generator]
    if stop_gate is not None:
        reason = (
            f"Not evaluated: {stop_gate} stopped selection. {by_rule[stop_gate].rationale}"
        )
        return {route: reason for route in unchosen}
    if result.generator == "timegan":
        return {
            "ctgan": "TimeGAN was selected; CTGAN is its documented fallback",
            "wgan_gp": "TimeGAN was selected, so type routing did not run",
        }
    reasons = {"timegan": by_rule["temporal_test"].rationale}
    other = "wgan_gp" if result.generator == "ctgan" else "ctgan"
    reasons[other] = by_rule["type_routing"].rationale
    return reasons


def _log_skipped_routes(result: SelectionResult, run: RunDir) -> None:
    for route, reason in _route_reasons(result).items():
        log_decision(
            "model_selection",
            "route_skipped",
            {"route": route, "attempt": 1},
            "NOT_SELECTED",
            reason,
            run=run,
        )


def _explain(
    result: SelectionResult, request: ModelSelectionInput, run: RunDir, client
) -> SelectionExplanation:
    """Ask the LLM to explain the logged steps. It must echo the rule's generator, never pick."""
    expected = result.generator or "none"
    context = json.dumps(
        {
            "dataset": request.config.dataset_name,
            "status": result.status,
            "generator": expected,
            "steps": [asdict(step) for step in result.steps],
            "routes_not_chosen": _route_reasons(result),
        },
        default=str,
    )
    return run_llm_step(
        Stage.MODEL_SELECTION.value,
        EXPLAIN_PROMPT,
        context,
        SelectionExplanation,
        LLMStepOptions(
            fallback=SelectionExplanation(generator=expected, explanation=result.rationale),
            allowed_options={"generator": (expected,)},
            expected_rule_hash=_rule_hash(request.config),
            current_rule_hash=lambda: _rule_hash(run.load_config()),
        ),
        run,
        client=client,
    )


def _log_steps(result: SelectionResult, attempt: int, run: RunDir) -> None:
    for step in result.steps:
        log_decision(
            "model_selection",
            step.rule,
            {**step.inputs, "attempt": attempt},
            step.outcome,
            step.rationale,
            run=run,
        )


def _skipped(result: SelectionResult) -> GeneratorOutput:
    if result.generator and not result.trainable:
        reason = (
            f"{result.generator} selected but is not implemented. "
            f"Documented fallback: {result.fallback}."
        )
    else:
        reason = result.rationale
    return GeneratorOutput(
        status=Status.SKIPPED,
        reason=reason,
        generator=result.generator,
        rationale=result.rationale,
    )


def _fallback_attempt(
    result: SelectionResult, attempt: int, run: RunDir, *, exhausted: bool
) -> GeneratorOutput:
    """SKIPPED only when the ladder is exhausted and no fallback generator is implemented."""
    fallback = result.fallback or "wgan_gp"
    if not exhausted or fallback in IMPLEMENTED_GENERATORS:
        raise RuntimeError(
            "_fallback_attempt requires an exhausted ladder and an unimplemented fallback"
        )
    reason = f"Attempt {attempt} uses fallback {fallback}, which is not implemented."
    log_decision(
        "model_selection", "fallback",
        {"attempt": attempt, "primary": result.generator, "fallback": fallback},
        "SKIPPED", reason, run=run,
    )
    return GeneratorOutput(
        status=Status.SKIPPED, reason=reason, generator=fallback, rationale=reason,
    )


def _bounds_json(bounds: dict) -> dict:
    return {name: [low, high] for name, (low, high) in bounds.items()}


def _clamp(defaults: dict[str, int], bounds: dict) -> dict[str, int] | None:
    settings = dict(defaults)
    for name, value in defaults.items():
        if name not in bounds:
            continue
        low, high = bounds[name]
        settings[name] = int(round(min(max(value, low), high)))
    if settings["batch_size"] % settings["pac"] != 0:
        return None
    return settings


def _failed_hard_checks(request: ModelSelectionInput) -> list[str]:
    return _failed_checks_from_report(request.previous_report)


def _retry_limits(
    settings: dict[str, int],
    gan_bounds: dict,
    privacy: bool,
    *,
    rake_allowed: Sequence[str] = (),
) -> RetryLimits | None:
    """Inclusive ranges and options the LLM may use, or None if nothing is tunable.

    A setting with no config bound is pinned to its current value, and so is one whose bound
    holds no integer or does not contain the current value. After a privacy failure epochs may
    not rise: a longer-trained generator copies more real rows.

    "Nothing tunable" (return None) means every numeric setting is pinned AND the categorical
    options cannot move. sampling_mode and rule_filter always have at least two legal values in
    RETRY_OPTIONS, so None is practically unreachable; this function always returns RetryLimits.
    """
    numeric: dict[str, tuple[int, int]] = {}
    for name in SETTING_NAMES:
        current = int(settings[name])
        if name in gan_bounds:
            bound_low, bound_high = gan_bounds[name]
            low, high = math.ceil(bound_low), math.floor(bound_high)
        else:
            low = high = current
        if name == "epochs" and privacy:
            high = min(high, current)
        if low > high or not low <= current <= high:
            low = high = current
        numeric[name] = (low, high)
    # Options always remain movable (see docstring); never return None.
    return RetryLimits(
        numeric=numeric,
        options={name: tuple(values) for name, values in RETRY_OPTIONS.items()},
        rake_allowed=tuple(rake_allowed),
    )


def _rule_retry(
    settings: dict[str, int], request: ModelSelectionInput
) -> tuple[dict[str, int], str, str]:
    """The deterministic retry, with no logging: (new settings, rule name, rationale).

    More epochs inside gan_bounds when a fidelity check failed, unless a privacy check failed
    too. The LLM proposer uses this as its fallback.

    Kept for Step 4 compatibility; Step 5 wires `_ladder_retry` in its place.
    """
    failed = _failed_hard_checks(request)
    privacy = _privacy_failed(failed)
    fidelity = _matches(failed, FIDELITY_TOKENS)
    bounds = request.config.gan_bounds
    previous = settings["epochs"]
    if privacy or not fidelity or "epochs" not in bounds:
        rationale = "no privacy-safe adjustment" if privacy else "retry has no legal adjustment"
        return settings, "retry_unchanged", rationale
    low, high = bounds["epochs"]
    proposed = min(previous + EPOCH_STEP, int(round(high)))
    proposed = max(proposed, int(round(low)))
    if proposed == previous:
        return settings, "retry_unchanged", "epochs are already at the gan_bounds limit"
    updated = {**settings, "epochs": proposed}
    rationale = f"epochs move from {previous} to {proposed} inside gan_bounds"
    return updated, "hyperparameter_retry", rationale


def _copy_seven(settings: dict) -> dict:
    return {
        "epochs": int(settings["epochs"]),
        "batch_size": int(settings["batch_size"]),
        "pac": int(settings["pac"]),
        "discriminator_steps": int(settings["discriminator_steps"]),
        "sampling_mode": settings["sampling_mode"],
        "rule_filter": bool(settings["rule_filter"]),
        "rake_columns": list(settings["rake_columns"]),
    }


def _report_ladder_signals(report: QualityReport | None) -> tuple[list[str], dict[str, float]]:
    """Hard failed check names, and all column_distance:<col> values from the report."""
    if report is None:
        return [], {}
    hard_failed: list[str] = []
    distances: dict[str, float] = {}
    for level in report.levels:
        for check in level.checks:
            if check.name.startswith("column_distance:"):
                distances[check.name.split(":", 1)[1]] = float(check.value)
            if check.hard and not check.passed:
                hard_failed.append(check.name)
            elif (
                not check.hard
                and not check.passed
                and check.name.startswith("column_distance:")
            ):
                # Soft per-column failures participate in row-1 matching.
                hard_failed.append(check.name)
    # Preserve order, drop duplicate names while keeping first occurrence.
    seen: set[str] = set()
    ordered: list[str] = []
    for name in hard_failed:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered, distances


def _numeric_bound(
    name: str, value: int, gan_bounds: dict, previous: int
) -> int:
    """Clamp into gan_bounds when present; otherwise pin to previous (Q2)."""
    if name not in gan_bounds:
        return int(previous)
    low, high = gan_bounds[name]
    low_i, high_i = math.ceil(low), math.floor(high)
    return int(min(max(int(value), low_i), high_i))


def _snap_batch(batch: int, pac: int, gan_bounds: dict, previous_batch: int) -> int:
    """Snap batch_size to a multiple of pac inside its bound (or pin previous if impossible)."""
    if pac < 1:
        return int(previous_batch)
    if "batch_size" in gan_bounds:
        low, high = gan_bounds["batch_size"]
        low_i, high_i = math.ceil(low), math.floor(high)
    else:
        return int(previous_batch)
    batch = int(min(max(int(batch), low_i), high_i))
    if batch % pac == 0:
        return batch
    snapped = batch // pac * pac
    if snapped < low_i:
        snapped += pac
    if low_i <= snapped <= high_i:
        return int(snapped)
    return int(previous_batch)


def _apply_ladder_invariants(
    candidate: dict,
    previous: dict,
    gan_bounds: dict,
    *,
    privacy_failed: bool,
    allowed_rake: Sequence[str],
) -> dict:
    """Global invariants (d): privacy epochs, bounds, batch%pac, options, rake subset."""
    out = _copy_seven(candidate)
    for name in SETTING_NAMES:
        out[name] = _numeric_bound(name, int(out[name]), gan_bounds, int(previous[name]))
    if privacy_failed:
        out["epochs"] = min(int(out["epochs"]), int(previous["epochs"]))
        if "epochs" in gan_bounds:
            out["epochs"] = _numeric_bound(
                "epochs", out["epochs"], gan_bounds, int(previous["epochs"])
            )
    out["batch_size"] = _snap_batch(
        int(out["batch_size"]), int(out["pac"]), gan_bounds, int(previous["batch_size"])
    )
    if out["sampling_mode"] not in RETRY_OPTIONS["sampling_mode"]:
        out["sampling_mode"] = previous["sampling_mode"]
    if out["rule_filter"] not in RETRY_OPTIONS["rule_filter"]:
        out["rule_filter"] = bool(previous["rule_filter"])
    allowed = set(allowed_rake)
    rake: list[str] = []
    seen: set[str] = set()
    for name in out["rake_columns"]:
        if name in allowed and name not in seen:
            seen.add(name)
            rake.append(name)
    out["rake_columns"] = rake
    return out


def _row1_rake(
    previous: dict,
    failed: list[str],
    distances: dict[str, float],
    discrete_columns: Sequence[str],
    allowed_rake: Sequence[str],
) -> list[str]:
    discrete = set(discrete_columns)
    allowed = set(allowed_rake)
    cat_distances = {col: distances[col] for col in distances if col in discrete}
    # Distance desc, then name asc (alphabetical tie-break).
    worst_order = sorted(cat_distances, key=lambda col: (-cat_distances[col], col))
    triggering: list[str] = []
    for name in failed:
        if name.startswith("column_distance:"):
            col = name.split(":", 1)[1]
            if col in discrete and col not in triggering:
                triggering.append(col)
    if "worst_column_distance" in failed and not triggering and worst_order:
        triggering = [worst_order[0]]
    # Triggering columns in worst-first order, then next two worst categoricals.
    picked: list[str] = []
    for col in worst_order:
        if col in triggering and col not in picked:
            picked.append(col)
    for col in triggering:
        if col not in picked:
            picked.append(col)
    extras = 0
    for col in worst_order:
        if col in picked:
            continue
        picked.append(col)
        extras += 1
        if extras >= 2:
            break
    for col in previous["rake_columns"]:
        if col not in picked:
            picked.append(col)
    return [col for col in picked if col in allowed]


def _ladder_retry(
    previous_settings: dict,
    history: list[AttemptRecord],
    request: ModelSelectionInput,
    discrete_columns: list[str],
    allowed_rake: list[str],
) -> LadderResult:
    """Deterministic retry ladder. Pure: no logging, no I/O, no mutation of inputs."""
    previous = _copy_seven(previous_settings)
    history_keys = {settings_key(record.settings) for record in history}
    failed, distances = _report_ladder_signals(request.previous_report)
    privacy_failed = bool(set(failed) & PRIVACY_CHECKS)
    gan_bounds = request.config.gan_bounds
    discrete = set(discrete_columns)

    candidates: list[tuple[int, str, dict]] = []

    # Row 1: worst_column_distance or discrete column_distance:<col>.
    row1_triggers = [
        name
        for name in failed
        if name == "worst_column_distance"
        or (
            name.startswith("column_distance:")
            and name.split(":", 1)[1] in discrete
        )
    ]
    if row1_triggers:
        trigger = row1_triggers[0]
        proposed = _copy_seven(previous)
        proposed["sampling_mode"] = "unconditional"
        proposed["rake_columns"] = _row1_rake(
            previous, failed, distances, discrete_columns, allowed_rake
        )
        candidates.append((1, trigger, proposed))

    # Row 2: any rule:* check.
    rule_triggers = [name for name in failed if name.startswith("rule:")]
    if rule_triggers:
        proposed = _copy_seven(previous)
        proposed["rule_filter"] = True
        candidates.append((2, rule_triggers[0], proposed))

    # Row 3: mean distance / correlations / detectability.
    row3_triggers = [name for name in failed if name in ROW3_CHECKS]
    if row3_triggers:
        proposed = _copy_seven(previous)
        proposed["discriminator_steps"] = min(
            int(previous["discriminator_steps"]) + 2, LADDER_STEPS_CAP
        )
        proposed["pac"] = 5
        proposed["epochs"] = min(int(previous["epochs"]) + EPOCH_STEP, LADDER_EPOCHS_CAP)
        candidates.append((3, row3_triggers[0], proposed))

    # Row 4: privacy.
    privacy_triggers = [name for name in failed if name in PRIVACY_CHECKS]
    if privacy_triggers:
        proposed = _copy_seven(previous)
        proposed["epochs"] = max(int(previous["epochs"]) - EPOCH_STEP, LADDER_EPOCHS_FLOOR)
        proposed["batch_size"] = min(int(previous["batch_size"]) * 2, LADDER_BATCH_CAP)
        candidates.append((4, privacy_triggers[0], proposed))

    # Diagnosis-derived candidates after matching table rows, before the catch-all.
    diagnosis = (
        request.previous_report.diagnosis
        if request.previous_report is not None
        else None
    )
    diag_rung = 6
    if diagnosis is not None:
        allowed = set(allowed_rake)
        for adjustment in dict.fromkeys(diagnosis.adjustments):
            proposed = _copy_seven(previous)
            if adjustment == "change_sampling":
                proposed["sampling_mode"] = (
                    "unconditional"
                    if previous["sampling_mode"] == "conditional"
                    else "conditional"
                )
            elif adjustment == "rebalance_categories":
                rake = list(previous["rake_columns"])
                for col in diagnosis.columns:
                    if col in allowed and col not in rake:
                        rake.append(col)
                proposed["rake_columns"] = rake
            elif adjustment == "more_epochs":
                if privacy_failed:
                    continue
                proposed["epochs"] = int(previous["epochs"]) + EPOCH_STEP
            elif adjustment == "change_pac":
                proposed["pac"] = 5
            else:
                continue
            candidates.append((diag_rung, f"diagnosis:{adjustment}", proposed))
            diag_rung += 1

    # Row 5: catch-all last resort (Q4).
    proposed = _copy_seven(previous)
    proposed["epochs"] = int(previous["epochs"]) + EPOCH_STEP
    candidates.append((5, "none", proposed))

    for rung, trigger, raw in candidates:
        applied = _apply_ladder_invariants(
            raw,
            previous,
            gan_bounds,
            privacy_failed=privacy_failed,
            allowed_rake=allowed_rake,
        )
        key = settings_key(applied)
        if key == settings_key(previous):
            continue
        if key in history_keys:
            continue
        changed = [
            name
            for name in SEVEN_KNOBS
            if (
                name == "rake_columns"
                and sorted(applied[name]) != sorted(previous[name])
            )
            or (name != "rake_columns" and applied[name] != previous[name])
        ]
        rationale = (
            f"ladder rung {rung} on {trigger}: "
            + ", ".join(f"{name}={applied[name]!r}" for name in changed)
        )
        return LadderResult(
            settings=applied,
            trigger=trigger,
            rung=rung,
            exhausted=False,
            rationale=rationale,
        )

    return LadderResult(
        settings=_copy_seven(previous),
        trigger="none",
        rung=5,
        exhausted=True,
        rationale="every rung is used up",
    )


def _normalize(
    answer: RetrySettings,
    limits: RetryLimits | dict[str, tuple[int, int]],
    *,
    rake_allowed: Sequence[str] | None = None,
) -> dict | None:
    """Make a proposal a legal CTGAN setting, or None if that is not possible inside the limits.

    The runtime already enforced the numeric bounds; the clip is a safety net. CTGAN raises
    unless batch_size is a multiple of pac, so a non-divisible batch is snapped down (or up to
    the first multiple inside the range). sampling_mode / rule_filter outside RETRY_OPTIONS, or
    rake_columns outside the allowed set or with duplicates, are rejected (no silent drops).
    """
    if isinstance(limits, RetryLimits):
        numeric = limits.numeric
        allowed = list(limits.rake_allowed if rake_allowed is None else rake_allowed)
        options = limits.options
    else:
        numeric = limits
        allowed = list(rake_allowed or ())
        options = RETRY_OPTIONS
    values: dict = {}
    for name in SETTING_NAMES:
        low, high = numeric[name]
        values[name] = int(min(max(getattr(answer, name), low), high))
    if any(value < 1 for value in values.values()):
        return None
    batch, pac = values["batch_size"], values["pac"]
    if batch % pac != 0:
        low, high = numeric["batch_size"]
        snapped = batch // pac * pac
        if snapped < low:
            snapped += pac
        if not low <= snapped <= high:
            return None
        values["batch_size"] = snapped
    sampling_mode = answer.sampling_mode
    rule_filter = answer.rule_filter
    if sampling_mode not in options["sampling_mode"]:
        return None
    if rule_filter not in options["rule_filter"]:
        return None
    rake = list(answer.rake_columns)
    if len(rake) != len(set(rake)):
        return None
    allowed_set = set(allowed)
    if any(name not in allowed_set for name in rake):
        return None
    values["sampling_mode"] = sampling_mode
    values["rule_filter"] = bool(rule_filter)
    values["rake_columns"] = rake
    return values


def _failed_check_summaries(request: ModelSelectionInput) -> list[dict]:
    report = request.previous_report
    if report is None:
        return []
    return [
        {
            "fill_level": level.fill_level,
            "name": check.name,
            "value": check.value,
            "limit": check.threshold,
        }
        for level in report.levels
        for check in level.checks
        if check.hard and not check.passed
    ]


def _worst_columns(
    request: ModelSelectionInput, discrete_columns: Sequence[str]
) -> list[dict]:
    """Up to three worst column_distance columns with kind categorical or numeric."""
    _failed, distances = _report_ladder_signals(request.previous_report)
    if not distances:
        return []
    discrete = set(discrete_columns)
    ordered = sorted(distances, key=lambda col: (-distances[col], col))[:3]
    return [
        {
            "column": col,
            "distance": distances[col],
            "kind": "categorical" if col in discrete else "numeric",
        }
        for col in ordered
    ]


def _retry_settings_answer(settings: dict, reason: str) -> RetrySettings:
    """Build a RetrySettings from a seven-knob settings dict."""
    return RetrySettings(
        epochs=int(settings["epochs"]),
        pac=int(settings["pac"]),
        discriminator_steps=int(settings["discriminator_steps"]),
        batch_size=int(settings["batch_size"]),
        sampling_mode=settings["sampling_mode"],
        rule_filter=bool(settings["rule_filter"]),
        rake_columns=list(settings["rake_columns"]),
        reason=reason,
    )


def _diagnosis_suggestions(request: ModelSelectionInput, allowed_rake: Sequence[str]) -> dict:
    """Map diagnosis adjustments to suggested knob moves for the LLM context only."""
    report = request.previous_report
    if report is None or report.diagnosis is None:
        return {}
    failed = _failed_hard_checks(request)
    privacy = bool(set(failed) & PRIVACY_CHECKS)
    allowed = set(allowed_rake)
    suggestions: dict = {}
    for adjustment in dict.fromkeys(report.diagnosis.adjustments):
        if adjustment == "change_sampling":
            suggestions[adjustment] = "flip sampling_mode"
        elif adjustment == "rebalance_categories":
            suggestions[adjustment] = [
                col for col in report.diagnosis.columns if col in allowed
            ]
        elif adjustment == "more_epochs" and not privacy:
            suggestions[adjustment] = f"epochs +{EPOCH_STEP}"
        elif adjustment == "change_pac":
            suggestions[adjustment] = "pac toward 5"
    return suggestions


def _propose_retry(
    request: ModelSelectionInput,
    run: RunDir,
    client,
    *,
    discrete_columns: list[str],
    allowed_rake: list[str],
) -> tuple[dict, bool]:
    """Propose seven knobs via LLM inside limits; the ladder is the fallback and safety net.

    Returns (applied_settings, exhausted). exhausted is True only when every ladder rung is used up.
    """
    history = attempt_history(request, run)
    previous = _copy_seven(history[-1].settings)
    ladder = _ladder_retry(previous, history, request, discrete_columns, allowed_rake)
    failed = _failed_hard_checks(request)
    privacy = bool(set(failed) & PRIVACY_CHECKS)
    limits = _retry_limits(
        previous,
        request.config.gan_bounds,
        privacy,
        rake_allowed=allowed_rake,
    )
    # limits is always RetryLimits (options always movable), so the LLM is called every retry.
    assert limits is not None
    applied = _copy_seven(ladder.settings)
    reason = ladder.rationale
    source = "ladder"
    proposed = None
    trigger = ladder.trigger
    rung = ladder.rung
    exhausted = ladder.exhausted

    report = request.previous_report
    diagnosis = report.diagnosis.model_dump() if report and report.diagnosis else None
    fallback = _retry_settings_answer(ladder.settings, LADDER_PREFIX + ladder.rationale)
    context = json.dumps(
        {
            "failed_hard_checks": _failed_check_summaries(request),
            "worst_columns": _worst_columns(request, discrete_columns),
            "diagnosis": diagnosis,
            "current_settings": previous,
            "limits": {name: list(edges) for name, edges in limits.numeric.items()},
            "options": {name: list(values) for name, values in limits.options.items()},
            "allowed_rake_columns": list(allowed_rake),
            "history": [
                {
                    "attempt": record.attempt,
                    "settings": record.settings,
                    "failed_checks": list(record.failed_checks),
                }
                for record in history
            ],
            "privacy_failed": privacy,
        },
        default=str,
    )
    answer = run_llm_step(
        Stage.MODEL_SELECTION.value,
        RETRY_PROMPT,
        context,
        RetrySettings,
        LLMStepOptions(
            fallback=fallback,
            numeric_bounds={name: limits.numeric[name] for name in SETTING_NAMES},
            allowed_options={
                "sampling_mode": RETRY_OPTIONS["sampling_mode"],
                "rule_filter": RETRY_OPTIONS["rule_filter"],
            },
            expected_rule_hash=_rule_hash(request.config),
            current_rule_hash=lambda: _rule_hash(run.load_config()),
        ),
        run,
        client=client,
    )
    # F4: LADDER_PREFIX marks the deterministic fallback; do not compare free-text reasons.
    from_llm = not answer.reason.startswith(LADDER_PREFIX)
    if from_llm:
        proposed = {
            "epochs": answer.epochs,
            "pac": answer.pac,
            "discriminator_steps": answer.discriminator_steps,
            "batch_size": answer.batch_size,
            "sampling_mode": answer.sampling_mode,
            "rule_filter": bool(answer.rule_filter),
            "rake_columns": list(answer.rake_columns),
        }
        normalized = _normalize(answer, limits)
        reject_reason = None
        if normalized is None:
            reject_reason = "could not be made a valid CTGAN setting"
        else:
            key = settings_key(normalized)
            if key == settings_key(previous) or key in {
                settings_key(record.settings) for record in history
            }:
                reject_reason = "repeats an earlier attempt"
        if reject_reason is not None:
            log_decision(
                "model_selection",
                "retry_rejected",
                {
                    "attempt": request.attempt,
                    "proposed": proposed,
                    "limits": {name: list(edges) for name, edges in limits.numeric.items()},
                },
                "REJECTED",
                reject_reason,
                run=run,
            )
            if reject_reason == "repeats an earlier attempt" and normalized is not None:
                augmented = [
                    *history,
                    AttemptRecord(
                        attempt=-1, settings=normalized, failed_checks=list(failed)
                    ),
                ]
                ladder = _ladder_retry(
                    previous, augmented, request, discrete_columns, allowed_rake
                )
                applied = _copy_seven(ladder.settings)
                reason = ladder.rationale
                source = "ladder"
                trigger = ladder.trigger
                rung = ladder.rung
                exhausted = ladder.exhausted
            else:
                applied = _copy_seven(ladder.settings)
                reason = ladder.rationale
                source = "ladder"
                trigger = ladder.trigger
                rung = ladder.rung
                exhausted = ladder.exhausted
        else:
            applied = _copy_seven(normalized)
            reason = answer.reason
            source = "llm"
            trigger = ladder.trigger
            rung = ladder.rung
            exhausted = False
            # Never keep an LLM answer that matches history (g).
            if settings_key(applied) in {settings_key(record.settings) for record in history}:
                augmented = [
                    *history,
                    AttemptRecord(attempt=-1, settings=applied, failed_checks=list(failed)),
                ]
                ladder = _ladder_retry(
                    previous, augmented, request, discrete_columns, allowed_rake
                )
                applied = _copy_seven(ladder.settings)
                reason = ladder.rationale
                source = "ladder"
                trigger = ladder.trigger
                rung = ladder.rung
                exhausted = ladder.exhausted

    if exhausted:
        log_decision(
            "model_selection",
            "retry_unchanged",
            {
                "attempt": request.attempt,
                "source": source,
                "trigger": trigger,
                "previous_settings": previous,
                "proposed": proposed,
                "applied": applied,
                "limits": {name: list(edges) for name, edges in limits.numeric.items()},
                "options": {name: list(values) for name, values in limits.options.items()},
                "failed_checks": failed,
                "privacy_failed": privacy,
                "rung": rung,
                "previous_epochs": previous["epochs"],
                "epochs": applied["epochs"],
                "bound": _bounds_json(request.config.gan_bounds).get("epochs"),
            },
            "UNCHANGED",
            reason if reason else "every rung is used up",
            run=run,
        )
        return applied, True

    log_decision(
        "model_selection",
        "hyperparameter_retry",
        {
            "attempt": request.attempt,
            "source": source,
            "trigger": trigger,
            "previous_settings": previous,
            "proposed": proposed,
            "applied": applied,
            "limits": {name: list(edges) for name, edges in limits.numeric.items()},
            "options": {name: list(values) for name, values in limits.options.items()},
            "failed_checks": failed,
            "privacy_failed": privacy,
            "rung": rung,
            "previous_epochs": previous["epochs"],
            "epochs": applied["epochs"],
            "bound": _bounds_json(request.config.gan_bounds).get("epochs"),
        },
        "ADJUSTED",
        reason,
        run=run,
    )
    return applied, False


# Formerly logged; diagnosis adjustments are now ladder/LLM suggestions (Step 5).
UNSUPPORTED_ADJUSTMENTS = ()


def _log_unsupported_adjustments(request: ModelSelectionInput, run: RunDir) -> None:
    """No-op: unsupported_adjustment lines must never be written."""
    return None


def _matches(names: list[str], tokens: tuple[str, ...]) -> bool:
    return any(token in name for name in names for token in tokens)


def _privacy_failed(failed: list[str]) -> bool:
    """Exact names from the privacy tool; a substring match missed exact_train_copies."""
    return bool(set(failed) & PRIVACY_CHECKS)


def _train_ctgan(
    request, run, train: pd.DataFrame, settings: dict[str, int], result: SelectionResult
):
    from agentic_gan.tools.ctgan import save_model, train_ctgan

    dataset = request.config.dataset_name
    target = request.config.target
    positive = _positive_class(request.config.positive_class, train[target])
    discrete = _discrete_columns(train, target, dataset)
    folder = run.subdir("synthetic")
    attempt = request.attempt
    checkpoint = folder / f"ctgan_attempt_{attempt}.pkl"
    loss_path = folder / f"ctgan_loss_attempt_{attempt}.csv"
    knobs = _generator_knobs(dataset, attempt, settings)
    trainable = _clamp({name: knobs[name] for name in CTGAN_DEFAULTS}, request.config.gan_bounds)
    if trainable is None:
        reason = "gan_bounds make batch_size not divisible by pac, so CTGAN is not trained."
        log_decision(
            "model_selection", "hyperparameter_bounds",
            {"attempt": attempt, "gan_bounds": _bounds_json(request.config.gan_bounds)},
            "SKIPPED", reason, run=run,
        )
        return GeneratorOutput(status=Status.SKIPPED, reason=reason, rationale=reason)
    knobs.update(trainable)
    log_decision(
        "model_selection", "hyperparameters",
        {"attempt": attempt, **knobs, "seed": request.config.split_seed, "generator": "ctgan"},
        "SET", "CTGAN settings are inside gan_bounds.", run=run,
    )
    model = train_ctgan(
        train,
        target,
        discrete_columns=discrete,
        log_path=loss_path,
        seed=request.config.split_seed,
        **trainable,
    )
    save_model(model, checkpoint)
    positives = request.preprocess.baseline_train_positives
    negatives = request.preprocess.baseline_train_negatives
    counts = [fill_rows(level, positives, negatives) for level in request.config.fill_levels]
    # One draw covers every fill level; each level takes the first rows, so the sets are nested.
    rows = sample_for_fill(
        model,
        train,
        train[train[target].eq(positive)],
        counts,
        sampling_mode=knobs["sampling_mode"],
        rule_filter=knobs["rule_filter"],
        rake_columns=knobs["rake_columns"],
        rules=RULES.get(dataset, ()),
        seed=request.config.split_seed,
    )
    if IS_SYNTHETIC not in rows.columns:
        rows = rows.copy()
        rows[IS_SYNTHETIC] = 1
    sets = []
    for level, count in zip(request.config.fill_levels, counts, strict=True):
        sample = rows.head(count)
        if len(sample) != count:
            raise ValueError(
                f"synthetic rows {len(sample)} do not match fill_rows {count} at {level}"
            )
        path = folder / f"fill_{_level_token(level)}_attempt_{attempt}.csv"
        sample.to_csv(path, index=False)
        sets.append(SyntheticSet(fill_level=level, rows=count, path=path.resolve()))
    return GeneratorOutput(
        status=Status.SUCCESS,
        generator="ctgan",
        rationale=result.rationale,
        hyperparameters={"attempt": attempt, **knobs, "seed": request.config.split_seed},
        checkpoint_path=checkpoint.resolve(),
        loss_history_path=loss_path.resolve(),
        synthetic_sets=sets,
    )


def _generator_knobs(dataset: str, attempt: int, settings: dict) -> dict:
    """The seven generator options of this attempt.

    Attempt 1 is the dataset's recipe. A later attempt takes every option its settings carry
    (a retry may move any of the seven) and keeps the recipe's value for the rest.
    """
    recipe = asdict(recipe_for(dataset))
    if attempt == 1:
        return recipe
    return {name: settings.get(name, value) for name, value in recipe.items()}


def _level_token(level: float) -> str:
    return f"{level:.2f}".replace(".", "p")


def _discrete_columns(train: pd.DataFrame, target: str, dataset: str) -> list[str]:
    discrete_names = set(DATASETS[dataset].discrete) if dataset in DATASETS else set()
    discrete = []
    for name in train.columns:
        if name == target:
            continue
        if name in discrete_names or name.endswith(("_missing", "_indicator")):
            discrete.append(name)
        elif not pd.api.types.is_numeric_dtype(train[name]):
            discrete.append(name)
    discrete.append(target)
    return discrete
