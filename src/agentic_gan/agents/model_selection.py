"""Choose a generator by rule, train CTGAN when that rule says so, and log every step."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Sequence
from dataclasses import asdict

import pandas as pd
from pydantic import Field

from agentic_gan.llm import LLMStepOptions, run_llm_step
from agentic_gan.orchestrator.decisions import log_decision, log_progress
from agentic_gan.orchestrator.manifest import RunDir
from agentic_gan.schemas import (
    GeneratorOutput,
    ModelSelectionInput,
    RunConfig,
    Schema,
    Stage,
    Status,
    SyntheticSet,
    fill_rows,
)
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
    "A CTGAN generator failed a synthetic-data quality gate. Propose the CTGAN settings for the "
    "next attempt: epochs, pac, discriminator_steps and batch_size. Use only the failed checks, "
    "the quality diagnosis and the current settings in the context. Every value must lie inside "
    "the inclusive range given for that setting in `limits`; a setting whose range has equal low "
    "and high cannot change, so return its current value. batch_size must be divisible by pac. If "
    "`privacy_failed` is true the generator may be copying real rows, so do not raise epochs. "
    "Explain in `reason` which failed checks or columns your changes target. You never change a "
    "rule, threshold, range or the choice of generator."
)


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
    reason: str = Field(min_length=1)


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
    }
    return hashlib.sha256(json.dumps(locked, sort_keys=True).encode()).hexdigest()


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
    if request.attempt >= 3:
        return _fallback_attempt(result, request.attempt, run)

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
        settings = _propose_retry(settings, request, run, client)
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


def _fallback_attempt(result: SelectionResult, attempt: int, run: RunDir) -> GeneratorOutput:
    fallback = result.fallback or "wgan_gp"
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
    report = request.previous_report
    if report is None:
        return []
    return [
        check.name
        for level in report.levels
        for check in level.checks
        if check.hard and not check.passed
    ]


def _retry_limits(
    settings: dict[str, int], gan_bounds: dict, privacy: bool
) -> dict[str, tuple[int, int]] | None:
    """Inclusive integer range the LLM may use for each setting, or None if nothing is tunable.

    A setting with no config bound is pinned to its current value, and so is one whose bound
    holds no integer or does not contain the current value. After a privacy failure epochs may
    not rise: a longer-trained generator copies more real rows.
    """
    limits: dict[str, tuple[int, int]] = {}
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
        limits[name] = (low, high)
    if all(low == high for low, high in limits.values()):
        return None
    return limits


def _rule_retry(
    settings: dict[str, int], request: ModelSelectionInput
) -> tuple[dict[str, int], str, str]:
    """The deterministic retry, with no logging: (new settings, rule name, rationale).

    More epochs inside gan_bounds when a fidelity check failed, unless a privacy check failed
    too. The LLM proposer uses this as its fallback.
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


def _normalize(answer: RetrySettings, limits: dict[str, tuple[int, int]]) -> dict[str, int] | None:
    """Make a proposal a legal CTGAN setting, or None if that is not possible inside the limits.

    The runtime already enforced the numeric bounds; the clip is a safety net. CTGAN raises
    unless batch_size is a multiple of pac, so a non-divisible batch is snapped down (or up to
    the first multiple inside the range).
    """
    values = {}
    for name in SETTING_NAMES:
        low, high = limits[name]
        values[name] = int(min(max(getattr(answer, name), low), high))
    if any(value < 1 for value in values.values()):
        return None
    batch, pac = values["batch_size"], values["pac"]
    if batch % pac != 0:
        low, high = limits["batch_size"]
        snapped = batch // pac * pac
        if snapped < low:
            snapped += pac
        if not low <= snapped <= high:
            return None
        values["batch_size"] = snapped
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


def _propose_retry(
    settings: dict[str, int], request: ModelSelectionInput, run: RunDir, client
) -> dict[str, int]:
    """Let the LLM propose settings inside gan_bounds; the rule retry is the fallback."""
    rule_settings, _rule_name, rule_reason = _rule_retry(settings, request)
    failed = _failed_hard_checks(request)
    privacy = bool(set(failed) & PRIVACY_CHECKS)
    limits = _retry_limits(settings, request.config.gan_bounds, privacy)
    applied, reason, source, proposed = rule_settings, rule_reason, "rule", None
    _log_unsupported_adjustments(request, run)
    if limits is not None:
        fallback = RetrySettings(**rule_settings, reason="Rule-based default: " + rule_reason)
        report = request.previous_report
        diagnosis = report.diagnosis.model_dump() if report and report.diagnosis else None
        context = json.dumps(
            {
                "attempt": request.attempt,
                "dataset": request.config.dataset_name,
                "failed_hard_checks": _failed_check_summaries(request),
                "diagnosis": diagnosis,
                "current_settings": settings,
                "limits": {name: list(edges) for name, edges in limits.items()},
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
                numeric_bounds={name: limits[name] for name in SETTING_NAMES},
                expected_rule_hash=_rule_hash(request.config),
                current_rule_hash=lambda: _rule_hash(run.load_config()),
            ),
            run,
            client=client,
        )
        if answer.reason != fallback.reason:
            proposed = {name: getattr(answer, name) for name in SETTING_NAMES}
            normalized = _normalize(answer, limits)
            if normalized is None:
                log_decision(
                    "model_selection",
                    "retry_rejected",
                    {
                        "attempt": request.attempt,
                        "proposed": proposed,
                        "limits": {name: list(edges) for name, edges in limits.items()},
                    },
                    "REJECTED",
                    "could not be made a valid CTGAN setting",
                    run=run,
                )
            else:
                applied, reason, source = normalized, answer.reason, "llm"
    changed = any(applied[name] != settings[name] for name in SETTING_NAMES)
    log_decision(
        "model_selection",
        "hyperparameter_retry" if changed else "retry_unchanged",
        {
            "attempt": request.attempt,
            "source": source,
            "previous_settings": settings,
            "proposed": proposed,
            "applied": applied,
            "limits": {name: list(edges) for name, edges in limits.items()} if limits else None,
            "failed_checks": failed,
            "privacy_failed": privacy,
            # Kept for readers of the earlier one-setting log line.
            "previous_epochs": settings["epochs"],
            "epochs": applied["epochs"],
            "bound": _bounds_json(request.config.gan_bounds).get("epochs"),
        },
        "ADJUSTED" if changed else "UNCHANGED",
        reason,
        run=run,
    )
    return applied


# Diagnosis adjustments with no CTGAN setting behind them. They are reported, never applied.
UNSUPPORTED_ADJUSTMENTS = ("rebalance_categories", "change_sampling")


def _log_unsupported_adjustments(request: ModelSelectionInput, run: RunDir) -> None:
    report = request.previous_report
    if report is None or report.diagnosis is None:
        return
    for adjustment in dict.fromkeys(report.diagnosis.adjustments):
        if adjustment in UNSUPPORTED_ADJUSTMENTS:
            log_decision(
                "model_selection",
                "unsupported_adjustment",
                {"attempt": request.attempt, "adjustment": adjustment},
                "NOT_APPLIED",
                "No CTGAN setting implements this adjustment, so it is reported only.",
                run=run,
            )


def _matches(names: list[str], tokens: tuple[str, ...]) -> bool:
    return any(token in name for name in names for token in tokens)


def _privacy_failed(failed: list[str]) -> bool:
    """Exact names from the privacy tool; a substring match missed exact_train_copies."""
    return bool(set(failed) & PRIVACY_CHECKS)


def _train_ctgan(
    request, run, train: pd.DataFrame, settings: dict[str, int], result: SelectionResult
):
    from agentic_gan.tools.ctgan import sample_synthetic, save_model, train_ctgan

    target = request.config.target
    positive = _positive_class(request.config.positive_class, train[target])
    discrete = _discrete_columns(train, target, request.config.dataset_name)
    folder = run.subdir("synthetic")
    attempt = request.attempt
    checkpoint = folder / f"ctgan_attempt_{attempt}.pkl"
    loss_path = folder / f"ctgan_loss_attempt_{attempt}.csv"
    log_decision(
        "model_selection", "hyperparameters",
        {"attempt": attempt, **settings, "seed": request.config.split_seed, "generator": "ctgan"},
        "SET", "CTGAN settings are inside gan_bounds.", run=run,
    )
    log_progress(
        "model_selection",
        f"train CTGAN attempt={attempt} epochs={settings['epochs']} rows={len(train)}",
        tool="train_ctgan",
        next_action="save checkpoint and sample fill levels",
    )
    model = train_ctgan(
        train,
        target,
        discrete_columns=discrete,
        log_path=loss_path,
        seed=request.config.split_seed,
        **settings,
    )
    save_model(model, checkpoint)
    sets = []
    positives = request.preprocess.baseline_train_positives
    negatives = request.preprocess.baseline_train_negatives
    for level in request.config.fill_levels:
        count = fill_rows(level, positives, negatives)
        log_progress(
            "model_selection",
            f"sample fill={level:g} rows={count}",
            tool="sample_synthetic",
            next_action="write synthetic CSV",
        )
        sample = sample_synthetic(
            model,
            count,
            target_value=positive,
            seed=request.config.split_seed,
            drop_duplicates_of=train,
        )
        if len(sample) != count:
            raise ValueError(
                f"synthetic rows {len(sample)} do not match fill_rows {count} at {level}"
            )
        if IS_SYNTHETIC not in sample.columns:
            sample = sample.copy()
            sample[IS_SYNTHETIC] = 1
        path = folder / f"fill_{_level_token(level)}_attempt_{attempt}.csv"
        sample.to_csv(path, index=False)
        sets.append(SyntheticSet(fill_level=level, rows=count, path=path.resolve()))
    return GeneratorOutput(
        status=Status.SUCCESS,
        generator="ctgan",
        rationale=result.rationale,
        hyperparameters={"attempt": attempt, **settings, "seed": request.config.split_seed},
        checkpoint_path=checkpoint.resolve(),
        loss_history_path=loss_path.resolve(),
        synthetic_sets=sets,
    )


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
