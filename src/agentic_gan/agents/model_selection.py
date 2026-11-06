"""Choose a generator by rule, train CTGAN when that rule says so, and log every step."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from agentic_gan.orchestrator.decisions import log_decision, log_progress
from agentic_gan.orchestrator.manifest import RunDir
from agentic_gan.schemas import (
    GeneratorOutput,
    ModelSelectionInput,
    Status,
    SyntheticSet,
    fill_rows,
)
from agentic_gan.tools.selection import (
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
FIDELITY_TOKENS = ("column_distance", "correlation", "ks", "js", "fidelity")
PRIVACY_TOKENS = ("dcr", "duplicate", "privacy")
# A constant cohort is the horizon id, not a ratio the generator should learn.
HORIZON_LABEL = "cohort"


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


def run_model_selection(request: ModelSelectionInput, run: RunDir) -> GeneratorOutput:
    """Select from the baseline training split only. Validation and test are not read."""
    train = pd.read_csv(request.preprocess.baseline.train_path)
    if IS_SYNTHETIC in train.columns:
        raise ValueError(f"training data must not already contain {IS_SYNTHETIC!r}")
    described = describe_characteristics(train, request)
    model_frame = train.drop(columns=["source_row"], errors="ignore")
    if described.dataset == "polish" and not described.horizons_pooled:
        model_frame = model_frame.drop(columns=[HORIZON_LABEL], errors="ignore")
    result = select_generator(described, min_rows=request.config.min_rows)
    _log_steps(result, request.attempt, run)

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
        settings = _retry_settings(settings, request, run)
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


def _retry_settings(
    settings: dict[str, int], request: ModelSelectionInput, run: RunDir
) -> dict[str, int]:
    failed = _failed_hard_checks(request)
    privacy = _matches(failed, PRIVACY_TOKENS)
    fidelity = _matches(failed, FIDELITY_TOKENS)
    bounds = request.config.gan_bounds
    previous = settings["epochs"]
    if privacy or not fidelity or "epochs" not in bounds:
        rationale = "no privacy-safe adjustment" if privacy else "retry has no legal adjustment"
        _log_retry(
            "retry_unchanged", request.attempt, previous, previous, bounds, failed, rationale, run
        )
        return settings
    low, high = bounds["epochs"]
    proposed = min(previous + EPOCH_STEP, int(round(high)))
    proposed = max(proposed, int(round(low)))
    if proposed == previous:
        _log_retry(
            "retry_unchanged", request.attempt, previous, previous, bounds, failed,
            "epochs are already at the gan_bounds limit", run,
        )
        return settings
    updated = {**settings, "epochs": proposed}
    _log_retry(
        "hyperparameter_retry", request.attempt, previous, proposed, bounds, failed,
        f"epochs move from {previous} to {proposed} inside gan_bounds", run,
    )
    return updated


def _matches(names: list[str], tokens: tuple[str, ...]) -> bool:
    return any(token in name for name in names for token in tokens)


def _log_retry(rule, attempt, previous, proposed, bounds, failed, rationale, run: RunDir) -> None:
    log_decision(
        "model_selection", rule,
        {
            "attempt": attempt,
            "previous_epochs": previous,
            "epochs": proposed,
            "bound": _bounds_json(bounds).get("epochs"),
            "failed_checks": failed,
        },
        "ADJUSTED" if rule == "hyperparameter_retry" else "UNCHANGED",
        rationale,
        run=run,
    )


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
