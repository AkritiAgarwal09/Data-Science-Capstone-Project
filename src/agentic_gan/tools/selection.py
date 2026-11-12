"""Deterministic generator selection. No training, no disk, and no decision log.

The agent in ``agents/model_selection.py`` turns this result into a ``GeneratorOutput`` and
writes one ``log_decision`` line per step. Column names never create a time index: a field
called Time, TransactionDT, month, or cohort is not a sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

STEP_ORDER: tuple[str, ...] = (
    "dcgan_excluded",
    "size_gate",
    "horizon_gate",
    "missing_gate",
    "leakage_gate",
    "minority_gate",
    "temporal_test",
    "type_routing",
    "fallback",
)

# Default of Credit Card stores six months wide: repayment status, bill, and amount paid.
# File order is September-first; the names below are the chronological block the test requires.
DEFAULT_WIDE_TEMPORAL: tuple[str, ...] = tuple(
    [f"PAY_{month}" for month in range(1, 7)]
    + [f"BILL_AMT{month}" for month in range(1, 7)]
    + [f"PAY_AMT{month}" for month in range(1, 7)]
)

TIME_INDEX_CONDITIONS: tuple[str, ...] = (
    "entity_key",
    "time_index",
    "min_steps",
    "repeated_observation",
)


@dataclass(frozen=True)
class DatasetStructure:
    """Facts the training frame may confirm and may not invent."""

    leakage: tuple[str, ...]
    one_row_one_entity: bool
    wide_temporal: tuple[str, ...]
    repeated_observation: bool
    steps: int | None
    non_sequence_time: tuple[str, ...]


DATASET_STRUCTURE: dict[str, DatasetStructure] = {
    "bank_marketing": DatasetStructure(
        leakage=("duration",),
        one_row_one_entity=False,
        wide_temporal=(),
        repeated_observation=False,
        steps=None,
        non_sequence_time=("month", "day_of_month"),
    ),
    "default": DatasetStructure(
        leakage=(),
        one_row_one_entity=True,
        wide_temporal=DEFAULT_WIDE_TEMPORAL,
        repeated_observation=True,
        steps=6,
        non_sequence_time=(),
    ),
    "polish": DatasetStructure(
        leakage=(),
        one_row_one_entity=False,
        wide_temporal=(),
        repeated_observation=False,
        steps=None,
        non_sequence_time=("cohort",),
    ),
    "ieee_cis": DatasetStructure(
        leakage=(),
        one_row_one_entity=False,
        wide_temporal=(),
        repeated_observation=False,
        steps=None,
        # A transaction clock and a row id. Neither is a within-entity sequence.
        non_sequence_time=("TransactionDT", "TransactionID"),
    ),
}


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    kind: str
    missing: int


@dataclass(frozen=True)
class DataCharacteristics:
    dataset: str
    rows: int
    target: str
    minority_count: int
    minority_rate: float
    columns: tuple[ColumnSpec, ...]
    tabular: bool = True
    entity_in_table: bool = False
    wide_steps: int | None = None
    repeated_observation: bool = False
    time_index_ready: bool = False
    leaky_columns: tuple[str, ...] = ()
    horizons_pooled: bool = False


@dataclass(frozen=True)
class DecisionStep:
    rule: str
    inputs: dict[str, Any]
    outcome: str
    rationale: str


@dataclass(frozen=True)
class SelectionResult:
    status: str
    generator: str | None
    trainable: bool
    fallback: str | None
    comparator: str | None
    scope: str
    rationale: str
    steps: tuple[DecisionStep, ...]


@dataclass(frozen=True)
class TemporalFacts:
    entity_in_table: bool
    wide_steps: int | None
    repeated_observation: bool
    time_index_ready: bool
    leakage: tuple[str, ...]


def temporal_facts(dataset: str, columns: set[str]) -> TemporalFacts:
    """Resolve temporal structure for a dataset. Unknown datasets fail closed."""
    spec = DATASET_STRUCTURE.get(dataset)
    if spec is None:
        return TemporalFacts(False, None, False, False, ())
    ready = bool(spec.wide_temporal) and all(name in columns for name in spec.wide_temporal)
    return TemporalFacts(
        entity_in_table=spec.one_row_one_entity,
        wide_steps=spec.steps if ready else None,
        repeated_observation=spec.repeated_observation and ready,
        time_index_ready=ready,
        leakage=spec.leakage,
    )


def select_generator(
    data: DataCharacteristics,
    *,
    min_rows: int = 2000,
    minority_min: int = 500,
    min_steps: int = 5,
) -> SelectionResult:
    """Gates, then the temporal test, then type routing. One input has one result."""
    context = {
        "dataset": data.dataset,
        "rows": data.rows,
        "minority_count": data.minority_count,
        "minority_rate": data.minority_rate,
    }
    steps: list[DecisionStep] = []

    def add(rule: str, outcome: str, rationale: str, extra: dict[str, Any] | None = None) -> None:
        inputs = dict(context)
        if extra:
            inputs.update(extra)
        steps.append(DecisionStep(rule, inputs, outcome, rationale))

    def rest_from(rule: str) -> None:
        started = False
        for name in STEP_ORDER:
            if name == rule:
                started = True
                continue
            if started:
                add(name, "NOT_REACHED", "An earlier gate stopped selection.")

    def skipped(rationale: str) -> SelectionResult:
        return SelectionResult(
            status="skipped",
            generator=None,
            trainable=False,
            fallback=None,
            comparator=None,
            scope="",
            rationale=rationale,
            steps=tuple(steps),
        )

    def chosen(
        generator: str,
        trainable: bool,
        fallback: str,
        rationale: str,
        scope: str = "",
    ) -> SelectionResult:
        comparator = "ctgan" if generator != "ctgan" else "wgan_gp"
        return SelectionResult(
            status="selected",
            generator=generator,
            trainable=trainable,
            fallback=fallback,
            comparator=comparator,
            scope=scope,
            rationale=rationale,
            steps=tuple(steps),
        )

    if not data.tabular:
        add(
            "dcgan_excluded",
            "EXCLUDED",
            "The table is not tabular. Image and diffusion generators stay out of scope.",
        )
        rest_from("dcgan_excluded")
        return skipped("Image and diffusion generators are out of scope for this project.")

    add(
        "dcgan_excluded",
        "EXCLUDED",
        "DCGAN is an image generator and is excluded for tabular data.",
    )

    if data.rows < min_rows:
        rationale = (
            f"Training rows {data.rows} are below min_rows {min_rows}, so no GAN is selected."
        )
        add("size_gate", "FAIL", rationale, {"min_rows": min_rows})
        rest_from("size_gate")
        return skipped(rationale)
    add(
        "size_gate",
        "PASS",
        f"Training rows {data.rows} meet min_rows {min_rows}.",
        {"min_rows": min_rows},
    )

    if data.horizons_pooled:
        rationale = "More than one Polish horizon is in the table. One horizon file is required."
        add("horizon_gate", "FAIL", rationale)
        rest_from("horizon_gate")
        return skipped(rationale)
    add("horizon_gate", "PASS", "One horizon, or this dataset has no horizon files.")

    missing = [column.name for column in data.columns if column.missing > 0]
    if missing:
        rationale = (
            f"{len(missing)} columns still have missing values, so selection waits for repair."
        )
        add(
            "missing_gate",
            "FAIL",
            rationale,
            {"missing_column_count": len(missing), "missing_columns": missing[:20]},
        )
        rest_from("missing_gate")
        return skipped(rationale)
    add("missing_gate", "PASS", "No unresolved missing values remain in the generator table.")

    if data.leaky_columns:
        listed = ", ".join(data.leaky_columns)
        rationale = f"Leakage columns are still in the generator table: {listed}."
        add("leakage_gate", "FAIL", rationale, {"leaky_columns": list(data.leaky_columns)})
        rest_from("leakage_gate")
        return skipped(rationale)
    add("leakage_gate", "PASS", "No listed leakage column is in the generator table.")

    if data.minority_count < minority_min:
        add(
            "minority_gate",
            "FORBIDDEN_CLASS_ONLY",
            (
                f"Minority count {data.minority_count} is below {minority_min}, so a class-only "
                "generator is forbidden. Conditional selection continues."
            ),
            {"minority_min": minority_min},
        )
    else:
        add(
            "minority_gate",
            "PASS",
            f"Minority count {data.minority_count} is at least {minority_min}.",
            {"minority_min": minority_min},
        )

    conditions = {
        "entity_key": "PASS" if data.entity_in_table else "FAIL",
        "time_index": "PASS" if data.time_index_ready and data.wide_steps is not None else "FAIL",
        "min_steps": (
            "PASS" if data.wide_steps is not None and data.wide_steps >= min_steps else "FAIL"
        ),
        "repeated_observation": "PASS" if data.repeated_observation else "FAIL",
    }
    if all(value == "PASS" for value in conditions.values()):
        scope = "monthly payment, bill, and payment-amount variables only"
        rationale = "All four temporal conditions passed. TimeGAN covers the monthly block only."
        add(
            "temporal_test",
            "PASS",
            rationale,
            {**conditions, "generator": "timegan", "wide_steps": data.wide_steps},
        )
        add("type_routing", "NOT_REACHED", "TimeGAN was selected, so type routing does not run.")
        add(
            "fallback",
            "DOCUMENTED",
            "TimeGAN is not implemented. Documented fallback is CTGAN.",
            {"fallback": "ctgan"},
        )
        return chosen("timegan", False, "ctgan", rationale, scope)

    failed = [name for name, value in conditions.items() if value == "FAIL"]
    add(
        "temporal_test",
        "FAIL",
        f"Temporal test failed on {', '.join(failed)}. TimeGAN is not selected.",
        {**conditions, "wide_steps": data.wide_steps},
    )

    kinds = {"numeric": 0, "categorical": 0, "binary": 0}
    for column in data.columns:
        if column.name == data.target:
            continue
        kinds[column.kind] = kinds.get(column.kind, 0) + 1
    mixed = kinds["categorical"] > 0 or kinds["binary"] > 0
    if mixed:
        rationale = "The table has categorical or indicator columns, so CTGAN is selected."
        add("type_routing", "SELECTED", rationale, {**kinds, "generator": "ctgan"})
        add(
            "fallback",
            "NOT_REACHED",
            "The WGAN-GP backup is attempt 3, after two quality-gate failures.",
        )
        return chosen("ctgan", True, "wgan_gp", rationale)

    rationale = "Every feature is numeric, so WGAN-GP is selected. Documented fallback is CTGAN."
    add("type_routing", "SELECTED", rationale, {**kinds, "generator": "wgan_gp"})
    add(
        "fallback",
        "DOCUMENTED",
        "WGAN-GP is implemented. CTGAN is the fallback if WGAN-GP fails.",
        {"fallback": "ctgan"},
    )
    return chosen("wgan_gp", True, "ctgan", rationale)


def backup_eligible(result: SelectionResult) -> tuple[bool, str]:
    """Return (eligible, reason) for a WGAN-GP backup on attempt 3.

    False when: minority_gate is FORBIDDEN_CLASS_ONLY, result is skipped, or fallback != wgan_gp.
    """
    if result.status == "skipped":
        return False, "selection was skipped"
    if result.fallback != "wgan_gp":
        return False, f"fallback is {result.fallback!r}, not wgan_gp"
    for step in result.steps:
        if step.rule == "minority_gate" and step.outcome == "FORBIDDEN_CLASS_ONLY":
            return False, (
                f"minority_gate is FORBIDDEN_CLASS_ONLY "
                f"(minority_count {step.inputs.get('minority_count')} "
                f"< {step.inputs.get('minority_min')})"
            )
    return True, "eligible"
