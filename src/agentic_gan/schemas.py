"""Shared data objects for the orchestrator and the agents.

These are the typed handoffs from docs/architecture.md section 7. Every agent output is an
`Outcome`: a status, warnings, an optional reason and file paths. The orchestrator validates each
output at the handoff and rejects one that fails.

Schemas only describe data. They never touch the disk: checking that a file exists, or that its
hash matches, is the manifest's job (orchestrator/manifest.py).
"""

from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)


class Status(StrEnum):
    SUCCESS = "SUCCESS"
    RETRY = "RETRY"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class Stage(StrEnum):
    """Pipeline stages, declared in run order."""

    SPLIT = "split"
    PREPROCESSING = "preprocessing"
    MODEL_SELECTION = "model_selection"
    QUALITY_CHECK = "quality_check"
    TRAINING = "training"
    EVALUATION = "evaluation"


class Arm(StrEnum):
    RAW = "raw"
    BASELINE = "baseline"
    SMOTE = "smote"
    GAN = "gan"


class Verdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"


STAGE_ORDER: tuple[Stage, ...] = tuple(Stage)

# Share of the training class gap filled with synthetic rows. 0 is the baseline itself.
FillLevel = Annotated[float, Field(gt=0, le=1)]

# Names the runs/<run_id>/ folder, so it must be safe as one path component.
RunId = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", max_length=100)]


def fill_rows(fill_level: float, positives: int, negatives: int) -> int:
    """Rows to add at a fill level: that share of (negatives - positives), rounded."""
    return round(fill_level * max(negatives - positives, 0))


class Schema(BaseModel):
    """Base for every object here: unknown fields are errors and objects are immutable."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, protected_namespaces=()
    )

    def updated(self, **changes: Any) -> Self:
        """Copy with changes, validated again (`model_copy(update=...)` skips validation)."""
        return type(self).model_validate({**dict(self), **changes})


# --- run config -------------------------------------------------------------------------------


class SplitRatios(Schema):
    train: float = Field(0.6, gt=0, lt=1)
    val: float = Field(0.2, gt=0, lt=1)
    test: float = Field(0.2, gt=0, lt=1)

    @model_validator(mode="after")
    def _sum_to_one(self) -> Self:
        if abs(self.train + self.val + self.test - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to 1")
        return self


class QualityThresholds(Schema):
    """The quality gate's limits. The fixed numbers are the starting values from architecture.md
    section 5.3.

    With ``calibrate`` on, the quality agent replaces the fidelity and rule limits by ``margin``
    times the dataset's own real-vs-real values (issue 101); the fixed numbers are then the
    fallback. The two augmentation-gain checks are reported but decide nothing unless
    ``utility_gain_hard`` is set. All of it is part of the rule hash, so an LLM cannot change it.
    """

    mean_column_distance_max: float = Field(0.10, ge=0)
    column_distance_max: float = Field(0.25, ge=0)
    correlation_diff_max: float = Field(0.10, ge=0)
    tstr_auc_gap_max: float = Field(0.05, ge=0)
    dcr_ratio_min: float = Field(1.0, ge=0)
    exact_duplicates_max: int = Field(0, ge=0)
    sanity_violations_max: int = Field(0, ge=0)
    margin: float = Field(3.0, ge=1)
    calibrate: bool = True
    utility_gain_hard: bool = False


class RunConfig(Schema):
    """Everything a run needs. Saved to runs/<run_id>/config.yaml; a run is rebuilt from it."""

    dataset_name: str = Field(min_length=1)
    dataset_path: Path
    target: str = Field(min_length=1)
    # Quote it in YAML: an unquoted yes/no is a boolean and is rejected here.
    positive_class: StrictStr | StrictInt
    excluded_features: list[str] = Field(default_factory=list)
    split_ratios: SplitRatios = Field(default_factory=SplitRatios)
    split_seed: int = 42
    seeds: list[int] = Field(default_factory=lambda: [42, 43, 44, 45, 46], min_length=1)
    # One fill level by default. In results/cross_dataset.csv (4 datasets, SMOTENC and CTGAN)
    # 10% was the best of 10, 25 and 50% for 7 of 8 dataset and arm pairs, and the loss against
    # the cleaned baseline grew with every step up. List several levels to sweep them.
    fill_levels: list[FillLevel] = Field(default_factory=lambda: [0.10], min_length=1)
    quality_gate: QualityThresholds = Field(default_factory=QualityThresholds)
    sanity_rules: list[str] = Field(default_factory=list)
    max_attempts: int = Field(3, ge=1)
    max_stage_retries: int = Field(1, ge=0)
    gan_bounds: dict[str, tuple[float, float]] = Field(default_factory=dict)
    candidate_models: list[str] = Field(
        default_factory=lambda: [
            "Gradient boosting",
            "LightGBM",
            "CatBoost",
            "Stacking ensemble",
        ],
        min_length=1,
    )
    min_rows: int = Field(2000, ge=1)
    calibrate: bool = False
    training_protocol_path: Path | None = None
    training_cv_folds: int = Field(0, ge=0)
    training_cv_repeats: int = Field(3, ge=1)

    @field_validator("training_cv_folds")
    @classmethod
    def _cv_folds(cls, value: int) -> int:
        if value == 1:
            raise ValueError("training_cv_folds must be 0 (off) or at least 2")
        return value

    @field_validator("seeds", "candidate_models")
    @classmethod
    def _unique(cls, value: list) -> list:
        if len(set(value)) != len(value):
            raise ValueError("values must be unique")
        return value

    @field_validator("fill_levels")
    @classmethod
    def _increasing(cls, value: list[float]) -> list[float]:
        if any(a >= b for a, b in pairwise(value)):
            raise ValueError("fill levels must be strictly increasing")
        return value

    @field_validator("gan_bounds")
    @classmethod
    def _bounds_ordered(cls, value: dict[str, tuple[float, float]]) -> dict:
        bad = [name for name, (low, high) in value.items() if low > high]
        if bad:
            raise ValueError(f"lower bound above upper bound for: {', '.join(bad)}")
        return value

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)

    @classmethod
    def from_yaml(cls, text: str) -> Self:
        return cls.model_validate(yaml.safe_load(text))

    @classmethod
    def from_file(cls, path: Path) -> Self:
        return cls.from_yaml(Path(path).read_text())


# --- outcomes ---------------------------------------------------------------------------------


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, list | dict) and not value)


def _collect_paths(value: object) -> list[Path]:
    if isinstance(value, Path):
        return [value]
    if isinstance(value, BaseModel):
        children = [getattr(value, name) for name in type(value).model_fields]
    elif isinstance(value, dict):
        children = list(value.values())
    elif isinstance(value, list | tuple):
        children = list(value)
    else:
        return []
    return [path for child in children for path in _collect_paths(child)]


class Outcome(Schema):
    """Fields shared by every agent output.

    FAILED and SKIPPED need a reason. SUCCESS needs every field in `required_on_success`; the
    payload is optional otherwise, so a failed stage can still return a valid object.
    """

    required_on_success: ClassVar[tuple[str, ...]] = ()

    status: Status
    warnings: list[str] = Field(default_factory=list)
    reason: str | None = None

    @model_validator(mode="after")
    def _check_status(self) -> Self:
        if self.status in (Status.FAILED, Status.SKIPPED) and not self.reason:
            raise ValueError(f"status {self.status} needs a reason")
        if self.status is Status.SUCCESS:
            missing = [n for n in self.required_on_success if _is_missing(getattr(self, n))]
            if missing:
                raise ValueError(f"status SUCCESS but missing: {', '.join(missing)}")
        return self

    def artifact_paths(self) -> list[Path]:
        """Every file path in this object, nested ones included, without repeats."""
        return list(dict.fromkeys(_collect_paths(self)))


class ArmFiles(Schema):
    train_path: Path
    val_path: Path
    test_path: Path


class SyntheticSet(Schema):
    fill_level: FillLevel
    rows: int = Field(ge=1)
    path: Path


class SplitOutput(Outcome):
    """Orchestrator (S0) to Pre-processing."""

    required_on_success: ClassVar[tuple[str, ...]] = (
        "train_path",
        "val_path",
        "test_path",
        "target",
        "seed",
        "train_positives",
        "train_negatives",
    )

    train_path: Path | None = None
    val_path: Path | None = None
    test_path: Path | None = None
    target: str | None = None
    seed: int | None = None
    train_positives: int | None = Field(None, ge=0)
    train_negatives: int | None = Field(None, ge=0)


class TreatmentPlan(Schema):
    """Portable repair recipe; contains no fitted statistics or executable objects."""

    version: Literal[1] = 1
    dataset_config: dict[str, Any]
    treatments: dict[str, Literal["drop", "median", "mode", "model_missing", "keep_category"]]
    excluded: list[str]
    source_columns: list[str]


class PreprocessOutput(Outcome):
    """Pre-processing to Model Selection. raw and baseline are two file sets (see section 1)."""

    required_on_success: ClassVar[tuple[str, ...]] = (
        "raw",
        "baseline",
        "profile_path",
        "treatment_log_path",
        "transformer_path",
        "transformer_fit_rows",
        "baseline_train_positives",
        "baseline_train_negatives",
    )

    raw: ArmFiles | None = None
    baseline: ArmFiles | None = None
    profile_path: Path | None = None
    plot_paths: list[Path] = Field(default_factory=list)
    treatment_log_path: Path | None = None
    transformer_path: Path | None = None
    treatment_plan_path: Path | None = None
    treatment_plan_sha256: str | None = None
    transformer_fit_rows: int | None = Field(None, ge=1)
    # Class counts of the baseline training file, after any duplicate rows were dropped. Fill
    # levels are computed from these, not from SplitOutput, which counts the split before cleaning.
    baseline_train_positives: int | None = Field(None, ge=0)
    baseline_train_negatives: int | None = Field(None, ge=0)


class GeneratorOutput(Outcome):
    """Model Selection to Quality Evaluation."""

    required_on_success: ClassVar[tuple[str, ...]] = (
        "generator",
        "rationale",
        "checkpoint_path",
        "loss_history_path",
        "synthetic_sets",
    )

    generator: str | None = None
    rationale: str | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    checkpoint_path: Path | None = None
    loss_history_path: Path | None = None
    synthetic_sets: list[SyntheticSet] = Field(default_factory=list)

    @field_validator("synthetic_sets")
    @classmethod
    def _levels_unique(cls, value: list[SyntheticSet]) -> list[SyntheticSet]:
        levels = [s.fill_level for s in value]
        if len(set(levels)) != len(levels):
            raise ValueError("one synthetic set per fill level")
        return value


class CheckResult(Schema):
    name: str = Field(min_length=1)
    value: float
    threshold: float
    passed: bool
    hard: bool = True


class LevelReport(Schema):
    """Quality checks for the synthetic set at one fill level. Only hard checks decide."""

    fill_level: FillLevel
    checks: list[CheckResult] = Field(min_length=1)
    passed: bool

    @model_validator(mode="after")
    def _passed_matches_checks(self) -> Self:
        if self.passed != all(c.passed for c in self.checks if c.hard):
            raise ValueError("passed does not match the hard checks")
        return self


# What a diagnosis may suggest changing. Model selection turns these into settings inside the
# config bounds; the numbers are never chosen here.
Adjustment = Literal["more_epochs", "change_pac", "rebalance_categories", "change_sampling"]


class QualityDiagnosis(Schema):
    """Why a quality report failed and what to try next. It explains the verdict, never sets it."""

    failed_checks: list[str]
    columns: list[str]
    likely_cause: str
    adjustments: list[Adjustment]

    @model_validator(mode="after")
    def _filled_in(self) -> Self:
        if not (self.failed_checks and self.likely_cause.strip() and self.adjustments):
            raise ValueError("a diagnosis needs failed checks, a likely cause and an adjustment")
        return self


class QualityReport(Outcome):
    """Quality Evaluation to Orchestrator. A FAIL verdict is a result, not a failed stage.

    On a FAIL, `diagnosis` rides along to model selection in the retry request.
    """

    required_on_success: ClassVar[tuple[str, ...]] = ("verdict", "levels", "report_path")

    verdict: Verdict | None = None
    levels: list[LevelReport] = Field(default_factory=list)
    report_path: Path | None = None
    plot_paths: list[Path] = Field(default_factory=list)
    diagnosis: QualityDiagnosis | None = None

    @model_validator(mode="after")
    def _verdict_matches_levels(self) -> Self:
        levels = [lv.fill_level for lv in self.levels]
        if len(set(levels)) != len(levels):
            raise ValueError("one report per fill level")
        if self.verdict is not None:
            expected = Verdict.PASS if self.passing_levels else Verdict.FAIL
            if self.verdict is not expected:
                raise ValueError(f"verdict {self.verdict} does not match the levels")
        return self

    @property
    def passing_levels(self) -> list[float]:
        return [lv.fill_level for lv in self.levels if lv.passed]


class ModelRecord(Outcome):
    """One fitted model for an arm and seed. fill_level applies to smote and gan only."""

    required_on_success: ClassVar[tuple[str, ...]] = (
        "model_path",
        "val_predictions_path",
        "test_predictions_path",
    )

    arm: Arm
    model: str = Field(min_length=1)
    seed: int
    fill_level: FillLevel | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    model_path: Path | None = None
    val_predictions_path: Path | None = None
    test_predictions_path: Path | None = None
    threshold: float | None = Field(None, ge=0, le=1)
    protocol_id: str | None = None
    selected: bool = False
    validation_probability_range: float | None = Field(None, ge=0, le=1)
    training_rows: int | None = Field(None, ge=1)
    synthetic_rows: int = Field(0, ge=0)
    val_metrics: dict[str, float] = Field(default_factory=dict)
    test_metrics: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _fill_level_matches_arm(self) -> Self:
        augmented = self.arm in (Arm.SMOTE, Arm.GAN)
        if augmented and self.fill_level is None:
            raise ValueError(f"arm {self.arm} needs a fill level")
        if not augmented and self.fill_level is not None:
            raise ValueError(f"arm {self.arm} has no fill level")
        return self


class TrainingOutput(Outcome):
    """Model Training to Evaluation."""

    required_on_success: ClassVar[tuple[str, ...]] = ("records",)

    records: list[ModelRecord] = Field(default_factory=list)
    metrics_path: Path | None = None
    validation_summary_path: Path | None = None
    protocol_path: Path | None = None
    cv_metrics_path: Path | None = None
    cv_predictions_path: Path | None = None

    @model_validator(mode="after")
    def _records_consistent(self) -> Self:
        keys = [(r.arm, r.fill_level, r.model, r.seed) for r in self.records]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate record for the same arm, fill level, model and seed")
        if self.status is Status.SUCCESS and any(r.status is Status.FAILED for r in self.records):
            raise ValueError("status SUCCESS but a record FAILED")
        return self


class EvaluationOutput(Outcome):
    """Evaluation to Orchestrator."""

    required_on_success: ClassVar[tuple[str, ...]] = (
        "metrics_table_paths",
        "report_path",
        "numbers_verified",
    )

    metrics_table_paths: list[Path] = Field(default_factory=list)
    report_path: Path | None = None
    numbers_verified: bool | None = None

    @model_validator(mode="after")
    def _numbers_verified(self) -> Self:
        if self.status is Status.SUCCESS and self.numbers_verified is False:
            raise ValueError("status SUCCESS but the report numbers were not verified")
        return self


# --- agent inputs -----------------------------------------------------------------------------


def _need_success(**outcomes: Outcome) -> None:
    bad = [name for name, outcome in outcomes.items() if outcome.status is not Status.SUCCESS]
    if bad:
        raise ValueError(f"upstream output is not SUCCESS: {', '.join(bad)}")


class PreprocessInput(Schema):
    config: RunConfig
    split: SplitOutput

    @model_validator(mode="after")
    def _upstream_ok(self) -> Self:
        _need_success(split=self.split)
        return self


class ModelSelectionInput(Schema):
    """attempt counts quality-gate attempts from 1. A retry carries the report that failed."""

    config: RunConfig
    preprocess: PreprocessOutput
    attempt: int = Field(1, ge=1)
    previous_report: QualityReport | None = None

    @model_validator(mode="after")
    def _attempt_consistent(self) -> Self:
        _need_success(preprocess=self.preprocess)
        if self.attempt > self.config.max_attempts:
            raise ValueError("attempt is above max_attempts")
        if (self.attempt == 1) != (self.previous_report is None):
            raise ValueError("a retry needs the previous report, and a first attempt has none")
        if self.previous_report is not None and self.previous_report.verdict is not Verdict.FAIL:
            raise ValueError("previous_report must be a FAIL verdict")
        return self


class QualityInput(Schema):
    config: RunConfig
    preprocess: PreprocessOutput
    generator: GeneratorOutput

    @model_validator(mode="after")
    def _upstream_ok(self) -> Self:
        _need_success(preprocess=self.preprocess, generator=self.generator)
        return self


class TrainingRequest(Schema):
    """Orchestrator to Model Training. No accepted synthetic sets means the GAN arm is not run."""

    config: RunConfig
    treatment_plan_path: Path | None = None
    treatment_plan_sha256: str | None = None
    raw: ArmFiles
    baseline: ArmFiles
    accepted_synthetic: list[SyntheticSet] = Field(default_factory=list)

    @model_validator(mode="after")
    def _levels_valid(self) -> Self:
        levels = [s.fill_level for s in self.accepted_synthetic]
        if len(set(levels)) != len(levels):
            raise ValueError("one synthetic set per fill level")
        unknown = [lv for lv in levels if lv not in self.config.fill_levels]
        if unknown:
            raise ValueError(f"fill levels not in the config grid: {unknown}")
        return self


class EvaluationInput(Schema):
    """Files are passed as paths: state_path is the run's state.json."""

    config: RunConfig
    training: TrainingOutput
    quality: QualityReport | None = None
    treatment_log_path: Path
    decision_log_path: Path
    manifest_path: Path
    state_path: Path

    @model_validator(mode="after")
    def _upstream_ok(self) -> Self:
        _need_success(training=self.training)
        return self


class VolumeDecision(Schema):
    """Fill level chosen by the volume rule. 0.0 means real data only."""

    level: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1)


# What each stage takes and returns, for checking the handoffs.
STAGE_IO: dict[Stage, tuple[type[Schema], type[Outcome]]] = {
    Stage.SPLIT: (RunConfig, SplitOutput),
    Stage.PREPROCESSING: (PreprocessInput, PreprocessOutput),
    Stage.MODEL_SELECTION: (ModelSelectionInput, GeneratorOutput),
    Stage.QUALITY_CHECK: (QualityInput, QualityReport),
    Stage.TRAINING: (TrainingRequest, TrainingOutput),
    Stage.EVALUATION: (EvaluationInput, EvaluationOutput),
}
