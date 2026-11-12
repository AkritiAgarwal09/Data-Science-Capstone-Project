"""Tests for the shared schemas: valid and invalid examples."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_gan.schemas import (
    STAGE_IO,
    STAGE_ORDER,
    Arm,
    ArmFiles,
    CheckResult,
    EvaluationInput,
    EvaluationOutput,
    GeneratorOutput,
    LevelReport,
    ModelRecord,
    ModelSelectionInput,
    Outcome,
    PreprocessInput,
    PreprocessOutput,
    QualityInput,
    QualityReport,
    RunConfig,
    SplitOutput,
    Stage,
    Status,
    SyntheticSet,
    TrainingOutput,
    TrainingRequest,
    Verdict,
    VolumeDecision,
    fill_rows,
)

# --- factories for valid objects --------------------------------------------------------------


def make_config(**overrides) -> RunConfig:
    fields = {
        "dataset_name": "bank_marketing",
        "dataset_path": Path("data/bank-full.csv"),
        "target": "y",
        "positive_class": "yes",
    }
    return RunConfig(**{**fields, **overrides})


def arm_files(name: str) -> ArmFiles:
    return ArmFiles(
        train_path=Path(f"data/{name}_train.csv"),
        val_path=Path(f"data/{name}_val.csv"),
        test_path=Path(f"data/{name}_test.csv"),
    )


def synthetic(level: float, rows: int = 100) -> SyntheticSet:
    return SyntheticSet(fill_level=level, rows=rows, path=Path(f"synthetic/fill_{level}.csv"))


def make_split(**overrides) -> SplitOutput:
    fields = {
        "status": Status.SUCCESS,
        "train_path": Path("data/train.csv"),
        "val_path": Path("data/val.csv"),
        "test_path": Path("data/test.csv"),
        "target": "y",
        "seed": 42,
        "train_positives": 3173,
        "train_negatives": 23953,
    }
    return SplitOutput(**{**fields, **overrides})


def make_preprocess(**overrides) -> PreprocessOutput:
    fields = {
        "status": Status.SUCCESS,
        "raw": arm_files("raw"),
        "baseline": arm_files("baseline"),
        "profile_path": Path("profile/profile.json"),
        "plot_paths": [Path("profile/missing.png")],
        "treatment_log_path": Path("profile/treatment_log.csv"),
        "transformer_path": Path("profile/transformer.joblib"),
        "transformer_fit_rows": 27126,
        "baseline_train_positives": 3173,
        "baseline_train_negatives": 23953,
    }
    return PreprocessOutput(**{**fields, **overrides})


def make_generator(**overrides) -> GeneratorOutput:
    fields = {
        "status": Status.SUCCESS,
        "generator": "ctgan",
        "rationale": "mixed types, no entity or time key, enough rows",
        "hyperparameters": {"epochs": 300},
        "checkpoint_path": Path("synthetic/ctgan.pkl"),
        "loss_history_path": Path("synthetic/loss.csv"),
        "synthetic_sets": [synthetic(0.10), synthetic(0.25)],
    }
    return GeneratorOutput(**{**fields, **overrides})


def check(name: str = "tstr_auc_gap", passed: bool = True, hard: bool = True) -> CheckResult:
    return CheckResult(name=name, value=0.02, threshold=0.05, passed=passed, hard=hard)


def level_report(level: float, passed: bool = True) -> LevelReport:
    return LevelReport(fill_level=level, checks=[check(passed=passed)], passed=passed)


def make_quality(passes: tuple[bool, ...] = (True, False), **overrides) -> QualityReport:
    grid = (0.10, 0.25, 0.50, 0.75, 1.00)
    fields = {
        "status": Status.SUCCESS,
        "verdict": Verdict.PASS if any(passes) else Verdict.FAIL,
        "levels": [level_report(lv, ok) for lv, ok in zip(grid, passes, strict=False)],
        "report_path": Path("quality/quality_report.json"),
    }
    return QualityReport(**{**fields, **overrides})


def record(arm: Arm = Arm.BASELINE, model: str = "logistic_regression", seed: int = 42, **kw):
    stem = f"models/{arm}_{model}_{seed}_{kw.get('fill_level', 0)}"
    fields = {
        "status": Status.SUCCESS,
        "arm": arm,
        "model": model,
        "seed": seed,
        "model_path": Path(f"{stem}.joblib"),
        "val_predictions_path": Path(f"{stem}_val.csv"),
        "test_predictions_path": Path(f"{stem}_test.csv"),
    }
    return ModelRecord(**{**fields, **kw})


def make_training(**overrides) -> TrainingOutput:
    fields = {
        "status": Status.SUCCESS,
        "records": [
            record(Arm.RAW),
            record(Arm.BASELINE),
            record(Arm.SMOTE, fill_level=0.10),
            record(Arm.GAN, fill_level=0.10),
        ],
    }
    return TrainingOutput(**{**fields, **overrides})


def make_evaluation(**overrides) -> EvaluationOutput:
    fields = {
        "status": Status.SUCCESS,
        "metrics_table_paths": [Path("report/metrics.csv")],
        "report_path": Path("report/final_report.md"),
        "numbers_verified": True,
    }
    return EvaluationOutput(**{**fields, **overrides})


VALID_OUTPUTS = [
    make_split,
    make_preprocess,
    make_generator,
    make_quality,
    make_training,
    make_evaluation,
]

# --- enums and helpers ------------------------------------------------------------------------


def test_status_has_exactly_the_four_values():
    assert {s.value for s in Status} == {"SUCCESS", "RETRY", "FAILED", "SKIPPED"}


def test_stage_order_is_the_pipeline_order():
    assert [s.value for s in STAGE_ORDER] == [
        "split",
        "preprocessing",
        "model_selection",
        "quality_check",
        "training",
        "evaluation",
    ]


def test_arms_are_named_not_lettered():
    assert [a.value for a in Arm] == ["raw", "baseline", "smote", "gan"]


def test_every_stage_has_an_input_and_an_output_type():
    assert set(STAGE_IO) == set(Stage)
    assert all(issubclass(output, Outcome) for _, output in STAGE_IO.values())


# Slide 8 of the 10/09 deck: rows added per fill level, from the training class counts.
SLIDE_8_GRIDS = {
    "default": (3982, 14018, [1004, 2509, 5018, 7527, 10036]),
    "bank": (3173, 23953, [2078, 5195, 10390, 15585, 20780]),
    "polish": (1255, 24788, [2353, 5883, 11766, 17650, 23533]),
    "fraud": (295, 170589, [17029, 42574, 85147, 127720, 170294]),
}


@pytest.mark.parametrize("name", SLIDE_8_GRIDS)
def test_fill_rows_reproduces_the_slide_grid(name):
    positives, negatives, expected = SLIDE_8_GRIDS[name]
    levels = [0.10, 0.25, 0.50, 0.75, 1.00]
    assert [fill_rows(lv, positives, negatives) for lv in levels] == expected


def test_fill_rows_is_zero_when_there_is_no_class_gap():
    assert fill_rows(0.5, positives=10, negatives=5) == 0


# --- RunConfig --------------------------------------------------------------------------------


def test_config_defaults():
    config = make_config()
    assert (config.split_ratios.train, config.split_ratios.val, config.split_ratios.test) == (
        0.6,
        0.2,
        0.2,
    )
    assert config.fill_levels == [0.10]
    assert config.max_attempts == 3
    assert len(config.seeds) == 5


def test_config_yaml_round_trip():
    config = make_config(
        positive_class=1,
        excluded_features=["duration"],
        fill_levels=[0.25, 0.5],
        gan_bounds={"epochs": (100, 600)},
        sanity_rules=["age >= 18"],
    )
    assert RunConfig.from_yaml(config.to_yaml()) == config


def test_config_yaml_round_trip_keeps_a_yes_string_quoted():
    assert RunConfig.from_yaml(make_config().to_yaml()).positive_class == "yes"


def test_config_from_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(make_config().to_yaml())
    assert RunConfig.from_file(path) == make_config()


def test_config_rejects_an_unquoted_yes():
    text = "dataset_name: b\ndataset_path: d.csv\ntarget: y\npositive_class: yes\n"
    with pytest.raises(ValidationError):
        RunConfig.from_yaml(text)


@pytest.mark.parametrize(
    "overrides",
    [
        {"split_ratios": {"train": 0.5, "val": 0.2, "test": 0.2}},
        {"split_ratios": {"train": 0.0, "val": 0.5, "test": 0.5}},
        {"seeds": []},
        {"seeds": [1, 1]},
        {"fill_levels": []},
        {"fill_levels": [0.0, 0.5]},
        {"fill_levels": [0.5, 1.5]},
        {"fill_levels": [0.5, 0.25]},
        {"fill_levels": [0.25, 0.25]},
        {"positive_class": True},
        {"max_attempts": 0},
        {"max_stage_retries": -1},
        {"gan_bounds": {"epochs": (600, 100)}},
        {"candidate_models": []},
        {"candidate_models": ["xgboost", "xgboost"]},
        {"quality_gate": {"tstr_auc_gap_max": -0.1}},
        {"dataset_name": ""},
        {"min_rows": 0},
        {"retry_limit": 3},
    ],
    ids=lambda o: next(iter(o)) + ":" + str(next(iter(o.values())))[:20],
)
def test_config_rejects_invalid_values(overrides):
    with pytest.raises(ValidationError):
        make_config(**overrides)


def test_config_is_immutable():
    with pytest.raises(ValidationError):
        make_config().target = "other"


# --- status rules shared by every output ------------------------------------------------------


def test_failed_needs_a_reason_but_no_payload():
    assert SplitOutput(status=Status.FAILED, reason="schema check failed").seed is None
    with pytest.raises(ValidationError, match="needs a reason"):
        SplitOutput(status=Status.FAILED)


def test_skipped_needs_a_reason():
    with pytest.raises(ValidationError, match="needs a reason"):
        GeneratorOutput(status=Status.SKIPPED)
    out = GeneratorOutput(status=Status.SKIPPED, reason="dataset has fewer than min_rows rows")
    assert out.synthetic_sets == []


def test_retry_needs_nothing():
    assert SplitOutput(status=Status.RETRY).status is Status.RETRY


def test_success_needs_its_payload():
    with pytest.raises(ValidationError, match="missing: seed"):
        make_split(seed=None)


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        make_split(bogus=1)


def test_outputs_are_immutable():
    with pytest.raises(ValidationError):
        make_split().seed = 1


def test_negative_counts_are_rejected():
    with pytest.raises(ValidationError):
        make_split(train_positives=-1)


@pytest.mark.parametrize("factory", VALID_OUTPUTS)
def test_outputs_round_trip_through_json(factory):
    obj = factory()
    assert type(obj).model_validate_json(obj.model_dump_json()) == obj


# --- PreprocessOutput and GeneratorOutput -----------------------------------------------------


def test_preprocess_needs_both_arm_file_sets():
    with pytest.raises(ValidationError, match="baseline"):
        make_preprocess(baseline=None)


def test_arm_files_need_all_three_paths():
    with pytest.raises(ValidationError):
        ArmFiles(train_path=Path("a.csv"), val_path=Path("b.csv"))


def test_preprocess_fit_rows_must_be_positive():
    with pytest.raises(ValidationError):
        make_preprocess(transformer_fit_rows=0)


@pytest.mark.parametrize("field", ["baseline_train_positives", "baseline_train_negatives"])
def test_preprocess_needs_the_baseline_class_counts(field):
    with pytest.raises(ValidationError, match=field):
        make_preprocess(**{field: None})


@pytest.mark.parametrize("field", ["baseline_train_positives", "baseline_train_negatives"])
def test_baseline_class_counts_cannot_be_negative(field):
    with pytest.raises(ValidationError):
        make_preprocess(**{field: -1})


def test_baseline_counts_can_differ_from_the_split_counts():
    # Cleaning may drop duplicate training rows, so the fill-level counts come from the baseline.
    pre = make_preprocess(baseline_train_positives=3150, baseline_train_negatives=23900)
    split = make_split()
    assert fill_rows(1.0, pre.baseline_train_positives, pre.baseline_train_negatives) == 20750
    assert fill_rows(1.0, split.train_positives, split.train_negatives) == 20780


def test_a_failed_preprocess_needs_no_counts():
    assert PreprocessOutput(status=Status.FAILED, reason="x").baseline_train_positives is None


def test_generator_needs_at_least_one_synthetic_set():
    with pytest.raises(ValidationError, match="synthetic_sets"):
        make_generator(synthetic_sets=[])


def test_generator_rejects_two_sets_at_one_level():
    with pytest.raises(ValidationError, match="one synthetic set per fill level"):
        make_generator(synthetic_sets=[synthetic(0.10), synthetic(0.10)])


@pytest.mark.parametrize("level", [0.0, 1.2, -0.1])
def test_synthetic_set_level_must_be_in_range(level):
    with pytest.raises(ValidationError):
        synthetic(level)


def test_synthetic_set_needs_rows():
    with pytest.raises(ValidationError):
        synthetic(0.1, rows=0)


# --- QualityReport ----------------------------------------------------------------------------


def test_quality_report_lists_passing_levels():
    report = make_quality(passes=(True, False, True))
    assert report.verdict is Verdict.PASS
    assert report.passing_levels == [0.10, 0.50]


def test_quality_report_with_no_passing_level_is_a_fail_verdict():
    report = make_quality(passes=(False, False))
    assert report.verdict is Verdict.FAIL
    assert report.passing_levels == []


def test_verdict_must_match_the_levels():
    with pytest.raises(ValidationError, match="does not match the levels"):
        make_quality(passes=(False, False), verdict=Verdict.PASS)
    with pytest.raises(ValidationError, match="does not match the levels"):
        make_quality(passes=(True, False), verdict=Verdict.FAIL)


def test_level_passed_must_match_its_hard_checks():
    with pytest.raises(ValidationError, match="hard checks"):
        LevelReport(fill_level=0.1, checks=[check(passed=False)], passed=True)


def test_a_failing_soft_check_does_not_fail_the_level():
    report = LevelReport(
        fill_level=0.1, checks=[check(), check("note", passed=False, hard=False)], passed=True
    )
    assert report.passed


def test_check_values_cannot_be_nan():
    with pytest.raises(ValidationError):
        CheckResult(name="ks", value=float("nan"), threshold=0.1, passed=True)


def test_quality_report_rejects_two_reports_for_one_level():
    with pytest.raises(ValidationError, match="one report per fill level"):
        make_quality(levels=[level_report(0.1), level_report(0.1)])


def test_a_crashed_quality_stage_needs_no_verdict():
    assert QualityReport(status=Status.FAILED, reason="timeout").verdict is None


# --- TrainingRequest, ModelRecord and TrainingOutput ------------------------------------------


def make_request(**overrides) -> TrainingRequest:
    fields = {"config": make_config(), "raw": arm_files("raw"), "baseline": arm_files("base")}
    return TrainingRequest(**{**fields, **overrides})


def test_training_request_without_synthetic_data_means_no_gan_arm():
    assert make_request().accepted_synthetic == []


def test_training_request_accepts_levels_from_the_grid():
    config = make_config(fill_levels=[0.10, 0.25, 0.50])
    request = make_request(config=config, accepted_synthetic=[synthetic(0.10), synthetic(0.50)])
    assert [s.fill_level for s in request.accepted_synthetic] == [0.10, 0.50]


def test_training_request_accepts_the_default_level():
    request = make_request(accepted_synthetic=[synthetic(0.10)])
    assert [s.fill_level for s in request.accepted_synthetic] == [0.10]


def test_training_request_rejects_a_level_outside_the_grid():
    with pytest.raises(ValidationError, match="not in the config grid"):
        make_request(accepted_synthetic=[synthetic(0.33)])


def test_training_request_rejects_duplicate_levels():
    with pytest.raises(ValidationError, match="one synthetic set per fill level"):
        make_request(accepted_synthetic=[synthetic(0.10), synthetic(0.10)])


@pytest.mark.parametrize("arm", [Arm.SMOTE, Arm.GAN])
def test_augmented_arms_need_a_fill_level(arm):
    with pytest.raises(ValidationError, match="needs a fill level"):
        record(arm)
    assert record(arm, fill_level=0.25).fill_level == 0.25


@pytest.mark.parametrize("arm", [Arm.RAW, Arm.BASELINE])
def test_raw_and_baseline_have_no_fill_level(arm):
    with pytest.raises(ValidationError, match="has no fill level"):
        record(arm, fill_level=0.25)


def test_a_record_needs_its_files_when_it_succeeds():
    with pytest.raises(ValidationError, match="model_path"):
        record(model_path=None)


def test_a_model_that_cannot_run_is_a_skipped_record():
    skipped = ModelRecord(
        status=Status.SKIPPED,
        arm=Arm.RAW,
        model="logistic_regression",
        seed=42,
        reason="model rejects NaN",
    )
    assert make_training(records=[record(Arm.BASELINE), skipped]).status is Status.SUCCESS


def test_training_output_rejects_duplicate_records():
    with pytest.raises(ValidationError, match="duplicate record"):
        make_training(records=[record(Arm.BASELINE), record(Arm.BASELINE)])


def test_same_model_at_two_fill_levels_is_not_a_duplicate():
    both = [record(Arm.GAN, fill_level=0.10), record(Arm.GAN, fill_level=0.25)]
    assert len(make_training(records=both).records) == 2


def test_training_cannot_succeed_with_a_failed_record():
    failed = ModelRecord(
        status=Status.FAILED, arm=Arm.BASELINE, model="xgboost", seed=1, reason="diverged"
    )
    with pytest.raises(ValidationError, match="a record FAILED"):
        make_training(records=[failed])


def test_training_needs_records_to_succeed():
    with pytest.raises(ValidationError, match="records"):
        make_training(records=[])


# --- EvaluationOutput and VolumeDecision ------------------------------------------------------


def test_evaluation_cannot_succeed_with_unverified_numbers():
    with pytest.raises(ValidationError, match="not verified"):
        make_evaluation(numbers_verified=False)


def test_evaluation_failure_is_valid_with_a_reason():
    out = make_evaluation(
        status=Status.FAILED, reason="number check failed", numbers_verified=False
    )
    assert out.status is Status.FAILED


def test_volume_decision_allows_real_data_only():
    assert VolumeDecision(level=0.0, reason="no reliable gain").level == 0.0


@pytest.mark.parametrize("kwargs", [{"level": 1.2, "reason": "x"}, {"level": 0.5, "reason": ""}])
def test_volume_decision_rejects_bad_values(kwargs):
    with pytest.raises(ValidationError):
        VolumeDecision(**kwargs)


# --- agent inputs -----------------------------------------------------------------------------


def test_preprocess_input_needs_a_successful_split():
    assert PreprocessInput(config=make_config(), split=make_split()).split.seed == 42
    failed = SplitOutput(status=Status.FAILED, reason="bad schema")
    with pytest.raises(ValidationError, match="not SUCCESS"):
        PreprocessInput(config=make_config(), split=failed)


def test_first_attempt_has_no_previous_report():
    first = ModelSelectionInput(config=make_config(), preprocess=make_preprocess())
    assert first.attempt == 1
    with pytest.raises(ValidationError, match="first attempt has none"):
        ModelSelectionInput(
            config=make_config(),
            preprocess=make_preprocess(),
            previous_report=make_quality(passes=(False,)),
        )


def test_a_retry_carries_the_failed_report():
    retry = ModelSelectionInput(
        config=make_config(),
        preprocess=make_preprocess(),
        attempt=2,
        previous_report=make_quality(passes=(False, False)),
    )
    assert retry.previous_report.verdict is Verdict.FAIL


def test_a_retry_without_a_report_is_rejected():
    with pytest.raises(ValidationError, match="needs the previous report"):
        ModelSelectionInput(config=make_config(), preprocess=make_preprocess(), attempt=2)


def test_a_retry_after_a_passing_report_is_rejected():
    with pytest.raises(ValidationError, match="must be a FAIL verdict"):
        ModelSelectionInput(
            config=make_config(),
            preprocess=make_preprocess(),
            attempt=2,
            previous_report=make_quality(passes=(True,)),
        )


def test_attempt_cannot_pass_the_retry_cap():
    with pytest.raises(ValidationError, match="above max_attempts"):
        ModelSelectionInput(
            config=make_config(max_attempts=2),
            preprocess=make_preprocess(),
            attempt=3,
            previous_report=make_quality(passes=(False,)),
        )


def test_quality_input_needs_a_successful_generator():
    skipped = GeneratorOutput(status=Status.SKIPPED, reason="too few rows")
    with pytest.raises(ValidationError, match="generator"):
        QualityInput(config=make_config(), preprocess=make_preprocess(), generator=skipped)


def test_evaluation_input_needs_successful_training_and_allows_no_quality_report():
    kwargs = {
        "config": make_config(),
        "treatment_log_path": Path("profile/treatment_log.csv"),
        "decision_log_path": Path("decisions.jsonl"),
        "manifest_path": Path("manifest.json"),
        "state_path": Path("state.json"),
    }
    assert EvaluationInput(training=make_training(), **kwargs).quality is None
    failed = TrainingOutput(status=Status.FAILED, reason="crash")
    with pytest.raises(ValidationError, match="training"):
        EvaluationInput(training=failed, **kwargs)


# --- artifact_paths ---------------------------------------------------------------------------


def test_artifact_paths_collects_nested_paths():
    paths = set(make_preprocess().artifact_paths())
    assert paths == {
        Path("data/raw_train.csv"),
        Path("data/raw_val.csv"),
        Path("data/raw_test.csv"),
        Path("data/baseline_train.csv"),
        Path("data/baseline_val.csv"),
        Path("data/baseline_test.csv"),
        Path("profile/profile.json"),
        Path("profile/missing.png"),
        Path("profile/treatment_log.csv"),
        Path("profile/transformer.joblib"),
    }


def test_artifact_paths_ignores_non_path_fields():
    generator = make_generator(hyperparameters={"checkpoint": "not/a/path.pkl"})
    assert Path("not/a/path.pkl") not in generator.artifact_paths()


def test_artifact_paths_has_no_repeats():
    same = Path("data/shared.csv")
    split = make_split(train_path=same, val_path=same, test_path=same)
    assert split.artifact_paths() == [same]


def test_a_skipped_record_adds_no_paths():
    skipped = ModelRecord(
        status=Status.SKIPPED, arm=Arm.RAW, model="xgboost", seed=1, reason="model rejects NaN"
    )
    training = make_training(records=[record(Arm.BASELINE), skipped])
    assert len(training.artifact_paths()) == 3


def test_a_failed_output_has_no_paths():
    assert SplitOutput(status=Status.FAILED, reason="x").artifact_paths() == []


# --- updated ----------------------------------------------------------------------------------


def test_updated_returns_a_changed_copy_and_keeps_the_original():
    split = make_split()
    changed = split.updated(seed=7)
    assert (changed.seed, split.seed) == (7, 42)
    assert changed.train_path == split.train_path


def test_updated_validates_the_result():
    with pytest.raises(ValidationError):
        make_split().updated(train_positives=-1)
    with pytest.raises(ValidationError, match="needs a reason"):
        make_split().updated(status=Status.FAILED)


def test_updated_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        make_split().updated(bogus=1)
