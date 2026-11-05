"""Generator selection for Bank, Default, Polish, and IEEE-CIS, without fitting a GAN."""

import json
import os
from functools import partial
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError
from test_quality_agent import V1_ANSWER, V1_DRIVERS

import agentic_gan.orchestrator.graph as graph_module
from agentic_gan.agents.model_selection import (
    CTGAN_DEFAULTS,
    SETTING_NAMES,
    RetrySettings,
    SelectionExplanation,
    _retry_limits,
    _route_reasons,
    _rule_hash,
    _rule_retry,
    run_model_selection,
)
from agentic_gan.llm import FakeLLM, RuleOverrideError
from agentic_gan.orchestrator.decisions import read_decisions
from agentic_gan.orchestrator.manifest import create_run
from agentic_gan.orchestrator.stubs import default_agents
from agentic_gan.schemas import (
    ArmFiles,
    CheckResult,
    LevelReport,
    ModelSelectionInput,
    PreprocessOutput,
    QualityDiagnosis,
    QualityReport,
    RunConfig,
    Stage,
    Status,
    Verdict,
    fill_rows,
)
from agentic_gan.tools.quality_privacy import PRIVACY_CHECKS
from agentic_gan.tools.selection import (
    DATASET_STRUCTURE,
    DEFAULT_WIDE_TEMPORAL,
    STEP_ORDER,
    ColumnSpec,
    DataCharacteristics,
    DatasetStructure,
    select_generator,
    temporal_facts,
)


def _column(name, kind, missing=0):
    return ColumnSpec(name, kind, missing)


def _data(**overrides) -> DataCharacteristics:
    fields = {
        "dataset": "bank_marketing",
        "rows": 27126,
        "target": "y",
        "minority_count": 3173,
        "minority_rate": 3173 / 27126,
        "columns": (
            _column("age", "numeric"),
            _column("job", "categorical"),
            _column("y", "numeric"),
        ),
    }
    fields.update(overrides)
    return DataCharacteristics(**fields)


def _step(result, rule):
    found = [item for item in result.steps if item.rule == rule]
    assert len(found) == 1
    return found[0]


def _assert_dcgan_excluded(result):
    assert result.generator != "dcgan"
    assert [item.rule for item in result.steps] == list(STEP_ORDER)
    assert _step(result, "dcgan_excluded").outcome == "EXCLUDED"


def test_steps_stay_in_order_and_repeat():
    data = _data()
    assert select_generator(data) == select_generator(data)


def test_bank_mixed_table_selects_ctgan():
    result = select_generator(_data())
    _assert_dcgan_excluded(result)
    assert result.status == "selected"
    assert result.generator == "ctgan"
    assert result.trainable is True
    assert result.fallback == "wgan_gp"
    assert result.comparator == "wgan_gp"
    assert _step(result, "temporal_test").outcome == "FAIL"
    assert _step(result, "type_routing").outcome == "SELECTED"
    assert _step(result, "type_routing").inputs["generator"] == "ctgan"


def test_bank_duration_stops_at_leakage():
    result = select_generator(_data(leaky_columns=("duration",)))
    _assert_dcgan_excluded(result)
    assert result.status == "skipped"
    assert result.generator is None
    assert _step(result, "leakage_gate").outcome == "FAIL"
    assert _step(result, "temporal_test").outcome == "NOT_REACHED"


def test_too_few_rows_stops_at_size():
    result = select_generator(_data(rows=1500, minority_count=200, minority_rate=200 / 1500))
    _assert_dcgan_excluded(result)
    assert result.status == "skipped"
    assert _step(result, "size_gate").outcome == "FAIL"
    assert _step(result, "horizon_gate").outcome == "NOT_REACHED"


def test_default_complete_block_selects_timegan():
    result = select_generator(_data(
        dataset="default",
        rows=18000,
        minority_count=4000,
        minority_rate=4000 / 18000,
        columns=(
            _column("SEX", "categorical"),
            _column("PAY_1", "categorical"),
            _column("BILL_AMT1", "numeric"),
            _column("y", "numeric"),
        ),
        entity_in_table=True,
        wide_steps=6,
        repeated_observation=True,
        time_index_ready=True,
    ))
    _assert_dcgan_excluded(result)
    assert result.generator == "timegan"
    assert result.trainable is False
    assert result.fallback == "ctgan"
    assert "monthly" in result.scope
    conditions = _step(result, "temporal_test").inputs
    assert conditions["entity_key"] == "PASS"
    assert conditions["time_index"] == "PASS"
    assert conditions["min_steps"] == "PASS"
    assert conditions["repeated_observation"] == "PASS"
    assert _step(result, "type_routing").outcome == "NOT_REACHED"


def test_default_broken_block_selects_ctgan():
    result = select_generator(_data(
        dataset="default",
        rows=18000,
        minority_count=4000,
        minority_rate=4000 / 18000,
        columns=(_column("SEX", "categorical"), _column("y", "numeric")),
        entity_in_table=True,
        wide_steps=None,
        repeated_observation=False,
        time_index_ready=False,
    ))
    _assert_dcgan_excluded(result)
    assert result.generator == "ctgan"
    assert result.trainable is True
    assert _step(result, "temporal_test").inputs["time_index"] == "FAIL"


def test_missing_monthly_column_clears_the_time_index():
    complete = temporal_facts("default", set(DEFAULT_WIDE_TEMPORAL))
    broken = temporal_facts("default", set(DEFAULT_WIDE_TEMPORAL) - {"PAY_1"})
    assert complete.time_index_ready is True
    assert complete.wide_steps == 6
    assert broken.time_index_ready is False
    assert broken.wide_steps is None
    assert broken.entity_in_table is True


def test_polish_pooled_horizons_stop():
    result = select_generator(_data(
        dataset="polish",
        rows=30000,
        target="bankrupt",
        minority_count=1500,
        minority_rate=1500 / 30000,
        columns=(_column("A1", "numeric"), _column("cohort", "categorical")),
        horizons_pooled=True,
    ))
    _assert_dcgan_excluded(result)
    assert result.status == "skipped"
    assert result.generator is None
    assert _step(result, "horizon_gate").outcome == "FAIL"
    assert _step(result, "missing_gate").outcome == "NOT_REACHED"


def test_polish_unrepaired_stops_at_missing():
    result = select_generator(_data(
        dataset="polish",
        rows=6000,
        target="bankrupt",
        minority_count=300,
        minority_rate=300 / 6000,
        columns=(_column("A1", "numeric", missing=40), _column("bankrupt", "numeric")),
    ))
    _assert_dcgan_excluded(result)
    assert result.status == "skipped"
    assert _step(result, "missing_gate").outcome == "FAIL"
    assert _step(result, "temporal_test").outcome == "NOT_REACHED"


def test_polish_indicators_select_ctgan():
    result = select_generator(_data(
        dataset="polish",
        rows=6000,
        target="bankrupt",
        minority_count=800,
        minority_rate=800 / 6000,
        columns=(
            _column("A1", "numeric"),
            _column("A1_missing", "binary"),
            _column("bankrupt", "numeric"),
        ),
    ))
    _assert_dcgan_excluded(result)
    assert result.generator == "ctgan"
    assert result.trainable is True
    assert result.comparator == "wgan_gp"
    assert _step(result, "temporal_test").outcome == "FAIL"


def test_polish_all_numeric_documents_wgan_fallback():
    result = select_generator(_data(
        dataset="polish",
        rows=6000,
        target="bankrupt",
        minority_count=800,
        minority_rate=800 / 6000,
        columns=(
            _column("A1", "numeric"),
            _column("A2", "numeric"),
            _column("bankrupt", "numeric"),
        ),
    ))
    _assert_dcgan_excluded(result)
    assert result.generator == "wgan_gp"
    assert result.trainable is False
    assert result.fallback == "ctgan"
    assert "CTGAN" in result.rationale
    assert _step(result, "fallback").outcome == "DOCUMENTED"


def test_ieee_as_measured_stops_at_missing_before_the_temporal_test():
    # The missing gate stops the procedure. Temporal failure is asserted on the repaired fixture.
    result = select_generator(_data(
        dataset="ieee_cis",
        rows=590_540,
        target="isFraud",
        minority_count=12_398,
        minority_rate=12_398 / 590_540,
        columns=(
            _column("TransactionDT", "numeric"),
            _column("ProductCD", "categorical", missing=1000),
            _column("isFraud", "numeric"),
        ),
    ))
    _assert_dcgan_excluded(result)
    assert result.status == "skipped"
    assert result.generator is None
    assert _step(result, "missing_gate").outcome == "FAIL"
    assert _step(result, "temporal_test").outcome == "NOT_REACHED"


def test_ieee_repaired_mixed_table_selects_ctgan_not_timegan():
    result = select_generator(_data(
        dataset="ieee_cis",
        rows=590_540,
        target="isFraud",
        minority_count=12_398,
        minority_rate=12_398 / 590_540,
        columns=(
            _column("TransactionDT", "numeric"),
            _column("TransactionAmt", "numeric"),
            _column("ProductCD", "categorical"),
            _column("isFraud", "numeric"),
        ),
    ))
    _assert_dcgan_excluded(result)
    assert result.generator == "ctgan"
    assert result.trainable is True
    temporal = _step(result, "temporal_test").inputs
    assert temporal["time_index"] == "FAIL"
    assert temporal["repeated_observation"] == "FAIL"
    structure: DatasetStructure = DATASET_STRUCTURE["ieee_cis"]
    assert "TransactionDT" in structure.non_sequence_time


def test_minority_below_500_forbids_a_class_only_generator_and_continues():
    result = select_generator(_data(rows=2000, minority_count=295, minority_rate=295 / 2000))
    _assert_dcgan_excluded(result)
    assert result.generator == "ctgan"
    assert _step(result, "minority_gate").outcome == "FORBIDDEN_CLASS_ONLY"
    assert _step(result, "type_routing").outcome == "SELECTED"


def test_image_data_selects_no_generator():
    result = select_generator(_data(tabular=False))
    assert result.status == "skipped"
    assert result.generator is None
    assert _step(result, "dcgan_excluded").outcome == "EXCLUDED"
    assert _step(result, "size_gate").outcome == "NOT_REACHED"


def _labels(rows, positives):
    return [1] * positives + [0] * (rows - positives)


def _config(tmp_path, **overrides):
    source = tmp_path / "source.csv"
    source.write_text("age,y\n1,1\n")
    fields = {
        "dataset_name": "bank_marketing",
        "dataset_path": source,
        "target": "y",
        "positive_class": "yes",
        "fill_levels": [0.25, 0.5],
    }
    fields.update(overrides)
    return RunConfig(**fields)


def _request(tmp_path, config, frame, positives, negatives, attempt=1, previous=None, run_id="sel"):
    run = create_run(config, run_id=run_id, runs_dir=tmp_path / "runs")
    train_path = tmp_path / f"{run_id}_train.csv"
    frame.to_csv(train_path, index=False)
    unread = tmp_path / f"{run_id}_unread.csv"
    unread.write_text("not-a-training-table\n")
    profile = tmp_path / f"{run_id}_profile.json"
    profile.write_text("{}\n")
    files = ArmFiles(
        train_path=train_path.resolve(),
        val_path=unread.resolve(),
        test_path=unread.resolve(),
    )
    preprocess = PreprocessOutput(
        status=Status.SUCCESS,
        raw=files,
        baseline=files,
        profile_path=profile.resolve(),
        treatment_log_path=profile.resolve(),
        transformer_path=profile.resolve(),
        transformer_fit_rows=max(len(frame), 1),
        baseline_train_positives=positives,
        baseline_train_negatives=negatives,
    )
    request = ModelSelectionInput(
        config=config,
        preprocess=preprocess,
        attempt=attempt,
        previous_report=previous,
    )
    return request, run, train_path.resolve()


def _bank_frame(rows=2000, positives=500, **extra):
    frame = pd.DataFrame({
        "source_row": range(rows),
        "age": range(rows),
        "job": ["admin"] * rows,
        "y": _labels(rows, positives),
    })
    for name, values in extra.items():
        frame[name] = values
    return frame


class _Gan:
    def __init__(self):
        self.trains = []
        self.samples = []

    def train(self, train_df, target_col, **kwargs):
        self.trains.append((train_df.copy(), kwargs))
        log_path = kwargs.get("log_path")
        if log_path is not None:
            Path(log_path).write_text("Generator Loss,Discriminator Loss\n0.1,0.2\n")
        return object()

    def save(self, model, path):
        Path(path).write_bytes(b"ctgan")

    def sample(self, model, n, target_value=1, seed=None, drop_duplicates_of=None, **kwargs):
        self.samples.append((n, target_value, list(drop_duplicates_of.columns)))
        frame = pd.DataFrame({name: [0] * n for name in drop_duplicates_of.columns})
        frame[drop_duplicates_of.columns[-1]] = target_value
        return frame


@pytest.fixture
def gan(monkeypatch):
    tool = _Gan()
    monkeypatch.setattr("agentic_gan.tools.ctgan.train_ctgan", tool.train)
    monkeypatch.setattr("agentic_gan.tools.ctgan.save_model", tool.save)
    monkeypatch.setattr("agentic_gan.tools.ctgan.sample_synthetic", tool.sample)
    return tool


def _reads(monkeypatch):
    seen = []
    real = pd.read_csv

    def track(path, *args, **kwargs):
        seen.append(Path(path).resolve())
        return real(path, *args, **kwargs)

    monkeypatch.setattr("agentic_gan.agents.model_selection.pd.read_csv", track)
    return seen


def _fail_report(*names, diagnosis=None):
    checks = [
        CheckResult(name=name, value=1.0, threshold=0.1, passed=False) for name in names
    ]
    assert all(check.hard for check in checks)
    return QualityReport(
        status=Status.SUCCESS,
        verdict=Verdict.FAIL,
        levels=[LevelReport(fill_level=0.25, passed=False, checks=checks)],
        report_path=Path("quality.json"),
        diagnosis=diagnosis,
    )


def test_bank_attempt_trains_ctgan_on_the_training_split_only(tmp_path, monkeypatch, gan):
    seen = _reads(monkeypatch)
    frame = _bank_frame()
    request, run, train_path = _request(tmp_path, _config(tmp_path), frame, 500, 1500)
    output = run_model_selection(request, run)
    assert output.status is Status.SUCCESS
    assert output.generator == "ctgan"
    assert output.synthetic_sets
    expected = [fill_rows(level, 500, 1500) for level in request.config.fill_levels]
    assert [item.rows for item in output.synthetic_sets] == expected
    assert [count for count, _value, _cols in gan.samples] == expected
    assert gan.samples[0][1] == 1
    trained_columns = list(gan.trains[0][0].columns)
    assert "source_row" not in trained_columns
    assert seen == [train_path]
    rules = set(read_decisions(run).rule)
    for rule in ("dcgan_excluded", "size_gate", "missing_gate", "leakage_gate", "minority_gate",
                 "temporal_test", "type_routing"):
        assert rule in rules


def test_duration_does_not_train(tmp_path, gan):
    frame = _bank_frame(duration=range(2000))
    request, run, _path = _request(tmp_path, _config(tmp_path), frame, 500, 1500)
    output = run_model_selection(request, run)
    assert output.status is Status.SKIPPED
    assert _logged(run, "leakage_gate") == "FAIL"
    assert gan.trains == []


def test_small_table_does_not_train(tmp_path, gan):
    frame = _bank_frame(rows=100, positives=20)
    request, run, _path = _request(tmp_path, _config(tmp_path), frame, 20, 80)
    output = run_model_selection(request, run)
    assert output.status is Status.SKIPPED
    assert _logged(run, "size_gate") == "FAIL"
    assert gan.trains == []


def test_missing_values_do_not_train(tmp_path, gan):
    frame = _bank_frame()
    frame.loc[0, "age"] = None
    request, run, _path = _request(tmp_path, _config(tmp_path), frame, 500, 1500)
    output = run_model_selection(request, run)
    assert output.status is Status.SKIPPED
    assert _logged(run, "missing_gate") == "FAIL"
    assert gan.trains == []


def test_default_selects_timegan_and_does_not_train(tmp_path, gan):
    rows, positives = 2000, 600
    frame = pd.DataFrame({name: range(rows) for name in DEFAULT_WIDE_TEMPORAL})
    frame["SEX"] = 1
    frame["y"] = _labels(rows, positives)
    config = _config(tmp_path, dataset_name="default", target="y", positive_class=1)
    request, run, _path = _request(tmp_path, config, frame, positives, rows - positives)
    output = run_model_selection(request, run)
    assert output.status is Status.SKIPPED
    assert output.generator == "timegan"
    assert "ctgan" in output.reason
    assert gan.trains == []


def test_constant_polish_cohort_does_not_make_ratios_mixed(tmp_path, gan):
    rows, positives = 2000, 600
    frame = pd.DataFrame({
        "A1": range(rows),
        "A2": range(rows),
        "cohort": ["3year"] * rows,
        "bankrupt": _labels(rows, positives),
    })
    config = _config(tmp_path, dataset_name="polish", target="bankrupt", positive_class=1)
    request, run, _path = _request(tmp_path, config, frame, positives, rows - positives)
    output = run_model_selection(request, run)
    assert output.status is Status.SKIPPED
    assert output.generator == "wgan_gp"
    assert "CTGAN" in output.rationale
    assert gan.trains == []


def test_pooled_polish_cohorts_do_not_train(tmp_path, gan):
    rows, positives = 2000, 600
    frame = pd.DataFrame({
        "A1": range(rows),
        "cohort": ["1year", "3year"] * (rows // 2),
        "bankrupt": _labels(rows, positives),
    })
    config = _config(tmp_path, dataset_name="polish", target="bankrupt", positive_class=1)
    request, run, _path = _request(tmp_path, config, frame, positives, rows - positives)
    output = run_model_selection(request, run)
    assert output.status is Status.SKIPPED
    assert output.generator is None
    assert _logged(run, "horizon_gate") == "FAIL"
    assert gan.trains == []


def test_retry_raises_epochs_inside_bounds_when_fidelity_fails(tmp_path, gan):
    frame = _bank_frame()
    config = _config(tmp_path, gan_bounds={"epochs": (100, 600)})
    first, run, _path = _request(tmp_path, config, frame, 500, 1500, run_id="first")
    second, _same, _path = _request(
        tmp_path, config, frame, 500, 1500, attempt=2,
        previous=_fail_report("column_distance"), run_id="second",
    )
    run_model_selection(first, run)
    output = run_model_selection(second, _same)
    assert output.status is Status.SUCCESS
    assert gan.trains[0][1]["epochs"] == 300
    assert gan.trains[1][1]["epochs"] == 400
    assert _logged(_same, "hyperparameter_retry") == "ADJUSTED"


def test_retry_without_bounds_keeps_the_defaults(tmp_path, gan):
    frame = _bank_frame()
    config = _config(tmp_path)
    request, run, _path = _request(
        tmp_path, config, frame, 500, 1500, attempt=2, previous=_fail_report("column_distance"),
    )
    output = run_model_selection(request, run)
    assert output.status is Status.SUCCESS
    assert gan.trains[0][1]["epochs"] == 300
    assert _logged(run, "retry_unchanged") == "UNCHANGED"


def test_privacy_failure_does_not_add_epochs(tmp_path, gan):
    frame = _bank_frame()
    config = _config(tmp_path, gan_bounds={"epochs": (100, 600)})
    request, run, _path = _request(
        tmp_path, config, frame, 500, 1500, attempt=2,
        previous=_fail_report("column_distance", "dcr_ratio"),
    )
    run_model_selection(request, run)
    assert gan.trains[0][1]["epochs"] == 300
    assert _logged(run, "retry_unchanged") == "UNCHANGED"


def test_exact_train_copies_does_not_add_epochs(tmp_path, gan):
    # The old substring match ("dcr", "duplicate", "privacy") missed this check name.
    config = _config(tmp_path, gan_bounds={"epochs": (100, 600)})
    request, run, _path = _request(
        tmp_path, config, _bank_frame(), 500, 1500, attempt=2,
        previous=_fail_report("column_distance", "exact_train_copies"),
    )
    run_model_selection(request, run)
    assert gan.trains[0][1]["epochs"] == 300
    assert _logged(run, "retry_unchanged") == "UNCHANGED"


@pytest.mark.parametrize("check", sorted(PRIVACY_CHECKS))
def test_every_privacy_check_name_blocks_more_epochs(tmp_path, gan, check):
    config = _config(tmp_path, gan_bounds={"epochs": (100, 600)})
    request, run, _path = _request(
        tmp_path, config, _bank_frame(), 500, 1500, attempt=2,
        previous=_fail_report("column_distance", check),
    )
    run_model_selection(request, run)
    assert gan.trains[0][1]["epochs"] == 300
    assert _logged(run, "retry_unchanged") == "UNCHANGED"


def test_third_attempt_documents_wgan_and_does_not_train(tmp_path, gan):
    frame = _bank_frame()
    request, run, _path = _request(
        tmp_path, _config(tmp_path), frame, 500, 1500, attempt=3,
        previous=_fail_report("column_distance"),
    )
    output = run_model_selection(request, run)
    assert output.status is Status.SKIPPED
    assert output.generator == "wgan_gp"
    assert gan.trains == []


def test_impossible_batch_bounds_do_not_train(tmp_path, gan):
    frame = _bank_frame()
    config = _config(tmp_path, gan_bounds={"batch_size": (10, 25)})
    request, run, _path = _request(tmp_path, config, frame, 500, 1500)
    output = run_model_selection(request, run)
    assert output.status is Status.SKIPPED
    assert gan.trains == []


def test_zero_class_gap_does_not_train(tmp_path, gan):
    frame = _bank_frame(rows=2000, positives=1000)
    request, run, _path = _request(tmp_path, _config(tmp_path), frame, 1000, 1000)
    output = run_model_selection(request, run)
    assert output.status is Status.SKIPPED
    assert "no rows" in output.reason
    assert gan.trains == []


def test_class_count_disagreement_raises(tmp_path, gan):
    frame = _bank_frame()
    request, run, _path = _request(tmp_path, _config(tmp_path), frame, 400, 1600)
    with pytest.raises(ValueError, match="disagree"):
        run_model_selection(request, run)


def _logged(run, rule):
    matched = read_decisions(run).query("rule == @rule")
    assert len(matched) >= 1
    return matched.outcome.iloc[-1]


# --- LLM step scaffolding: output models and the locked rule hash -----------------------------


def test_rule_hash_is_stable_for_equal_configs_and_moves_with_the_locked_fields(tmp_path):
    base = _config(tmp_path, gan_bounds={"epochs": (100, 600)})
    same = _config(tmp_path, gan_bounds={"epochs": (100, 600)})
    assert _rule_hash(base) == _rule_hash(same)
    assert _rule_hash(base) != _rule_hash(_config(tmp_path, min_rows=2500,
                                                  gan_bounds={"epochs": (100, 600)}))
    assert _rule_hash(base) != _rule_hash(_config(tmp_path, gan_bounds={"epochs": (100, 700)}))
    assert _rule_hash(base) != _rule_hash(_config(tmp_path, gan_bounds={"pac": (5, 10),
                                                                        "epochs": (100, 600)}))


def test_rule_hash_survives_the_config_yaml_round_trip(tmp_path):
    config = _config(tmp_path, min_rows=1500, gan_bounds={"epochs": (100, 600), "pac": (5, 10)})
    run = create_run(config, run_id="hash", runs_dir=tmp_path / "runs")
    assert _rule_hash(run.load_config()) == _rule_hash(config)


def test_selection_explanation_rejects_unknown_fields_and_empty_text():
    SelectionExplanation(generator="ctgan", explanation="Mixed column types, so CTGAN.")
    with pytest.raises(ValidationError):
        SelectionExplanation(generator="ctgan", explanation="")
    with pytest.raises(ValidationError):
        SelectionExplanation(generator="ctgan", explanation="ok", rule="type_routing")
    with pytest.raises(ValidationError):
        SelectionExplanation(explanation="no generator given")


def test_retry_settings_reject_unknown_fields_and_an_empty_reason():
    RetrySettings(epochs=400, pac=10, discriminator_steps=1, batch_size=500, reason="more epochs")
    with pytest.raises(ValidationError):
        RetrySettings(epochs=400, pac=10, discriminator_steps=1, batch_size=500, reason="")
    with pytest.raises(ValidationError):
        RetrySettings(
            epochs=400, pac=10, discriminator_steps=1, batch_size=500, reason="ok", seed=1
        )
    with pytest.raises(ValidationError):
        RetrySettings(epochs=400, pac=10, discriminator_steps=1, reason="missing batch_size")


# --- Explanation step ----------------------------------------------------------------------------


def test_first_attempt_asks_the_llm_to_explain_and_logs_the_answer(tmp_path, gan):
    fake = FakeLLM([{"generator": "ctgan", "explanation": "Mixed columns, so CTGAN."}])
    request, run, _path = _request(tmp_path, _config(tmp_path), _bank_frame(), 500, 1500)
    output = run_model_selection(request, run, client=fake)
    assert len(fake.calls) == 1
    assert output.generator == "ctgan"
    steps = read_decisions(run).query("rule == 'llm_step' and agent == 'model_selection'")
    assert len(steps) == 1
    assert steps.outcome.iloc[0] == {
        "generator": "ctgan", "explanation": "Mixed columns, so CTGAN."
    }


def _llm_steps(run):
    return read_decisions(run).query("rule == 'llm_step' and agent == 'model_selection'")


EXPLAIN_ANSWER = {"generator": "ctgan", "explanation": "Mixed columns, so CTGAN."}


def test_explanation_does_not_change_the_generator_output(tmp_path, gan):
    fake = FakeLLM([EXPLAIN_ANSWER])
    explained, run, _path = _request(
        tmp_path, _config(tmp_path), _bank_frame(), 500, 1500, run_id="explained",
    )
    with_llm = run_model_selection(explained, run, client=fake)
    plain, other, _path = _request(
        tmp_path, _config(tmp_path), _bank_frame(), 500, 1500, run_id="plain",
    )
    without = run_model_selection(plain, other)
    assert len(fake.calls) == 1
    assert fake.calls[0][2] is SelectionExplanation
    assert _llm_steps(run).outcome.iloc[0] == EXPLAIN_ANSWER
    assert with_llm.generator == without.generator == "ctgan"
    assert with_llm.hyperparameters == without.hyperparameters
    assert [item.rows for item in with_llm.synthetic_sets] == [
        item.rows for item in without.synthetic_sets
    ]


def test_explanation_falls_back_when_the_llm_is_down(tmp_path, gan):
    fake = FakeLLM([RuntimeError("offline"), RuntimeError("offline")])
    request, run, _path = _request(tmp_path, _config(tmp_path), _bank_frame(), 500, 1500)
    output = run_model_selection(request, run, client=fake)
    assert len(fake.calls) == 2
    assert _logged(run, "llm_step") == "LLM_FALLBACK"
    assert output.generator == "ctgan"
    assert output.status is Status.SUCCESS


def test_an_explanation_that_names_another_generator_is_rejected(tmp_path, gan):
    wrong = {"generator": "wgan_gp", "explanation": "All numeric, so WGAN-GP."}
    fake = FakeLLM([wrong, wrong])
    request, run, _path = _request(tmp_path, _config(tmp_path), _bank_frame(), 500, 1500)
    output = run_model_selection(request, run, client=fake)
    assert len(fake.calls) == 2
    assert _logged(run, "llm_step") == "LLM_FALLBACK"
    assert output.generator == "ctgan"


def test_a_skipped_selection_is_still_explained(tmp_path, gan):
    fake = FakeLLM([{"generator": "none", "explanation": "Too few rows for a GAN."}])
    frame = _bank_frame(rows=100, positives=20)
    request, run, _path = _request(tmp_path, _config(tmp_path), frame, 20, 80)
    output = run_model_selection(request, run, client=fake)
    assert len(fake.calls) == 1
    assert fake.calls[0][2] is SelectionExplanation
    assert len(_llm_steps(run)) == 1
    assert _llm_steps(run).outcome.iloc[0]["generator"] == "none"
    assert output.status is Status.SKIPPED
    assert gan.trains == []


def test_a_retry_does_not_ask_for_an_explanation(tmp_path, gan):
    fake = FakeLLM([PROPOSAL])
    request, run, _path = _retry_request(tmp_path, {"epochs": (100, 600)}, "column_distance")
    run_model_selection(request, run, client=fake)
    assert len(fake.calls) == 1
    assert fake.calls[0][2] is RetrySettings


def test_timegan_explanation_must_echo_timegan(tmp_path, gan):
    rows, positives = 2000, 600
    frame = pd.DataFrame({name: range(rows) for name in DEFAULT_WIDE_TEMPORAL})
    frame["SEX"] = 1
    frame["y"] = _labels(rows, positives)
    config = _config(tmp_path, dataset_name="default", target="y", positive_class=1)
    request, run, _path = _request(tmp_path, config, frame, positives, rows - positives)
    fake = FakeLLM([{"generator": "timegan", "explanation": "The monthly block is a sequence."}])
    output = run_model_selection(request, run, client=fake)
    assert len(fake.calls) == 1
    assert fake.calls[0][2] is SelectionExplanation
    assert _llm_steps(run).outcome.iloc[0]["generator"] == "timegan"
    assert output.generator == "timegan"
    assert output.status is Status.SKIPPED
    assert gan.trains == []


# --- Retry limits: what the LLM proposer may move ---------------------------------------------

PINNED = {name: (value, value) for name, value in CTGAN_DEFAULTS.items()}


def test_retry_limits_are_none_when_nothing_is_bounded():
    assert _retry_limits(CTGAN_DEFAULTS, {}, privacy=False) is None
    assert _retry_limits(CTGAN_DEFAULTS, {}, privacy=True) is None


def test_retry_limits_open_only_the_bounded_setting():
    limits = _retry_limits(CTGAN_DEFAULTS, {"epochs": (100, 600)}, privacy=False)
    assert limits == {**PINNED, "epochs": (100, 600)}


def test_retry_limits_cap_epochs_at_the_current_value_after_a_privacy_failure():
    limits = _retry_limits(CTGAN_DEFAULTS, {"epochs": (100, 600)}, privacy=True)
    assert limits == {**PINNED, "epochs": (100, CTGAN_DEFAULTS["epochs"])}


def test_retry_limits_round_fractional_edges_inward_to_integers():
    limits = _retry_limits(CTGAN_DEFAULTS, {"epochs": (99.5, 600.4)}, privacy=False)
    assert limits["epochs"] == (100, 600)
    assert all(isinstance(edge, int) for edge in limits["epochs"])


def test_retry_limits_return_every_bounded_range():
    bounds = {
        "epochs": (100, 600),
        "pac": (5, 10),
        "discriminator_steps": (1, 3),
        "batch_size": (100, 1000),
    }
    assert _retry_limits(CTGAN_DEFAULTS, bounds, privacy=False) == bounds


def test_retry_limits_pin_a_setting_whose_bound_cannot_hold_it():
    # No integer between 10.2 and 10.8; and 300 epochs lies outside (400, 600).
    limits = _retry_limits(
        CTGAN_DEFAULTS, {"pac": (10.2, 10.8), "epochs": (400, 600), "batch_size": (100, 1000)},
        privacy=False,
    )
    assert limits["pac"] == (10, 10)
    assert limits["epochs"] == (300, 300)
    assert limits["batch_size"] == (100, 1000)


# --- Retry proposer ------------------------------------------------------------------------------

PROPOSAL = {"epochs": 500, "pac": 10, "discriminator_steps": 1, "batch_size": 500,
            "reason": "more training"}


def _retry_request(tmp_path, gan_bounds, *failed, diagnosis=None, run_id="sel"):
    config = _config(tmp_path, gan_bounds=gan_bounds)
    return _request(
        tmp_path, config, _bank_frame(), 500, 1500, attempt=2,
        previous=_fail_report(*failed, diagnosis=diagnosis), run_id=run_id,
    )


@pytest.mark.parametrize(
    ("gan_bounds", "failed"),
    [
        ({"epochs": (100, 600)}, ("column_distance",)),
        ({"epochs": (100, 600)}, ("column_distance", "dcr_ratio")),
        ({"epochs": (100, 300)}, ("column_distance",)),
        ({"epochs": (100, 350)}, ("column_distance",)),
        ({"epochs": (100, 600), "pac": (5, 10)}, ("column_distance",)),
    ],
)
def test_rule_retry_always_lies_inside_the_retry_limits(tmp_path, gan_bounds, failed):
    # The runtime validates the fallback against numeric_bounds, so this must hold.
    request, _run, _path = _retry_request(tmp_path, gan_bounds, *failed)
    rule_settings, _name, _reason = _rule_retry(CTGAN_DEFAULTS, request)
    privacy = bool(set(failed) & PRIVACY_CHECKS)
    limits = _retry_limits(CTGAN_DEFAULTS, gan_bounds, privacy)
    assert limits is not None
    for name, (low, high) in limits.items():
        assert low <= rule_settings[name] <= high


def test_llm_proposal_inside_the_limits_is_trained(tmp_path, gan):
    fake = FakeLLM([PROPOSAL])
    request, run, _path = _retry_request(tmp_path, {"epochs": (100, 600)}, "column_distance")
    output = run_model_selection(request, run, client=fake)
    assert len(fake.calls) == 1  # attempt 2: no explanation call, one proposal call
    assert output.status is Status.SUCCESS
    assert gan.trains[0][1]["epochs"] == 500
    assert _logged(run, "hyperparameter_retry") == "ADJUSTED"


def test_llm_proposal_outside_the_limits_falls_back_to_the_rule(tmp_path, gan):
    fake = FakeLLM([{**PROPOSAL, "epochs": 900}, {**PROPOSAL, "epochs": 900}])
    request, run, _path = _retry_request(tmp_path, {"epochs": (100, 600)}, "column_distance")
    run_model_selection(request, run, client=fake)
    assert len(fake.calls) == 2  # one retry on the invalid answer
    assert gan.trains[0][1]["epochs"] == 400
    assert _logged(run, "llm_step") == "LLM_FALLBACK"
    assert _logged(run, "hyperparameter_retry") == "ADJUSTED"


# --- Retry logging: one line per retry, with its source -----------------------------------------

RETRY_RULES = {"hyperparameter_retry", "retry_unchanged"}


def _retry_lines(run):
    decisions = read_decisions(run)
    return decisions[decisions.rule.isin(RETRY_RULES)]


def test_an_accepted_proposal_logs_one_retry_line_from_the_llm(tmp_path, gan):
    request, run, _path = _retry_request(tmp_path, {"epochs": (100, 600)}, "column_distance")
    run_model_selection(request, run, client=FakeLLM([PROPOSAL]))
    lines = _retry_lines(run)
    assert len(lines) == 1
    line = lines.iloc[0]
    assert line.rule == "hyperparameter_retry" and line.outcome == "ADJUSTED"
    assert line.inputs["source"] == "llm"
    assert line.inputs["proposed"] == {k: PROPOSAL[k] for k in CTGAN_DEFAULTS}
    assert line.inputs["applied"] == {**CTGAN_DEFAULTS, "epochs": 500}
    assert line.inputs["previous_settings"] == CTGAN_DEFAULTS
    assert line.inputs["limits"]["epochs"] == [100, 600]
    assert line.inputs["privacy_failed"] is False
    assert line.inputs["failed_checks"] == ["column_distance"]
    assert line.rationale == PROPOSAL["reason"]
    # The old one-setting keys are still there.
    assert (line.inputs["previous_epochs"], line.inputs["epochs"]) == (300, 500)
    assert line.inputs["bound"] == [100, 600]


def test_a_fallback_logs_one_retry_line_from_the_rule(tmp_path, gan):
    bad = {**PROPOSAL, "epochs": 900}
    request, run, _path = _retry_request(tmp_path, {"epochs": (100, 600)}, "column_distance")
    run_model_selection(request, run, client=FakeLLM([bad, bad]))
    lines = _retry_lines(run)
    assert len(lines) == 1
    line = lines.iloc[0]
    assert line.rule == "hyperparameter_retry" and line.outcome == "ADJUSTED"
    assert line.inputs["source"] == "rule"
    assert line.inputs["proposed"] is None
    assert line.inputs["applied"] == {**CTGAN_DEFAULTS, "epochs": 400}
    assert "epochs move from 300 to 400" in line.rationale


def test_the_no_llm_path_also_logs_exactly_one_retry_line(tmp_path, gan):
    fake = FakeLLM([])
    request, run, _path = _retry_request(tmp_path, {}, "column_distance")
    run_model_selection(request, run, client=fake)
    assert fake.calls == []
    lines = _retry_lines(run)
    assert len(lines) == 1
    line = lines.iloc[0]
    assert line.rule == "retry_unchanged" and line.outcome == "UNCHANGED"
    assert line.inputs["source"] == "rule"
    assert line.inputs["limits"] is None
    assert line.rationale == "retry has no legal adjustment"


def test_unsupported_adjustments_are_reported_once_and_not_applied(tmp_path, gan):
    diagnosis = QualityDiagnosis(
        failed_checks=["column_distance"],
        columns=["job"],
        likely_cause="Job categories are off.",
        adjustments=["more_epochs", "rebalance_categories"],
    )
    report = _fail_report("column_distance").model_copy(update={"diagnosis": diagnosis})
    config = _config(tmp_path, gan_bounds={"epochs": (100, 600)})
    request, run, _path = _request(
        tmp_path, config, _bank_frame(), 500, 1500, attempt=2, previous=report,
    )
    run_model_selection(request, run, client=FakeLLM([PROPOSAL]))
    decisions = read_decisions(run)
    unsupported = decisions.query("rule == 'unsupported_adjustment'")
    assert len(unsupported) == 1
    assert unsupported.iloc[0].inputs["adjustment"] == "rebalance_categories"
    assert unsupported.iloc[0].outcome == "NOT_APPLIED"
    # It comes before the retry line, and more_epochs produced no such line.
    assert unsupported.index[0] < _retry_lines(run).index[0]
    assert gan.trains[0][1]["epochs"] == 500


# --- Retry proposer: full FakeLLM contract ------------------------------------------------------

ANSWER = {
    "epochs": 400,
    "pac": 10,
    "discriminator_steps": 1,
    "batch_size": 500,
    "reason": "more training",
}


def _answer(**overrides):
    return {**ANSWER, **overrides}


def _trained(gan):
    return gan.trains[-1][1]


def test_fail_report_builds_hard_checks_and_can_carry_a_diagnosis():
    assert CheckResult.model_fields["hard"].default is True
    diagnosis = QualityDiagnosis(
        failed_checks=["column_distance"],
        columns=["age"],
        likely_cause="Undertrained.",
        adjustments=["more_epochs"],
    )
    report = _fail_report("column_distance", "exact_train_copies", diagnosis=diagnosis)
    assert all(check.hard for check in report.levels[0].checks)
    assert report.diagnosis is diagnosis


def test_a_proposal_inside_the_bounds_is_used_and_logged_from_the_llm(tmp_path, gan):
    fake = FakeLLM([_answer(epochs=500)])
    request, run, _path = _retry_request(tmp_path, {"epochs": (100, 600)}, "column_distance")
    run_model_selection(request, run, client=fake)
    assert len(fake.calls) == 1
    assert _trained(gan)["epochs"] == 500
    line = _retry_lines(run).iloc[0]
    assert len(_retry_lines(run)) == 1
    assert line.rule == "hyperparameter_retry"
    assert line.inputs["source"] == "llm"


def test_a_proposal_outside_the_bounds_is_rejected(tmp_path, gan):
    # The issue's required out-of-bounds case: the runtime rejects it, the rule's 400 is used.
    bad = _answer(epochs=900)
    fake = FakeLLM([bad, bad])
    request, run, _path = _retry_request(tmp_path, {"epochs": (100, 600)}, "column_distance")
    run_model_selection(request, run, client=fake)
    assert len(fake.calls) == 2
    assert _logged(run, "llm_step") == "LLM_FALLBACK"
    assert _trained(gan)["epochs"] == 400
    assert _retry_lines(run).iloc[0].inputs["source"] == "rule"


def test_changing_an_unbounded_setting_is_rejected(tmp_path, gan):
    moved_pac = _answer(pac=5)
    fake = FakeLLM([moved_pac, moved_pac])
    request, run, _path = _retry_request(tmp_path, {"epochs": (100, 600)}, "column_distance")
    run_model_selection(request, run, client=fake)
    assert len(fake.calls) == 2
    assert _logged(run, "llm_step") == "LLM_FALLBACK"
    assert _trained(gan)["pac"] == 10
    assert _trained(gan)["epochs"] == 400
    assert _retry_lines(run).iloc[0].inputs["source"] == "rule"


def test_a_non_divisible_batch_snaps_down_when_the_multiple_is_in_range(tmp_path, gan):
    # 500 // 8 * 8 = 496, which sits inside (480, 520).
    fake = FakeLLM([_answer(pac=8, batch_size=500)])
    request, run, _path = _retry_request(
        tmp_path,
        {"epochs": (100, 600), "pac": (5, 12), "batch_size": (480, 520)},
        "column_distance",
    )
    run_model_selection(request, run, client=fake)
    assert len(fake.calls) == 1
    assert _trained(gan)["pac"] == 8
    assert _trained(gan)["batch_size"] == 496
    assert _retry_lines(run).iloc[0].inputs["source"] == "llm"


def test_a_non_divisible_batch_snaps_up_when_the_round_down_is_below_the_bound(tmp_path, gan):
    # 496 is below 500, so the next multiple 504 is used.
    fake = FakeLLM([_answer(pac=8, batch_size=500)])
    request, run, _path = _retry_request(
        tmp_path,
        {"epochs": (100, 600), "pac": (5, 12), "batch_size": (500, 520)},
        "column_distance",
    )
    run_model_selection(request, run, client=fake)
    assert len(fake.calls) == 1
    assert _trained(gan)["batch_size"] == 504
    assert _retry_lines(run).iloc[0].inputs["source"] == "llm"


def test_an_impossible_batch_snap_is_rejected_and_logged(tmp_path, gan):
    # 496 < 500 and 504 > 503: no legal multiple inside the batch_size range.
    fake = FakeLLM([_answer(pac=8, batch_size=500)])
    request, run, _path = _retry_request(
        tmp_path,
        {"epochs": (100, 600), "pac": (5, 12), "batch_size": (500, 503)},
        "column_distance",
    )
    run_model_selection(request, run, client=fake)
    assert len(fake.calls) == 1
    assert _logged(run, "retry_rejected") == "REJECTED"
    assert _trained(gan)["pac"] == 10
    assert _trained(gan)["batch_size"] == 500
    assert _trained(gan)["epochs"] == 400
    assert _retry_lines(run).iloc[0].inputs["source"] == "rule"


@pytest.mark.parametrize("check", ["dcr_ratio", "exact_train_copies"])
def test_a_privacy_failure_rejects_raising_epochs(tmp_path, gan, check):
    raised = _answer(epochs=500)
    fake = FakeLLM([raised, raised])
    request, run, _path = _retry_request(
        tmp_path, {"epochs": (100, 600)}, "column_distance", check,
    )
    run_model_selection(request, run, client=fake)
    assert len(fake.calls) == 2
    assert _logged(run, "llm_step") == "LLM_FALLBACK"
    assert _trained(gan)["epochs"] == 300
    assert _retry_lines(run).iloc[0].inputs["source"] == "rule"


@pytest.mark.parametrize("check", ["dcr_ratio", "exact_train_copies"])
def test_a_privacy_failure_accepts_lowering_epochs(tmp_path, gan, check):
    fake = FakeLLM([_answer(epochs=200, reason="shorter training after copies")])
    request, run, _path = _retry_request(
        tmp_path, {"epochs": (100, 600)}, "column_distance", check,
    )
    run_model_selection(request, run, client=fake)
    assert len(fake.calls) == 1
    assert _trained(gan)["epochs"] == 200
    assert _retry_lines(run).iloc[0].inputs["source"] == "llm"


def test_empty_gan_bounds_make_no_llm_call_and_keep_the_defaults(tmp_path, gan):
    fake = FakeLLM([])
    request, run, _path = _retry_request(tmp_path, {}, "column_distance")
    run_model_selection(request, run, client=fake)
    assert len(fake.calls) == 0
    assert {name: _trained(gan)[name] for name in CTGAN_DEFAULTS} == CTGAN_DEFAULTS
    line = _retry_lines(run).iloc[0]
    assert len(_retry_lines(run)) == 1
    assert line.rule == "retry_unchanged"


def test_attempt_1_explains_and_attempt_2_proposes(tmp_path, gan):
    first_llm = FakeLLM([EXPLAIN_ANSWER])
    first, run, _path = _request(
        tmp_path, _config(tmp_path), _bank_frame(), 500, 1500, run_id="explain",
    )
    run_model_selection(first, run, client=first_llm)
    assert len(first_llm.calls) == 1
    assert first_llm.calls[0][2] is SelectionExplanation

    second_llm = FakeLLM([_answer(epochs=500)])
    second, retry_run, _path = _retry_request(
        tmp_path, {"epochs": (100, 600)}, "column_distance",
    )
    run_model_selection(second, retry_run, client=second_llm)
    assert len(second_llm.calls) == 1
    assert second_llm.calls[0][2] is RetrySettings


def test_change_sampling_is_the_unsupported_adjustment_that_gets_logged(tmp_path, gan):
    diagnosis = QualityDiagnosis(
        failed_checks=["column_distance"],
        columns=["job"],
        likely_cause="Job categories are off.",
        adjustments=["more_epochs", "change_sampling"],
    )
    fake = FakeLLM([_answer(epochs=500)])
    request, run, _path = _retry_request(
        tmp_path, {"epochs": (100, 600)}, "column_distance", diagnosis=diagnosis,
    )
    run_model_selection(request, run, client=fake)
    assert len(fake.calls) == 1
    unsupported = read_decisions(run).query("rule == 'unsupported_adjustment'")
    assert len(unsupported) == 1
    assert unsupported.iloc[0].inputs["adjustment"] == "change_sampling"


def test_attempt_2_writes_exactly_one_retry_line(tmp_path, gan):
    changed, run, _path = _retry_request(
        tmp_path, {"epochs": (100, 600)}, "column_distance", run_id="changed",
    )
    run_model_selection(changed, run, client=FakeLLM([_answer(epochs=500)]))
    lines = _retry_lines(run)
    assert len(lines) == 1
    assert lines.iloc[0].rule == "hyperparameter_retry"

    unchanged, other, _path = _retry_request(tmp_path, {}, "column_distance", run_id="unchanged")
    run_model_selection(unchanged, other, client=FakeLLM([]))
    idle = _retry_lines(other)
    assert len(idle) == 1
    assert idle.iloc[0].rule == "retry_unchanged"


# --- Unchosen routes: why TimeGAN / WGAN-GP / CTGAN lost ---------------------------------------


def test_bank_mixed_skips_timegan_and_wgan_gp_only():
    reasons = _route_reasons(select_generator(_data()))
    assert set(reasons) == {"timegan", "wgan_gp"}
    assert "TimeGAN is not selected" in reasons["timegan"]
    assert "CTGAN is selected" in reasons["wgan_gp"]


def test_default_complete_block_skips_ctgan_and_wgan_gp():
    reasons = _route_reasons(select_generator(_data(
        dataset="default",
        rows=18000,
        minority_count=4000,
        minority_rate=4000 / 18000,
        columns=(
            _column("SEX", "categorical"),
            _column("PAY_1", "categorical"),
            _column("BILL_AMT1", "numeric"),
            _column("y", "numeric"),
        ),
        entity_in_table=True,
        wide_steps=6,
        repeated_observation=True,
        time_index_ready=True,
    )))
    assert set(reasons) == {"ctgan", "wgan_gp"}
    assert reasons["ctgan"] == "TimeGAN was selected; CTGAN is its documented fallback"
    assert reasons["wgan_gp"] == "TimeGAN was selected, so type routing did not run"


def test_polish_all_numeric_skips_timegan_and_ctgan_from_type_routing():
    reasons = _route_reasons(select_generator(_data(
        dataset="polish",
        rows=6000,
        target="bankrupt",
        minority_count=800,
        minority_rate=800 / 6000,
        columns=(
            _column("A1", "numeric"),
            _column("A2", "numeric"),
            _column("bankrupt", "numeric"),
        ),
    )))
    assert set(reasons) == {"timegan", "ctgan"}
    assert "TimeGAN is not selected" in reasons["timegan"]
    assert "WGAN-GP is selected" in reasons["ctgan"]
    assert "CTGAN" in reasons["ctgan"]


def test_too_few_rows_skips_every_route_at_the_size_gate():
    reasons = _route_reasons(select_generator(
        _data(rows=1500, minority_count=200, minority_rate=200 / 1500)
    ))
    assert set(reasons) == {"timegan", "wgan_gp", "ctgan"}
    prefix = "Not evaluated: size_gate stopped selection."
    assert all(reason.startswith(prefix) for reason in reasons.values())
    assert "below min_rows" in next(iter(reasons.values()))


def test_non_tabular_skips_every_route_at_dcgan_excluded():
    reasons = _route_reasons(select_generator(_data(tabular=False)))
    assert set(reasons) == {"timegan", "wgan_gp", "ctgan"}
    assert all(
        reason.startswith("Not evaluated: dcgan_excluded stopped selection.")
        for reason in reasons.values()
    )


def test_route_skipped_is_logged_once_per_unchosen_route_on_attempt_1_only(tmp_path, gan):
    first, run, _path = _request(tmp_path, _config(tmp_path), _bank_frame(), 500, 1500)
    run_model_selection(first, run)
    first_routes = read_decisions(run).query("rule == 'route_skipped'")
    assert sorted(first_routes.inputs.map(lambda item: item["route"])) == ["timegan", "wgan_gp"]
    assert set(first_routes.outcome) == {"NOT_SELECTED"}
    assert (first_routes.inputs.map(lambda item: item["attempt"]) == 1).all()

    second, same, _path = _request(
        tmp_path, _config(tmp_path), _bank_frame(), 500, 1500, attempt=2,
        previous=_fail_report("column_distance"), run_id="second",
    )
    run_model_selection(second, same)
    second_routes = read_decisions(same).query("rule == 'route_skipped'")
    assert second_routes.empty


# --- Rule lock, grounding, precedence, live -----------------------------------------------------

STOP_GATES = ("size_gate", "horizon_gate", "missing_gate", "leakage_gate")
LIVE_BOUNDS = {
    "epochs": (100, 600),
    "pac": (5, 10),
    "discriminator_steps": (1, 3),
    "batch_size": (100, 1000),
}


class _ConfigEditingLLM:
    """Rewrites a locked config field before answering, the way the quality-agent test does."""

    def __init__(self, run, answer):
        self.run = run
        self.answer = answer

    def complete(self, system_prompt, context, output_model):
        text = self.run.config_path.read_text()
        self.run.config_path.write_text(text.replace("min_rows: 2000", "min_rows: 2500"))
        return self.answer


def test_changing_min_rows_during_the_explain_call_is_rejected(tmp_path, gan):
    request, run, _path = _request(tmp_path, _config(tmp_path), _bank_frame(), 500, 1500)
    with pytest.raises(RuleOverrideError):
        run_model_selection(request, run, client=_ConfigEditingLLM(run, EXPLAIN_ANSWER))
    assert "RULE_OVERRIDE_REJECTED" in read_decisions(run).outcome.tolist()
    assert gan.trains == []


def test_changing_min_rows_during_the_proposer_call_is_rejected(tmp_path, gan):
    request, run, _path = _retry_request(tmp_path, {"epochs": (100, 600)}, "column_distance")
    with pytest.raises(RuleOverrideError):
        run_model_selection(request, run, client=_ConfigEditingLLM(run, _answer(epochs=500)))
    assert "RULE_OVERRIDE_REJECTED" in read_decisions(run).outcome.tolist()
    assert gan.trains == []


def test_the_proposer_context_is_grounded_in_the_diagnosis_and_has_no_rows(tmp_path, gan):
    diagnosis = QualityDiagnosis(**V1_ANSWER)
    fake = FakeLLM([_answer(epochs=500)])
    request, run, _path = _retry_request(
        tmp_path,
        {"epochs": (100, 600)},
        "mean_column_distance",
        "worst_column_distance",
        "real_vs_synthetic_auc",
        diagnosis=diagnosis,
    )
    run_model_selection(request, run, client=fake)
    assert len(fake.calls) == 1
    _prompt, context, model = fake.calls[0]
    assert model is RetrySettings
    assert "source_row" not in context
    for name in V1_DRIVERS:
        assert name in context
    for name in V1_ANSWER["failed_checks"]:
        assert name in context
    assert "AUC 0.86" in context
    payload = json.loads(context)
    assert set(payload) == {
        "attempt",
        "dataset",
        "failed_hard_checks",
        "diagnosis",
        "current_settings",
        "limits",
        "privacy_failed",
    }
    failed_names = [item["name"] for item in payload["failed_hard_checks"]]
    assert failed_names == [
        "mean_column_distance",
        "worst_column_distance",
        "real_vs_synthetic_auc",
    ]
    assert payload["diagnosis"]["likely_cause"] == V1_ANSWER["likely_cause"]
    assert set(payload["diagnosis"]["columns"]) == set(V1_DRIVERS)


def _multi_fail(**overrides):
    fields = {
        "rows": 100,
        "minority_count": 10,
        "minority_rate": 0.1,
        "columns": (_column("age", "numeric", missing=5), _column("y", "numeric")),
        "leaky_columns": ("duration",),
        "horizons_pooled": True,
    }
    fields.update(overrides)
    return _data(**fields)


def test_several_failing_gates_stop_at_size_and_do_not_override_each_other():
    result = select_generator(_multi_fail())
    assert result.status == "skipped"
    names = [step.rule for step in result.steps]
    assert names == list(STEP_ORDER)
    by_rule = dict(zip(names, result.steps, strict=True))
    assert by_rule["size_gate"].outcome == "FAIL"
    later = STEP_ORDER[STEP_ORDER.index("size_gate") + 1 :]
    assert all(by_rule[name].outcome == "NOT_REACHED" for name in later)
    for name in STOP_GATES:
        assert names.count(name) == 1


@pytest.mark.parametrize("cleared", range(len(STOP_GATES)))
def test_clearing_earlier_gates_lets_the_next_one_stop_selection(cleared):
    overrides = {}
    if cleared >= 1:
        overrides.update(rows=27126, minority_count=3173, minority_rate=3173 / 27126)
    if cleared >= 2:
        overrides["horizons_pooled"] = False
    if cleared >= 3:
        overrides["columns"] = (_column("age", "numeric"), _column("y", "numeric"))
    expected = STOP_GATES[cleared]
    result = select_generator(_multi_fail(**overrides))
    names = [step.rule for step in result.steps]
    by_rule = dict(zip(names, result.steps, strict=True))
    assert result.status == "skipped"
    assert by_rule[expected].outcome == "FAIL"
    assert [name for name in STOP_GATES if by_rule[name].outcome == "FAIL"] == [expected]
    later = STEP_ORDER[STEP_ORDER.index(expected) + 1 :]
    assert all(by_rule[name].outcome == "NOT_REACHED" for name in later)
    for name in STOP_GATES:
        assert names.count(name) == 1


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("RUN_LIVE_LLM"), reason="set RUN_LIVE_LLM=1 to call Vertex")
def test_live_gemini_explains_the_rule_and_proposes_a_legal_retry(tmp_path, gan):
    """One real Vertex call per step; run with RUN_LIVE_LLM=1 and -m live."""
    first, run, _path = _request(
        tmp_path, _config(tmp_path), _bank_frame(), 500, 1500, run_id="live-explain",
    )
    explained = run_model_selection(first, run)
    explain_step = read_decisions(run).query("rule == 'llm_step'").outcome.iloc[0]
    assert explain_step != "LLM_FALLBACK", "the Vertex explain call failed; see decisions.jsonl"
    assert explained.generator == "ctgan"
    assert explain_step["generator"] == "ctgan"

    second, retry_run, _path = _retry_request(
        tmp_path, LIVE_BOUNDS, "column_distance", run_id="live-retry",
    )
    retried = run_model_selection(second, retry_run)
    retry_step = read_decisions(retry_run).query("rule == 'llm_step'").outcome.iloc[-1]
    assert retry_step != "LLM_FALLBACK", "the Vertex proposer call failed; see decisions.jsonl"
    settings = {name: retried.hyperparameters[name] for name in SETTING_NAMES}
    for name, (low, high) in LIVE_BOUNDS.items():
        assert low <= settings[name] <= high
    assert settings["batch_size"] % settings["pac"] == 0
    assert retried.generator == "ctgan"


def test_model_selection_in_the_graph_uses_the_fake_explanation(tmp_path, gan):
    frame = _bank_frame()
    positives, negatives = 500, 1500

    def preprocess(request, run):
        data = run.subdir("data")
        train_path = data / "baseline_train.csv"
        frame.to_csv(train_path, index=False)
        unread = data / "unread.csv"
        unread.write_text("not-a-training-table\n")
        files = ArmFiles(
            train_path=train_path.resolve(),
            val_path=unread.resolve(),
            test_path=unread.resolve(),
        )
        dummy = run.subdir("profile") / "dummy.json"
        dummy.write_text("{}\n")
        return PreprocessOutput(
            status=Status.SUCCESS,
            raw=files,
            baseline=files,
            profile_path=dummy.resolve(),
            treatment_log_path=dummy.resolve(),
            transformer_path=dummy.resolve(),
            transformer_fit_rows=len(frame),
            baseline_train_positives=positives,
            baseline_train_negatives=negatives,
        )

    fake = FakeLLM([EXPLAIN_ANSWER])
    agents = default_agents()
    agents[Stage.PREPROCESSING] = preprocess
    agents[Stage.MODEL_SELECTION] = partial(run_model_selection, client=fake)
    state, run = graph_module.run_pipeline(
        _config(tmp_path), runs_dir=tmp_path / "runs", agents=agents
    )
    assert state.is_complete
    assert state.attempts(Stage.MODEL_SELECTION) == 1
    assert len(fake.calls) == 1
    assert fake.calls[0][2] is SelectionExplanation
    steps = read_decisions(run).query("rule == 'llm_step' and agent == 'model_selection'")
    assert steps.outcome.iloc[0] == EXPLAIN_ANSWER
    assert gan.trains
