"""Generator selection for Bank, Default, Polish, and IEEE-CIS, without fitting a GAN."""

from pathlib import Path

import pandas as pd
import pytest

from agentic_gan.agents.model_selection import run_model_selection
from agentic_gan.orchestrator.decisions import read_decisions
from agentic_gan.orchestrator.manifest import create_run
from agentic_gan.schemas import (
    ArmFiles,
    CheckResult,
    LevelReport,
    ModelSelectionInput,
    PreprocessOutput,
    QualityReport,
    RunConfig,
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


def _fail_report(*names):
    checks = [
        CheckResult(name=name, value=1.0, threshold=0.1, passed=False) for name in names
    ]
    return QualityReport(
        status=Status.SUCCESS,
        verdict=Verdict.FAIL,
        levels=[LevelReport(fill_level=0.25, passed=False, checks=checks)],
        report_path=Path("quality.json"),
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
