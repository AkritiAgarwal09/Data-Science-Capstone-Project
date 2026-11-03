"""Training handoffs, frozen settings, provenance, and fold-local evaluation."""

import json
import shutil
from dataclasses import asdict

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from agentic_gan.agents import training as training
from agentic_gan.llm import FakeLLM, RuleOverrideError
from agentic_gan.orchestrator.decisions import read_decisions
from agentic_gan.orchestrator.graph import run_pipeline
from agentic_gan.orchestrator.manifest import create_run
from agentic_gan.orchestrator.stubs import default_agents
from agentic_gan.schemas import (
    Arm,
    ArmFiles,
    CheckResult,
    GeneratorOutput,
    LevelReport,
    PreprocessOutput,
    QualityReport,
    RunConfig,
    Stage,
    Status,
    SyntheticSet,
    TrainingRequest,
    Verdict,
    fill_rows,
)
from agentic_gan.tools import baselines as b
from agentic_gan.tools.preprocessing import Preprocessor, load_dataset, split_dataset
from agentic_gan.utils.datasets import DATASETS
from agentic_gan.utils.files import sha256_file


def tiny_suite(seed=42):
    return {"Logistic": LogisticRegression(random_state=seed), "Dummy": DummyClassifier()}


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "build_models", tiny_suite)
    return make_experiment(tmp_path)


def make_experiment(tmp_path, *, cv_folds=3, missing=False):
    source = tmp_path / "default.csv"
    rng = np.random.default_rng(11)
    pd.DataFrame(
        {
            "LIMIT_BAL": rng.normal(100, 30, 120),
            "SEX": [1, 2] * 60,
            "PAY_1": rng.integers(0, 3, 120),
            "y": [0, 0, 0, 1] * 30,
        }
    ).to_csv(source, index=False)
    if missing:
        data = pd.read_csv(source)
        data.loc[::6, "LIMIT_BAL"] = np.nan
        data.to_csv(source, index=False)
    cfg = RunConfig(
        dataset_name="default",
        dataset_path=source,
        target="y",
        positive_class=1,
        candidate_models=list(b.build_models()),
        seeds=[42, 43],
        fill_levels=[0.25, 0.5],
    )
    run = create_run(cfg, runs_dir=tmp_path / "runs")
    dataset = DATASETS["default"]
    data = load_dataset(dataset, source)
    ids = split_dataset(data, dataset)
    raw = {part: data.loc[index] for part, index in ids.items()}
    pre = Preprocessor(dataset).fit(raw["train"])
    clean = {part: pre.clean(frame) for part, frame in raw.items()}
    arms = {}
    for arm, tables in (("raw", raw), ("baseline", clean)):
        paths = {}
        for part, table in tables.items():
            path = run.subdir("data") / f"{arm}_{part}.csv"
            table.to_csv(path, index_label="source_row")
            paths[f"{'val' if part == 'validation' else part}_path"] = path
        arms[arm] = ArmFiles(**paths)
    protocol_path = tmp_path / "protocol.json"
    models = list(b.build_models())
    table = pd.DataFrame({"model": models, "threshold": [0.27] * len(models), "status": "success"})
    metadata = {
        "source_sha256": sha256_file(source),
        "config": asdict(dataset),
        "split_hashes": {part: b._digest(index) for part, index in ids.items()},
        "versions": b._versions(),
    }
    b.freeze_protocol(
        protocol_path,
        table,
        metadata,
        dataset="default",
        seed=42,
        cv_folds=cv_folds,
        y_val=raw["validation"].y,
        y_test=raw["test"].y,
    )
    cfg = cfg.updated(training_protocol_path=protocol_path)
    extras = []
    for level in cfg.fill_levels:
        n = fill_rows(level, int(clean["train"].y.sum()), int((clean["train"].y == 0).sum()))
        extra = clean["train"].query("y == 1").sample(n, replace=True, random_state=42).copy()
        extra["LIMIT_BAL"] += 0.123
        extra["is_synthetic"] = 1
        path = run.subdir("synthetic") / f"gan_{level}.csv"
        extra.to_csv(path, index=False)
        extras.append(SyntheticSet(fill_level=level, rows=n, path=path))
    request = TrainingRequest(config=cfg, **arms, accepted_synthetic=extras[:1])
    return request, run, raw, clean, extras


def test_all_combinations_predictions_and_manifest(experiment, monkeypatch):
    request, run, raw, _, _ = experiment
    before = sha256_file(request.config.training_protocol_path)
    monkeypatch.setattr(b, "select_f1_threshold", lambda *a: pytest.fail("Retuned thresholds"))
    result = training.run_training(request, run)
    assert result.status is Status.SUCCESS
    assert len(result.records) == 2 * 2 * (2 + 2 + 1)
    assert {r.arm for r in result.records} == set(Arm)
    assert {r.seed for r in result.records} == {42, 43}
    assert {r.threshold for r in result.records} == {0.27}
    assert sha256_file(request.config.training_protocol_path) == before
    metrics = pd.read_csv(result.metrics_path)
    assert metrics.columns[-6:].tolist() == list(training.METRICS)
    assert set(metrics.arm) == {"original", "cleaned", "smotenc", "ctgan"}
    assert len(metrics) == 2 * len(result.records)
    for record in result.records:
        assert record.status is Status.SUCCESS, record.reason
        artifact = joblib.load(record.model_path)
        estimator = artifact["estimator"]
        assert "source_row" not in artifact["features"]
        encoder = estimator.steps[0][1]
        assert encoder.fit_rows_ == len(raw["train"])
        for part, path in (
            ("validation", record.val_predictions_path),
            ("test", record.test_predictions_path),
        ):
            saved = pd.read_csv(path)
            assert saved.source_row.tolist() == raw[part].index.tolist()
            assert saved.is_synthetic.eq(0).all()
            assert np.array_equal(saved.prediction, saved.probability.ge(0.27).astype(int))
            calculated = b.metric_row(saved.y_true, saved.probability, threshold=0.27)
            reported = record.val_metrics if part == "validation" else record.test_metrics
            assert reported == pytest.approx(calculated)
    summary = pd.read_csv(result.validation_summary_path)
    assert summary.pr_auc_count.eq(2).all()
    assert not any("test" in c for c in summary)
    manifest = run.load_manifest().register(run, Stage.TRAINING, result)
    assert manifest.verify(run) == []
    assert set(result.artifact_paths()) == {run.root / item.path for item in manifest.entries}


@pytest.mark.parametrize("arm", ["raw", "baseline"])
@pytest.mark.parametrize("part", ["val_path", "test_path"])
@pytest.mark.parametrize("flag", [1, np.nan, 2])
def test_synthetic_or_ambiguous_holdout_rejected(experiment, arm, part, flag):
    request, run, *_ = experiment
    path = getattr(getattr(request, arm), part)
    data = pd.read_csv(path)
    data["is_synthetic"] = flag
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="synthetic rows|provenance"):
        training.run_training(request, run)
    assert not list(run.subdir("models").iterdir())


@pytest.mark.parametrize(
    "change,match",
    [
        ("missing_id", "source_row IDs"),
        ("duplicate_id", "duplicate source_row"),
        ("overlap", "share source rows"),
        ("reorder", "split_hashes"),
        ("feature", "raw validation differs"),
        ("clean_feature", "cleaned validation differs"),
    ],
)
def test_row_identity_and_content_checked(experiment, change, match):
    request, run, *_ = experiment
    path = request.baseline.val_path if change == "clean_feature" else request.raw.val_path
    data = pd.read_csv(path)
    if change == "missing_id":
        data = data.drop(columns="source_row")
    elif change == "duplicate_id":
        data.loc[1, "source_row"] = data.source_row.iloc[0]
    elif change == "overlap":
        data.loc[0, "source_row"] = pd.read_csv(request.raw.train_path).source_row.iloc[0]
    elif change == "reorder":
        data = data.iloc[::-1]
    else:
        data.loc[0, "LIMIT_BAL"] += 100
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match=match):
        training.run_training(request, run)


@pytest.mark.parametrize(
    "change,match",
    [
        ("count", "row count"),
        ("flag", "flagged"),
        ("target", "positive class"),
        ("id", "Synthetic schema"),
        ("nan", "nonmissing"),
    ],
)
def test_invalid_accepted_rows_rejected(experiment, change, match):
    request, run, *_ = experiment
    path = request.accepted_synthetic[0].path
    data = pd.read_csv(path)
    if change == "count":
        data = data.iloc[:-1]
    elif change == "flag":
        data.loc[0, "is_synthetic"] = 0
    elif change == "target":
        data.loc[0, "y"] = 0
    elif change == "id":
        data["source_row"] = range(len(data))
    else:
        data.loc[0, "LIMIT_BAL"] = np.nan
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match=match):
        training.run_training(request, run)


@pytest.mark.parametrize(
    "override",
    [
        {"calibrate": True},
        {"excluded_features": ["SEX"]},
        {"candidate_models": ["Dummy"]},
        {"positive_class": 0},
        {"split_seed": 43},
    ],
)
def test_configuration_drift_fails_before_fitting(experiment, override):
    request, run, *_ = experiment
    request = request.updated(config=request.config.updated(**override))
    with pytest.raises(ValueError):
        training.run_training(request, run)
    assert not list(run.subdir("models").iterdir())


def test_protocol_checksum_checked(experiment):
    request, run, *_ = experiment
    path = request.config.training_protocol_path
    protocol = json.loads(path.read_text())
    protocol["thresholds"]["Dummy"] = 0.9
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="checksum"):
        training.run_training(request, run)


def test_missing_protocol_explains_how_to_freeze_thresholds(experiment, tmp_path):
    request, run, *_ = experiment
    missing = tmp_path / "not_frozen.json"
    request = request.updated(config=request.config.updated(training_protocol_path=missing))
    with pytest.raises(FileNotFoundError, match="--dataset default --arm both --cv-folds 3"):
        training.run_training(request, run)


def test_training_discovers_validation_only_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "build_models", tiny_suite)
    monkeypatch.setattr(training, "ROOT", tmp_path)
    request, run, *_ = make_experiment(tmp_path, cv_folds=0)
    protocol_dir = tmp_path / "results"
    protocol_dir.mkdir()
    protocol_path = protocol_dir / "baseline_protocol_default_42_cv0.json"
    shutil.copyfile(request.config.training_protocol_path, protocol_path)
    request = request.updated(config=request.config.updated(training_protocol_path=None))

    result = training.run_training(request, run)

    assert result.status is Status.SUCCESS
    assert {record.threshold for record in result.records} == {0.27}
    assert json.loads(result.protocol_path.read_text())["cv_folds"] == 0


def test_no_accepted_gan_keeps_other_arms(experiment):
    request, run, *_ = experiment
    request = request.updated(accepted_synthetic=[])
    result = training.run_training(request, run)
    assert result.status is Status.SUCCESS
    assert {r.arm for r in result.records} == {Arm.RAW, Arm.BASELINE, Arm.SMOTE}


def test_encoder_ignores_synthetic_extremes_and_metadata(experiment):
    _, _, _, clean, _ = experiment
    real = clean["train"]
    extra = real.iloc[:3].copy()
    extra.LIMIT_BAL = 1e12
    X, y = training._model_data(real, DATASETS["default"], synthetic=extra)
    encoder = training.ArmEncoder(DATASETS["default"]).fit(X, y)
    assert encoder.fit_rows_ == len(real)
    scaler = encoder.encoder_.encoder.named_transformers_["numeric"]
    assert scaler.mean_[0] == pytest.approx(real.LIMIT_BAL.mean())
    assert np.isfinite(encoder.transform(X)).all()
    assert "is_synthetic" not in encoder.encoder_.columns


class BrokenClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        raise RuntimeError("deliberate fit failure")


def test_failed_fit_is_returned_and_other_models_continue(tmp_path, monkeypatch):
    monkeypatch.setattr(
        b,
        "build_models",
        lambda seed=42: {
            "Broken": BrokenClassifier(),
            "Dummy": DummyClassifier(),
        },
    )
    request, run, *_ = make_experiment(tmp_path)
    result = training.run_training(request, run)
    assert result.status is Status.FAILED
    assert "model fits failed" in result.reason
    for record in result.records:
        assert record.status is (Status.FAILED if record.model == "Broken" else Status.SUCCESS)
        if record.model == "Broken":
            assert "deliberate fit failure" in record.reason
            assert record.model_path is None


def test_repeated_cv_is_fold_local_and_excludes_test(experiment, monkeypatch):
    request, run, raw, _, _ = experiment
    request = request.updated(
        config=request.config.updated(training_cv_folds=2, training_cv_repeats=2, seeds=[42])
    )
    calls = []
    original = training.smotenc_rows

    def capture(train, *args, **kwargs):
        calls.append(len(train))
        return original(train, *args, **kwargs)

    monkeypatch.setattr(training, "smotenc_rows", capture)
    result = training.run_training(request, run)
    assert result.status is Status.SUCCESS
    cv = pd.read_csv(result.cv_metrics_path)
    assert "f1" not in cv and "threshold" not in cv
    assert set(cv.query("arm == 'ctgan'").status) == {"SKIPPED"}
    assert result.warnings
    predicted = pd.read_csv(result.cv_predictions_path)
    combined = set(raw["train"].index) | set(raw["validation"].index)
    assert set(predicted.source_row) == combined
    assert not set(predicted.source_row) & set(raw["test"].index)
    assert predicted.is_synthetic.eq(0).all()
    # Every real row is evaluated once per repeat per model/arm/level.
    counts = predicted.groupby(["arm", "fill_level", "model", "seed", "source_row"]).size()
    assert counts.eq(2).all()
    assert calls == [len(raw["train"])] + [len(combined) // 2] * 4


def test_cv_gan_factory_receives_only_fold_training_rows(experiment):
    request, run, raw, _, _ = experiment
    request = request.updated(
        config=request.config.updated(training_cv_folds=2, training_cv_repeats=1, seeds=[42])
    )
    seen = []

    def factory(train, count, seed):
        seen.append(set(train.index))
        out = train.query("y == 1").sample(count, replace=True, random_state=seed).copy()
        return out.assign(is_synthetic=1)

    result = training.run_training(request, run, cv_gan_factory=factory)
    assert result.status is Status.SUCCESS
    cv = pd.read_csv(result.cv_predictions_path)
    for fold, fit_ids in enumerate(seen):
        held = set(cv.loc[cv.arm.eq("ctgan") & cv.fold.eq(fold), "source_row"])
        assert not held & fit_ids
        assert not fit_ids & set(raw["test"].index)
        assert held | fit_ids == set(raw["train"].index) | set(raw["validation"].index)


def test_real_four_model_suite_smoke(tmp_path):
    request, run, *_ = make_experiment(tmp_path)
    request = request.updated(config=request.config.updated(seeds=[42], fill_levels=[0.25]))
    result = training.run_training(request, run)
    assert len(result.records) == 16
    assert result.status is Status.SUCCESS, [(r.model, r.reason) for r in result.records]
    assert {r.model for r in result.records} == set(b.build_models())


def test_default_graph_dispatches_to_training_agent(experiment, monkeypatch):
    from agentic_gan.orchestrator import graph

    request, _, *_ = experiment
    calls = []
    stub = default_agents()[Stage.TRAINING]

    def capture(incoming, run):
        calls.append(incoming)
        return stub(incoming, run)

    from agentic_gan.agents.preprocessing import preprocess
    from agentic_gan.llm import FakeLLM

    monkeypatch.setattr(
        graph, "preprocess", lambda req, run: preprocess(req, run, client=FakeLLM([]))
    )
    from agentic_gan.agents.evaluation import run_evaluation

    monkeypatch.setattr(
        graph, "run_evaluation", lambda req, run: run_evaluation(req, run, client=FakeLLM([]))
    )
    monkeypatch.setattr(graph, "run_training", capture)
    monkeypatch.setattr(graph, "run_quality", default_agents()[Stage.QUALITY_CHECK])
    state, _ = graph.run_pipeline(
        request.config, runs_dir=request.config.dataset_path.parent / "dispatch_runs"
    )
    assert state.is_complete
    assert len(calls) == 1
    assert isinstance(calls[0], TrainingRequest)


def test_repeated_run_is_deterministic(experiment):
    request, run, *_ = experiment
    request = request.updated(config=request.config.updated(seeds=[42]))
    first = training.run_training(request, run)
    before = pd.read_csv(first.metrics_path)
    predictions = {
        r.val_predictions_path: r.val_predictions_path.read_bytes() for r in first.records
    }
    second = training.run_training(request, run)
    pd.testing.assert_frame_equal(before, pd.read_csv(second.metrics_path))
    assert all(path.read_bytes() == content for path, content in predictions.items())


def test_orchestrator_passes_only_quality_accepted_levels(experiment):
    request, _, _, _, extras = experiment
    config = request.config.updated(seeds=[42])
    agents = default_agents()
    captured = []

    def preprocessing(incoming, run):
        arms = {}
        for arm in ("raw", "baseline"):
            fields = {}
            for field in ("train_path", "val_path", "test_path"):
                source = getattr(getattr(request, arm), field)
                path = run.subdir("data") / source.name
                shutil.copy2(source, path)
                fields[field] = path
            arms[arm] = ArmFiles(**fields)
        paths = {}
        for name in ("profile_path", "treatment_log_path", "transformer_path"):
            paths[name] = run.subdir("profile") / f"{name}.txt"
            paths[name].write_text("fixture")
        return PreprocessOutput(
            status=Status.SUCCESS,
            **arms,
            **paths,
            transformer_fit_rows=72,
            baseline_train_positives=18,
            baseline_train_negatives=54,
        )

    def generator(incoming, run):
        sets = []
        for item in extras:
            path = run.subdir("synthetic") / item.path.name
            shutil.copy2(item.path, path)
            sets.append(item.updated(path=path))
        checkpoint = run.subdir("synthetic") / "checkpoint.txt"
        checkpoint.write_text("fixture")
        return GeneratorOutput(
            status=Status.SUCCESS,
            generator="ctgan",
            rationale="fixture",
            checkpoint_path=checkpoint,
            loss_history_path=checkpoint,
            synthetic_sets=sets,
        )

    def quality(incoming, run):
        path = run.subdir("quality") / "gate.json"
        path.write_text("{}")
        return QualityReport(
            status=Status.SUCCESS,
            verdict=Verdict.PASS,
            report_path=path,
            levels=[
                LevelReport(
                    fill_level=lv,
                    passed=passed,
                    checks=[
                        CheckResult(name="fixture", value=int(passed), threshold=1, passed=passed)
                    ],
                )
                for lv, passed in ((0.25, True), (0.5, False))
            ],
        )

    def train(incoming, run):
        captured.append(incoming)
        return training.run_training(incoming, run)

    agents.update(
        {
            Stage.PREPROCESSING: preprocessing,
            Stage.MODEL_SELECTION: generator,
            Stage.QUALITY_CHECK: quality,
            Stage.TRAINING: train,
        }
    )
    state, run = run_pipeline(
        config, agents=agents, runs_dir=request.config.dataset_path.parent / "graph_runs"
    )
    assert state.is_complete
    assert [s.fill_level for s in captured[0].accepted_synthetic] == [0.25]
    metrics = pd.read_csv(run.subdir("models") / "metrics.csv")
    assert set(metrics.query("arm == 'ctgan'").fill_level) == {0.25}
    assert set(metrics.query("arm == 'smotenc'").fill_level) == {0.25, 0.5}
    assert run.load_manifest().verify(run) == []


# --- LLM run plan -------------------------------------------------------------------------

PROTOCOL = {"protocol_id": "p1", "thresholds": {"A": 0.3}, "models": {"A": {}}}
SIZES = {
    "train_rows": 90,
    "train_positives": 22,
    "validation_positives": 500,
    "test_positives": 500,
}


def _quality(tmp_path, passed_by_level):
    failed = CheckResult(name="worst_column_distance", value=0.2, threshold=0.1, passed=False)
    ok = CheckResult(name="worst_column_distance", value=0.05, threshold=0.1, passed=True)
    levels = [
        LevelReport(fill_level=lv, checks=[ok if p else failed], passed=p)
        for lv, p in passed_by_level.items()
    ]
    return QualityReport(
        status=Status.SUCCESS,
        verdict=Verdict.PASS if any(passed_by_level.values()) else Verdict.FAIL,
        levels=levels,
        report_path=tmp_path / "quality.json",
    )


def _plan(**changes):
    plan = {
        "smote_levels": [0.25],
        "gan_levels": [0.5],
        "seeds": [42],
        "repeated_cv": False,
        "reason": "Keep the run small",
    }
    return {**plan, **changes}


def _llm_decisions(run):
    decisions = read_decisions(run)
    return decisions[decisions.rule == "llm_step"]


@pytest.fixture
def planning(experiment, tmp_path):
    request, run, _, _, extras = experiment
    request = request.updated(accepted_synthetic=extras)  # both levels have a synthetic set
    quality = _quality(tmp_path, {0.25: False, 0.5: True})
    return request, run, quality


def test_plan_within_the_allowed_values_is_used_and_logged(planning):
    request, run, quality = planning
    fake = FakeLLM([_plan()])
    plan = training.plan_training(request, run, PROTOCOL, SIZES, quality, client=fake)
    assert plan.gan_levels == [0.5] and plan.seeds == [42] and plan.smote_levels == [0.25]
    assert len(fake.calls) == 1
    context = json.loads(fake.calls[0][1])
    assert context["allowed"]["gan_levels"] == [0.5]
    assert context["quality"]["levels"][0]["failed_hard_checks"] == ["worst_column_distance"]
    assert context["small_evaluation_warning"] is None
    assert _llm_decisions(run).iloc[0].outcome["gan_levels"] == [0.5]


def test_plan_that_trains_on_a_failed_level_is_rejected(planning):
    request, run, quality = planning
    bad = _plan(gan_levels=[0.25, 0.5])  # 0.25 failed the quality gate
    fake = FakeLLM([bad, bad])
    plan = training.plan_training(request, run, PROTOCOL, SIZES, quality, client=fake)
    assert plan == training.default_plan(request, [0.5])
    assert len(fake.calls) == 2  # one retry, then the fallback
    assert _llm_decisions(run).iloc[0].outcome == "LLM_FALLBACK"


@pytest.mark.parametrize(
    "bad",
    [
        _plan(seeds=[999]),
        _plan(smote_levels=[0.75]),
        _plan(seeds=[]),
        _plan(gan_levels=[0.5, 0.5]),
        {**_plan(), "models": ["Logistic"]},
        {**_plan(), "threshold": 0.9},
    ],
)
def test_invalid_plans_fall_back_to_the_default(planning, bad):
    request, run, quality = planning
    plan = training.plan_training(
        request, run, PROTOCOL, SIZES, quality, client=FakeLLM([bad, bad])
    )
    assert plan == training.default_plan(request, [0.5])
    assert _llm_decisions(run).iloc[0].outcome == "LLM_FALLBACK"


def test_llm_error_falls_back_to_the_default(planning):
    request, run, quality = planning
    fake = FakeLLM([TimeoutError("timed out"), TimeoutError("timed out")])
    plan = training.plan_training(request, run, PROTOCOL, SIZES, quality, client=fake)
    assert plan == training.default_plan(request, [0.5])
    assert _llm_decisions(run).iloc[0].outcome == "LLM_FALLBACK"


def test_without_a_quality_report_every_accepted_level_is_allowed(planning):
    request, run, _ = planning
    plan = training.plan_training(
        request, run, PROTOCOL, SIZES, None, client=FakeLLM([_plan(gan_levels=[0.25, 0.5])])
    )
    assert plan.gan_levels == [0.25, 0.5]


def test_small_evaluation_set_is_flagged_to_the_llm(planning):
    request, run, quality = planning
    fake = FakeLLM([_plan()])
    sizes = {**SIZES, "test_positives": 30}
    training.plan_training(request, run, PROTOCOL, sizes, quality, client=fake)
    warning = json.loads(fake.calls[0][1])["small_evaluation_warning"]
    assert "test" in warning and "validation" not in warning


def test_changed_rules_during_planning_are_rejected(planning):
    request, run, quality = planning

    class EditsTheConfig(FakeLLM):
        def complete(self, system_prompt, context, output_model):
            run.config_path.write_text(run.config_path.read_text().replace("- 43", "- 99", 1))
            return super().complete(system_prompt, context, output_model)

    with pytest.raises(RuleOverrideError):
        training.plan_training(
            request, run, PROTOCOL, SIZES, quality, client=EditsTheConfig([_plan()])
        )
    assert _llm_decisions(run).iloc[0].outcome == "RULE_OVERRIDE_REJECTED"


def test_configured_repeated_cv_cannot_be_dropped_by_the_plan(planning):
    request, run, quality = planning
    request = request.updated(config=request.config.updated(training_cv_folds=3))
    run.config_path.write_text(request.config.to_yaml())
    bad = _plan(repeated_cv=False)
    plan = training.plan_training(
        request, run, PROTOCOL, SIZES, quality, client=FakeLLM([bad, bad])
    )
    assert plan.repeated_cv is True  # the fallback keeps the configured CV
    assert _llm_decisions(run).iloc[0].outcome == "LLM_FALLBACK"


def _run_with_plan(request, run, quality, answers):
    fake = FakeLLM(answers)
    result = training.run_training(request, run, quality=quality, client=fake)
    return result, fake, pd.read_csv(result.metrics_path)


def test_run_trains_only_what_the_plan_allows(planning):
    request, run, quality = planning
    result, fake, metrics = _run_with_plan(request, run, quality, [_plan()])
    assert result.status is Status.SUCCESS
    assert len(fake.calls) == 1
    assert set(metrics.seed) == {42}
    assert set(metrics.query("arm == 'smotenc'").fill_level) == {0.25}
    assert set(metrics.query("arm == 'ctgan'").fill_level) == {0.5}
    assert {"original", "cleaned"} <= set(metrics.arm)
    saved = json.loads((run.subdir("models") / "training_plan.json").read_text())
    assert saved["gan_levels"] == [0.5] and saved["seeds"] == [42]


def test_run_never_trains_ctgan_on_a_failed_level(planning):
    request, run, quality = planning
    bad = _plan(gan_levels=[0.25, 0.5])  # 0.25 failed the gate
    result, _, metrics = _run_with_plan(request, run, quality, [bad, bad])
    assert result.status is Status.SUCCESS
    assert set(metrics.query("arm == 'ctgan'").fill_level) == {0.5}
    assert _llm_decisions(run).iloc[0].outcome == "LLM_FALLBACK"


def test_plan_without_synthetic_arms_still_runs_the_real_arms(planning):
    request, run, quality = planning
    plan = _plan(smote_levels=[], gan_levels=[])
    _, _, metrics = _run_with_plan(request, run, quality, [plan])
    assert set(metrics.arm) == {"original", "cleaned"}


def test_run_without_a_quality_report_makes_no_llm_call(planning):
    request, run, _ = planning
    fake = FakeLLM([])  # any call would fail
    result = training.run_training(request, run, client=fake)
    assert result.status is Status.SUCCESS and not fake.calls
    metrics = pd.read_csv(result.metrics_path)
    assert set(metrics.query("arm == 'ctgan'").fill_level) == {0.25, 0.5}
    decisions = read_decisions(run)
    assert (decisions.outcome.astype(str).str.contains("seeds")).any()


def test_run_reads_the_quality_agents_saved_report(planning):
    request, run, quality = planning
    folder = run.subdir("quality")
    for attempt in (1, 2):  # the newest report is used
        report = quality if attempt == 2 else _quality(folder, {0.25: True, 0.5: True})
        (folder / f"quality_report_attempt_{attempt}.json").write_text(report.model_dump_json())
    fake = FakeLLM([_plan()])
    result = training.run_training(request, run, client=fake)
    metrics = pd.read_csv(result.metrics_path)
    assert len(fake.calls) == 1
    assert json.loads(fake.calls[0][1])["allowed"]["gan_levels"] == [0.5]
    assert set(metrics.query("arm == 'ctgan'").fill_level) == {0.5}


def test_frozen_thresholds_are_the_same_whatever_the_plan(planning):
    request, run, quality = planning
    result, _, metrics = _run_with_plan(request, run, quality, [_plan()])
    protocol = json.loads(result.protocol_path.read_text())
    used = metrics.dropna(subset=["threshold"]).groupby("model").threshold.unique()
    assert all(len(values) == 1 for values in used)
    assert {m: v[0] for m, v in used.items()} == pytest.approx(protocol["thresholds"])
