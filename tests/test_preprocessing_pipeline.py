"""Real split -> fake-LLM repair -> training, including fold-local refitting."""

import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from test_training_agent import make_experiment, tiny_suite

from agentic_gan.agents import training
from agentic_gan.agents.preprocessing import preprocess
from agentic_gan.llm import FakeLLM
from agentic_gan.orchestrator import graph
from agentic_gan.orchestrator.manifest import create_run
from agentic_gan.orchestrator.split import split_for_training
from agentic_gan.schemas import (
    Arm,
    PreprocessInput,
    Stage,
    Status,
    SyntheticSet,
    TrainingRequest,
    fill_rows,
)
from agentic_gan.tools.preprocessing import TreatmentPreprocessor


def prepare(tmp_path, monkeypatch):
    monkeypatch.setattr(training.baselines, "build_models", tiny_suite)
    request, run, *_ = make_experiment(tmp_path, missing=True)
    config = request.config.updated(training_cv_folds=2, training_cv_repeats=1, seeds=[42])
    run = create_run(config, runs_dir=tmp_path / "integrated_runs")
    split = split_for_training(config, run)
    output = preprocess(PreprocessInput(config=config, split=split), run, client=FakeLLM([]))
    request = TrainingRequest(
        config=config,
        raw=output.raw,
        baseline=output.baseline,
        treatment_plan_path=output.treatment_plan_path,
        treatment_plan_sha256=output.treatment_plan_sha256,
    )
    return request, run, output


def test_plan_runs_all_arms_and_fold_local_cv(tmp_path, monkeypatch):
    request, run, output = prepare(tmp_path, monkeypatch)
    cleaned = pd.read_csv(output.baseline.train_path)
    assert "LIMIT_BAL__missing" in cleaned
    count = fill_rows(0.25, int(cleaned.y.sum()), int(cleaned.y.eq(0).sum()))
    extra = cleaned.query("y == 1").sample(count, replace=True, random_state=1)
    path = run.subdir("synthetic") / "accepted.csv"
    extra.drop(columns="source_row").assign(is_synthetic=1).to_csv(path, index=False)
    request = request.updated(
        accepted_synthetic=[SyntheticSet(fill_level=0.25, rows=count, path=path)]
    )
    fits = []
    original = TreatmentPreprocessor.fit

    def capture(self, frame):
        fits.append(frame.copy())
        return original(self, frame)

    monkeypatch.setattr(TreatmentPreprocessor, "fit", capture)
    result = training.run_training(request, run)
    assert result.status is Status.SUCCESS, [(r.arm, r.reason) for r in result.records]
    assert {r.arm for r in result.records} == set(Arm)
    # The combined 96-row CV population fits only 48-row fold training subsets.
    assert max(len(frame) for frame in fits) == len(cleaned) < 72
    assert any(len(frame) <= 48 for frame in fits)
    assert any(frame.LIMIT_BAL.isna().any() for frame in fits if len(frame) <= 48)
    for record in result.records:
        if record.arm is Arm.RAW:
            continue
        encoder = joblib.load(record.model_path)["estimator"].steps[0][1]
        assert isinstance(encoder.encoder_, TreatmentPreprocessor)
        assert encoder.fit_rows_ == len(cleaned)
        assert "LIMIT_BAL__missing" in encoder.encoder_.columns
    cv = pd.read_csv(result.cv_predictions_path)
    test_ids = set(pd.read_csv(output.raw.test_path).source_row)
    assert not set(cv.source_row) & test_ids
    assert run.load_manifest().register(run, Stage.PREPROCESSING, output).verify(run) == []


def test_default_graph_uses_real_split_repair_and_training(tmp_path, monkeypatch):
    request, _, _ = prepare(tmp_path, monkeypatch)
    observed = []

    def fake_preprocess(incoming, run):
        result = preprocess(incoming, run, client=FakeLLM([]))
        observed.append(result)
        return result

    from agentic_gan.agents.evaluation import run_evaluation

    monkeypatch.setattr(
        graph, "run_evaluation", lambda req, run: run_evaluation(req, run, client=FakeLLM([]))
    )
    monkeypatch.setattr(graph, "preprocess", fake_preprocess)
    state, run = graph.run_pipeline(request.config, runs_dir=tmp_path / "pipeline_runs")
    assert state.is_complete, run.decisions_path.read_text()
    assert len(observed) == 1
    assert (
        "Negative or inconclusive results" in (run.subdir("report") / "final_report.md").read_text()
    )
    assert run.load_manifest().verify(run) == []
    metrics = pd.read_csv(run.subdir("models") / "metrics.csv")
    assert set(metrics.arm) == {"original", "cleaned", "smotenc"}
    assert metrics.status.eq("SUCCESS").all()


@pytest.mark.parametrize("tamper", ["plan", "checksum", "cleaned", "raw"])
def test_plan_integrity_and_data_checks_remain_enforced(tmp_path, monkeypatch, tamper):
    request, run, _ = prepare(tmp_path, monkeypatch)
    if tamper == "plan":
        payload = json.loads(request.treatment_plan_path.read_text())
        payload["treatments"]["LIMIT_BAL"] = "drop"
        request.treatment_plan_path.write_text(json.dumps(payload))
    elif tamper == "checksum":
        request = request.updated(treatment_plan_sha256="0" * 64)
    else:
        files = request.raw if tamper == "raw" else request.baseline
        data = pd.read_csv(files.val_path)
        data.loc[0, "LIMIT_BAL"] = -123456
        data.to_csv(files.val_path, index=False)
    with pytest.raises(ValueError, match="checksum|differs from"):
        training.run_training(request, run)
    assert not list(run.subdir("models").iterdir())


def test_changed_dataset_rules_rejected_even_with_updated_checksum(tmp_path, monkeypatch):
    request, run, _ = prepare(tmp_path, monkeypatch)
    payload = json.loads(request.treatment_plan_path.read_text())
    payload["dataset_config"]["imputation"] = "mean"
    request.treatment_plan_path.write_text(json.dumps(payload))
    request = request.updated(
        treatment_plan_sha256=hashlib.sha256(request.treatment_plan_path.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="dataset rules changed"):
        training.run_training(request, run)


def test_stacking_refits_repairs_on_inner_real_rows(tmp_path, monkeypatch):
    request, run, output = prepare(tmp_path, monkeypatch)
    plan = training._load_plan(request, run)
    config = training.load_configs()["default"]
    raw = pd.read_csv(output.raw.train_path).drop(columns="source_row")
    extra = pd.read_csv(output.baseline.train_path).drop(columns="source_row").iloc[:6].copy()
    extra.LIMIT_BAL = 1e12
    X, y = training._model_data(raw, config, synthetic=extra)
    fits = []
    original = TreatmentPreprocessor.fit

    def capture(self, frame):
        fits.append(frame.copy())
        return original(self, frame)

    monkeypatch.setattr(TreatmentPreprocessor, "fit", capture)
    model = StackingClassifier(
        estimators=[("dummy", DummyClassifier()), ("logistic", LogisticRegression())],
        final_estimator=LogisticRegression(),
        cv=2,
    )
    model = training.baselines._with_preprocessing(model, training.ArmEncoder(config, plan=plan))
    model.fit(X, y)
    assert np.isfinite(model.predict_proba(X)).all()
    assert any(len(frame) < len(raw) for frame in fits)
    assert all(len(frame) <= len(raw) for frame in fits)
    assert all(frame.LIMIT_BAL.max() < 1e12 for frame in fits)
    assert all("LIMIT_BAL__missing" not in frame for frame in fits)


def run_dataset_pipeline(tmp_path, monkeypatch, dataset, source):
    from dataclasses import asdict

    from agentic_gan.schemas import RunConfig

    b = training.baselines
    monkeypatch.setattr(b, "build_models", lambda seed=42: {"Dummy": DummyClassifier()})
    path = tmp_path / "frozen.json"
    cfg = RunConfig(
        dataset_name=dataset.name,
        dataset_path=source,
        target=dataset.target,
        positive_class="yes" if dataset.name == "bank_marketing" else 1,
        candidate_models=["Dummy"],
        seeds=[42],
        fill_levels=[0.25],
        training_protocol_path=path,
    )
    run = create_run(cfg, runs_dir=tmp_path / "dataset_runs")
    split = split_for_training(cfg, run)
    tables = {
        part: pd.read_csv(getattr(split, field))
        for part, field in [
            ("train", "train_path"),
            ("validation", "val_path"),
            ("test", "test_path"),
        ]
    }
    labels = {}
    for part, table in tables.items():
        labels[part] = (
            table[dataset.target].map({"yes": 1, "no": 0})
            if (dataset.name == "bank_marketing")
            else table[dataset.target]
        )
    b.freeze_protocol(
        path,
        pd.DataFrame({"model": ["Dummy"], "threshold": [0.27], "status": ["success"]}),
        {
            "source_sha256": b.source_hash_for(dataset, source),
            "config": asdict(dataset),
            "split_hashes": {
                part: b._digest(table.source_row.tolist()) for part, table in tables.items()
            },
            "versions": b._versions(),
        },
        dataset=dataset.name,
        seed=42,
        cv_folds=0,
        y_val=labels["validation"],
        y_test=labels["test"],
    )
    output = preprocess(PreprocessInput(config=cfg, split=split), run, client=FakeLLM([]))
    result = training.run_training(
        TrainingRequest(
            config=cfg,
            raw=output.raw,
            baseline=output.baseline,
            treatment_plan_path=output.treatment_plan_path,
            treatment_plan_sha256=output.treatment_plan_sha256,
        ),
        run,
    )
    assert result.status is Status.SUCCESS, [(r.arm, r.reason) for r in result.records]
    assert {r.arm for r in result.records} == {Arm.RAW, Arm.BASELINE, Arm.SMOTE}
    return output, result


@pytest.mark.parametrize("name", ["bank_marketing", "polish"])
def test_dataset_loaders_feed_plan_training(tmp_path, monkeypatch, name):
    import zipfile

    from agentic_gan.utils.datasets import DATASETS

    dataset = DATASETS[name]
    if name == "bank_marketing":
        source = tmp_path / "bank.csv"
        pd.DataFrame(
            {
                "day": np.arange(120) % 30 + 1,
                "duration": np.arange(120),
                "pdays": [-1, 2, 3, 4] * 30,
                "job": ["unknown", "admin"] * 60,
                "education": ["unknown"] * 120,
                "contact": ["unknown"] * 120,
                "poutcome": ["unknown"] * 120,
                "balance": np.arange(120, dtype=float),
                "y": ["no", "no", "no", "yes"] * 30,
            }
        ).to_csv(source, index=False)
    else:
        source = tmp_path / "polish.zip"
        with zipfile.ZipFile(source, "w") as archive:
            for cohort in dataset.cohorts:
                header = "@relation polish\n" + "".join(
                    f"@attribute attr{i} numeric\n" for i in range(1, 65)
                )
                header += "@attribute class {0,1}\n@data\n"
                rows = []
                for i in range(24):
                    values = [
                        "?" if j == 1 and i % 6 == 0 else str(i + j / 10) for j in range(1, 65)
                    ]
                    rows.append(",".join([*values, str(int(i % 4 == 3))]))
                archive.writestr(cohort, header + "\n".join(rows))
    output, result = run_dataset_pipeline(tmp_path, monkeypatch, dataset, source)
    for record in result.records:
        if record.arm is Arm.RAW:
            continue
        encoder = joblib.load(record.model_path)["estimator"].steps[0][1].encoder_
        if name == "bank_marketing":
            assert "duration" not in encoder.columns
        else:
            assert "A1__missing" in encoder.columns
    assert not pd.read_csv(output.baseline.val_path).isna().any().any()
