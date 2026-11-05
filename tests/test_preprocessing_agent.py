import json
import pickle

import pandas as pd
import pytest

from agentic_gan.agents.preprocessing import preprocess
from agentic_gan.llm import FakeLLM
from agentic_gan.orchestrator.decisions import read_decisions
from agentic_gan.orchestrator.manifest import create_run
from agentic_gan.schemas import PreprocessInput, RunConfig, SplitOutput, Status


def setup_run(tmp_path, name, train, hold):
    target = {"polish": "bankrupt", "ieee_cis": "isFraud"}.get(name, "y")
    source = tmp_path / "source.csv"
    train.to_csv(source, index=False)
    config = RunConfig(dataset_name=name, dataset_path=source, target=target, positive_class=1)
    run = create_run(config, runs_dir=tmp_path / "runs")
    paths = {}
    for part, frame in (("train", train), ("val", hold), ("test", hold)):
        path = run.subdir("data") / f"{part}.csv"
        frame.to_csv(path, index=False)
        paths[f"{part}_path"] = path
    split = SplitOutput(
        status=Status.SUCCESS,
        **paths,
        target=target,
        seed=42,
        train_positives=1,
        train_negatives=len(train) - 1,
    )
    return PreprocessInput(config=config, split=split), run


def test_invalid_choice_retries_then_falls_back_and_train_only_median(tmp_path):
    train = pd.DataFrame({"x": [None, *range(1, 40)], "bankrupt": [0, 1] * 20})
    hold = pd.DataFrame({"x": [None, 1000000], "bankrupt": [0, 1]})
    request, run = setup_run(tmp_path, "polish", train, hold)
    fake = FakeLLM([{"treatment": "delete_everything", "reason": "bad"}] * 2)
    result = preprocess(request, run, client=fake)
    assert len(fake.calls) == 2
    assert "LLM_FALLBACK" in read_decisions(run).outcome.tolist()
    assert pd.read_csv(result.baseline.val_path).x.iloc[0] == 20
    assert pd.isna(pd.read_csv(result.raw.val_path).x.iloc[0])
    profile = json.loads(result.profile_path.read_text())
    assert profile["rows"] == 40
    assert "skew" in profile["columns"]["x"]
    assert result.transformer_fit_rows == 40


@pytest.mark.parametrize("name,target", [("polish", "bankrupt"), ("default", "y")])
def test_model_missing_serializes_and_deduplicates_train_only(tmp_path, name, target):
    train = pd.DataFrame(
        {"x": [1.0, 2.0, None, 4.0, 1.0], "z": [2, 4, 6, 8, 2], target: [0, 1, 0, 1, 0]}
    )
    hold = pd.DataFrame({"x": [None, None], "z": [100, 100], target: [0, 0]})
    request, run = setup_run(tmp_path, name, train, hold)
    fake = FakeLLM([{"treatment": "model_missing", "reason": "Frequent gaps need a flag"}])
    result = preprocess(request, run, client=fake)
    assert result.transformer_fit_rows == 4
    assert len(pd.read_csv(result.raw.train_path)) == 5
    cleaned = pd.read_csv(result.baseline.val_path)
    assert len(cleaned) == 2
    assert cleaned.x__missing.eq(1).all()
    assert cleaned.x.between(1, 4).all()
    processor = pickle.loads(result.transformer_path.read_bytes())
    pd.testing.assert_frame_equal(processor.clean(hold), cleaned, check_dtype=False)
    assert len(processor.models) == 1


def test_ieee_threshold_and_flags_cannot_be_overridden(tmp_path):
    train = pd.DataFrame(
        {
            "TransactionID": range(10),
            "TransactionDT": range(10),
            "x": [None] * 3 + list(range(7)),
            "z": [None] * 2 + list(range(8)),
            "isFraud": [0, 1] * 5,
        }
    )
    request, run = setup_run(tmp_path, "ieee_cis", train, train)
    fake = FakeLLM([{"treatment": "retain", "reason": "Override"}] * 20)
    result = preprocess(request, run, client=fake)
    cleaned = pd.read_csv(result.baseline.train_path)
    assert "x" not in cleaned
    assert "z" in cleaned
    assert "x__missing" in cleaned
    assert "TransactionID" not in cleaned
    assert "TransactionDT" not in cleaned
    assert not cleaned.isna().any().any()


def test_bank_fixed_semantics_and_duration(tmp_path):
    train = pd.DataFrame(
        {
            "day": [1, 2, 3],
            "pdays": [-1, 3, 5],
            "duration": [2, 4, 6],
            "job": ["unknown", "admin", "admin"],
            "education": ["unknown"] * 3,
            "contact": ["unknown"] * 3,
            "poutcome": ["unknown"] * 3,
            "y": [0, 1, 0],
        }
    )
    request, run = setup_run(tmp_path, "bank_marketing", train, train)
    result = preprocess(request, run, client=FakeLLM([]))
    baseline = pd.read_csv(result.baseline.train_path)
    raw = pd.read_csv(result.raw.train_path)
    assert "duration" not in baseline and "duration" not in raw
    assert baseline.pdays.iloc[0] == -1
    assert baseline.job.iloc[0] == "unknown"
    assert "day" in raw
    assert "day_of_month" in baseline


def test_llm_failure_and_extra_fields_use_default(tmp_path):
    train = pd.DataFrame({"x": [None, 2, 3], "y": [0, 1, 0]})
    request, run = setup_run(tmp_path, "default", train, train)
    fake = FakeLLM(
        [
            RuntimeError("offline"),
            {"treatment": "model_missing", "reason": "ok", "max_missing": 1.0},
        ]
    )
    result = preprocess(request, run, client=fake)
    assert result.status == Status.SUCCESS
    assert "LLM_FALLBACK" in read_decisions(run).outcome.tolist()


def test_retry_accepts_valid_choice_and_preserves_provenance(tmp_path):
    train = pd.DataFrame(
        {"source_row": [10, 20, 30], "x": [1.0, None, 3.0], "z": [1, 2, 3], "y": [0, 1, 0]}
    )
    request, run = setup_run(tmp_path, "default", train, train)
    fake = FakeLLM(
        [
            {"treatment": "drop", "reason": "Fixed provenance"},
            {"treatment": "mode", "reason": "Invalid tier"},
            {"treatment": "model_missing", "reason": "Valid tier"},
        ]
    )
    result = preprocess(request, run, client=fake)
    assert len(fake.calls) == 3
    assert pd.read_csv(result.raw.train_path).source_row.tolist() == [10, 20, 30]
    assert pd.read_csv(result.baseline.train_path).source_row.tolist() == [10, 20, 30]
    processor = pickle.loads(result.transformer_path.read_bytes())
    assert processor.models["x"][0] == ["z"]
    assert "source_row" not in processor.columns


def test_changed_rules_are_rejected(tmp_path):
    from agentic_gan.llm import RuleOverrideError

    train = pd.DataFrame({"x": [1.0, None, 3.0], "y": [0, 1, 0]})
    request, run = setup_run(tmp_path, "default", train, train)

    class MutatingLLM:
        def complete(self, *args):
            changed = request.config.updated(excluded_features=["x"])
            run.config_path.write_text(changed.to_yaml())
            return {"treatment": "model_missing", "reason": "Changed rules"}

    with pytest.raises(RuleOverrideError):
        preprocess(request, run, client=MutatingLLM())
    assert "RULE_OVERRIDE_REJECTED" in read_decisions(run).outcome.tolist()
