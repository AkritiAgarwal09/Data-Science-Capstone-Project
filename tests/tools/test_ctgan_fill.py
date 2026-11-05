import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentic_gan.tools.ctgan_fill import (
    FILL_LEVELS,
    FILL_NAMES,
    FillPaths,
    fill_paths,
    fill_sizes,
    load_training_frame,
    range_lines,
    rule_mask,
    run_fill,
    selection_lines,
    selection_record,
)
from agentic_gan.utils.ctgan_settings import load_ctgan_settings


def make_default_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A Default-shaped cleaned training table: 23 predictors, y, source_row."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.25).astype(int)
    frame = {
        "LIMIT_BAL": rng.integers(10_000, 500_000, n),
        "SEX": rng.choice([1, 2], n),
        "EDUCATION": rng.choice([1, 2, 3], n),
        "MARRIAGE": rng.choice([1, 2], n),
        "AGE": rng.integers(21, 70, n),
    }
    for month in range(1, 7):
        frame[f"PAY_{month}"] = rng.choice([-1, 0, 1, 2], n)
    for month in range(1, 7):
        frame[f"BILL_AMT{month}"] = rng.integers(-5_000, 200_000, n)
    for month in range(1, 7):
        frame[f"PAY_AMT{month}"] = rng.integers(0, 20_000, n)
    frame["y"] = y
    data = pd.DataFrame(frame)
    data.insert(0, "source_row", np.arange(n))
    return data


def temp_paths(tmp_path: Path, train: pd.DataFrame, horizon: str | None = None) -> FillPaths:
    tmp_path.mkdir(parents=True, exist_ok=True)
    train_file = tmp_path / "train_cleaned.csv"
    train.to_csv(train_file, index=False)
    return FillPaths(
        dataset="default",
        version="v1",
        horizon=horizon,
        train_file=train_file,
        synthetic_dir=tmp_path / "synthetic",
        artifacts=tmp_path / "artifacts",
        file_stem="default_ctgan",
        model_name="ctgan_default.pkl",
    )


def test_fill_paths_pick_suffixes_for_version_and_horizon():
    v1 = fill_paths("default", "v1")
    assert v1.artifacts.name == "artifacts"
    assert v1.fill_file("fill50").name == "default_ctgan_fill50.csv"
    assert v1.model_name == "ctgan_default.pkl"

    v2 = fill_paths("polish", "v2")
    assert v2.artifacts.name == "artifacts_v2"
    assert v2.fill_file("fill10").name == "polish_ctgan_v2_fill10.csv"

    pinned = fill_paths("polish", "v2", "3year")
    assert pinned.artifacts.name == "artifacts_v2_3year"
    assert pinned.fill_file("fill25").name == "polish_3year_ctgan_v2_fill25.csv"
    assert pinned.model_name == "ctgan_polish_3year_v2.pkl"


def test_fill_paths_refuse_unknown_inputs():
    with pytest.raises(SystemExit, match="unknown dataset"):
        fill_paths("nope", "v1")
    with pytest.raises(SystemExit, match="v1 or v2"):
        fill_paths("default", "v3")


def test_load_training_frame_drops_source_row_and_refuses_gaps(tmp_path):
    frame = make_default_frame()
    paths = temp_paths(tmp_path, frame)
    model = load_training_frame(paths)
    assert "source_row" not in model.columns
    assert len(model) == len(frame)

    frame.loc[3, "AGE"] = np.nan
    broken = temp_paths(tmp_path / "broken", frame)
    with pytest.raises(SystemExit, match="missing values"):
        load_training_frame(broken)


def test_fill_sizes_use_the_class_gap():
    frame = make_default_frame().drop(columns=["source_row"])
    sizes = fill_sizes(frame, "y")
    positives = int(frame["y"].sum())
    gap = len(frame) - 2 * positives
    assert sizes == {level: round(level * gap) for level in FILL_LEVELS}


def test_selection_record_for_default_names_timegan_and_the_dc4_override():
    frame = make_default_frame(2500).drop(columns=["source_row"])
    record = selection_record(frame, "default", "y")
    assert record["status"] == "selected"
    assert record["generator"] == "timegan"
    assert record["trainable"] is False
    assert record["fallback"] == "ctgan"
    assert record["trained"] == "ctgan"
    assert "DC-4" in record["override"]
    rules = [step["rule"] for step in record["steps"]]
    assert rules[0] == "dcgan_excluded" and "temporal_test" in rules
    json.dumps(record)  # every input is JSON serialisable


def test_selection_record_for_a_small_table_is_a_refused_run():
    frame = make_default_frame(300).drop(columns=["source_row"])
    record = selection_record(frame, "default", "y")
    assert record["status"] == "skipped"
    assert "refused" in record["override"]


def test_rule_mask_applies_the_dataset_rules():
    frame = make_default_frame(20).drop(columns=["source_row"])
    frame.loc[0, "PAY_AMT2"] = -5
    frame.loc[1, "PAY_3"] = 11
    mask = rule_mask("default")(frame)
    assert not mask.iloc[0] and not mask.iloc[1] and mask.iloc[2:].all()
    # a dataset without rules keeps every row
    assert rule_mask("nothing_registered")(frame).all()


def test_diagnostic_lines_cope_with_missing_and_present_meta():
    assert selection_lines({})[0].startswith("selection rule: not recorded")
    assert "not recorded" in range_lines({})[1]
    meta = {
        "selection": {
            "status": "selected", "generator": "wgan_gp", "trainable": False,
            "fallback": "ctgan", "rationale": "Every feature is numeric.",
            "override": "CTGAN was trained as the documented comparator (rule PB-3).",
            "steps": [
                {"rule": "size_gate", "outcome": "PASS"},
                {"rule": "minority_gate", "outcome": "FORBIDDEN_CLASS_ONLY"},
                {"rule": "type_routing", "outcome": "SELECTED"},
            ],
        },
        "out_of_range_before_clip": {
            "A1": {"below": 0.20, "above": 0.05, "outside": 0.25, "rows": 1000},
            "A2": {"below": 0.0, "above": 0.001, "outside": 0.001, "rows": 1000},
        },
    }
    selected = selection_lines(meta)
    assert "wgan_gp" in selected[0] and "PB-3" in selected[2]
    assert "minority_gate FORBIDDEN_CLASS_ONLY" in selected[3]
    ranged = range_lines(meta)
    assert any(line.startswith("  A1: 25.0% outside") for line in ranged)
    assert not any(line.startswith("  A2:") for line in ranged)


def test_run_fill_end_to_end_on_a_small_table(tmp_path):
    frame = make_default_frame(400)
    paths = temp_paths(tmp_path, frame)
    settings = load_ctgan_settings("v1", epochs=1)
    settings = type(settings)(**{**settings.__dict__, "batch_size": 50, "pac": 5})
    meta = run_fill(paths, settings)

    sizes = fill_sizes(frame.drop(columns=["source_row"]), "y")
    for level in FILL_LEVELS:
        file = paths.fill_file(FILL_NAMES[level])
        assert file.exists()
        written = pd.read_csv(file)
        assert len(written) == sizes[level]
        assert (written["y"] == 1).all() and (written["is_synthetic"] == 1).all()
    assert (paths.artifacts / "run_meta.json").exists()
    assert (paths.artifacts / "ctgan_default.pkl").exists()
    assert meta["rules_applied"] == ["pay_status_codes", "pay_amount_non_negative", "static_codes"]
    assert meta["selection"]["trained"] == "ctgan"
    numeric = {"LIMIT_BAL", "AGE"} | {f"BILL_AMT{m}" for m in range(1, 7)} | {
        f"PAY_AMT{m}" for m in range(1, 7)
    }
    assert set(meta["out_of_range_before_clip"]) == numeric
    assert meta["validation_and_test_read"] is False
