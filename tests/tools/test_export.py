"""The GAN-augmented training tables written from a finished run."""

import json

import pandas as pd
import pytest
from test_training_agent import _quality

from agentic_gan.orchestrator.decisions import read_decisions
from agentic_gan.orchestrator.manifest import create_run
from agentic_gan.schemas import RunConfig
from agentic_gan.tools.export import export_augmented


@pytest.fixture
def run(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("x,y\n0,0\n1,1\n")
    config = RunConfig(
        dataset_name="export_fixture", dataset_path=source, target="y", positive_class=1
    )
    run = create_run(config, runs_dir=tmp_path / "runs")
    pd.DataFrame({"source_row": [7, 8, 9], "x": [1.0, 2.0, 3.0], "y": [0, 0, 1]}).to_csv(
        run.subdir("data") / "baseline_train.csv", index=False
    )
    return run


def add_attempt(run, tmp_path, attempt, passed_by_level, marker):
    """A quality report and a synthetic file per level; `marker` tells the attempts apart."""
    report = _quality(tmp_path, passed_by_level)
    (run.subdir("quality") / f"quality_report_attempt_{attempt}.json").write_text(
        report.model_dump_json()
    )
    for level in passed_by_level:
        token = f"{level:.2f}".replace(".", "p")
        pd.DataFrame({"x": [marker, marker], "y": [1, 1], "is_synthetic": [1, 1]}).to_csv(
            run.subdir("synthetic") / f"fill_{token}_attempt_{attempt}.csv", index=False
        )


def test_each_accepted_level_gets_real_and_synthetic_rows_from_the_newest_attempt(run, tmp_path):
    add_attempt(run, tmp_path, 1, {0.1: False, 0.25: False}, marker=111.0)
    add_attempt(run, tmp_path, 2, {0.1: False, 0.25: True}, marker=222.0)

    paths = export_augmented(run)

    assert [p.name for p in paths] == ["augmented_train_fill_0p25.csv"]
    table = pd.read_csv(paths[0])
    assert "source_row" not in table
    assert table.is_synthetic.tolist() == [0, 0, 0, 1, 1]
    assert table.x.tolist() == [1.0, 2.0, 3.0, 222.0, 222.0]
    logged = read_decisions(run).query("agent == 'export'").iloc[0]
    assert logged.outcome == "EXPORTED" and logged.inputs["files"] == [paths[0].name]


def test_only_levels_in_the_training_plan_are_exported(run, tmp_path):
    add_attempt(run, tmp_path, 1, {0.1: True, 0.25: True}, marker=1.0)
    (run.subdir("models") / "training_plan.json").write_text(json.dumps({"gan_levels": [0.1]}))

    assert [p.name for p in export_augmented(run)] == ["augmented_train_fill_0p10.csv"]


def test_a_rejected_run_writes_nothing_and_says_why(run, tmp_path):
    add_attempt(run, tmp_path, 1, {0.1: False, 0.25: False}, marker=1.0)

    assert export_augmented(run) == []
    assert not list(run.subdir("report").glob("augmented_*"))
    assert read_decisions(run).query("agent == 'export'").iloc[0].outcome == "NO_SYNTHETIC_DATA"


def test_a_run_without_a_quality_report_writes_nothing(run):
    assert export_augmented(run) == []
    assert read_decisions(run).query("agent == 'export'").iloc[0].outcome == "NO_QUALITY_REPORT"
