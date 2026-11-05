"""The cross-dataset table built from runs' comparisons.csv files."""

import numpy as np
import pytest
from test_evaluation import results

from agentic_gan.tools.cross_dataset import COLUMNS, cross_dataset_table
from agentic_gan.tools.evaluation import compare_arms


def write_run(tmp_path, name, metrics):
    """A run folder whose report/comparisons.csv is what the evaluation agent writes."""
    comparison = compare_arms(metrics, reference="cleaned").assign(reference="cleaned")
    folder = tmp_path / name / "report"
    folder.mkdir(parents=True)
    comparison.to_csv(folder / "comparisons.csv", index=False)
    return tmp_path / name


def test_one_row_per_dataset_arm_and_level_with_the_verdict(tmp_path):
    bank = results()
    other = results().assign(dataset="polish")
    other = other[~other.arm.eq("ctgan")]  # a dataset where the GAN arm was not available
    runs = [write_run(tmp_path, "bank", bank), write_run(tmp_path, "polish", other)]

    table = cross_dataset_table(runs)

    assert list(table.columns) == COLUMNS
    keys = table[["dataset", "arm", "fill_level"]].apply(tuple, axis=1)
    assert not keys.duplicated().any()
    assert set(table.dataset) == {"bank_marketing", "polish"}
    assert not (table.dataset.eq("polish") & table.arm.eq("ctgan")).any()
    bank_ctgan = table.query("dataset == 'bank_marketing' and arm == 'ctgan'")
    row = bank_ctgan[bank_ctgan.fill_level.eq(0.25)].iloc[0]
    # The fixture's CTGAN arm beats original by 0.02 on test, and cleaned equals original.
    assert row.lift == pytest.approx(0.02, abs=0.003)
    assert row.verdict == "reliable gain" and row.n_seeds == 5
    assert table.query("arm == 'cleaned'").verdict.eq("reference").all()
    assert table.query("arm == 'cleaned'").lift.isna().all()
    loss = table.query("dataset == 'bank_marketing' and arm == 'smotenc' and fill_level == 0.5")
    assert loss.verdict.iloc[0] == "reliable loss"
    assert np.isclose(
        table.query("arm == 'original'").test_mean.iloc[0], np.mean([0.48, 0.50, 0.46, 0.49, 0.47])
    )
