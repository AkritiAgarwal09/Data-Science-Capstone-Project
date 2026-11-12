"""Verify poster totals and variable coverage without restricted IEEE-CIS files."""

from pathlib import Path

import pandas as pd
import pytest

from agentic_gan.tools.dataset_summary import Counts, summarize_dataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("dataset", "rows", "features", "responses", "rate"),
    [
        ("bank_marketing", 45_211, 16, 5_289, "11.7%"),
        ("default", 30_000, 23, 6_636, "22.1%"),
        ("polish", 43_405, 64, 2_091, "4.8%"),
    ],
)
def test_poster_totals_and_cleaned_variable_coverage(dataset, rows, features, responses, rate):
    summary = summarize_dataset(ROOT, dataset)

    assert summary.total == summary.source == Counts(rows, responses)
    assert summary.features == features
    assert f"{summary.total.rate:.1%}" == rate
    assert list(summary.splits) == ["train", "validation", "test"]
    cleaned = ROOT / "data" / "interim" / dataset / "train_cleaned.csv"
    assert set(pd.read_csv(cleaned, nrows=0)) <= {row[0].strip("`") for row in summary.variables}
    assert sum(row[3] == "Feature" for row in summary.variables) == features
