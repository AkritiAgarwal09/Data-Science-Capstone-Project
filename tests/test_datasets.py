import csv
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from agentic_gan.utils.datasets import DATASETS, load_configs

ROOT = Path(__file__).resolve().parents[1]
IEEE = DATASETS["ieee_cis"]
SPLIT_FILE = ROOT / "splits" / "ieee_cis_time_split.csv"


def split_rows() -> dict[str, dict]:
    with SPLIT_FILE.open() as f:
        return {row["part"]: row for row in csv.DictReader(f)}


def test_entries_without_extra_settings_keep_their_defaults(tmp_path):
    settings = {"tiny": {"file": "a.csv", "target": "y", "discrete": ["c"], "split": {"seed": 1}}}
    path = tmp_path / "datasets.json"
    path.write_text(json.dumps(settings))
    config = load_configs(path)["tiny"]
    assert config.discrete == ("c",)
    assert config.id_column is None and config.merge is None
    assert config.duplicates is None and config.sample is None
    for name in ("bank_marketing", "default", "polish"):
        older = DATASETS[name]
        assert older.id_column is None and older.merge is None
        assert older.duplicates is None and older.sample is None


def test_ieee_cis_tables_and_keys():
    assert IEEE.target == "isFraud"
    assert IEEE.id_column == "TransactionID"
    assert IEEE.file == "data/external/train_transaction.csv"
    assert IEEE.merge == {
        "file": "data/external/train_identity.csv",
        "on": "TransactionID",
        "how": "left",
    }


def test_ieee_cis_data_stays_under_folders_git_ignores():
    assert IEEE.file.startswith("data/external/")
    assert IEEE.merge["file"].startswith("data/external/")
    ignored = (ROOT / ".gitignore").read_text().splitlines()
    assert "data/external/" in ignored
    assert "data/interim/ieee_cis/" in ignored  # cleaned and synthetic rows are derived data


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_no_ieee_cis_data_is_tracked():
    files = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    leaked = [
        f
        for f in files
        if f.startswith(("data/external/", "data/interim/ieee_cis/"))
        or Path(f).name in {"train_transaction.csv", "train_identity.csv"}
    ]
    assert leaked == []


def test_ieee_cis_discrete_columns():
    discrete = IEEE.discrete
    assert len(discrete) == len(set(discrete)) == 49
    assert {"ProductCD", "card1", "card4", "addr1", "M1", "M9"} <= set(discrete)
    assert {"DeviceType", "DeviceInfo", "id_12", "id_38"} <= set(discrete)
    # the numeric identity columns, the key, the time and the target are not discrete
    assert not {"id_01", "id_11", IEEE.id_column, IEEE.split["time_column"], IEEE.target} & set(
        discrete
    )


def test_ieee_cis_duplicate_policy_and_sample():
    assert IEEE.duplicates == {
        "drop": "exact",
        "ignore_columns": ["TransactionID"],
        "keep": "first",
        "before_split": True,
    }
    assert IEEE.sample == {
        "part": "train",
        "rows": 100_000,
        "stratify": IEEE.target,
        "seed": 42,
        "max_missing": 0.2,
    }


def test_ieee_cis_time_split_is_a_rule_and_matches_the_split_file():
    s = IEEE.split
    assert s["strategy"] == "time" and s["time_column"] == "TransactionDT"
    assert s["train"] + s["validation"] + s["test"] == pytest.approx(1.0)
    first, second = s["cut_points"]
    assert first < second

    rows = split_rows()
    assert rows["train"]["rule"] == f"TransactionDT < {first}"
    assert rows["validation"]["rule"] == f"{first} <= TransactionDT < {second}"
    assert rows["test"]["rule"] == f"TransactionDT >= {second}"
    assert int(rows["train"]["transaction_dt_to"]) < first
    assert int(rows["validation"]["transaction_dt_from"]) >= first
    assert int(rows["validation"]["transaction_dt_to"]) < second
    assert int(rows["test"]["transaction_dt_from"]) >= second

    total = sum(int(rows[p]["rows"]) for p in ("train", "validation", "test"))
    assert total == 590_540 - 3  # the file, less the 3 exact copies
    for part in ("train", "validation", "test"):
        assert int(rows[part]["rows"]) / total == pytest.approx(s[part], abs=0.001)
    assert sum(int(rows[p]["fraud_rows"]) for p in ("train", "validation", "test")) == 20_663


def test_ieee_cis_split_file_holds_counts_and_rules_only():
    with SPLIT_FILE.open() as f:
        header = next(csv.reader(f))
    assert header == [
        "part",
        "rule",
        "transaction_dt_from",
        "transaction_dt_to",
        "rows",
        "fraud_rows",
        "fraud_rate",
    ]
    rows = split_rows()
    assert set(rows) == {"train", "validation", "test", "train_sample"}


def test_ieee_cis_sample_comes_from_the_train_part_and_keeps_its_fraud_rate():
    rows = split_rows()
    sample, train = rows["train_sample"], rows["train"]
    assert int(sample["rows"]) == IEEE.sample["rows"]
    assert int(sample["rows"]) < int(train["rows"])
    assert float(sample["fraud_rate"]) == pytest.approx(float(train["fraud_rate"]), abs=0.0005)
    assert int(sample["transaction_dt_to"]) <= int(train["transaction_dt_to"])
