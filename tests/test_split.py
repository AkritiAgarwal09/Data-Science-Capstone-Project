"""Tests for step S0: schema validation and the train/validation/test split."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from agentic_gan.orchestrator.manifest import create_run
from agentic_gan.orchestrator.split import (
    MIN_CLASS_ROWS,
    SchemaError,
    split_data,
    split_row_ids,
    validate_schema,
)
from agentic_gan.schemas import STAGE_IO, PreprocessInput, RunConfig, SplitOutput, Stage, Status


def make_df(n: int = 1000, positives: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = np.array(["yes"] * positives + ["no"] * (n - positives))
    rng.shuffle(y)
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "age": rng.integers(18, 90, n),
            "balance": rng.normal(size=n),
            "job": rng.choice(["admin", "blue-collar", "services"], n),
            "duration": rng.integers(1, 500, n),
            "y": y,
        }
    )


def make_config(path: Path = Path("unused.csv"), **overrides) -> RunConfig:
    fields = {
        "dataset_name": "bank_marketing",
        "dataset_path": path,
        "target": "y",
        "positive_class": "yes",
    }
    return RunConfig(**{**fields, **overrides})


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "bank.csv"
    make_df().to_csv(path, index=False)
    return path


@pytest.fixture
def config(csv_path):
    return make_config(csv_path)


@pytest.fixture
def run(config, tmp_path):
    return create_run(config, runs_dir=tmp_path / "runs")


# --- validate_schema --------------------------------------------------------------------------


def test_a_valid_dataset_passes():
    validate_schema(make_config(), make_df())


def test_a_dataset_with_excluded_features_present_passes():
    validate_schema(make_config(excluded_features=["duration"]), make_df())


def drop_target(df):
    return df.drop(columns="y")


def blank_a_target(df):
    df.loc[0, "y"] = None
    return df


def add_a_third_value(df):
    df.loc[0, "y"] = "maybe"
    return df


def duplicate_a_column(df):
    df.columns = ["id", "age", "age", "job", "duration", "y"]
    return df


def add_an_empty_column(df):
    df["empty"] = np.nan
    return df


@pytest.mark.parametrize(
    ("change_data", "overrides", "message"),
    [
        (drop_target, {}, "target column 'y' not found"),
        (blank_a_target, {}, "target has 1 missing values"),
        (add_a_third_value, {}, "exactly 2 values, found 3"),
        (lambda df: df, {"positive_class": "true"}, "positive class 'true' not among"),
        (lambda df: make_df(positives=MIN_CLASS_ROWS - 1), {}, "each class needs at least 3"),
        (lambda df: df, {"excluded_features": ["nope"]}, "excluded features not in the data"),
        (lambda df: df, {"excluded_features": ["y"]}, "target cannot be an excluded feature"),
        (duplicate_a_column, {}, "duplicate column names"),
        (add_an_empty_column, {}, "columns with no values"),
        (lambda df: df.iloc[0:0], {}, "no rows"),
        (lambda df: df[["y"]], {}, "no feature columns"),
    ],
    ids=[
        "no target column",
        "missing target value",
        "three target values",
        "positive class absent",
        "too few positives",
        "unknown excluded feature",
        "target excluded",
        "duplicate column",
        "empty column",
        "no rows",
        "no features",
    ],
)
def test_a_bad_dataset_is_rejected(change_data, overrides, message):
    with pytest.raises(SchemaError, match=message):
        validate_schema(make_config(**overrides), change_data(make_df()))


def test_every_problem_is_listed_at_once():
    config = make_config(excluded_features=["nope"])
    with pytest.raises(SchemaError) as error:
        validate_schema(config, drop_target(make_df()))
    assert "target column 'y' not found" in str(error.value)
    assert "excluded features not in the data" in str(error.value)


def test_schema_error_is_a_value_error():
    assert issubclass(SchemaError, ValueError)


# --- split_row_ids ----------------------------------------------------------------------------


def test_the_split_follows_the_two_step_recipe_used_for_eda_and_the_baselines():
    df = make_df()
    positive = df["y"] == "yes"
    train, hold = train_test_split(df.index, test_size=0.4, stratify=positive, random_state=42)
    val, test = train_test_split(hold, test_size=0.5, stratify=positive.loc[hold], random_state=42)
    ids = split_row_ids(make_config(), df)
    assert list(ids["train"]) == list(train)
    assert list(ids["val"]) == list(val)
    assert list(ids["test"]) == list(test)


def test_the_parts_cover_every_row_once():
    df = make_df()
    ids = split_row_ids(make_config(), df)
    everything = np.concatenate([ids["train"], ids["val"], ids["test"]])
    assert sorted(everything) == list(df.index)


def test_default_part_sizes_are_60_20_20():
    ids = split_row_ids(make_config(), make_df())
    assert [len(ids[p]) for p in ("train", "val", "test")] == [600, 200, 200]


@pytest.mark.parametrize(
    ("ratios", "sizes"),
    [
        ((0.7, 0.15, 0.15), (700, 150, 150)),
        ((0.6, 0.1, 0.3), (600, 100, 300)),
        ((0.5, 0.4, 0.1), (500, 400, 100)),
    ],
)
def test_custom_ratios_are_used(ratios, sizes):
    train, val, test = ratios
    config = make_config(split_ratios={"train": train, "val": val, "test": test})
    ids = split_row_ids(config, make_df())
    for part, expected in zip(("train", "val", "test"), sizes, strict=True):
        assert abs(len(ids[part]) - expected) <= 1


def test_each_part_keeps_the_class_balance():
    df = make_df()
    ids = split_row_ids(make_config(), df)
    positives = {p: int((df.loc[i, "y"] == "yes").sum()) for p, i in ids.items()}
    assert positives == {"train": 60, "val": 20, "test": 20}


def test_the_same_seed_gives_the_same_split_and_another_seed_a_different_one():
    df = make_df()
    first = split_row_ids(make_config(), df)
    again = split_row_ids(make_config(), df)
    other = split_row_ids(make_config(split_seed=7), df)
    assert list(first["test"]) == list(again["test"])
    assert list(first["test"]) != list(other["test"])


def test_the_split_does_not_depend_on_how_the_classes_are_labelled():
    df = make_df()
    numeric = df.assign(y=(df["y"] == "yes").astype(int))
    by_text = split_row_ids(make_config(), df)
    by_number = split_row_ids(make_config(positive_class=1), numeric)
    assert all(list(by_text[p]) == list(by_number[p]) for p in by_text)


# --- split_data -------------------------------------------------------------------------------


def test_split_data_writes_the_three_files(config, run):
    out = split_data(config, run)
    assert out.status is Status.SUCCESS
    for path in (out.train_path, out.val_path, out.test_path):
        assert path.is_file()
        assert path.parent == run.subdir("data")


def test_the_files_hold_every_source_row_once_with_provenance(config, run):
    out = split_data(config, run)
    parts = [pd.read_csv(p) for p in (out.train_path, out.val_path, out.test_path)]
    assert all(list(part.columns) == ["source_row", *make_df().columns] for part in parts)
    assert sorted(pd.concat(parts)["source_row"]) == list(range(1000))
    assert sorted(pd.concat(parts)["id"]) == list(range(1000))


def test_excluded_features_stay_in_the_files(csv_path, tmp_path):
    config = make_config(csv_path, excluded_features=["duration"])
    run = create_run(config, runs_dir=tmp_path / "runs")
    assert "duration" in pd.read_csv(split_data(config, run).train_path).columns


def test_the_output_reports_the_target_seed_and_training_class_counts(config, run):
    out = split_data(config, run)
    train = pd.read_csv(out.train_path)
    assert out.target == "y"
    assert out.seed == 42
    assert out.train_positives == int((train["y"] == "yes").sum()) == 60
    assert out.train_negatives == int((train["y"] == "no").sum()) == 540


def test_a_small_positive_class_is_a_warning(config, run):
    out = split_data(config, run)
    assert any(w.startswith("val has only 20 positives") for w in out.warnings)
    assert len(out.warnings) == 3


def test_no_warning_when_every_part_has_enough_positives(tmp_path):
    path = tmp_path / "big.csv"
    make_df(n=3000, positives=1500).to_csv(path, index=False)
    config = make_config(path)
    run = create_run(config, runs_dir=tmp_path / "runs")
    assert split_data(config, run).warnings == []


def test_a_bad_dataset_writes_nothing(tmp_path):
    path = tmp_path / "bad.csv"
    make_df().drop(columns="y").to_csv(path, index=False)
    config = make_config(path)
    run = create_run(config, runs_dir=tmp_path / "runs")
    with pytest.raises(SchemaError):
        split_data(config, run)
    assert list(run.subdir("data").iterdir()) == []


def test_a_missing_dataset_file_is_an_error(config, run, tmp_path):
    with pytest.raises(FileNotFoundError):
        split_data(config.updated(dataset_path=tmp_path / "nope.csv"), run)


# --- fit with the rest of the orchestrator ----------------------------------------------------


def test_the_output_is_the_split_stage_output_and_feeds_preprocessing(config, run):
    out = split_data(config, run)
    assert isinstance(out, STAGE_IO[Stage.SPLIT][1])
    assert isinstance(out, SplitOutput)
    assert PreprocessInput(config=config, split=out).split == out


def test_the_files_can_be_registered_and_verified_in_the_manifest(config, run):
    out = split_data(config, run)
    manifest = run.load_manifest().register(run, Stage.SPLIT, out)
    assert len(manifest.entries) == 3
    assert manifest.verify(run) == []
