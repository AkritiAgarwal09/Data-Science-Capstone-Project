"""IEEE-CIS preprocessing without redistributing the competition data."""

import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

from agentic_gan.agents import training as training_agent
from agentic_gan.agents.training import _model_data
from agentic_gan.orchestrator.manifest import create_run
from agentic_gan.schemas import ArmFiles, RunConfig, Status, TrainingRequest
from agentic_gan.tools import baselines
from agentic_gan.tools import preprocessing as p
from agentic_gan.utils.datasets import DATASETS


@pytest.fixture
def ieee(tmp_path):
    config = replace(DATASETS["ieee_cis"], split={
        **DATASETS["ieee_cis"].split, "cut_points": [100, 200],
    }, sample={**DATASETS["ieee_cis"].sample, "rows": 20})
    tx = pd.DataFrame({
        "TransactionID": np.arange(1000, 1060),
        "TransactionDT": [*range(40), *range(100, 110), *range(200, 210)],
        "isFraud": [0, 0, 0, 1] * 15,
        "TransactionAmt": np.arange(60, dtype=float) + 10,
        "ProductCD": ["W", "C"] * 30,
        "card1": np.arange(60) % 5,
        "V1": [np.nan] * 40 + list(range(20)),
        "C1": np.arange(60, dtype=float),
    })
    sample, _ = train_test_split(
        list(range(40)), train_size=20, stratify=tx.isFraud.iloc[:40], random_state=42
    )
    tx.loc[sample[:2], "TransactionAmt"] = np.nan
    # A different-time repeat stays; an exact row except for ID is removed.
    duplicate = tx.iloc[[6]].copy()
    duplicate.TransactionID = 9999
    tx = pd.concat([tx, duplicate], ignore_index=True)
    ident = pd.DataFrame({"TransactionID": np.arange(1000, 1060, 2), "id_01": 7.0,
                          "DeviceType": "desktop"}).iloc[::-1]
    # Keep the duplicate identical after the identity join, as #50 deduplicates
    # on the merged table (with TransactionID omitted).
    ident = pd.concat(
        [ident, pd.DataFrame({"TransactionID": [9999], "id_01": [7.0], "DeviceType": ["desktop"]})],
        ignore_index=True,
    )
    folder = tmp_path / "data/external"
    folder.mkdir(parents=True)
    tx_path, id_path = folder / "train_transaction.csv", folder / "train_identity.csv"
    tx.to_csv(tx_path, index=False)
    ident.to_csv(id_path, index=False)
    cfg = tmp_path / "datasets.json"
    settings = asdict(config)
    settings.pop("name")
    cfg.write_text(json.dumps({"ieee_cis": settings}))
    return config, tx_path, id_path, cfg, tmp_path


def test_merge_preserves_transaction_order_and_identity_absence(ieee):
    cfg, tx, identity, *_ = ieee
    data = p.load_dataset(cfg, tx)
    assert len(data) == 61
    assert data.TransactionID.tolist() == pd.read_csv(tx).TransactionID.tolist()
    assert data.has_identity.iloc[:4].tolist() == [1, 0, 1, 0]
    assert data.id_01.iloc[0] == 7
    assert pd.isna(data.id_01.iloc[1])
    assert data.index.name == "source_row"


@pytest.mark.parametrize("table,change", [
    ("transaction", "duplicate"), ("identity", "duplicate"),
    ("transaction", "null"), ("identity", "null"),
    ("identity", "label"), ("identity", "overlap"), ("identity", "foreign_key"),
])
def test_ambiguous_or_wrong_merge_rejected(ieee, table, change):
    cfg, tx, identity, *_ = ieee
    path = tx if table == "transaction" else identity
    data = pd.read_csv(path)
    if change == "duplicate":
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    elif change == "null":
        data.loc[0, "TransactionID"] = np.nan
    elif change == "label":
        data["isFraud"] = 1
    elif change == "overlap":
        data["TransactionAmt"] = 1
    else:
        data.loc[0, "TransactionID"] = -1
    data.to_csv(path, index=False)
    with pytest.raises(ValueError):
        p.load_dataset(cfg, tx)


def test_exact_duplicates_cutoff_boundaries_and_training_sample(ieee):
    cfg, tx, *_ = ieee
    data = p.load_dataset(cfg, tx)
    full = p.split_dataset(data, cfg, sample=False)
    assert {part: len(ids) for part, ids in full.items()} == {
        "train": 40,
        "validation": 10,
        "test": 10,
    }
    assert 60 not in sum(full.values(), [])
    assert 40 in full["validation"] and 50 in full["test"]
    selected = p.split_dataset(data, cfg)
    expected, _ = train_test_split(
        full["train"],
        train_size=20,
        stratify=data.loc[full["train"], "isFraud"],
        random_state=42,
    )
    assert selected["train"] == sorted(expected)
    assert selected["validation"] == full["validation"]
    assert selected["test"] == full["test"]
    assert len(set(sum(selected.values(), []))) == 40
    assert data.loc[selected["train"], "isFraud"].sum() == 5
    repeated = data.iloc[[2]].copy()
    repeated.TransactionID = 8888
    repeated.TransactionDT = 205  # same other values at a later time is NOT a duplicate
    repeated.index = pd.Index([77], name="source_row")
    with_repeat = pd.concat([data, repeated])
    assert 77 in p.split_dataset(with_repeat, cfg, sample=False)["test"]


@pytest.mark.parametrize("change", ["missing_time", "bad_cuts", "too_big_sample"])
def test_invalid_split_fails(ieee, change):
    cfg, tx, *_ = ieee
    data = p.load_dataset(cfg, tx)
    if change == "missing_time":
        data.loc[0, "TransactionDT"] = np.nan
    elif change == "bad_cuts":
        cfg = replace(cfg, split={**cfg.split, "cut_points": [200, 100]})
    else:
        cfg = replace(cfg, sample={**cfg.sample, "rows": 1000})
    with pytest.raises(ValueError):
        p.split_dataset(data, cfg)


def test_training_only_statistics_flags_and_excluded_features(ieee):
    cfg, tx, *_ = ieee
    data = p.load_dataset(cfg, tx)
    ids = p.split_dataset(data, cfg)
    processor = p.Preprocessor(cfg).fit(data.loc[ids["train"]])
    before = dict(processor.fill_values)
    held = data.loc[ids["validation"]].copy()
    held.TransactionAmt = 1e9
    held.loc[40, "TransactionAmt"] = np.nan
    held.loc[41, "C1"] = np.nan  # holdout-only gaps cannot create new features
    held.ProductCD = "UNSEEN"
    cleaned = processor.clean(held)
    assert cleaned.loc[40, "TransactionAmt"] == pytest.approx(
        data.loc[ids["train"], "TransactionAmt"].median()
    )
    assert cleaned.loc[40, "TransactionAmt__missing"] == 1
    assert "C1__missing" not in cleaned
    assert cleaned.ProductCD.eq(p.OTHER_CATEGORY).all()
    assert "has_identity" in cleaned
    excluded = {"TransactionID", "TransactionDT", "source_row", "V1", "id_01", "DeviceType"}
    assert not excluded & set(cleaned)
    assert not cleaned.isna().any().any()
    assert processor.fill_values == before
    encoded = processor.transform_cleaned(cleaned)
    assert np.isfinite(encoded.to_numpy()).all()
    assert encoded.isFraud.tolist() == held.isFraud.tolist()
    # Calling transform_cleaned preserves the existing flags, rather than recomputing from fills.
    matrix = processor.transform_features(cleaned)
    assert sparse.isspmatrix_csr(matrix)
    np.testing.assert_allclose(matrix.toarray(), encoded.drop(columns="isFraud"))
    numeric = processor.encoder.named_transformers_["numeric"]
    amount_index = processor.numeric.index("TransactionAmt")
    clean_train = processor.clean(data.loc[ids["train"]])
    assert numeric.mean_[amount_index] == pytest.approx(clean_train.TransactionAmt.mean())


def test_baseline_arms_use_issue_50_time_split_without_id_or_clock(ieee):
    config, tx, _, config_path, root = ieee
    original, _, original_metadata = baselines.prepare_inputs(
        "ieee_cis", "original", root, 42, config_path=config_path
    )
    cleaned, _, cleaned_metadata = baselines.prepare_inputs(
        "ieee_cis", "cleaned", root, 42, config_path=config_path
    )
    for features in (original[0], cleaned[0]):
        assert "TransactionID" not in features
        assert "TransactionDT" not in features
    assert len(original[0]) == len(cleaned[0]) == 20
    assert set(original[2].index) == set(cleaned[2].index) == set(range(40, 50))
    assert set(original[4].index) == set(cleaned[4].index) == set(range(50, 60))
    assert original_metadata["source_sha256"] == cleaned_metadata["source_sha256"]

    before = baselines.source_hash_for(config, tx)
    identity_path = root / config.merge["file"]
    identity = pd.read_csv(identity_path)
    identity.loc[0, "id_01"] = 9.0
    identity.to_csv(identity_path, index=False)
    assert baselines.source_hash_for(config, tx) != before


def test_training_model_matrix_drops_ieee_join_key_and_split_clock():
    table = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "TransactionDT": [100, 200],
            "TransactionAmt": [1.5, 2.5],
            "isFraud": [0, 1],
        }
    )
    features, target = _model_data(table, DATASETS["ieee_cis"], original=True)
    assert list(features.columns) == ["TransactionAmt", "is_synthetic"]
    assert target.tolist() == [0, 1]


def test_ieee_original_encoder_keeps_high_cardinality_one_hot_sparse():
    train = pd.DataFrame({
        "card1": np.arange(1000),
        "amount": np.linspace(0, 1, 1000),
    })
    transformer = baselines._minimal_transformer(
        train, {"card1"}, sparse_output=True
    ).fit(train)
    train_matrix = transformer.transform(train)
    unseen_matrix = transformer.transform(
        pd.DataFrame({"card1": [1001], "amount": [0.5]})
    )
    assert sparse.issparse(train_matrix)
    assert sparse.issparse(unseen_matrix)
    assert train_matrix.shape == (1000, 1001)
    assert unseen_matrix.shape == (1, 1001)


def test_ieee_baseline_protocol_hands_off_to_training(ieee, monkeypatch):
    config, tx, _, config_path, root = ieee
    monkeypatch.setattr(
        baselines,
        "build_models",
        lambda seed=42: {"Dummy": DummyClassifier(strategy="prior")},
    )
    inputs, transformer, metadata = baselines.prepare_inputs(
        "ieee_cis", "original", root, 42, config_path=config_path
    )
    baseline_table = baselines.run_models(
        *inputs, preprocessor=transformer, cv_folds=3
    )
    protocol_path = root / "results" / "baseline_protocol_ieee_cis_42_cv3.json"
    protocol = baselines.freeze_protocol(
        protocol_path,
        baseline_table,
        metadata,
        dataset="ieee_cis",
        seed=42,
        cv_folds=3,
        y_val=inputs[3],
        y_test=inputs[5],
    )

    monkeypatch.setattr(training_agent, "load_configs", lambda path=None: {"ieee_cis": config})
    run_config = RunConfig(
        dataset_name="ieee_cis",
        dataset_path=tx,
        target="isFraud",
        positive_class=1,
        candidate_models=["Dummy"],
        fill_levels=[0.05],
        training_protocol_path=protocol_path,
    )
    run = create_run(run_config, runs_dir=root / "runs")
    raw = p.load_dataset(config, tx)
    ids = p.split_dataset(raw, config)
    processor = p.Preprocessor(config).fit(raw.loc[ids["train"]])
    arms = {}
    for arm, frames in (
        ("raw", {part: raw.loc[part_ids] for part, part_ids in ids.items()}),
        (
            "baseline",
            {
                part: processor.clean(raw.loc[part_ids])
                for part, part_ids in ids.items()
            },
        ),
    ):
        paths = {}
        for part, frame in frames.items():
            filename = f"{arm}_{part}.csv"
            path = run.subdir("data") / filename
            frame.to_csv(path, index_label="source_row")
            field = "val_path" if part == "validation" else f"{part}_path"
            paths[field] = path
        arms[arm] = ArmFiles(**paths)
    request = TrainingRequest(config=run_config, raw=arms["raw"], baseline=arms["baseline"])

    result = training_agent.run_training(request, run)
    assert result.status is Status.SUCCESS
    assert {record.threshold for record in result.records} == {
        protocol["thresholds"]["Dummy"]
    }
    assert {record.status for record in result.records} == {Status.SUCCESS, Status.SKIPPED}


def test_category_cap_reserved_buckets_and_missing_values(ieee):
    cfg, tx, *_ = ieee
    data = p.load_dataset(cfg, tx).iloc[:40].copy()
    data = pd.concat([data, data.iloc[:30]], ignore_index=True)
    data.card1 = np.arange(70)
    data.loc[0, "card1"] = np.nan
    processor = p.Preprocessor(cfg).fit(data)
    clean = processor.clean(data)
    assert len(processor.category_levels["card1"]) == 50
    assert len(set(clean.card1)) <= 52
    assert clean.card1.iloc[0] == p.MISSING_CATEGORY
    assert p.OTHER_CATEGORY in set(clean.card1)
    shifted = data.copy()
    shifted.card1 = 100000
    assert processor.clean(shifted).card1.eq(p.OTHER_CATEGORY).all()


def test_full_artifact_roundtrip_and_column_log(ieee):
    cfg, tx, _, config_path, root = ieee
    raw = p.load_dataset(cfg, tx)
    ids = p.split_dataset(raw, cfg)
    full = p.split_dataset(raw, cfg, sample=False)
    counts = p._split_counts(raw, full, ids, cfg)
    split_folder = root / "splits"
    split_folder.mkdir()
    pd.DataFrame([{"part": key, **value} for key, value in counts.items()]).to_csv(
        split_folder / "ieee_cis_time_split.csv", index=False
    )
    output = p.run_preprocessing("ieee_cis", root=root, config_path=config_path)
    tables, processor = p.load_preprocessed(output)
    settings = json.loads((output / "encoder_settings.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["training_rows"] == 20
    assert manifest["duplicates_dropped"] == 1
    assert manifest["reference_counts_verified"]
    assert settings["category_limit"] == 50
    assert settings["feature_names"] == processor.feature_names
    assert {"has_identity", "TransactionAmt__missing"} <= set(settings["gan_discrete_columns"])
    for part, table in tables.items():
        assert table.index.tolist() == ids[part]
        matrix = sparse.load_npz(output / f"{part}_encoded.npz")
        rows = pd.read_csv(output / f"{part}_encoded_rows.csv")
        assert matrix.shape == (len(table), len(settings["feature_names"]))
        assert rows.source_row.tolist() == ids[part]
        assert rows.isFraud.tolist() == table.isFraud.tolist()
        loaded, labels, source_rows = p.load_encoded(output, part)
        assert loaded.shape == matrix.shape
        assert labels.tolist() == rows.isFraud.tolist()
        assert source_rows.tolist() == rows.source_row.tolist()
        assert np.isfinite(matrix.data).all()
        np.testing.assert_allclose(matrix.toarray(), processor.transform_features(table).toarray())
        assert not {"TransactionID", "TransactionDT"} & set(table)
    log = pd.read_csv(output / "treatment_log.csv")
    assert log.column.is_unique
    assert set(log.column) == set(raw.columns) | set(processor.columns)
    assert log.set_index("column").loc["TransactionID", "treatment"] == "Exclude from features"
    assert log.set_index("column").loc["V1", "treatment"] == "Drop value column"
    old = {part: table.copy() for part, table in tables.items()}
    p.run_preprocessing("ieee_cis", root=root, config_path=config_path)
    for part, new in p.load_preprocessed(output)[0].items():
        pd.testing.assert_frame_equal(old[part], new)
    with (output / "train_cleaned.csv").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="artifact changed"):
        p.load_preprocessed(output)


def test_reference_count_mismatch_stops_before_output(ieee):
    _, _, _, config_path, root = ieee
    folder = root / "splits"
    folder.mkdir()
    pd.DataFrame([{"part": "train", "rows": 999}]).to_csv(
        folder / "ieee_cis_time_split.csv", index=False
    )
    with pytest.raises(ValueError, match="reference split parts"):
        p.run_preprocessing("ieee_cis", root=root, config_path=config_path)
    assert not (root / "data/interim/ieee_cis").exists()


def test_ieee_llm_plan_reaches_training_with_fixed_time_sample(ieee, monkeypatch):
    from test_preprocessing_pipeline import run_dataset_pipeline

    from agentic_gan.agents import preprocessing as agent
    from agentic_gan.utils import datasets

    config, source, _, _, root = ieee
    settings = {"ieee_cis": config}
    monkeypatch.setattr(datasets, "load_configs", lambda: settings)
    monkeypatch.setattr(agent, "load_configs", lambda: settings)
    monkeypatch.setattr(training_agent, "load_configs", lambda: settings)
    output, _ = run_dataset_pipeline(root, monkeypatch, config, source)
    train = pd.read_csv(output.baseline.train_path)
    assert len(train) == config.sample["rows"]
    assert "V1" not in train and "id_01" not in train
    assert "id_01__missing" in train
    assert not {"TransactionID", "TransactionDT"} & set(train)
    assert not train.isna().any().any()
