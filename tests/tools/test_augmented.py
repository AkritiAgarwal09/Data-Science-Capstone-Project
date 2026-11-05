import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from agentic_gan.tools import augmented, baselines
from agentic_gan.tools.augmented import (
    check_holdouts,
    load_cleaned,
    plot_summary,
    run_experiment,
    save_results,
    summarize,
)
from agentic_gan.utils.datasets import DATASETS
from agentic_gan.utils.synthetic import fill_counts

CONFIG = DATASETS["bank_marketing"]


@pytest.fixture(scope="module")
def tables():
    full = load_cleaned(CONFIG)
    return {
        "train": full["train"].groupby("y", group_keys=False).sample(frac=0.15, random_state=0),
        "validation": full["validation"].sample(1500, random_state=0),
        "test": full["test"].sample(1500, random_state=0),
    }


def test_holdout_check_rejects_synthetic_flag_and_overlap(tables):
    check_holdouts(tables, CONFIG)
    flagged = {**tables, "test": tables["test"].assign(is_synthetic=1)}
    with pytest.raises(ValueError, match="is_synthetic"):
        check_holdouts(flagged, CONFIG)
    leaked = {**tables, "validation": pd.concat([tables["validation"], tables["train"].head(5)])}
    with pytest.raises(ValueError, match="share source rows"):
        check_holdouts(leaked, CONFIG)


def test_smotenc_experiment_end_to_end(tables, tmp_path, monkeypatch):
    def models(seed):
        return {"test model": DummyClassifier(strategy="prior")}

    monkeypatch.setattr(baselines, "build_models", models)
    monkeypatch.setattr(augmented, "build_models", models)
    metadata = {
        "source_sha256": "fixture",
        "split_hashes": {},
        "config": {},
        "versions": baselines._versions(),
    }
    for seed in (1, 2):
        baselines.freeze_protocol(
            tmp_path / f"baseline_protocol_bank_marketing_{seed}_cv0.json",
            pd.DataFrame({"model": ["test model"], "status": ["success"], "threshold": [0.1]}),
            metadata,
            dataset="bank_marketing",
            seed=seed,
            cv_folds=0,
            y_val=tables["validation"].y,
            y_test=tables["test"].y,
        )
    monkeypatch.setattr(baselines, "select_f1_threshold", lambda *a: pytest.fail("Retuned"))
    out = run_experiment(
        tables,
        CONFIG,
        "smotenc",
        "smotenc",
        seeds=(1, 2),
        fill_levels=(0.10,),
        model_names=("test model",),
        protocol_dir=tmp_path,
    )
    assert set(out.arm) == {"real_only", "smotenc"}
    assert (out.status == "success").all()
    assert set(out.threshold) == {0.1}
    assert set(out.cv_folds) == {0}
    assert out.groupby(["arm", "seed"]).selected.sum().eq(1).all()
    expected = fill_counts(tables["train"]["y"], (0.10,))[0.10]
    assert list(out[out.arm == "smotenc"].n_synthetic.unique()) == [expected]
    summary = summarize(out)
    assert {"val_pr_auc_mean", "test_f1_std"} <= set(summary.columns)


def test_plot_summary_writes_png(tmp_path):
    rows = []
    for arm, fills in (("real_only", [0.0]), ("smotenc", [0.1, 0.25, 0.5])):
        for fill in fills:
            rows.append(
                {
                    "arm": arm,
                    "fill_level": fill,
                    "test_pr_auc_mean": 0.45 - fill / 10,
                    "test_pr_auc_std": 0.01,
                    "test_f1_mean": 0.4,
                    "test_f1_std": 0.01,
                }
            )
    out = tmp_path / "plot.png"
    plot_summary(pd.DataFrame(rows), out, {"test_pr_auc": 0.49, "test_f1": 0.36})
    assert out.stat().st_size > 0


def test_legacy_threshold_results_cannot_be_mixed(tmp_path):
    path = tmp_path / "comparison.csv"
    pd.DataFrame({"dataset": ["bank_marketing"], "seed": [42]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Legacy 0.5"):
        save_results(pd.DataFrame(), path)


def test_noor_baseline_reads_the_protocol_selected_for_seed(tmp_path, monkeypatch):
    protocol_dir = tmp_path / "protocols"
    protocol_dir.mkdir()
    (protocol_dir / "baseline_protocol_default_42_cv0.json").write_text(
        '{"protocol_id": "frozen-cv0"}'
    )
    results = tmp_path / "baselines.csv"
    pd.DataFrame(
        {
            "dataset": ["default", "default"],
            "arm": ["cleaned", "cleaned"],
            "seed": [42, 42],
            "protocol_id": ["old", "frozen-cv0"],
            "selected": [True, True],
            "test_pr_auc": [0.1, 0.6],
            "test_f1": [0.2, 0.5],
        }
    ).to_csv(results, index=False)

    assert augmented.noor_baseline("default", results, protocol_dir) == {
        "test_pr_auc": 0.6,
        "test_f1": 0.5,
    }


def test_augmented_models_cannot_drift_from_frozen_suite(tables, tmp_path):
    with pytest.raises(ValueError, match="complete frozen"):
        run_experiment(
            tables,
            CONFIG,
            "smotenc",
            "smotenc",
            seeds=(42,),
            model_names=("LightGBM",),
            protocol_dir=tmp_path,
        )
