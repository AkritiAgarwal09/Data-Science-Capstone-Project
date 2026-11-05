"""Evaluation tools: arm comparison, the reliable-gain rule, segment metrics, number check."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentic_gan.agents.training import _save_metrics
from agentic_gan.schemas import Arm, ModelRecord, Status
from agentic_gan.tools.evaluation import (
    compare_arms,
    reliable_gain,
    segment_metrics,
    verify_report_numbers,
)

SEEDS = [42, 43, 44, 45, 46]
ORIGINAL = {
    "validation": [0.50, 0.52, 0.48, 0.51, 0.49],
    "test": [0.48, 0.50, 0.46, 0.49, 0.47],
}
# Paired difference from the original arm, per seed, by split.
NOISE = [0.002, -0.002, 0.001, -0.001, 0.0]
LIFTS = {
    ("smotenc", 0.25): {  # swings both ways: no lift beyond the noise
        "validation": [0.02, -0.02, 0.03, -0.03, 0.0],
        "test": [0.0, 0.01, -0.01, 0.0, 0.0],
    },
    ("smotenc", 0.5): {  # about -0.05 and -0.04 on every seed
        "validation": [-0.05 + n for n in NOISE],
        "test": [-0.04 + n for n in NOISE],
    },
    ("ctgan", 0.25): {  # about +0.03 and +0.02 on every seed
        "validation": [0.03 + n for n in NOISE],
        "test": [0.02 + n for n in NOISE],
    },
    ("ctgan", 0.5): {  # reliable on validation only
        "validation": [0.03 + n for n in NOISE],
        "test": [0.03, -0.02, 0.04, -0.03, 0.01],
    },
}


def arm_rows(arm, fill_level, scores):
    """Rows as the training agent saves them: two models per seed, one of them selected."""
    rows = []
    for split, values in scores.items():
        for seed, score in zip(SEEDS, values, strict=True):
            for model, selected, pr_auc in (("CatBoost", True, score), ("LightGBM", False, 0.99)):
                rows.append(
                    {
                        "dataset": "bank_marketing",
                        "arm": arm,
                        "fill_level": fill_level,
                        "seed": seed,
                        "model": model,
                        "split": split,
                        "status": "SUCCESS",
                        "selected": selected,
                        "pr_auc": pr_auc,
                        "f1": pr_auc / 2,
                        "precision": pr_auc / 4,
                        "recall": pr_auc * 2,
                    }
                )
    return rows


def results():
    rows = arm_rows("original", 0.0, ORIGINAL)
    rows += arm_rows("cleaned", 0.0, ORIGINAL)
    for (arm, level), lift in LIFTS.items():
        scores = {
            split: [o + d for o, d in zip(ORIGINAL[split], lift[split], strict=True)]
            for split in lift
        }
        rows += arm_rows(arm, level, scores)
    # A failed fit that carries the selected flag must not count.
    rows.append({**rows[0], "model": "XGBoost", "status": "FAILED", "pr_auc": np.nan})
    return pd.DataFrame(rows)


def pick(table, arm, fill_level, split, metric="pr_auc"):
    row = table[
        table.arm.eq(arm)
        & np.isclose(table.fill_level, fill_level)
        & table.split.eq(split)
        & table.metric.eq(metric)
    ]
    assert len(row) == 1
    return row.iloc[0]


def test_compare_arms_gives_mean_spread_and_paired_difference():
    table = compare_arms(results())

    original = pick(table, "original", 0.0, "validation")
    assert original.n_seeds == 5
    assert original["mean"] == pytest.approx(0.50)
    assert original["sd"] == pytest.approx(np.std(ORIGINAL["validation"], ddof=1))
    assert math.isnan(original.diff_mean) and original.n_paired == 0

    gan = pick(table, "ctgan", 0.25, "validation")
    lift = LIFTS[("ctgan", 0.25)]["validation"]
    assert gan["mean"] == pytest.approx(0.50 + 0.03)
    assert gan.diff_mean == pytest.approx(np.mean(lift))
    assert gan.diff_sd == pytest.approx(np.std(lift, ddof=1))
    assert gan.n_paired == 5
    # The unselected LightGBM rows (0.99) and the failed fit are left out; other metrics follow.
    assert pick(table, "ctgan", 0.25, "test", "f1").diff_mean == pytest.approx(0.02 / 2)
    assert pick(table, "ctgan", 0.25, "test", "recall").diff_mean == pytest.approx(0.02 * 2)


def test_compare_arms_orders_arms_then_fill_levels():
    arms = compare_arms(results())[["arm", "fill_level"]].drop_duplicates()
    assert arms.values.tolist() == [
        ["original", 0.0],
        ["cleaned", 0.0],
        ["smotenc", 0.25],
        ["smotenc", 0.5],
        ["ctgan", 0.25],
        ["ctgan", 0.5],
    ]


def test_compare_arms_pairs_only_the_seeds_both_arms_have():
    data = results()
    data = data[~(data.arm.eq("original") & data.seed.eq(44))]
    gan = pick(compare_arms(data), "ctgan", 0.25, "validation")
    assert (gan.n_seeds, gan.n_paired) == (5, 4)
    kept = [i for i, seed in enumerate(SEEDS) if seed != 44]
    assert gan.diff_mean == pytest.approx(
        np.mean([LIFTS[("ctgan", 0.25)]["validation"][i] for i in kept])
    )


def test_compare_arms_rejects_two_selected_models_for_one_seed():
    data = results()
    data.loc[data.model.eq("LightGBM"), "selected"] = True
    with pytest.raises(ValueError, match="more than one selected model"):
        compare_arms(data)
    with pytest.raises(ValueError, match="missing columns"):
        compare_arms(results().drop(columns="split"))


@pytest.mark.parametrize(
    ("arm", "level", "verdict"),
    [
        ("ctgan", 0.25, "reliable gain"),  # lift above the spread on both splits
        ("ctgan", 0.5, "inconclusive"),  # validation only
        ("smotenc", 0.25, "inconclusive"),  # lift inside the noise
        ("smotenc", 0.5, "reliable loss"),  # below minus the spread on both splits
        ("original", 0.0, "not enough paired seeds"),  # nothing to pair with
    ],
)
def test_reliable_gain_needs_a_lift_above_the_spread_on_validation_and_test(arm, level, verdict):
    table = compare_arms(results())
    result = reliable_gain(table, dataset="bank_marketing", arm=arm, fill_level=level)

    assert result.verdict == verdict
    assert result.reliable is (verdict == "reliable gain")
    if arm != "original":
        for split in ("validation", "test"):
            row = pick(table, arm, level, split)
            assert result.lift[split] == pytest.approx(row.diff_mean)
            assert result.spread[split] == pytest.approx(row.diff_sd)
            assert f"{split} lift {row.diff_mean:.4f} vs spread {row.diff_sd:.4f}" in result.reason


def test_reliable_gain_needs_at_least_three_paired_seeds():
    data = results()
    table = compare_arms(data[data.seed.isin(SEEDS[:2])])
    result = reliable_gain(table, dataset="bank_marketing", arm="ctgan", fill_level=0.25)
    assert result.verdict == "not enough paired seeds"


def test_comparison_reads_the_training_agents_own_metrics_file(tmp_path):
    def record(arm, level, seed, pr_auc, model="CatBoost", selected=True):
        scores = {"pr_auc": pr_auc, "f1": 0.4, "precision": 0.5, "recall": 0.4, "roc_auc": 0.8}
        return ModelRecord(
            status=Status.SUCCESS,
            arm=arm,
            fill_level=level,
            model=model,
            seed=seed,
            selected=selected,
            threshold=0.2,
            protocol_id="p",
            training_rows=100,
            model_path=Path("m.joblib"),
            val_predictions_path=Path("v.csv"),
            test_predictions_path=Path("t.csv"),
            val_metrics=scores,
            test_metrics=scores,
        )

    records = []
    for seed, base in zip(SEEDS[:3], [0.50, 0.52, 0.48], strict=True):
        records += [record(Arm.RAW, None, seed, base), record(Arm.GAN, 0.25, seed, base + 0.03)]
        records.append(record(Arm.RAW, None, seed, 0.9, model="LightGBM", selected=False))
    path, _ = _save_metrics(records, "bank_marketing", tmp_path)

    table = compare_arms(pd.read_csv(path))
    gan = pick(table, "ctgan", 0.25, "test")
    assert gan.diff_mean == pytest.approx(0.03) and gan.n_paired == 3
    assert reliable_gain(table, dataset="bank_marketing", arm="ctgan", fill_level=0.25).reliable


# --- segments ----------------------------------------------------------------------------------


def predictions():
    return pd.DataFrame(
        {
            "source_row": range(8),
            "y_true": [1, 1, 0, 0, 1, 0, 0, 0],
            "probability": [0.9, 0.4, 0.2, 0.1, 0.8, 0.7, 0.3, 0.05],
            "prediction": [1, 0, 0, 0, 1, 1, 0, 0],
        }
    )


def test_segment_metrics_scores_each_segment_and_flags_small_ones():
    features = pd.DataFrame({"source_row": range(8), "job": ["admin"] * 4 + ["student"] * 4})
    table = segment_metrics(predictions(), features, "job", min_positives=2).set_index("segment")

    admin, student = table.loc["admin"], table.loc["student"]
    assert (admin.n, admin.positives, bool(admin.small)) == (4, 2, False)
    assert (admin.precision, admin.recall) == (1.0, 0.5)
    assert admin.f1 == pytest.approx(2 / 3)
    assert admin.pr_auc == pytest.approx(1.0)
    assert (student.n, student.positives, bool(student.small)) == (4, 1, True)
    assert (student.precision, student.recall) == (0.5, 1.0)


def test_segment_metrics_bands_numeric_columns_and_skips_missing_ones():
    n = 20
    data = pd.DataFrame(
        {
            "source_row": range(n),
            "y_true": [1, 0] * (n // 2),
            "probability": np.linspace(0.05, 0.95, n),
            "prediction": [0, 1] * (n // 2),
        }
    )
    features = pd.DataFrame({"source_row": range(n), "age": range(20, 20 + n)})
    table = segment_metrics(data, features, "age")
    assert table.n.tolist() == [5, 5, 5, 5]
    assert segment_metrics(data, features, "income").empty


# --- number check ------------------------------------------------------------------------------


@pytest.fixture
def result_files(tmp_path):
    table = pd.DataFrame(
        {
            "arm": ["ctgan", "ctgan"],
            "fill_level": [0.25, 0.5],
            "n_seeds": [5, 5],
            "mean": [0.4812, 0.4701],
            "diff_mean": [0.0524, -0.0515],
        }
    )
    table.to_csv(tmp_path / "comparison.csv", index=False)
    (tmp_path / "counts.json").write_text(
        json.dumps({"test_rows": 8238, "ok": True, "x": [0.123456]})
    )
    return [tmp_path / "comparison.csv", tmp_path / "counts.json"]


def test_numbers_that_are_in_the_results_pass(result_files):
    report = (
        "1. Summary\n"
        "CTGAN v1 at 25% fill reached PR-AUC 0.48 (0.4812 exactly), a lift of 0.052 over the "
        "original arm across 5 seeds; at 0.5 fill it dropped by 0.05 (-0.0515). The test set "
        "has 8,238 rows and pr_auc2 is a column name. Another value is 0.123456."
    )
    check = verify_report_numbers(report, result_files)
    assert check.verified and check.unverified == ()
    assert check.checked == 10


@pytest.mark.parametrize(
    ("report", "unverified"),
    [
        ("PR-AUC reached 0.61 with the CTGAN arm.", ("0.61",)),
        ("It was run on 12 seeds.", ("12",)),  # a whole number must match exactly
        ("A 30% fill level was best.", ("30%",)),
        ("The lift was 0.052, then 0.07, then 0.07.", ("0.07",)),
    ],
)
def test_a_number_that_is_not_in_the_results_fails(result_files, report, unverified):
    check = verify_report_numbers(report, result_files)
    assert not check.verified
    assert check.unverified == unverified


def test_numbers_the_caller_vouches_for_are_allowed(result_files):
    report = "We trained 4 models."
    assert not verify_report_numbers(report, result_files).verified
    assert verify_report_numbers(report, result_files, allowed=[4]).verified


def test_numeric_segment_boundaries_are_grounded_for_report_verification(tmp_path):
    n = 24
    data = pd.DataFrame(
        {
            "source_row": range(n),
            "y_true": [0, 1] * (n // 2),
            "probability": np.linspace(0.05, 0.95, n),
            "prediction": [0, 1] * (n // 2),
        }
    )
    features = pd.DataFrame({"source_row": range(n), "age": range(21, 21 + n)})
    table = segment_metrics(data, features, "age")
    path = tmp_path / "segments.csv"
    table.to_csv(path, index=False)
    assert table.segment_lower_bound.min() == pytest.approx(20.999)
    assert verify_report_numbers("Age band (20.999, 26.75] is observed.", [path]).verified
    assert not verify_report_numbers("Age band (20.998, 26.75] is observed.", [path]).verified
    categorical = segment_metrics(data, features.assign(job="admin"), "job")
    assert categorical.segment_lower_bound.isna().all()


def test_missing_numeric_segment_values_are_not_a_string_nan_group():
    n = 24
    data = pd.DataFrame(
        {
            "source_row": range(n),
            "y_true": [0, 1] * (n // 2),
            "probability": np.linspace(0.05, 0.95, n),
            "prediction": [0, 1] * (n // 2),
        }
    )
    features = pd.DataFrame({"source_row": range(n), "age": [np.nan, *range(21, 21 + n - 1)]})
    table = segment_metrics(data, features, "age")
    assert table.n.sum() == n - 1
    assert "nan" not in table.segment.tolist()
