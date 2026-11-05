import json

import numpy as np
import pandas as pd
import pytest

from agentic_gan.schemas import CheckResult, QualityThresholds
from agentic_gan.tools.quality_fidelity import (
    BANK_RULES,
    CRAMERS_V,
    DEFAULT_RULES,
    LABEL_BALANCE,
    MEAN_DISTANCE,
    POLISH_RULES,
    RULES,
    SCHEMA_MATCH,
    SPEARMAN,
    WORST_DISTANCE,
    association_difference,
    column_kinds,
    cramers_v,
    fidelity_checks,
    ks_distance,
    main,
    real_vs_real,
    rule_violation_rate,
    schema_thresholds,
    spearman_matrix,
    suggest_thresholds,
    tv_distance,
)


def make_bank(n: int, seed: int, target: int = 1) -> pd.DataFrame:
    """A small Bank-like table that obeys the consistency rules."""
    rng = np.random.default_rng(seed)
    contacted = rng.random(n) < 0.3
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, n),
            "balance": rng.normal(1000, 500, n).round(),
            "job": rng.choice(
                ["admin", "technician", "services", "retired"], n, p=[0.4, 0.3, 0.2, 0.1]
            ),
            "marital": rng.choice(["married", "single", "divorced"], n, p=[0.5, 0.35, 0.15]),
            "pdays": np.where(contacted, rng.integers(1, 400, n), -1),
            "poutcome": np.where(
                contacted, rng.choice(["success", "failure", "other"], n), "unknown"
            ),
            "previously_contacted": contacted.astype(int),
            "previous": np.where(contacted, rng.integers(1, 5, n), 0),
            "y": target,
        }
    )


DISCRETE = ("job", "marital", "poutcome", "previously_contacted")


def run(real, synth, **kwargs):
    return fidelity_checks(real, synth, target="y", discrete=DISCRETE, rules=BANK_RULES, **kwargs)


def by_name(report):
    return {c.name: c for c in report.checks}


# --- column typing ----------------------------------------------------------------------------


def test_text_columns_are_categorical_even_when_not_object_dtype():
    df = pd.DataFrame(
        {
            "a": pd.Series(["x", "y", "x"]),  # str dtype in pandas 3
            "b": pd.Series(["x", "y", "x"], dtype="string"),
            "c": pd.Series(["x", "y", "x"], dtype=object),
            "d": [1.0, 2.0, 3.0],
            "e": [True, False, True],
            "f": [0, 1, 0],
        }
    )
    kinds = column_kinds(df, list(df.columns), discrete=["f"])
    assert kinds == {
        "a": "categorical",
        "b": "categorical",
        "c": "categorical",
        "d": "numeric",
        "e": "categorical",
        "f": "categorical",
    }


def test_numeric_codes_stay_numeric_unless_listed_as_discrete():
    df = pd.DataFrame({"code": [0, 1, 1, 0]})
    assert column_kinds(df, ["code"]) == {"code": "numeric"}
    assert column_kinds(df, ["code"], discrete=["code"]) == {"code": "categorical"}


# --- distances --------------------------------------------------------------------------------


def test_tv_distance_known_values():
    half = pd.Series(["a"] * 5 + ["b"] * 5)
    skewed = pd.Series(["a"] * 7 + ["b"] * 3)
    assert tv_distance(half, half) == 0.0
    assert tv_distance(half, skewed) == pytest.approx(0.2)
    # a category the real data never has counts as mass that is not matched
    assert tv_distance(pd.Series(["a", "a"]), pd.Series(["b", "b"])) == 1.0
    assert tv_distance(pd.Series(["a", "a"]), pd.Series(["a", "b"])) == pytest.approx(0.5)


def test_tv_distance_empty_inputs():
    empty = pd.Series([], dtype=str)
    assert tv_distance(empty, empty) == 0.0
    assert tv_distance(empty, pd.Series(["a"])) == 1.0


def test_ks_distance_known_values():
    same = pd.Series(np.arange(100.0))
    assert ks_distance(same, same) == 0.0
    assert ks_distance(same, same + 1000) == 1.0
    assert ks_distance(same, same + 50) == pytest.approx(0.5)
    assert ks_distance(pd.Series([], dtype=float), pd.Series([1.0])) == 1.0


# --- associations -----------------------------------------------------------------------------


def test_cramers_v_perfect_independent_and_constant():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.choice(list("abc"), 2000))
    assert cramers_v(x, x) == pytest.approx(1.0, abs=0.01)
    assert cramers_v(x, pd.Series(rng.choice(list("uvw"), 2000))) < 0.05
    assert cramers_v(x, pd.Series(["k"] * 2000)) == 0.0


def test_spearman_matrix_treats_a_constant_column_as_no_association():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4], "c": [5, 5, 5, 5]})
    matrix = spearman_matrix(df, ["a", "b", "c"])
    assert matrix.loc["a", "b"] == pytest.approx(1.0)
    assert matrix.loc["a", "c"] == 0.0


def test_association_difference():
    one = pd.DataFrame([[1.0, 0.8], [0.8, 1.0]], index=["a", "b"], columns=["a", "b"])
    other = pd.DataFrame([[1.0, 0.2], [0.2, 1.0]], index=["a", "b"], columns=["a", "b"])
    assert association_difference(one, one) == 0.0
    assert association_difference(one, other) == pytest.approx(0.6)
    assert association_difference(one.iloc[:1, :1], one.iloc[:1, :1]) is None


# --- the checks -------------------------------------------------------------------------------


def test_a_resample_of_the_same_distribution_passes_the_starting_thresholds():
    real, synth = make_bank(3000, 0), make_bank(3000, 1)
    report = run(real, synth)
    assert report.passed
    assert by_name(report)[MEAN_DISTANCE].value < 0.05


def test_identical_data_scores_zero_everywhere():
    real = make_bank(500, 0)
    report = run(real, real.copy())
    assert all(c.value == 0.0 for c in report.checks)


def test_results_are_schema_check_results_with_unique_names():
    report = run(make_bank(300, 0), make_bank(300, 1))
    assert all(isinstance(c, CheckResult) for c in report.checks)
    names = [c.name for c in report.checks]
    assert len(names) == len(set(names))
    expected = {MEAN_DISTANCE, WORST_DISTANCE, SPEARMAN, CRAMERS_V, LABEL_BALANCE, SCHEMA_MATCH}
    assert expected <= set(names)
    assert {"rule:pdays_poutcome", "rule:previously_contacted_pdays"} <= set(names)
    assert "column_distance:job" in names


def test_per_column_results_are_soft_and_do_not_decide():
    real = make_bank(1000, 0)
    synth = make_bank(1000, 1)
    synth["marital"] = "single"  # one badly distorted column
    report = run(real, synth)
    soft = {c.name: c for c in report.checks if not c.hard}
    assert soft["column_distance:marital"].passed is False
    assert report.worst_column == "marital"
    assert by_name(report)[WORST_DISTANCE].passed is False
    assert not report.passed


def test_per_column_results_can_be_left_out():
    report = run(make_bank(200, 0), make_bank(200, 1), per_column=False)
    assert not any(c.name.startswith("column_distance:") for c in report.checks)


def test_rules_count_rows_that_break_them():
    df = make_bank(10, 0)
    assert rule_violation_rate(df, BANK_RULES[0]) == 0.0
    broken = df.copy()
    broken.loc[broken.index[:2], "poutcome"] = "success"
    broken.loc[broken.index[:2], "pdays"] = -1  # pdays -1 with a known outcome
    rate = rule_violation_rate(broken, BANK_RULES[0])
    assert rate == pytest.approx(
        ((broken["pdays"] == -1) != (broken["poutcome"] == "unknown")).mean()
    )
    assert rate > 0


def test_both_directions_of_the_pdays_rule_are_caught():
    df = pd.DataFrame(
        {
            "pdays": [-1, 5, -1, 7],
            "poutcome": ["unknown", "success", "success", "unknown"],
            "previously_contacted": [0, 1, 0, 0],
        }
    )
    # rows 2 and 3 break the pdays/poutcome rule, one in each direction
    assert rule_violation_rate(df, BANK_RULES[0]) == 0.5
    # only row 3 breaks the previously_contacted rule: pdays is 7 but the flag is 0
    assert rule_violation_rate(df, BANK_RULES[1]) == 0.25


def test_the_other_bank_rules_catch_their_own_violations():
    df = pd.DataFrame({"previous": [0, 2, 0, 1], "pdays": [-1, 10, 30, -1]})
    # row 2 has a past contact count of 0 but pdays 30, row 3 the reverse
    assert rule_violation_rate(df, BANK_RULES[2]) == 0.5
    ranges = pd.DataFrame({"pdays": [-1, 0, 5, -3]})
    assert rule_violation_rate(ranges, BANK_RULES[3]) == 0.5  # 0 and -3 are not allowed


def test_a_rule_threshold_is_a_rate():
    real = make_bank(1000, 0)
    synth = make_bank(1000, 1)
    synth.loc[synth.index[:30], "poutcome"] = "success"
    synth.loc[synth.index[:30], "pdays"] = -1  # 3% break the pdays rule at most
    loose = {**schema_thresholds(rules=BANK_RULES), "rule:pdays_poutcome": 0.5}
    strict = {**loose, "rule:pdays_poutcome": 0.001}
    assert by_name(run(real, synth, thresholds=loose))["rule:pdays_poutcome"].passed
    assert not by_name(run(real, synth, thresholds=strict))["rule:pdays_poutcome"].passed


def test_rule_on_missing_columns_is_a_clear_error():
    with pytest.raises(ValueError, match="pdays"):
        rule_violation_rate(pd.DataFrame({"a": [1]}), BANK_RULES[0])


def _default_rows(n: int) -> pd.DataFrame:
    frame = {f"PAY_{m}": [0] * n for m in range(1, 7)}
    frame.update({f"PAY_AMT{m}": [100] * n for m in range(1, 7)})
    frame.update({"SEX": [1] * n, "EDUCATION": [2] * n, "MARRIAGE": [1] * n})
    return pd.DataFrame(frame)


def test_default_rules_catch_bad_codes_and_negative_payments():
    by_rule = {rule.name: rule for rule in DEFAULT_RULES}
    clean = _default_rows(4)
    for rule in DEFAULT_RULES:
        assert rule_violation_rate(clean, rule) == 0.0

    codes = clean.copy()
    codes.loc[0, "PAY_3"] = 9  # above the observed range
    codes.loc[1, "PAY_6"] = -3  # below it
    assert rule_violation_rate(codes, by_rule["pay_status_codes"]) == 0.5

    paid = clean.copy()
    paid.loc[2, "PAY_AMT1"] = -1
    assert rule_violation_rate(paid, by_rule["pay_amount_non_negative"]) == 0.25

    static = clean.copy()
    static.loc[3, "SEX"] = 3
    static.loc[0, "MARRIAGE"] = 7
    assert rule_violation_rate(static, by_rule["static_codes"]) == 0.5


def test_polish_rules_catch_missing_ratios_and_bad_labels():
    by_rule = {rule.name: rule for rule in POLISH_RULES}
    clean = pd.DataFrame({f"A{i}": [0.5, 1.5] for i in range(1, 65)})
    clean["bankrupt"] = [0, 1]
    clean["cohort"] = ["3year", "1year"]
    for rule in POLISH_RULES:
        assert rule_violation_rate(clean, rule) == 0.0

    gap = clean.copy()
    gap.loc[0, "A37"] = float("nan")
    assert rule_violation_rate(gap, by_rule["ratios_numeric"]) == 0.5

    with pytest.raises(ValueError, match="A64"):
        rule_violation_rate(clean.drop(columns=["A64"]), by_rule["ratios_numeric"])

    labels = clean.copy()
    labels.loc[1, "bankrupt"] = 2
    labels.loc[0, "cohort"] = "6year"
    assert rule_violation_rate(labels, by_rule["target_binary"]) == 0.5
    assert rule_violation_rate(labels, by_rule["cohort_label"]) == 0.5


def test_every_dataset_rule_set_is_registered():
    assert set(RULES) == {"bank_marketing", "default", "polish"}
    assert RULES["default"] is DEFAULT_RULES
    assert RULES["polish"] is POLISH_RULES


def test_label_balance_for_files_that_hold_only_the_positive_class():
    real, synth = make_bank(500, 0), make_bank(500, 1)
    assert by_name(run(real, synth))[LABEL_BALANCE].value == 0.0
    synth.loc[synth.index[:50], "y"] = 0  # 10% negatives
    check = by_name(run(real, synth))[LABEL_BALANCE]
    assert check.value == pytest.approx(0.1)
    assert not check.passed


def test_label_balance_with_an_expected_share():
    real = make_bank(500, 0)
    synth = make_bank(500, 1)
    synth.loc[synth.index[:400], "y"] = 0  # 20% positive
    check = by_name(run(real, synth, expected_positive_share=0.2))[LABEL_BALANCE]
    assert check.value == pytest.approx(0.0)


def test_missing_columns_fail_the_schema_check_and_the_rest_still_runs():
    real, synth = make_bank(500, 0), make_bank(500, 1).drop(columns=["job"])
    report = run(real, synth)
    assert by_name(report)[SCHEMA_MATCH].value == 1
    assert not by_name(report)[SCHEMA_MATCH].passed
    assert "column_distance:job" not in by_name(report)
    assert not report.passed


def test_extra_columns_such_as_is_synthetic_and_source_row_are_ignored():
    real = make_bank(300, 0).assign(source_row=range(300))
    synth = make_bank(300, 1).assign(is_synthetic=1)
    names = by_name(run(real, synth))
    assert "column_distance:source_row" not in names
    assert "column_distance:is_synthetic" not in names
    assert names[SCHEMA_MATCH].value == 0


def test_no_shared_columns_is_an_error():
    with pytest.raises(ValueError, match="no columns"):
        fidelity_checks(
            pd.DataFrame({"a": [1], "y": [1]}), pd.DataFrame({"b": [1], "y": [1]}), target="y"
        )


# --- thresholds and calibration ---------------------------------------------------------------


def test_schema_thresholds_come_from_quality_thresholds():
    q = QualityThresholds(
        mean_column_distance_max=0.2, column_distance_max=0.3, correlation_diff_max=0.15
    )
    t = schema_thresholds(q, rules=BANK_RULES)
    assert t[MEAN_DISTANCE] == 0.2
    assert t[WORST_DISTANCE] == 0.3
    assert t[SPEARMAN] == t[CRAMERS_V] == 0.15
    assert t["rule:pdays_poutcome"] == 0.0  # the starting count of 0, until calibrated


def test_suggested_thresholds_are_a_margin_over_the_baseline_with_floors():
    baseline = {
        MEAN_DISTANCE: 0.023,
        WORST_DISTANCE: 0.048,
        SPEARMAN: 0.01,
        CRAMERS_V: 0.031,
        "rule:pdays_poutcome": 0.0,
        LABEL_BALANCE: 0.0,
        SCHEMA_MATCH: 0.0,
    }
    t = suggest_thresholds(baseline)
    assert t[MEAN_DISTANCE] == 0.05  # 0.046 rounded up to the next 0.005
    assert t[WORST_DISTANCE] == 0.1  # 0.096 rounded up
    assert t[SPEARMAN] == 0.03  # the floor, 2x of 0.01 would be 0.02
    assert t[CRAMERS_V] == 0.065
    assert t["rule:pdays_poutcome"] == 0.005  # real data breaks the rule a little, never zero
    assert t[LABEL_BALANCE] == 0.01
    assert t[SCHEMA_MATCH] == 0.0
    assert suggest_thresholds(baseline, margin=3.0)[MEAN_DISTANCE] == 0.07


def test_real_vs_real_uses_only_rows_of_the_positive_class():
    train = pd.concat([make_bank(400, 0, target=1), make_bank(600, 1, target=0)], ignore_index=True)
    val = pd.concat([make_bank(150, 2, target=1), make_bank(250, 3, target=0)], ignore_index=True)
    report = real_vs_real(train, val, target="y", discrete=DISCRETE, rules=BANK_RULES)
    assert report.n_reference_rows == 400
    assert report.n_rows == 150


def test_suggested_thresholds_pass_real_data_and_fail_a_distorted_copy():
    train, val = make_bank(3000, 0), make_bank(1000, 1)
    baseline = real_vs_real(train, val, target="y", discrete=DISCRETE, rules=BANK_RULES)
    thresholds = suggest_thresholds(baseline.values)
    assert run(train, make_bank(1000, 2), thresholds=thresholds).passed
    distorted = make_bank(1000, 2)
    distorted["job"] = "admin"
    assert not run(train, distorted, thresholds=thresholds).passed


# --- scorecard command ------------------------------------------------------------------------


@pytest.fixture
def files(tmp_path):
    train = make_bank(3000, 0)
    val = make_bank(1000, 1)
    good = make_bank(1000, 2).assign(is_synthetic=1)
    bad = good.assign(job="admin")
    paths = {"train": tmp_path / "train.csv", "val": tmp_path / "val.csv"}
    train.to_csv(paths["train"], index=False)
    val.to_csv(paths["val"], index=False)
    good.to_csv(tmp_path / "good.csv", index=False)
    bad.to_csv(tmp_path / "bad.csv", index=False)
    return tmp_path


def score(files, *names, extra=()):
    return main(
        [
            *[str(files / name) for name in names],
            "--real-train", str(files / "train.csv"),
            "--real-val", str(files / "val.csv"),
            *extra,
        ]
    )  # fmt: skip


def test_scorecard_exit_status_and_table(files, capsys):
    assert score(files, "good.csv", "bad.csv") == 1
    out = capsys.readouterr().out
    assert "good" in out and "bad" in out
    assert "mean_column_distance" in out
    assert out.count("FAIL") == 1  # only the distorted file; the real-vs-real column passes


def test_scorecard_exit_status_zero_when_every_file_passes(files):
    assert score(files, "good.csv") == 0


def test_scorecard_with_no_files_prints_the_baseline_only(files, capsys):
    assert score(files) == 0
    assert "real-vs-real" in capsys.readouterr().out


def test_scorecard_per_column_table_and_json(files, capsys):
    out_json = files / "out" / "scorecard.json"
    assert score(files, "bad.csv", extra=["--per-column", "--json", str(out_json)]) == 1
    assert "job" in capsys.readouterr().out
    payload = json.loads(out_json.read_text())
    assert payload["files"]["bad"]["passed"] is False
    assert payload["files"]["bad"]["worst_column"] == "job"
    assert {c["name"] for c in payload["files"]["bad"]["checks"]} >= {MEAN_DISTANCE, WORST_DISTANCE}
    assert payload["suggested_thresholds"][MEAN_DISTANCE] >= 0.03
    CheckResult.model_validate(payload["real_vs_real"][0])


def test_scorecard_schema_thresholds_option(files, capsys):
    assert score(files, "good.csv", extra=["--thresholds", "schema"]) == 0
    assert "thresholds: schema" in capsys.readouterr().out


def test_scorecard_missing_file_is_a_clear_error(files):
    with pytest.raises(SystemExit, match="file not found"):
        score(files, "nope.csv")
