"""Fidelity and rule checks: how close a synthetic set is to real rows of the same class.

This is the fidelity half of the quality gate. The memorization and classifier checks are a
separate tool. The Quality Evaluation Agent calls it once per fill level and puts the returned
``CheckResult`` objects into that level's report.

The reference is the real **training** rows of the class the synthetic file holds. The Bank CTGAN
files hold only subscribers (``y == 1``), so they are compared with real train subscribers. To
know what "close" means, every check is also run real-vs-real: real validation rows of that class
against the same real training rows. That is the noise floor, and ``suggest_thresholds`` turns it
into thresholds.

Checks (names are stable, the agent and the reports use them):

* ``mean_column_distance`` and ``worst_column_distance``: per-column distance, KS statistic for a
  numeric column and total variation distance for a categorical one (both between 0 and 1). The
  mean over columns and the worst column. The distance of every column is also returned as a soft
  ``column_distance:<col>`` check, so a report can list the failing columns.
* ``spearman_diff`` and ``cramers_v_diff``: mean absolute difference of the pairwise associations,
  Spearman between numeric columns and bias-corrected Cramér's V between categorical columns.
  Numeric-categorical pairs are not scored.
* ``rule:<name>``: share of rows that break a consistency rule (``RULES``). Real data breaks some
  rules a little, so thresholds are rates, not zero.
* ``label_balance``: how far the share of positive rows is from the expected share (1.0 when the
  file holds only the positive class).
* ``schema_match``: number of real columns missing from the synthetic file.

A column is categorical when it is not numeric (with pandas 3 a text column is ``str``, not
``object``), or when it is listed in ``discrete`` (numeric codes such as ``previously_contacted``).

Run it on any synthetic file::

    uv run python -m agentic_gan.tools.quality_fidelity data/interim/bank_marketing/synthetic/*.csv
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from agentic_gan.schemas import CheckResult, QualityThresholds
from agentic_gan.utils.datasets import DATASETS

IS_SYNTHETIC = "is_synthetic"
SOURCE_ROW = "source_row"

MEAN_DISTANCE = "mean_column_distance"
WORST_DISTANCE = "worst_column_distance"
SPEARMAN = "spearman_diff"
CRAMERS_V = "cramers_v_diff"
LABEL_BALANCE = "label_balance"
SCHEMA_MATCH = "schema_match"

# Smallest threshold ``suggest_thresholds`` will propose, so a lucky tiny baseline does not give
# a threshold below the sampling noise of a few thousand rows.
SUGGESTION_FLOORS = {
    MEAN_DISTANCE: 0.03,
    WORST_DISTANCE: 0.06,
    SPEARMAN: 0.03,
    CRAMERS_V: 0.03,
    "rule": 0.005,
    LABEL_BALANCE: 0.01,
}


# --- consistency rules ------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """A row-level consistency rule. ``violated`` is True where a row breaks the rule."""

    name: str
    description: str
    violated: Callable[[pd.DataFrame], pd.Series]


BANK_RULES = (
    Rule(
        "pdays_poutcome",
        "pdays is -1 exactly when poutcome is unknown",
        lambda d: (d["pdays"] == -1) != (d["poutcome"] == "unknown"),
    ),
    Rule(
        "previously_contacted_pdays",
        "previously_contacted is 1 exactly when pdays is not -1",
        lambda d: (d["previously_contacted"] == 1) != (d["pdays"] != -1),
    ),
    Rule(
        "previous_pdays",
        "previous is above 0 exactly when pdays is not -1",
        lambda d: (d["previous"] > 0) != (d["pdays"] != -1),
    ),
    Rule(
        "pdays_range",
        "pdays is -1 or at least 1",
        lambda d: (d["pdays"] == 0) | (d["pdays"] < -1),
    ),
)

# Default of Credit Card Clients (rules DC-5 and DC-9). Repayment status is an ordinal code:
# -2 and -1 are paid, 0 is revolving, 1 to 8 are months of delay. Amount paid cannot be negative.
DEFAULT_PAY_CODES = frozenset(range(-2, 9))
_PAY_STATUS = [f"PAY_{month}" for month in range(1, 7)]
_PAY_AMOUNT = [f"PAY_AMT{month}" for month in range(1, 7)]
_DEFAULT_CODE_SETS = {
    "SEX": frozenset({1, 2}),
    "EDUCATION": frozenset(range(0, 7)),
    "MARRIAGE": frozenset(range(0, 4)),
}


def _any_column(frame: pd.DataFrame, columns: Sequence[str], test) -> pd.Series:
    broken = pd.Series(False, index=frame.index)
    for column in columns:
        broken |= test(frame[column])
    return broken


DEFAULT_RULES = (
    Rule(
        "pay_status_codes",
        "every PAY_1..PAY_6 value is a repayment code observed in the data (-2 to 8)",
        lambda d: _any_column(d, _PAY_STATUS, lambda s: ~s.isin(DEFAULT_PAY_CODES)),
    ),
    Rule(
        "pay_amount_non_negative",
        "every PAY_AMT1..PAY_AMT6 value is zero or more",
        lambda d: _any_column(d, _PAY_AMOUNT, lambda s: s < 0),
    ),
    Rule(
        "static_codes",
        "SEX, EDUCATION and MARRIAGE use the coded values from the source",
        lambda d: _any_column(
            d, list(_DEFAULT_CODE_SETS), lambda s: ~s.isin(_DEFAULT_CODE_SETS[s.name])
        ),
    ),
)

# Polish Bankruptcy (rule PB-9). The 64 ratios must be present and numeric, the target binary,
# and the cohort one of the five horizon labels. Duplicate rows are handled by the sampler.
POLISH_RATIOS = [f"A{i}" for i in range(1, 65)]
POLISH_COHORTS = frozenset({"1year", "2year", "3year", "4year", "5year"})


def _polish_ratio_rows_broken(d: pd.DataFrame) -> pd.Series:
    missing = [c for c in POLISH_RATIOS if c not in d.columns]
    if missing:
        raise KeyError(missing[0])
    numeric = d[POLISH_RATIOS].apply(pd.to_numeric, errors="coerce")
    return numeric.isna().any(axis=1)


POLISH_RULES = (
    Rule(
        "ratios_numeric",
        "all 64 ratio columns A1..A64 are present and every value is a finite number",
        _polish_ratio_rows_broken,
    ),
    Rule(
        "target_binary",
        "bankrupt is 0 or 1",
        lambda d: ~d["bankrupt"].isin({0, 1}),
    ),
    Rule(
        "cohort_label",
        "cohort is one of the five horizon labels",
        lambda d: ~d["cohort"].isin(POLISH_COHORTS),
    ),
)

RULES: dict[str, tuple[Rule, ...]] = {
    "bank_marketing": BANK_RULES,
    "default": DEFAULT_RULES,
    "polish": POLISH_RULES,
}


def rule_name(rule: Rule) -> str:
    return f"rule:{rule.name}"


def rule_violation_rate(df: pd.DataFrame, rule: Rule) -> float:
    """Share of rows in ``df`` that break ``rule``."""
    if df.empty:
        return 0.0
    try:
        return float(rule.violated(df).mean())
    except KeyError as error:
        raise ValueError(
            f"rule {rule.name!r} needs column {error}, which is not in the data"
        ) from error


# --- thresholds -------------------------------------------------------------------------------


def schema_thresholds(
    thresholds: QualityThresholds | None = None, rules: Sequence[Rule] = ()
) -> dict[str, float]:
    """The starting thresholds from ``QualityThresholds``, keyed by check name.

    ``sanity_violations_max`` is a count that starts at 0. For a rate that means no tolerance,
    which the real data itself fails, so the rule thresholds here are 0.0 until they are calibrated.
    """
    q = thresholds or QualityThresholds()
    out = {
        MEAN_DISTANCE: q.mean_column_distance_max,
        WORST_DISTANCE: q.column_distance_max,
        SPEARMAN: q.correlation_diff_max,
        CRAMERS_V: q.correlation_diff_max,
        LABEL_BALANCE: 0.01,
        SCHEMA_MATCH: 0.0,
    }
    out.update({rule_name(rule): float(q.sanity_violations_max) for rule in rules})
    return {key: float(value) for key, value in out.items()}


def suggest_thresholds(
    baseline: Mapping[str, float], margin: float = 2.0, step: float = 0.005
) -> dict[str, float]:
    """Thresholds from the real-vs-real values: ``margin`` times the baseline, rounded up.

    A synthetic set that is as close to the real training rows as real validation rows are passes
    with room to spare. A floor (``SUGGESTION_FLOORS``) keeps a threshold above the sampling noise.
    ``baseline`` maps check name to its real-vs-real value.
    """
    out = {}
    for name, value in baseline.items():
        if name == SCHEMA_MATCH:
            out[name] = 0.0
            continue
        floor = SUGGESTION_FLOORS["rule" if name.startswith("rule:") else name]
        out[name] = round(math.ceil(max(margin * value, floor) / step - 1e-9) * step, 4)
    return out


# --- column typing and per-column distance ----------------------------------------------------


def column_kinds(
    df: pd.DataFrame, columns: Sequence[str], discrete: Collection[str] = ()
) -> dict[str, str]:
    """``numeric`` or ``categorical`` per column. Not numeric (text, bool) means categorical."""
    kinds = {}
    for column in columns:
        series = df[column]
        is_number = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        kinds[column] = "numeric" if is_number and column not in discrete else "categorical"
    return kinds


def ks_distance(real: pd.Series, synth: pd.Series) -> float:
    """Kolmogorov-Smirnov statistic between two numeric samples (0 equal, 1 disjoint)."""
    a = pd.to_numeric(real, errors="coerce").dropna()
    b = pd.to_numeric(synth, errors="coerce").dropna()
    if a.empty or b.empty:
        return 0.0 if a.empty and b.empty else 1.0
    return float(ks_2samp(a, b).statistic)


def tv_distance(real: pd.Series, synth: pd.Series) -> float:
    """Total variation distance between two categorical samples (0 equal, 1 disjoint)."""
    if real.empty or synth.empty:
        return 0.0 if real.empty and synth.empty else 1.0
    p = real.value_counts(normalize=True)
    q = synth.value_counts(normalize=True)
    labels = p.index.union(q.index)
    gap = p.reindex(labels, fill_value=0.0) - q.reindex(labels, fill_value=0.0)
    return float(0.5 * gap.abs().sum())


def column_distances(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    columns: Sequence[str],
    discrete: Collection[str] = (),
) -> pd.DataFrame:
    """One row per column: its kind and its distance (KS if numeric, TV if categorical)."""
    kinds = column_kinds(real, columns, discrete)
    rows = []
    for column in columns:
        distance = ks_distance if kinds[column] == "numeric" else tv_distance
        rows.append((column, kinds[column], distance(real[column], synth[column])))
    return pd.DataFrame(rows, columns=["column", "kind", "distance"]).set_index("column")


# --- associations -----------------------------------------------------------------------------


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramér's V (Bergsma 2013) between two categorical columns, 0 to 1."""
    observed = pd.crosstab(x, y).to_numpy(dtype=float)
    n = observed.sum()
    rows, cols = observed.shape
    if n < 2 or rows < 2 or cols < 2:
        return 0.0
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / n
    chi2 = ((observed - expected) ** 2 / expected).sum()
    phi2 = max(0.0, chi2 / n - (cols - 1) * (rows - 1) / (n - 1))
    rows_c = rows - (rows - 1) ** 2 / (n - 1)
    cols_c = cols - (cols - 1) ** 2 / (n - 1)
    room = min(cols_c - 1, rows_c - 1)
    return float(math.sqrt(phi2 / room)) if room > 0 else 0.0


def spearman_matrix(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Spearman correlations between numeric columns. A constant column has no association (0)."""
    numbers = df[list(columns)].apply(pd.to_numeric, errors="coerce")
    return numbers.corr(method="spearman").fillna(0.0)


def cramers_v_matrix(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Cramér's V between every pair of categorical columns."""
    matrix = pd.DataFrame(0.0, index=list(columns), columns=list(columns))
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            matrix.loc[a, b] = matrix.loc[b, a] = cramers_v(df[a], df[b])
    return matrix


def association_difference(real: pd.DataFrame, synth: pd.DataFrame) -> float | None:
    """Mean absolute difference of two association matrices, off-diagonal. None below 2 columns."""
    if len(real.columns) < 2:
        return None
    other = synth.reindex(index=real.index, columns=real.columns).fillna(0.0)
    upper = np.triu_indices(len(real.columns), k=1)
    return float(np.abs(real.to_numpy()[upper] - other.to_numpy()[upper]).mean())


# --- label balance ----------------------------------------------------------------------------


def label_balance_gap(
    synth: pd.DataFrame, target: str, positive=1, expected_share: float = 1.0
) -> float:
    """Distance between the share of positive rows and the expected share."""
    if target not in synth.columns or synth.empty:
        return 1.0
    return abs(float((synth[target] == positive).mean()) - expected_share)


# --- the checks -------------------------------------------------------------------------------


@dataclass
class FidelityReport:
    """The checks for one synthetic set, plus the detail behind them."""

    checks: list[CheckResult]
    columns: pd.DataFrame  # per-column kind and distance
    n_rows: int
    n_reference_rows: int

    @property
    def passed(self) -> bool:
        """Only hard checks decide, as in ``LevelReport``."""
        return all(c.passed for c in self.checks if c.hard)

    @property
    def values(self) -> dict[str, float]:
        """Value of every hard check by name."""
        return {c.name: c.value for c in self.checks if c.hard}

    @property
    def worst_column(self) -> str:
        return str(self.columns["distance"].idxmax())

    def failing(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


def _check(
    name: str, value: float, thresholds: Mapping[str, float], hard: bool = True
) -> CheckResult:
    threshold = float(thresholds[name])
    value = float(value)
    return CheckResult(
        name=name, value=value, threshold=threshold, passed=value <= threshold, hard=hard
    )


def fidelity_checks(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    *,
    target: str,
    positive=1,
    discrete: Collection[str] = (),
    rules: Sequence[Rule] = (),
    thresholds: Mapping[str, float] | None = None,
    expected_positive_share: float = 1.0,
    per_column: bool = True,
) -> FidelityReport:
    """Score ``synth`` against ``real`` (real rows of the same class as the synthetic file).

    ``thresholds`` maps check name to its maximum, as ``schema_thresholds`` and
    ``suggest_thresholds`` return it. The default is the starting values from ``QualityThresholds``.
    """
    thresholds = dict(thresholds) if thresholds is not None else schema_thresholds(rules=rules)
    skip = {target, IS_SYNTHETIC, SOURCE_ROW}
    wanted = [c for c in real.columns if c not in skip]
    missing = [c for c in wanted if c not in synth.columns]
    columns = [c for c in wanted if c in synth.columns]
    if not columns:
        raise ValueError("the synthetic data shares no columns with the real data")

    distances = column_distances(real, synth, columns, discrete)
    kinds = distances["kind"].to_dict()
    checks = [
        _check(MEAN_DISTANCE, distances["distance"].mean(), thresholds),
        _check(WORST_DISTANCE, distances["distance"].max(), thresholds),
    ]
    for name, kind, matrix in (
        (SPEARMAN, "numeric", spearman_matrix),
        (CRAMERS_V, "categorical", cramers_v_matrix),
    ):
        cols = [c for c in columns if kinds[c] == kind]
        diff = (
            association_difference(matrix(real, cols), matrix(synth, cols))
            if len(cols) > 1
            else None
        )
        if diff is not None:
            checks.append(_check(name, diff, thresholds))
    for rule in rules:
        checks.append(_check(rule_name(rule), rule_violation_rate(synth, rule), thresholds))
    gap = label_balance_gap(synth, target, positive, expected_positive_share)
    checks.append(_check(LABEL_BALANCE, gap, thresholds))
    checks.append(_check(SCHEMA_MATCH, len(missing), thresholds))
    if per_column:
        for column, distance in distances["distance"].items():
            checks.append(
                CheckResult(
                    name=f"column_distance:{column}",
                    value=float(distance),
                    threshold=float(thresholds[WORST_DISTANCE]),
                    passed=float(distance) <= float(thresholds[WORST_DISTANCE]),
                    hard=False,
                )
            )
    return FidelityReport(checks, distances, n_rows=len(synth), n_reference_rows=len(real))


def real_vs_real(
    real_train: pd.DataFrame,
    real_val: pd.DataFrame,
    *,
    target: str,
    positive=1,
    **kwargs,
) -> FidelityReport:
    """The noise floor: real validation rows of the class against real training rows."""
    train = real_train[real_train[target] == positive]
    val = real_val[real_val[target] == positive]
    return fidelity_checks(train, val, target=target, positive=positive, **kwargs)


# --- scorecard --------------------------------------------------------------------------------


def _format_table(rows: list[list[str]]) -> str:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = [
        "  ".join(
            cell.ljust(w) if i == 0 else cell.rjust(w)
            for i, (cell, w) in enumerate(zip(row, widths, strict=True))
        )
        for row in rows
    ]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def _mark(check: CheckResult | None) -> str:
    if check is None:
        return "-"
    return f"{check.value:.3f}" + ("" if check.passed else " *")


def scorecard(
    baseline: FidelityReport,
    reports: Mapping[str, FidelityReport],
    thresholds: Mapping[str, float],
    per_column: bool = False,
) -> str:
    """A text table: a row per check, a column per synthetic file. ``*`` marks a failed check."""
    names = [c.name for c in baseline.checks if c.hard]
    by_file = {label: {c.name: c for c in report.checks} for label, report in reports.items()}
    base = {c.name: c for c in baseline.checks}
    header = ["check", "limit", "real-vs-real", *reports]
    rows = [header]
    for name in names:
        rows.append(
            [name, f"{thresholds[name]:.3f}", _mark(base[name])]
            + [_mark(by_file[label].get(name)) for label in reports]
        )
    rows.append(
        ["worst column", "", baseline.worst_column] + [r.worst_column for r in reports.values()]
    )
    rows.append(["rows", "", str(baseline.n_rows)] + [str(r.n_rows) for r in reports.values()])
    rows.append(
        ["verdict", "", "PASS" if baseline.passed else "FAIL"]
        + ["PASS" if r.passed else "FAIL" for r in reports.values()]
    )
    out = [_format_table(rows)]
    if per_column:
        limit = thresholds[WORST_DISTANCE]
        detail = [["column (distance)", "kind", "real-vs-real", *reports]]
        for column in baseline.columns.index:

            def cell(report: FidelityReport, column: str = column) -> str:
                value = (
                    report.columns.loc[column, "distance"]
                    if column in report.columns.index
                    else float("nan")
                )
                return f"{value:.3f}" + (" *" if value > limit else "")

            detail.append(
                [column, baseline.columns.loc[column, "kind"], cell(baseline)]
                + [cell(r) for r in reports.values()]
            )
        out += ["", _format_table(detail)]
    out.append("")
    out.append(
        f"* = above the limit ({baseline.n_reference_rows:,} real training rows as the reference)"
    )
    return "\n".join(out)


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"file not found: {path}")
    return pd.read_csv(path)


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Fidelity scorecard: score synthetic files against real training rows.",
        epilog="Exit status is 1 when any file fails a hard check.",
    )
    parser.add_argument("files", nargs="*", type=Path, help="synthetic CSV files to score")
    parser.add_argument("--dataset", default="bank_marketing", choices=sorted(DATASETS))
    parser.add_argument(
        "--real-train", type=Path, help="default: data/interim/<dataset>/train_cleaned.csv"
    )
    parser.add_argument(
        "--real-val", type=Path, help="default: data/interim/<dataset>/validation_cleaned.csv"
    )
    parser.add_argument(
        "--positive", default=1, type=int, help="positive class value of the target"
    )
    parser.add_argument(
        "--thresholds",
        choices=["suggested", "schema"],
        default="suggested",
        help="suggested: 2x the real-vs-real values (default); schema: QualityThresholds defaults",
    )
    parser.add_argument(
        "--margin", type=float, default=2.0, help="multiplier for suggested thresholds"
    )
    parser.add_argument(
        "--expected-positive-share",
        type=float,
        default=1.0,
        help="expected share of positive rows (1.0 when the files hold only the positive class)",
    )
    parser.add_argument(
        "--per-column", action="store_true", help="also print each column's distance"
    )
    parser.add_argument("--json", dest="json_path", type=Path, help="write the full result as JSON")
    args = parser.parse_args(argv)

    config = DATASETS[args.dataset]
    folder = root / "data" / "interim" / args.dataset
    real_train = _read(args.real_train or folder / "train_cleaned.csv")
    real_val = _read(args.real_val or folder / "validation_cleaned.csv")
    rules = RULES.get(args.dataset, ())
    common = {
        "target": config.target,
        "positive": args.positive,
        "discrete": config.discrete,
        "rules": rules,
    }

    baseline = real_vs_real(real_train, real_val, **common)
    suggested = suggest_thresholds(baseline.values, args.margin)
    thresholds = suggested if args.thresholds == "suggested" else schema_thresholds(rules=rules)
    baseline = real_vs_real(real_train, real_val, thresholds=thresholds, **common)

    train_class = real_train[real_train[config.target] == args.positive]
    reports = {
        path.stem: fidelity_checks(
            train_class,
            _read(path),
            thresholds=thresholds,
            expected_positive_share=args.expected_positive_share,
            **common,
        )
        for path in args.files
    }
    print(f"Fidelity scorecard: {args.dataset}, thresholds: {args.thresholds}")
    print(scorecard(baseline, reports, thresholds, per_column=args.per_column))
    if args.json_path:
        payload = {
            "dataset": args.dataset,
            "thresholds_source": args.thresholds,
            "thresholds": thresholds,
            "suggested_thresholds": suggested,
            "real_vs_real": [c.model_dump() for c in baseline.checks],
            "files": {
                label: {
                    "passed": report.passed,
                    "worst_column": report.worst_column,
                    "n_rows": report.n_rows,
                    "checks": [c.model_dump() for c in report.checks],
                }
                for label, report in reports.items()
            },
        }
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(payload, indent=2) + "\n")
    return 0 if all(report.passed for report in reports.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
