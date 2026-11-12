"""Generate a poster data summary from local source and cleaned split files."""

import argparse
import csv
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_gan.tools.preprocessing import (
    _sample_train,
    _split_counts,
    _verify_reference_counts,
    load_dataset,
    normalize,
    split_dataset,
)
from agentic_gan.utils.datasets import DATASETS
from agentic_gan.utils.files import write_text_atomic

PARTS = ("train", "validation", "test")
DATASET_NAMES = {
    "bank_marketing": "Bank Marketing",
    "default": "Default of Credit Card Clients",
    "polish": "Polish Companies Bankruptcy",
    "ieee_cis": "IEEE-CIS Fraud Detection",
}
POLISH_APPENDIX = "polish_attribute_definitions.csv"
MISSING_FLAG = "__missing"
ROW_ID = "source_row"
IEEE_MIN_CATEGORY_COUNT = 20

INTRODUCTIONS = {
    "bank_marketing": (
        "Each row records a client contact in a Portuguese bank’s telephone marketing campaign. "
        "The response is a term-deposit subscription, `y=yes` in the source and `y=1` "
        "in the cleaned splits. "
        "[UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)"
    ),
    "default": (
        "Each row represents a Taiwanese credit card client, with account and payment history "
        "from April through September 2005. The response is a default in the following month "
        "(`y=1`). [UCI Default of Credit Card Clients]"
        "(https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)"
    ),
    "polish": (
        "Each row is a Polish company’s financial statement in one of five forecasting cohorts "
        "from the EMIS data distributed by UCI. The response is bankruptcy within that cohort’s "
        "one- to five-year forecast horizon (`bankrupt=1`). The pooled counts refer to financial "
        "statements. [UCI Polish Companies Bankruptcy]"
        "(https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data)"
    ),
    "ieee_cis": (
        "Each row is a labeled online payment supplied by Vesta, with identity and device fields "
        "joined when available. The response is a fraudulent transaction (`isFraud=1`). "
        "[Kaggle IEEE-CIS Fraud Detection]"
        "(https://www.kaggle.com/competitions/ieee-fraud-detection/data)"
    ),
}

BANK_MEANINGS = {
    "age": "Client age in years",
    "job": "Occupation group",
    "marital": "Marital status",
    "education": "Education level",
    "default": "Existing credit in default",
    "balance": "Average annual account balance in euros",
    "housing": "Whether the client has a housing loan",
    "loan": "Whether the client has a personal loan",
    "contact": "Communication channel used for the most recent contact",
    "day_of_month": "Calendar day of the most recent contact",
    "month": "Calendar month of the most recent contact",
    "duration": "Length of the most recent call in seconds",
    "campaign": "Contacts with the client during the current campaign",
    "pdays": "Days since prior campaign contact, with −1 for no prior contact",
    "previous": "Contacts with the client before the current campaign",
    "poutcome": "Outcome of the prior campaign",
    "previously_contacted": "Prior campaign contact flag derived from `pdays`",
    "y": "Term-deposit subscription outcome",
}
DEFAULT_MEANINGS = {
    "LIMIT_BAL": "Granted credit limit in New Taiwan dollars",
    "SEX": "Client sex code, with 1 for male and 2 for female",
    "EDUCATION": "Education code. 1 graduate school, 2 university, 3 high school, 4 other",
    "MARRIAGE": "Marital status code, with 1 for married, 2 for single and 3 for other",
    "AGE": "Client age in years",
    "y": "Default in the following month",
}
for number, month in enumerate(("September", "August", "July", "June", "May", "April"), start=1):
    DEFAULT_MEANINGS[f"PAY_{number}"] = f"Repayment status code for {month} 2005"
    DEFAULT_MEANINGS[f"BILL_AMT{number}"] = f"{month} 2005 bill in New Taiwan dollars"
    DEFAULT_MEANINGS[f"PAY_AMT{number}"] = f"{month} 2005 payment in New Taiwan dollars"

IEEE_MEANINGS = {
    "TransactionID": "Transaction key used for the identity join",
    "TransactionDT": "Seconds since the competition’s reference time",
    "TransactionAmt": "Payment amount in US dollars",
    "ProductCD": "Product category code",
    "addr1": "Masked billing region code",
    "addr2": "Masked billing country code",
    "dist1": "Masked distance measure",
    "dist2": "Masked distance measure",
    "P_emaildomain": "Purchaser email domain",
    "R_emaildomain": "Recipient email domain",
    "DeviceType": "Broad device category",
    "DeviceInfo": "Device information reported with the identity record",
    "has_identity": "Whether a matching identity record was available",
    "isFraud": "Transaction fraud outcome",
}
IEEE_FAMILIES = {
    "card": "Masked payment-card attribute",
    "C": "Masked count-based measure",
    "D": "Masked elapsed-time measure",
    "M": "Masked match indicator",
    "V": "Masked Vesta-engineered measure",
    "id_": "Masked identity or device attribute",
}


@dataclass(frozen=True)
class Counts:
    rows: int
    responses: int

    @classmethod
    def from_labels(cls, labels: pd.Series) -> "Counts":
        return cls(len(labels), int(labels.sum()))

    @property
    def rate(self) -> float:
        return self.responses / self.rows


@dataclass(frozen=True)
class DatasetSummary:
    dataset: str
    features: int
    source: Counts
    splits: dict[str, Counts]
    variables: list[tuple[str, ...]]
    full_time_splits: dict[str, Counts] | None = None

    @property
    def total(self) -> Counts:
        return Counts(
            sum(part.rows for part in self.splits.values()),
            sum(part.responses for part in self.splits.values()),
        )


def _polish_meanings(root: Path) -> dict[str, str]:
    with (root / "docs" / POLISH_APPENDIX).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [row["variable"] for row in rows] != [f"A{i}" for i in range(1, 65)]:
        raise ValueError("The Polish appendix must define A1 through A64 in order")
    return {
        "cohort": "Source forecasting cohort, which determines the outcome horizon",
        "bankrupt": "Bankruptcy within the cohort’s forecasting horizon",
        **{row["variable"]: row["meaning"] for row in rows},
    }


def _ieee_meaning(column: str) -> str:
    if column.endswith(MISSING_FLAG):
        source = column.removesuffix(MISSING_FLAG)
        return f"Whether `{source}` was missing before cleaning"
    if column in IEEE_MEANINGS:
        return IEEE_MEANINGS[column]
    family = re.fullmatch(r"(card|[CDMV]|id_)\d+", column)
    if family:
        return IEEE_FAMILIES[family[1]]
    raise ValueError(f"Unknown IEEE-CIS variable {column!r}")


def _read_cleaned_splits(folder, target, labels, expected_ids):
    counts = {}
    columns = None
    for part in PARTS:
        path = folder / f"{part}_cleaned.csv"
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if columns is not None and columns != header:
            raise ValueError(f"{path} differs from the training schema")
        columns = header
        cleaned = pd.read_csv(path, usecols=[ROW_ID, target], index_col=ROW_ID)
        if not cleaned.index.is_unique or not cleaned.index.sort_values().equals(
            pd.Index(expected_ids[part]).sort_values()
        ):
            raise ValueError(f"{path} does not match the configured {part} split")
        if not np.array_equal(cleaned[target], labels.loc[cleaned.index]):
            raise ValueError(f"{path} contains labels that differ from the source")
        counts[part] = Counts.from_labels(cleaned[target])
    return columns, counts


def _category_label(value) -> str:
    if isinstance(value, (float, np.floating)) and value.is_integer():
        value = int(value)
    text = " ".join(str(value).split())
    return escape(text).replace("|", "&#124;").replace("`", "&#96;")


def _numeric_bound(value) -> str:
    number = float(value)
    if number.is_integer() and abs(number) < 1e12:
        return f"{int(number):,}"
    return f"{number:,.6g}"


def _profile(values: pd.Series, kind: str, min_category_count: int) -> tuple[str, str, str]:
    if kind == "numeric":
        values = values.replace([np.inf, -np.inf], np.nan)
    observed = values.dropna()
    if kind == "identifier":
        spread = "—"
    elif observed.empty:
        spread = "No observed values"
    elif kind == "numeric":
        spread = f"{_numeric_bound(observed.min())} to {_numeric_bound(observed.max())}"
    else:
        counts = observed.value_counts(sort=False).nlargest(3)
        spread = (
            ", ".join(
                f"{_category_label(value)} ({count / len(values):.2%})"
                for value, count in counts.items()
                if count >= min_category_count
            )
            or f"No category has {min_category_count} observations"
        )
    return f"{values.isna().mean():.2%}", f"{observed.nunique():,}", spread


def _variable_rows(root, config, raw, cleaned):
    dataset = config.name
    if dataset == "ieee_cis":
        meanings = _ieee_meaning
    else:
        definitions = {
            "bank_marketing": BANK_MEANINGS,
            "default": DEFAULT_MEANINGS,
            "polish": _polish_meanings(root) if dataset == "polish" else {},
        }[dataset]
        meanings = definitions.__getitem__
    identifiers = {ROW_ID, config.id_column}
    categorical = {*config.discrete, config.target, "has_identity"}
    excluded = {
        ROW_ID: "Excluded. Row provenance for joins and audits.",
        "duration": "Excluded. Call length is known after the call and leaks outcome information.",
        "TransactionID": "Excluded. Transaction key used for joins.",
        "TransactionDT": "Excluded. Used to define the chronological split.",
    }
    names = dict.fromkeys([*cleaned, *raw.columns])
    min_count = IEEE_MIN_CATEGORY_COUNT if dataset == "ieee_cis" else 1
    rows = []
    for name in names:
        if name == ROW_ID:
            values = raw.index.to_series()
            meaning = "Repository index of the original source row"
        elif name.endswith(MISSING_FLAG):
            source = raw[name.removesuffix(MISSING_FLAG)]
            values = (source.isna() | source.isin([np.inf, -np.inf])).astype(int)
            meaning = meanings(name)
        else:
            values = raw[name]
            meaning = meanings(name)

        if name in identifiers:
            kind = "identifier"
        elif name in categorical or name.endswith(MISSING_FLAG):
            kind = "categorical"
        else:
            kind = "numeric"

        role = excluded.get(name, "Feature")
        if name == config.target:
            role = "Target"
        elif name == "cohort":
            role = "Cohort metadata"
        elif name not in cleaned and name not in excluded:
            if dataset != "ieee_cis":
                raise ValueError(f"No exclusion reason for {dataset} variable {name!r}")
            role = f"Excluded. More than {config.sample['max_missing']:.0%} missing in training."
        rows.append((f"`{name}`", kind, meaning, role, *_profile(values, kind, min_count)))
    return rows


def summarize_dataset(root: Path, dataset: str) -> DatasetSummary:
    """Measure one dataset using the repository’s source, schema and split recipe."""
    config = DATASETS[dataset]
    raw = load_dataset(config, root / config.file)
    folder = root / "data" / "interim" / dataset
    full_counts = None
    if dataset == "ieee_cis":
        labels = raw[config.target]
        keys = raw[[config.target, config.split["time_column"]]]
        full_ids = split_dataset(raw, config, sample=False)
        split_ids = _sample_train(keys, config, full_ids)
        _verify_reference_counts(
            _split_counts(keys, full_ids, split_ids, config),
            root / "splits" / "ieee_cis_time_split.csv",
        )
        full_counts = {part: Counts.from_labels(labels.loc[ids]) for part, ids in full_ids.items()}
        profile = raw.loc[split_ids["train"]]
    else:
        normalized = normalize(raw, config, config.include_duration)
        labels = normalized[config.target]
        split_ids = split_dataset(normalized, config)
        profile = raw
        if dataset == "bank_marketing":
            profile = raw.rename(columns={"day_of_week": "day_of_month", "day": "day_of_month"})
            profile["previously_contacted"] = normalized["previously_contacted"]

    columns, split_counts = _read_cleaned_splits(folder, config.target, labels, split_ids)
    non_features = {ROW_ID, config.target, "duration", "cohort", config.id_column}
    if dataset == "ieee_cis":
        non_features.add(config.split["time_column"])
    return DatasetSummary(
        dataset=dataset,
        features=len(set(columns) - non_features),
        source=Counts.from_labels(labels),
        splits=split_counts,
        variables=_variable_rows(root, config, profile, columns),
        full_time_splits=full_counts,
    )


def _table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


def _counts_cells(counts, digits=2):
    return f"{counts.rows:,}", f"{counts.responses:,}", f"{counts.rate:.{digits}%}"


def _split_table(splits, total, *, sampled=False):
    return _table(
        ("Split", "Rows", "Responses", "Response rate"),
        [
            *(
                (
                    "Train (sample)" if part == "train" and sampled else part.capitalize(),
                    *_counts_cells(counts),
                )
                for part, counts in splits.items()
            ),
            ("**Total**", *_counts_cells(total)),
        ],
    )


def render_report(summaries: list[DatasetSummary]) -> str:
    """Render Markdown containing aggregate statistics only."""
    parts = [
        "# Dataset summaries",
        "Generated by `uv run python -m agentic_gan.tools.dataset_summary` from the local source "
        "files and the cleaned split schemas.",
        "## Poster overview",
        _table(
            ("Dataset", "Rows", "Features", "Responses", "Response rate"),
            [
                (
                    DATASET_NAMES[summary.dataset],
                    f"{summary.total.rows:,}",
                    f"{summary.features:,}",
                    f"{summary.total.responses:,}",
                    f"{summary.total.rate:.1%}",
                )
                for summary in summaries
            ],
        ),
        "Feature counts are before one-hot encoding and exclude outcomes and identifiers. "
        "Polish counts its 64 financial attributes separately from the cohort label. "
        "IEEE-CIS uses the 100,000-row training sample plus the full holdouts, and counts "
        "the missingness flags as features.",
        "Variable statistics use source values before imputation. Missing percentages and "
        "category shares use all profiled rows as the denominator. Distinct counts exclude "
        "missing values. Identifier values are omitted. Generated flags are summarized as "
        "binary categories.",
    ]
    for summary in summaries:
        dataset = summary.dataset
        parts += [f"## {DATASET_NAMES[dataset]}", INTRODUCTIONS[dataset]]
        if dataset == "ieee_cis":
            full = summary.full_time_splits
            train_end, validation_end = DATASETS[dataset].split["cut_points"]
            deduplicated = Counts(
                sum(counts.rows for counts in full.values()),
                sum(counts.responses for counts in full.values()),
            )
            rows, positives, rate = _counts_cells(summary.source)
            parts += [
                f"The labeled source has **{rows}** transactions and **{positives}** fraud "
                f"responses ({rate}). The full split follows the checked-in time cutoffs after "
                f"removing {summary.source.rows - deduplicated.rows:,} exact copies. "
                "[Fraud dataset recommendation](fraud_dataset_recommendation.md) · "
                "[Time-split counts](../splits/ieee_cis_time_split.csv)",
                "### Full chronological split",
                f"Train covers `TransactionDT` below {train_end:,}. Validation runs from "
                f"{train_end:,} to below {validation_end:,}. Test starts at {validation_end:,}.",
                _split_table(full, deduplicated),
                "### Analysis split used in the poster",
                "The earliest period contributes a response-stratified training sample of "
                "100,000 rows (seed 42). Validation and test retain their full time periods.",
                _split_table(summary.splits, summary.total, sampled=True),
                "### Variables",
                "The variable statistics use the 100,000 sampled training rows. Category labels "
                f"are shown only for groups with at least {IEEE_MIN_CATEGORY_COUNT} observations. "
                "The feature count includes both retained values and missingness indicators. "
                "Vesta masks the exact definitions of individual fields in several families.",
                "<details>\n<summary>IEEE-CIS variable table</summary>",
            ]
        else:
            parts += [
                "### Stratified 60/20/20 split",
                _split_table(summary.splits, summary.total),
                "### Variables",
                f"Variable statistics cover all {summary.total.rows:,} source rows.",
            ]
            if dataset == "bank_marketing":
                parts.append(
                    "`day_of_month` is the cleaned name for the source column `day_of_week`. "
                    "The feature count includes the derived `previously_contacted` flag. "
                    "Blank source categories appear as missing here and become `unknown` in "
                    "the cleaned data."
                )
            elif dataset == "default":
                parts.append(
                    "In UCI’s repayment code definitions, −1 means paid on time and positive "
                    "values indicate months overdue. The source extract also contains "
                    "additional codes."
                )
            elif dataset == "polish":
                parts.append(
                    f"The [Polish attribute appendix]({POLISH_APPENDIX}) gives the full UCI "
                    "definitions for A1–A64. The cleaned file also carries the cohort field."
                )
        parts.append(
            _table(
                (
                    "Variable",
                    "Type",
                    "Meaning",
                    "Role",
                    "Missing",
                    "Distinct",
                    "Range or top categories",
                ),
                summary.variables,
            )
        )
        if dataset == "ieee_cis":
            parts.append("</details>")
    return "\n\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, help="Defaults to docs/dataset_summaries.md")
    args = parser.parse_args()
    summaries = [summarize_dataset(args.root, dataset) for dataset in DATASET_NAMES]
    output = args.output or args.root / "docs" / "dataset_summaries.md"
    write_text_atomic(output, render_report(summaries))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
