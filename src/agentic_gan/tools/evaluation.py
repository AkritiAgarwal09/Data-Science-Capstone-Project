"""Deterministic evaluation tools: arm comparison, the reliable-gain rule, segments, number check.

The Evaluation agent's LLM only reads what these functions return; it never computes a number.
They read the training agent's `metrics.csv` (one row per arm, fill level, seed, model and split,
`selected` marking the model chosen on validation PR-AUC) and the prediction files it saves.

Reliable gain, as the rule is written down here: at a fill level, a metric has a reliable gain
when, on both validation and test, the mean paired lift over the original arm (same seeds) is
larger than the seed-to-seed spread of that lift (its sample standard deviation over the seeds),
and at least MIN_SEEDS seeds are paired. A lift below minus the spread on both is a reliable
loss; anything else, a lift inside the noise or one that differs between splits, is inconclusive.
"""

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from agentic_gan.tools.baselines import pr_auc_score

METRICS = ("pr_auc", "roc_auc", "f1", "accuracy", "precision", "recall")
SPLITS = ("validation", "test")
ARM_ORDER = ("original", "cleaned", "smotenc", "ctgan")
REFERENCE_ARM = "original"
MIN_SEEDS = 3
MAX_CATEGORIES = 10
_COLUMNS = ("dataset", "arm", "fill_level", "seed", "split", "status", "selected", *METRICS)
_ONE_ROW = ["dataset", "arm", "fill_level", "seed", "split"]
_GROUP = ["dataset", "arm", "fill_level", "split", "metric"]


def compare_arms(metrics: pd.DataFrame, *, reference: str = REFERENCE_ARM) -> pd.DataFrame:
    """Mean and seed-to-seed spread per dataset, arm, fill level, split and metric.

    Each arm, level and seed counts once, with the model the training agent selected. `diff_mean`
    and `diff_sd` are the paired difference from the reference arm at the same seeds (NaN for the
    reference itself); `n_paired` is how many seeds had a reference to pair with.
    """
    missing = [name for name in _COLUMNS if name not in metrics]
    if missing:
        raise ValueError(f"metrics table is missing columns: {missing}")
    rows = metrics[
        metrics["status"].astype(str).str.upper().eq("SUCCESS")
        & metrics["selected"].astype(str).str.lower().eq("true")
        & metrics["split"].isin(SPLITS)
    ]
    if rows.duplicated(_ONE_ROW).any():
        raise ValueError("more than one selected model per dataset, arm, fill level, seed, split")
    long = rows.melt(
        id_vars=_ONE_ROW, value_vars=list(METRICS), var_name="metric", value_name="value"
    ).dropna(subset=["value"])
    pair_on = ["dataset", "split", "metric", "seed"]
    base = long.loc[long["arm"] == reference].set_index(pair_on)["value"].rename("reference")
    if not base.index.is_unique:
        raise ValueError(f"the {reference} arm has more than one fill level or selected model")
    paired = long.join(base, on=pair_on)
    paired["diff"] = (paired["value"] - paired["reference"]).where(paired["arm"] != reference)
    table = paired.groupby(_GROUP, sort=False).agg(
        n_seeds=("value", "count"),
        mean=("value", "mean"),
        sd=("value", "std"),
        n_paired=("diff", "count"),
        diff_mean=("diff", "mean"),
        diff_sd=("diff", "std"),
    )
    table = table.reset_index()
    rank = table["arm"].map({arm: i for i, arm in enumerate(ARM_ORDER)}).fillna(len(ARM_ORDER))
    metric_rank = table["metric"].map({metric: i for i, metric in enumerate(METRICS)})
    order = table.assign(rank=rank, metric_rank=metric_rank).sort_values(
        ["dataset", "rank", "fill_level", "split", "metric_rank"]
    )
    return order.drop(columns=["rank", "metric_rank"]).reset_index(drop=True)


@dataclass(frozen=True)
class GainVerdict:
    """The reliable-gain verdict for one arm and fill level, with the numbers behind it."""

    dataset: str
    arm: str
    fill_level: float
    metric: str
    verdict: str  # reliable gain, reliable loss, inconclusive or not enough paired seeds
    lift: dict[str, float]  # mean paired lift over the reference arm, by split
    spread: dict[str, float]  # seed-to-seed spread of that lift, by split
    reason: str

    @property
    def reliable(self) -> bool:
        return self.verdict == "reliable gain"


def reliable_gain(
    comparison: pd.DataFrame,
    *,
    dataset: str,
    arm: str,
    fill_level: float = 0.0,
    metric: str = "pr_auc",
) -> GainVerdict:
    """Apply the reliable-gain rule (module docstring) to one row group of `compare_arms`."""
    rows = comparison[
        comparison["dataset"].eq(dataset)
        & comparison["arm"].eq(arm)
        & np.isclose(comparison["fill_level"], fill_level)
        & comparison["metric"].eq(metric)
    ].set_index("split")
    lift, spread = {}, {}
    for split in SPLITS:
        if split not in rows.index or rows.loc[split, "n_paired"] < MIN_SEEDS:
            reason = f"fewer than {MIN_SEEDS} seeds paired with the original arm on {split}"
            return GainVerdict(
                dataset, arm, fill_level, metric, "not enough paired seeds", lift, spread, reason
            )
        lift[split] = float(rows.loc[split, "diff_mean"])
        spread[split] = float(rows.loc[split, "diff_sd"])
    reason = "; ".join(f"{s} lift {lift[s]:.4f} vs spread {spread[s]:.4f}" for s in SPLITS)
    if all(lift[split] > spread[split] for split in SPLITS):
        verdict = "reliable gain"
    elif all(lift[split] < -spread[split] for split in SPLITS):
        verdict = "reliable loss"
    else:
        verdict = "inconclusive"
    return GainVerdict(dataset, arm, fill_level, metric, verdict, lift, spread, reason)


def segment_metrics(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    column: str,
    *,
    bins: int = 4,
    min_positives: int = 20,
) -> pd.DataFrame:
    """Precision, recall, F1 and PR-AUC of one model's predictions within segments of a column.

    `predictions` is a prediction file from the training agent (source_row, y_true, probability,
    prediction). `features` holds the same rows' `source_row` and the segment column; a numeric
    column with many values is cut into `bins` quantile bands. A column that is missing gives an
    empty table. Segments with fewer than `min_positives` positives are flagged `small`.
    """
    names = [
        "segment",
        "n",
        "positives",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "small",
        "segment_lower_bound",
        "segment_upper_bound",
    ]
    if column not in features:
        return pd.DataFrame(columns=names)
    values = (
        features.set_index("source_row")[column] if "source_row" in features else features[column]
    )
    if not values.index.is_unique:
        raise ValueError("features has repeated source_row values")
    bounds = {}
    if pd.api.types.is_numeric_dtype(values) and values.nunique() > MAX_CATEGORIES:
        bands = pd.qcut(values, bins, duplicates="drop")
        # Store the displayed bounds as numbers so report verification can ground
        # interval labels (including qcut's adjusted lower endpoint).
        bounds = {str(band): (float(band.left), float(band.right)) for band in bands.cat.categories}
        values = bands.astype(str).where(bands.notna())
    data = predictions.assign(segment=predictions["source_row"].map(values)).dropna(
        subset=["segment"]
    )
    rows = []
    for segment, part in data.groupby("segment", sort=False):
        y, predicted = part["y_true"], part["prediction"]
        positives = int(y.sum())
        has_both = 0 < positives < len(part)
        rows.append(
            {
                "segment": str(segment),
                "segment_lower_bound": bounds.get(str(segment), (np.nan, np.nan))[0],
                "segment_upper_bound": bounds.get(str(segment), (np.nan, np.nan))[1],
                "n": len(part),
                "positives": positives,
                "precision": precision_score(y, predicted, zero_division=0),
                "recall": recall_score(y, predicted, zero_division=0) if positives else np.nan,
                "f1": f1_score(y, predicted, zero_division=0) if positives else np.nan,
                "pr_auc": pr_auc_score(y, part["probability"]) if has_both else np.nan,
                "small": positives < min_positives,
            }
        )
    table = pd.DataFrame(rows, columns=names)
    return table.sort_values("n", ascending=False, ignore_index=True)


@dataclass(frozen=True)
class NumberCheck:
    """Which numbers in a report were found in the results."""

    verified: bool
    checked: int
    unverified: tuple[str, ...]


# Digits not glued to letters or a dot, with thousands separators, decimals and a percent sign.
_NUMBER = re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?!\w)")
_LIST_MARKER = re.compile(r"(?m)^[ \t]*(?:#+[ \t]*)?\d+[.)][ \t]+")


def _json_numbers(node) -> Iterable[float]:
    if isinstance(node, dict):
        node = list(node.values())
    if isinstance(node, list):
        for child in node:
            yield from _json_numbers(child)
    elif isinstance(node, int | float) and not isinstance(node, bool):
        yield node


def _result_numbers(result_files: Iterable[str | Path]) -> np.ndarray:
    values: list[float] = []
    for path in map(Path, result_files):
        if path.suffix == ".json":
            values.extend(_json_numbers(json.loads(path.read_text())))
        else:
            values.extend(pd.read_csv(path).select_dtypes("number").to_numpy().ravel())
    numbers = np.abs(np.asarray(values, dtype=float))
    return numbers[np.isfinite(numbers)]


def _found(token: str, known: np.ndarray) -> bool:
    percent = token.endswith("%")
    text = token.rstrip("%").replace(",", "")
    value = float(text)
    decimals = len(text.partition(".")[2])
    if "." not in text and not percent:
        return bool(np.any(np.abs(known - value) < 1e-9))  # a count or an id must match exactly
    options = [(value, decimals)] + ([(value / 100, decimals + 2)] if percent else [])
    return any(np.any(np.abs(known - v) <= 0.5 * 10.0**-d + 1e-9) for v, d in options)


def verify_report_numbers(
    report_text: str, result_files: Iterable[str | Path], *, allowed: Iterable[float] = ()
) -> NumberCheck:
    """Check that every number quoted in the report is in the results (CSV or JSON files).

    A decimal or percentage matches a result that rounds to it at the digits quoted (0.05 matches
    0.0524; 25% matches 0.25), ignoring sign so "dropped by 0.05" matches -0.05. A whole number
    such as a seed count must match exactly. `allowed` lists numbers the caller vouches for, such
    as the configured seeds. Numbers glued to letters (v1, pr_auc2) and list numbering are skipped.
    """
    known = np.concatenate([_result_numbers(result_files), np.abs(list(allowed))])
    tokens = _NUMBER.findall(_LIST_MARKER.sub("", report_text))
    unverified = tuple(dict.fromkeys(token for token in tokens if not _found(token, known)))
    return NumberCheck(not unverified, len(tokens), unverified)
