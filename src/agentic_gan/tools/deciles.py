"""Summarize ranked binary-classifier predictions in equal-sized groups."""

from collections.abc import Iterable
from numbers import Integral

import numpy as np
import pandas as pd


def decile_table(
    probability,
    y_true,
    prediction,
    features: pd.DataFrame,
    columns: Iterable[str],
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return one row per bin, highest probability first.

    Inputs must be in the same positional order. Ties retain that order, even across bin
    boundaries. Earlier bins get the extra rows when the split size is not divisible by
    n_bins. Precision is zero when no row meets the model's frozen threshold. Rates whose
    population is empty are missing. Cumulative lift compares the share of positives
    reached to the share of all rows reached. Null features do not compete for top category,
    and category shares use the entire bin as their denominator.
    """
    if isinstance(n_bins, bool) or not isinstance(n_bins, Integral) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")

    scores = pd.Series(probability)
    labels = pd.Series(y_true)
    predicted = pd.Series(prediction)
    if not len(scores) == len(labels) == len(predicted) == len(features):
        raise ValueError("probability, y_true, prediction and features must have the same length")
    if not scores.notna().all() or not scores.between(0, 1).all():
        raise ValueError("probability must be finite and between 0 and 1")
    if not labels.isin([0, 1]).all() or not predicted.isin([0, 1]).all():
        raise ValueError("y_true and prediction must be observed binary values")

    columns = list(columns)
    order = np.argsort(-scores.to_numpy(dtype=float), kind="stable")
    bins = np.array_split(order, n_bins)
    labels = labels.to_numpy(dtype=bool)
    predicted = predicted.to_numpy(dtype=bool)
    all_positives = int(labels.sum())
    all_negatives = len(labels) - all_positives
    rows = []
    for decile, indices in enumerate(bins, start=1):
        n = len(indices)
        actual, classified = labels[indices], predicted[indices]
        positives = int(actual.sum())
        tp = int((actual & classified).sum())
        fp = int((~actual & classified).sum())
        rows.append(
            {
                "decile": decile,
                "n": n,
                "positives": positives,
                "actual_rate": positives / n if n else np.nan,
                "mean_predicted_probability": scores.iloc[indices].mean(),
                "tp": tp,
                "fp": fp,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "tp_rate": tp / all_positives if all_positives else np.nan,
                "fp_rate": fp / all_negatives if all_negatives else np.nan,
                "share_of_positives": positives / all_positives if all_positives else np.nan,
            }
        )

    table = pd.DataFrame(rows)
    table["cumulative_share_of_positives"] = table.positives.cumsum() / all_positives
    cumulative_share_of_rows = table.n.cumsum() / len(scores)
    table["cumulative_lift"] = table.cumulative_share_of_positives / cumulative_share_of_rows

    for column in columns:
        values = features[column]
        groups = [values.iloc[indices] for indices in bins]
        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            table[f"{column}_mean"] = [group.mean() for group in groups]
        else:
            counts = [group.value_counts(sort=False) for group in groups]
            table[f"{column}_top_category"] = [
                count.idxmax() if count.sum() else pd.NA for count in counts
            ]
            table[f"{column}_top_share"] = [
                count.max() / len(group) if count.sum() else np.nan
                for count, group in zip(counts, groups, strict=True)
            ]
    return table
