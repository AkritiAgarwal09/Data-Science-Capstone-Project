"""Decile counts and rates for a ranked synthetic portfolio."""

import numpy as np
import pandas as pd
import pytest

from agentic_gan.tools.deciles import decile_table


def test_thousand_rows_with_ties_across_decile_boundaries():
    position = np.arange(1000)
    probability = ((999 - position) // 27) / 37
    assert probability[99] == probability[100]
    y_true = position % 100 < 100 - 10 * (position // 100)
    prediction = probability >= 0.61
    features = pd.DataFrame(
        {
            "balance": position,
            "channel": pd.Categorical(np.where(position % 100 < 70, "direct", "branch")),
        },
        index=position + 1000,
    )
    features.loc[1015, "channel"] = None

    table = decile_table(probability, y_true, prediction, features, ["balance", "channel"])

    assert table.decile.tolist() == list(range(1, 11))
    assert table.n.tolist() == [100] * 10
    assert table.positives.tolist() == list(range(100, 0, -10))
    assert table.actual_rate.is_monotonic_decreasing
    assert table.mean_predicted_probability.is_monotonic_decreasing
    assert table.tp.sum() == np.count_nonzero(y_true & prediction)
    assert table.fp.sum() == np.count_nonzero(~y_true & prediction)
    for i, row in enumerate(table.itertuples(index=False)):
        part = slice(i * 100, (i + 1) * 100)
        assert row.actual_rate == pytest.approx(y_true[part].mean())
        assert row.mean_predicted_probability == pytest.approx(probability[part].mean())
        assert row.tp == np.count_nonzero(y_true[part] & prediction[part])
        assert row.fp == np.count_nonzero(~y_true[part] & prediction[part])
        assert row.precision == pytest.approx(row.tp / (row.tp + row.fp) if row.tp + row.fp else 0)
        assert row.tp_rate == pytest.approx(row.tp / y_true.sum())
        assert row.fp_rate == pytest.approx(row.fp / (~y_true).sum())
        assert row.share_of_positives == pytest.approx(row.positives / y_true.sum())
        expected_cumulative = y_true[: part.stop].sum() / y_true.sum()
        assert row.cumulative_share_of_positives == pytest.approx(expected_cumulative)
        assert row.cumulative_lift == pytest.approx(expected_cumulative / (part.stop / 1000))
        assert row.balance_mean == pytest.approx(position[part].mean())
        assert row.channel_top_category == "direct"
        assert row.channel_top_share == pytest.approx(0.69 if i == 0 else 0.70)
    assert table.n.sum() == len(y_true)
    assert table.positives.sum() == y_true.sum()
    assert table.cumulative_share_of_positives.iloc[-1] == 1.0
    assert table.cumulative_lift.iloc[-1] == 1.0


def test_unsorted_scores_rank_stably_and_bin_sizes_differ_by_at_most_one():
    features = pd.DataFrame({"row": [1, 2, 3, 4, 5]})
    table = decile_table(
        [0.2, 0.9, 0.6, 0.6, 0.1],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        features,
        ["row"],
        n_bins=3,
    )
    assert table.n.tolist() == [2, 2, 1]
    assert table.row_mean.tolist() == [2.5, 2.5, 5.0]
    assert table.positives.tolist() == [2, 0, 0]


def test_small_split_has_explicit_empty_bins_and_undefined_positive_rates():
    table = decile_table([0.2, 0.1], [0, 0], [0, 0], pd.DataFrame(index=range(2)), [])
    assert table.n.tolist() == [1, 1] + [0] * 8
    assert table.precision.eq(0).all()
    assert table.tp_rate.isna().all()
    assert table.cumulative_share_of_positives.isna().all()
    assert table.actual_rate.iloc[2:].isna().all()


@pytest.mark.parametrize("probability", [[0.9, np.nan], [0.9, np.inf], [0.9, -0.1]])
def test_invalid_probabilities_are_rejected(probability):
    with pytest.raises(ValueError, match="probability must be finite"):
        decile_table(probability, [0, 1], [1, 0], pd.DataFrame(index=range(2)), [])
