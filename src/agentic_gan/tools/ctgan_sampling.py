"""How the pipeline samples CTGAN rows: the generator recipe and the V2 sampling steps.

V1 asked CTGAN for minority rows directly. On Bank Marketing that skewed ``job`` (technicians were
42% of the fake subscribers against 15% of the real ones). V2 fixed it on the sampling side:

* sample without conditioning on the target and keep the minority-class rows;
* drop rows that break the dataset's consistency rules (the same ``RULES`` the quality gate uses);
* draw the final rows from a large pool so that the category shares of ``rake_columns`` match the
  real minority rows (raking).

``GeneratorRecipe`` holds the seven options of one attempt, ``recipe_for`` gives the first
attempt's values for a dataset, and ``sample_for_fill`` returns the rows for every fill level at
once. The fill levels are the first ``n`` rows of that table, so they are nested.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from agentic_gan.tools.quality_fidelity import Rule

SAMPLING_MODES = ("conditional", "unconditional")
POOL_ROWS = 60_000
CONDITIONAL_POOL_FACTOR = 3  # conditional pool = this many times the largest fill
RAKE_BINS = 5  # a numeric rake column is cut into this many quantile bins of the real rows
RAKE_ITERATIONS = 50
UNCONDITIONAL_MAX_ROUNDS = 100  # the target is matched by rejection, so many rounds are normal


@dataclass(frozen=True)
class GeneratorRecipe:
    """The options of one CTGAN attempt: training settings, then three sampling options."""

    epochs: int
    batch_size: int
    pac: int
    discriminator_steps: int
    sampling_mode: str
    rule_filter: bool
    rake_columns: list[str]

    def __post_init__(self):
        if self.sampling_mode not in SAMPLING_MODES:
            raise ValueError(
                f"sampling_mode must be one of {SAMPLING_MODES}, not {self.sampling_mode!r}"
            )


def recipe_for(dataset: str) -> GeneratorRecipe:
    """The first attempt's options: the V2 recipe for Bank Marketing, V1 for every other dataset."""
    if dataset == "bank_marketing":
        return GeneratorRecipe(
            epochs=300,
            batch_size=500,
            pac=5,
            discriminator_steps=3,
            sampling_mode="unconditional",
            rule_filter=True,
            rake_columns=["job", "month", "poutcome", "marital", "day_of_month"],
        )
    return GeneratorRecipe(
        epochs=300,
        batch_size=500,
        pac=10,
        discriminator_steps=1,
        sampling_mode="conditional",
        rule_filter=False,
        rake_columns=[],
    )


def rule_row_filter(rules: Sequence[Rule]) -> Callable[[pd.DataFrame], pd.Series]:
    """A ``sample_synthetic`` row filter that keeps rows breaking none of ``rules``."""

    def keep(frame: pd.DataFrame) -> pd.Series:
        broken = pd.Series(False, index=frame.index)
        for rule in rules:
            broken |= rule.violated(frame)
        return ~broken

    return keep


def rake_weights(
    pool: pd.DataFrame,
    real: pd.DataFrame,
    columns: Sequence[str],
    iterations: int = RAKE_ITERATIONS,
) -> np.ndarray:
    """Row weights, summing to 1, that make the pool's category shares match ``real``."""
    weights = np.ones(len(pool))
    for _ in range(iterations):
        for col in columns:
            target = real[col].value_counts(normalize=True)
            current = pd.Series(weights, index=pool.index).groupby(pool[col]).sum()
            current = current / current.sum()
            ratio = (target / current).replace([np.inf, -np.inf], np.nan).fillna(1.0)
            weights = weights * pool[col].map(ratio).fillna(1.0).to_numpy()
    return weights / weights.sum()


def with_rake_bins(
    pool: pd.DataFrame, real: pd.DataFrame, columns: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Copies of both tables where each numeric rake column is replaced by a bin label.

    The bins are quantile bins of the real rows, so each holds about the same share of them.
    Returns the pool, the real rows and the column names to rake on.
    """
    pool, real = pool.copy(), real.copy()
    names = []
    for col in columns:
        if col not in pool.columns or col not in real.columns:
            raise ValueError(f"rake column {col!r} is not in the generated and real rows")
        if not pd.api.types.is_numeric_dtype(real[col]):
            names.append(col)
            continue
        edges = np.unique(np.quantile(real[col].dropna(), np.linspace(0, 1, RAKE_BINS + 1)))
        edges = np.array([-np.inf, np.inf]) if len(edges) < 2 else edges
        edges[0], edges[-1] = -np.inf, np.inf
        name = f"__bin_{col}"
        for frame in (pool, real):
            frame[name] = pd.cut(frame[col], edges).astype(str)
        names.append(name)
    return pool, real, names


def raked_sample(
    pool: pd.DataFrame, real: pd.DataFrame, n: int, columns: Sequence[str], seed: int
) -> pd.DataFrame:
    """``n`` pool rows, drawn without replacement, whose shares match ``real`` on ``columns``."""
    pool = pool.reset_index(drop=True)
    binned_pool, binned_real, names = with_rake_bins(pool, real, columns)
    weights = rake_weights(binned_pool, binned_real, names)
    if int((weights > 0).sum()) < n:
        raise ValueError(f"only {int((weights > 0).sum())} pool rows can be drawn for {n} wanted")
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(pool), size=n, replace=False, p=weights)
    return pool.iloc[rng.permutation(picked)].reset_index(drop=True)


def sample_for_fill(
    model,
    train: pd.DataFrame,
    positive_rows: pd.DataFrame,
    counts: Sequence[int] | Mapping[object, int],
    *,
    sampling_mode: str,
    rule_filter: bool,
    rake_columns: Sequence[str],
    rules: Sequence[Rule],
    pool_rows: int = POOL_ROWS,
    seed: int,
) -> pd.DataFrame:
    """Return ``max(counts)`` synthetic minority rows; each fill level takes the first n of them.

    ``train`` is the real training table (rows copying it are never returned) and
    ``positive_rows`` its minority-class rows, which the raking matches. With
    ``rule_filter``, rows that break any of ``rules`` are rejected while sampling.
    """
    # Imported here: the CTGAN tool loads torch, which the rest of this module does not need.
    from agentic_gan.tools import ctgan as ctgan_tool

    if sampling_mode not in SAMPLING_MODES:
        raise ValueError(f"sampling_mode must be one of {SAMPLING_MODES}, not {sampling_mode!r}")
    wanted = max(counts.values() if isinstance(counts, Mapping) else counts)
    if wanted < 1:
        raise ValueError("counts must include at least one fill level with a row")
    if positive_rows.empty:
        raise ValueError("positive_rows is empty, so there is nothing to match")
    target_value = positive_rows[model.target_col].iloc[0]
    rake = list(rake_columns)
    unconditional = sampling_mode == "unconditional"

    if rake:
        pool_n = max(pool_rows, wanted) if unconditional else CONDITIONAL_POOL_FACTOR * wanted
    else:
        pool_n = wanted
    options = {
        "target_value": target_value,
        "seed": seed,
        "drop_duplicates_of": train,
        "row_filter": rule_row_filter(rules) if rule_filter and rules else None,
        "conditional": not unconditional,
    }
    if unconditional:
        options["max_rounds"] = UNCONDITIONAL_MAX_ROUNDS
    pool = ctgan_tool.sample_synthetic(model, pool_n, **options)
    if len(pool) < wanted:
        raise ValueError(f"sampled {len(pool)} usable rows, fewer than the {wanted} wanted")
    if not rake:
        return pool.head(wanted).reset_index(drop=True)
    return raked_sample(pool, positive_rows, wanted, rake, seed)
