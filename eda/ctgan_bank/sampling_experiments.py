"""Sampling-side fixes for the Bank CTGAN v1 model, with no retraining.

Loads the saved v1 generator and builds four 10,390-row sets (the 50% fill), each scored with
ctgan_scoring.py and added to eda/ctgan_bank/artifacts/ctgan_runs.csv:

* ``v1_rules``: conditional sampling plus every Bank consistency rule as a row filter (v1 only
  enforced three of the four; ``pdays_poutcome`` was missing).
* ``v1_uncond_rules``: the same, but sampled without conditioning on the target and keeping the
  subscriber rows, to see whether the conditioning skews ``job``.
* ``v1_rake_rules``: a large rule-passing pool, reweighted so the job, month and poutcome shares
  match the real subscribers, then sampled without replacement.

    uv run python eda/ctgan_bank/sampling_experiments.py
"""

import numpy as np
import pandas as pd
from ctgan_scoring import ROOT, Scorer, save_runs

from agentic_gan.tools.ctgan import load_model, sample_synthetic
from agentic_gan.tools.quality_fidelity import BANK_RULES, rule_violation_rate

MODEL_FILE = ROOT / "eda" / "ctgan_bank" / "artifacts" / "ctgan_bank.pkl"
N_ROWS = 10_390
POOL_ROWS = 60_000
RAKE_COLUMNS = ("job", "month", "poutcome")
WIDE_RAKE_COLUMNS = (*RAKE_COLUMNS, "marital", "_day_bin")
DAY_BINS = [0, 5, 10, 15, 20, 25, 31]
SEED = 42


def passes_rules(df: pd.DataFrame) -> pd.Series:
    """True for rows that break none of the Bank consistency rules."""
    broken = pd.Series(False, index=df.index)
    for rule in BANK_RULES:
        broken |= rule.violated(df)
    return ~broken


def with_day_bin(df: pd.DataFrame) -> pd.DataFrame:
    """Copy with a helper column that bins day_of_month, so raking can match its shape too."""
    return df.assign(_day_bin=pd.cut(df["day_of_month"], DAY_BINS).astype(str))


def rake_weights(pool: pd.DataFrame, real: pd.DataFrame, columns, iterations: int = 50):
    """Row weights that make the pool's category shares match ``real`` on each column."""
    weights = np.ones(len(pool))
    for _ in range(iterations):
        for col in columns:
            target = real[col].value_counts(normalize=True)
            current = pd.Series(weights, index=pool.index).groupby(pool[col]).sum()
            current = current / current.sum()
            ratio = (target / current).replace([np.inf, -np.inf], np.nan).fillna(1.0)
            weights = weights * pool[col].map(ratio).fillna(1.0).to_numpy()
    return weights / weights.sum()


def main() -> None:
    scorer = Scorer()
    model = load_model(MODEL_FILE)
    train = scorer.real_train.drop(columns="source_row")
    settings = {"epochs": 300, "pac": 10, "discriminator_steps": 1}
    rows = []

    def run(label, table, **extra):
        table = table.reset_index(drop=True)
        rows.append(scorer.score(table, label, {**settings, **extra}))
        r = rows[-1]
        print(
            f"{label}: mean={r['mean_column_distance']:.3f} "
            f"worst={r['worst_column_distance']:.3f} ({r['worst_column']}) "
            f"technician={r['technician_share']:.3f} auc={r['real_vs_synth_auc']:.3f} "
            f"passed={r['passed']}", flush=True)

    pool = sample_synthetic(model, POOL_ROWS, seed=SEED, drop_duplicates_of=train,
                            row_filter=passes_rules)
    run("v1_rules", pool.head(N_ROWS), sampling="conditional + all rules")

    uncond = sample_synthetic(model, N_ROWS, seed=SEED, drop_duplicates_of=train,
                              row_filter=passes_rules, conditional=False, max_rounds=60)
    run("v1_uncond_rules", uncond, sampling="unconditional + all rules")

    weights = rake_weights(pool, scorer.reference, RAKE_COLUMNS)
    picked = np.random.default_rng(SEED).choice(len(pool), size=N_ROWS, replace=False, p=weights)
    run("v1_rake_rules", pool.iloc[picked],
        sampling="conditional + rules + raked job/month/poutcome")

    uncond_pool = sample_synthetic(model, POOL_ROWS, seed=SEED, drop_duplicates_of=train,
                                   row_filter=passes_rules, conditional=False, max_rounds=100)
    for label, pool_in, columns, text in (
        ("v1_uncond_rake3", uncond_pool, RAKE_COLUMNS,
         "unconditional + rules + raked job/month/poutcome"),
        ("v1_uncond_rake5", uncond_pool, WIDE_RAKE_COLUMNS,
         "unconditional + rules + raked job/month/poutcome/marital/day bins"),
    ):
        binned = with_day_bin(pool_in)
        real_binned = with_day_bin(scorer.reference)
        w = rake_weights(binned, real_binned, columns)
        idx = np.random.default_rng(SEED).choice(len(binned), size=N_ROWS, replace=False, p=w)
        run(label, pool_in.iloc[idx], sampling=text)

    broken = {r.name: rule_violation_rate(pool, r) for r in BANK_RULES}
    print("rule violations in the pool (should all be 0):", broken)
    table = save_runs(rows)
    shown = ["label", "mean_column_distance", "worst_column_distance", "worst_column",
             "technician_share", "real_vs_synth_auc", "passed"]
    print(table[shown].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
