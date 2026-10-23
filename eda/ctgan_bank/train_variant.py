"""Train one CTGAN setting on Bank, sample it three ways, and log a scored run for each.

Each call changes the training settings (epochs, pac, discriminator steps) and keeps everything
else as in v1. The trained model is saved, then three 10,390-row sets are sampled from it and
scored with ctgan_scoring.py into eda/ctgan_bank/artifacts/ctgan_runs.csv:

* ``<label>_cond_rules``: conditional sampling plus the Bank consistency rules (the v1 recipe);
* ``<label>_uncond_rules``: unconditional sampling plus the rules;
* ``<label>_uncond_rake5``: unconditional pool reweighted to the real job, month, poutcome,
  marital and day-of-month shares (the best recipe from sampling_experiments.py).

    uv run python eda/ctgan_bank/train_variant.py --label epochs600 --epochs 600
    uv run python eda/ctgan_bank/train_variant.py --label pac5 --pac 5
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from ctgan_scoring import RUNS_FILE, Scorer, save_runs
from sampling_experiments import (
    N_ROWS,
    POOL_ROWS,
    SEED,
    WIDE_RAKE_COLUMNS,
    passes_rules,
    rake_weights,
    with_day_bin,
)

from agentic_gan.tools.ctgan import sample_synthetic, save_model, train_ctgan
from agentic_gan.utils.datasets import DATASETS

CONFIG = DATASETS["bank_marketing"]
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def raked_sample(pool: pd.DataFrame, reference: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """``n`` pool rows whose shares match ``reference`` on the raking columns, in random order."""
    weights = rake_weights(with_day_bin(pool), with_day_bin(reference), WIDE_RAKE_COLUMNS)
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(pool), size=n, replace=False, p=weights)
    return pool.iloc[rng.permutation(picked)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--pac", type=int, default=10)
    parser.add_argument("--discriminator-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--runs-file", type=Path, default=RUNS_FILE)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    scorer = Scorer()
    train = scorer.real_train.drop(columns="source_row")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = train_ctgan(
        train, CONFIG.target, discrete_columns=list(CONFIG.discrete), epochs=args.epochs,
        batch_size=args.batch_size, pac=args.pac, discriminator_steps=args.discriminator_steps,
        seed=SEED, log_path=args.out_dir / f"ctgan_{args.label}_losses.csv",
    )
    save_model(model, args.out_dir / f"ctgan_{args.label}.pkl")

    settings = {
        "epochs": args.epochs, "pac": args.pac, "discriminator_steps": args.discriminator_steps,
        "batch_size": args.batch_size, "train_seconds": round(model.train_seconds, 1),
    }
    common = {"seed": SEED, "drop_duplicates_of": train, "row_filter": passes_rules}
    cond = sample_synthetic(model, N_ROWS, **common)
    uncond_pool = sample_synthetic(
        model, POOL_ROWS, conditional=False, max_rounds=100, **common
    )
    variants = [
        ("cond_rules", cond, "conditional + all rules"),
        ("uncond_rules", uncond_pool.head(N_ROWS), "unconditional + all rules"),
        ("uncond_rake5", raked_sample(uncond_pool, scorer.reference, N_ROWS, SEED),
         "unconditional + rules + raked job/month/poutcome/marital/day bins"),
    ]
    rows = []
    for suffix, table, text in variants:
        row = scorer.score(table.reset_index(drop=True), f"{args.label}_{suffix}",
                           {**settings, "sampling": text})
        rows.append(row)
        print(f"{row['label']}: mean={row['mean_column_distance']:.3f} "
              f"worst={row['worst_column_distance']:.3f} ({row['worst_column']}) "
              f"spearman={row['spearman_diff']:.3f} technician={row['technician_share']:.3f} "
              f"auc={row['real_vs_synth_auc']:.3f} passed={row['passed']}", flush=True)
    save_runs(rows, args.runs_file)


if __name__ == "__main__":
    main()
