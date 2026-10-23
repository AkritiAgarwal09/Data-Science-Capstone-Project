"""Generate the Bank CTGAN v2 synthetic files (fill 10/25/50) from a trained v2 model.

Samples a large unconditional pool that passes every Bank consistency rule, reweights it so the
job, month, poutcome, marital and day-of-month shares match the real subscribers, then picks the
50% fill without replacement in random order. The 10% and 25% files are the first rows of that
file, so the fill levels are nested, as in v1. Each file is scored and added to ctgan_runs.csv.

    uv run python eda/ctgan_bank/make_v2.py --label pac5_dstep3
"""

import argparse
import logging
from pathlib import Path

from ctgan_scoring import FOLDER, RUNS_FILE, Scorer, save_runs
from sampling_experiments import POOL_ROWS, SEED, passes_rules
from train_variant import ARTIFACTS, raked_sample

from agentic_gan.tools.ctgan import load_model, sample_synthetic
from agentic_gan.utils.datasets import DATASETS
from agentic_gan.utils.synthetic import fill_counts

CONFIG = DATASETS["bank_marketing"]
OUT_DIR = FOLDER / "synthetic"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="trained variant, e.g. pac5_dstep3")
    parser.add_argument("--pool-rows", type=int, default=POOL_ROWS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--runs-file", type=Path, default=RUNS_FILE)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    scorer = Scorer()
    train = scorer.real_train.drop(columns="source_row")
    model = load_model(ARTIFACTS / f"ctgan_{args.label}.pkl")
    counts = fill_counts(train[CONFIG.target])

    pool = sample_synthetic(
        model, args.pool_rows, conditional=False, max_rounds=100, seed=SEED,
        drop_duplicates_of=train, row_filter=passes_rules,
    )
    picked = raked_sample(pool, scorer.reference, max(counts.values()), SEED).reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    settings = {**model.params, "train_seconds": round(model.train_seconds, 1),
                "sampling": "unconditional + rules + raked job/month/poutcome/marital/day bins"}
    rows = []
    for fraction, n in counts.items():
        table = picked.head(n)
        name = f"bank_ctgan_v2_fill{round(fraction * 100)}"
        table.to_csv(args.out_dir / f"{name}.csv", index=False)
        row = scorer.score(table, f"v2_fill{round(fraction * 100)}", settings)
        rows.append(row)
        print(f"{name}: rows={n} mean={row['mean_column_distance']:.3f} "
              f"worst={row['worst_column_distance']:.3f} ({row['worst_column']}) "
              f"spearman={row['spearman_diff']:.3f} technician={row['technician_share']:.3f} "
              f"auc={row['real_vs_synth_auc']:.3f} passed={row['passed']}", flush=True)
    save_runs(rows, args.runs_file)


if __name__ == "__main__":
    main()
