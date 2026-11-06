"""Train CTGAN on the IEEE-CIS fraud rows and write the synthetic fill-level files (issue #90).

The model is fit on the training sample's fraud rows only (about 3,400), because the full 100,000
rows would take about 15 hours on CPU (see time_trial.py). Every generated row is then fraud, so
sampling needs no target condition. Fill levels follow the usual rule: a share of the gap between
the majority and minority class in the 100,000-row sample, as nested files (the 10% file is the
first rows of the 50% file).

    caffeinate -i uv run python eda/ctgan_ieee/run_ieee_ctgan.py --label v1
    caffeinate -i uv run python eda/ctgan_ieee/run_ieee_ctgan.py --label v2 --pac 5 \
        --discriminator-steps 3

Files go to data/interim/ieee_cis/synthetic/ (ignored by git: IEEE-CIS is derived data under
Kaggle's rules). The model and loss curve go to eda/ctgan_ieee/artifacts/.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from time_trial import CONFIG, FOLDER, load_sample

from agentic_gan.tools.ctgan import sample_synthetic, save_model, train_ctgan
from agentic_gan.utils.synthetic import fill_counts

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
OUT_DIR = FOLDER / "synthetic"
SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="v1", help="names the model and output files")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--pac", type=int, default=10)
    parser.add_argument("--discriminator-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    fraud, discrete = load_sample(0, SEED, fraud_only=True)
    target = pd.read_csv(FOLDER / "train_cleaned.csv", usecols=[CONFIG.target])[CONFIG.target]
    counts = fill_counts(target)

    model = train_ctgan(
        fraud, CONFIG.target, discrete_columns=discrete, epochs=args.epochs,
        batch_size=args.batch_size, pac=args.pac, discriminator_steps=args.discriminator_steps,
        seed=SEED, log_path=ARTIFACTS / f"ctgan_ieee_{args.label}_losses.csv",
    )
    save_model(model, ARTIFACTS / f"ctgan_ieee_{args.label}.pkl")

    pool = sample_synthetic(
        model, max(counts.values()), seed=SEED, drop_duplicates_of=fraud, conditional=False
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fraction, n in counts.items():
        pool.head(n).to_csv(OUT_DIR / f"ieee_ctgan_{args.label}_fill{round(fraction * 100)}.csv",
                            index=False)
    meta = {
        "label": args.label, "train_rows": len(fraud), "epochs": args.epochs, "pac": args.pac,
        "discriminator_steps": args.discriminator_steps, "batch_size": args.batch_size,
        "train_seconds": round(model.train_seconds, 1), "fill_rows": counts,
        "width": int(model.model._transformer.output_dimensions),
    }
    (ARTIFACTS / f"run_meta_{args.label}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(meta, flush=True)


if __name__ == "__main__":
    main()
