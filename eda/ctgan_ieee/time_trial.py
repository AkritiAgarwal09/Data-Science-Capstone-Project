"""Time CTGAN on a stratified sample of the IEEE-CIS training table (issue #90).

IEEE-CIS has 540 columns, about 1,100 categories and a 3% fraud rate, so CTGAN's internal width
is in the thousands. This trains for a few epochs on N rows, prints the width and the seconds, and
appends the run to eda/ctgan_ieee/artifacts/time_trials.csv. Run it twice with different epochs
(say 1 and 3): the difference is the time per epoch, and what is left is the one-off set-up cost.

    uv run python eda/ctgan_ieee/time_trial.py --rows 5000 --epochs 1
    uv run python eda/ctgan_ieee/time_trial.py --rows 5000 --epochs 3
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from agentic_gan.tools.ctgan import train_ctgan
from agentic_gan.utils.datasets import DATASETS

CONFIG = DATASETS["ieee_cis"]
ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "data" / "interim" / CONFIG.name
RESULTS = Path(__file__).resolve().parent / "artifacts" / "time_trials.csv"


def load_sample(
    rows: int, seed: int, fraud_only: bool = False
) -> tuple[pd.DataFrame, list[str]]:
    """The cleaned training sample: thinned to ``rows`` rows with the fraud share kept, or (with
    ``fraud_only``) just the fraud rows, so every generated row is fraud."""
    settings = json.loads((FOLDER / "encoder_settings.json").read_text())
    data = pd.read_csv(FOLDER / "train_cleaned.csv", index_col="source_row", low_memory=False)
    discrete = [c for c in settings["gan_discrete_columns"] if c in data.columns]
    data[discrete] = data[discrete].astype(str)
    if fraud_only:
        return data[data[CONFIG.target] == 1], discrete
    if rows < len(data):
        data = data.groupby(CONFIG.target, group_keys=False).sample(
            frac=rows / len(data), random_state=seed
        )
    return data, discrete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5_000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--pac", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraud-only", action="store_true", help="train on the fraud rows only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    data, discrete = load_sample(args.rows, args.seed, args.fraud_only)
    model = train_ctgan(
        data, CONFIG.target, discrete_columns=discrete, epochs=args.epochs,
        batch_size=args.batch_size, pac=args.pac, seed=args.seed,
    )
    row = {
        "rows": len(data), "fraud_rows": int(data[CONFIG.target].sum()), "columns": data.shape[1],
        "epochs": args.epochs, "batch_size": args.batch_size, "pac": args.pac,
        "fraud_only": args.fraud_only,
        "width": model.model._transformer.output_dimensions,
        "seconds": round(model.train_seconds, 1),
    }
    print(row, flush=True)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)


if __name__ == "__main__":
    main()
