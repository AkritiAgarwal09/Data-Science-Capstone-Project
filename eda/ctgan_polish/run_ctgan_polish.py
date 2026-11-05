"""Train CTGAN on the cleaned Polish training split and write minority fill files.

``--settings v1`` is the original Bank run: 300 epochs, batch size 500, pac 10,
one discriminator step, conditional sampling. ``--settings v2`` is Kyle's Bank
recipe (docs/notes/ctgan_bank_v2.md): pac 5, three discriminator steps,
unconditional sampling, and raking the discrete columns from a 60,000-row pool.
v2 files are named polish_ctgan_v2_fill*.csv and written beside the v1 files.

The cleaned table pools five horizon files. The written selection rules refuse a
pooled table (rule PB-1), so ``--horizon 3year`` keeps one cohort's rows and
writes polish_3year_ctgan*_fill*.csv with artifacts in artifacts*_3year/. On one
horizon the table is all numeric, the rule names WGAN-GP, and CTGAN is trained as
the documented comparator (rule PB-3). Either way run_meta.json records the rule's
choice, every rule step, and the override. The imputation in the cleaned table was
fitted on the pooled training split; a horizon run only filters rows afterwards.

    uv run python eda/ctgan_polish/run_ctgan_polish.py --settings v2
    uv run python eda/ctgan_polish/run_ctgan_polish.py --settings v2 --horizon 3year
"""

from __future__ import annotations

import argparse
import logging

from agentic_gan.tools.ctgan_fill import fill_paths, run_fill
from agentic_gan.utils.ctgan_settings import load_ctgan_settings
from agentic_gan.utils.datasets import DATASETS

DATASET = "polish"
HORIZONS = tuple(name.removesuffix(".arff") for name in DATASETS[DATASET].cohorts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", choices=["v1", "v2"], default="v1")
    parser.add_argument("--horizon", choices=HORIZONS, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"])
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_ctgan_settings(
        args.settings, epochs=args.epochs, device=args.device, seed=args.seed
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_fill(fill_paths(DATASET, settings.version, args.horizon), settings)


if __name__ == "__main__":
    main()
