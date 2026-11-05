"""Train CTGAN on the cleaned Default training split and write minority fill files.

``--settings v1`` is the original Bank run: 300 epochs, batch size 500, pac 10,
one discriminator step, conditional sampling. ``--settings v2`` is Kyle's Bank
recipe (docs/notes/ctgan_bank_v2.md): pac 5, three discriminator steps,
unconditional sampling, and raking the discrete columns from a 60,000-row pool.
v2 files are named default_ctgan_v2_fill*.csv and written beside the v1 files.

The written selection rules choose TimeGAN for Default and name CTGAN on the
flattened wide table as the fallback (rule DC-4). TimeGAN is not implemented, so
this script trains that fallback and records the rule's choice, every rule step,
and the override in run_meta.json. The Default consistency rules (repayment codes,
non-negative payments, coded static columns) drop any synthetic row that breaks them.

    uv run python eda/ctgan_default/run_ctgan_default.py --settings v2
"""

from __future__ import annotations

import argparse
import logging

from agentic_gan.tools.ctgan_fill import fill_paths, run_fill
from agentic_gan.utils.ctgan_settings import load_ctgan_settings

DATASET = "default"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", choices=["v1", "v2"], default="v1")
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
    run_fill(fill_paths(DATASET, settings.version), settings)


if __name__ == "__main__":
    main()
