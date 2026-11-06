"""Driver for the Polish e2e run."""
import argparse
import subprocess
import sys
from pathlib import Path

from agentic_gan.orchestrator.graph import config_for_dataset, run_pipeline
from agentic_gan.utils.datasets import DATASETS

POOLED_COHORTS = ("1year.arff", "2year.arff", "3year.arff", "4year.arff", "5year.arff")
YEAR3_COHORTS  = ("3year.arff",)

def _check_datasets_json(mode: str) -> None:
    actual = tuple(DATASETS["polish"].cohorts)
    expected = YEAR3_COHORTS if mode == "3year" else POOLED_COHORTS
    if actual != expected:
        sys.exit(
            f"ERROR: datasets.json cohorts={actual!r} does not match --mode {mode!r} "
            f"(expected {expected!r}).\n"
            f"Apply eda/polish_e2e/polish_3year_datasets.patch for 3year mode."
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["3year", "pooled"], default="3year")
    parser.add_argument("--smoke", action="store_true",
                        help="Skip CTGAN: set min_rows=1_000_000_000")
    parser.add_argument("--protocol", type=Path,
                        default=Path("results/e2e_polish_3year/baseline_protocol_polish_42_cv0.json"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    _check_datasets_json(args.mode)

    rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                   text=True).strip()
    dirty = subprocess.run(["git", "diff", "--quiet"], capture_output=True).returncode != 0
    print(f"revision: {rev}{'*' if dirty else ''}")

    config = config_for_dataset("polish")
    updates: dict = dict(
        gan_bounds={
            "epochs": (100.0, 600.0),
            "pac": (5.0, 10.0),
            "discriminator_steps": (1.0, 3.0),
            "batch_size": (250.0, 500.0),
        }
    )
    if args.mode == "3year":
        updates["training_protocol_path"] = args.protocol.resolve()
    if args.smoke:
        updates["min_rows"] = 1_000_000_000
        updates["fill_levels"] = [0.5]
        updates["seeds"] = [42]
    config = config.updated(**updates)

    state, run = run_pipeline(config, runs_dir=args.runs_dir)
    final = "SUCCESS" if state.is_complete else "FAILED"
    print(f"Run : {run.root}")
    print(f"Status: {final}")
    if final != "SUCCESS":
        sys.exit(1)

if __name__ == "__main__":
    main()
