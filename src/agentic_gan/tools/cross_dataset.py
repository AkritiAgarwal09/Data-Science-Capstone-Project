"""One results table across datasets, built from finished runs' report/comparisons.csv.

Each row is a dataset, arm and fill level: the mean test score over seeds, its seed-to-seed spread,
the paired lift over the reference arm, and the reliable-gain verdict from `tools/evaluation.py`.

    uv run python -m agentic_gan.tools.cross_dataset runs/<bank> runs/<polish> \
        --out results/cross_dataset.csv
"""

import argparse
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from agentic_gan.tools.evaluation import reliable_gain

COLUMNS = ["dataset", "arm", "fill_level", "n_seeds", "test_mean", "test_sd", "lift", "verdict"]


def cross_dataset_table(
    run_dirs: Iterable[Path], *, reference: str = "cleaned", metric: str = "pr_auc"
) -> pd.DataFrame:
    """Stack the runs' comparisons and keep one test-score row per dataset, arm and fill level."""
    comparisons = pd.concat(
        [pd.read_csv(Path(run) / "report" / "comparisons.csv") for run in run_dirs],
        ignore_index=True,
    )
    chosen = comparisons[comparisons.reference.eq(reference) & comparisons.metric.eq(metric)]
    rows = []
    for (dataset, arm, level), group in chosen.groupby(
        ["dataset", "arm", "fill_level"], sort=False
    ):
        test = group[group.split.eq("test")]
        if test.empty:
            continue
        test = test.iloc[0]
        if arm == reference:
            verdict, lift = "reference", float("nan")
        else:
            verdict = reliable_gain(
                chosen, dataset=dataset, arm=arm, fill_level=level, metric=metric
            ).verdict
            lift = test.diff_mean
        rows.append([dataset, arm, level, int(test.n_seeds), test["mean"], test.sd, lift, verdict])
    return pd.DataFrame(rows, columns=COLUMNS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dirs", nargs="+", type=Path, help="finished run folders")
    parser.add_argument("--reference", default="cleaned", help="arm the lifts are measured against")
    parser.add_argument("--out", type=Path, help="write the table here as CSV")
    args = parser.parse_args()
    table = cross_dataset_table(args.run_dirs, reference=args.reference)
    print(table.round(4).to_string(index=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
