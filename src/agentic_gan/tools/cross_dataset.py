"""One results table and chart across datasets.

Each row is a dataset, arm and fill level: the mean test score over seeds, its seed-to-seed spread,
the paired lift over the reference arm, and the reliable-gain verdict from `tools/evaluation.py`.
The input is either finished runs' report/comparisons.csv or the training agent's metrics table
(`results/training_all_arms.csv`, from `tools/training_grid.py`).

    uv run python -m agentic_gan.tools.cross_dataset runs/<bank> runs/<polish> \
        --out results/cross_dataset.csv
    uv run python -m agentic_gan.tools.cross_dataset --metrics results/training_all_arms.csv \
        --out results/cross_dataset.csv --plot results/cross_dataset.png
"""

import argparse
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from agentic_gan.tools.evaluation import ARM_ORDER, compare_arms, reliable_gain

COLUMNS = ["dataset", "arm", "fill_level", "n_seeds", "test_mean", "test_sd", "lift", "verdict"]


# Okabe-Ito hues, in a fixed order per arm (checked for colour-blind separation); each arm also
# has its own marker shape, so colour is never the only cue.
ARM_COLORS = {"original": "#CC79A7", "cleaned": "#E69F00", "smotenc": "#0072B2", "ctgan": "#009E73"}
ARM_MARKERS = {"original": "s", "cleaned": "D", "smotenc": "o", "ctgan": "^"}
DATASET_NAMES = {
    "bank_marketing": "Bank Marketing",
    "default": "Default",
    "polish": "Polish",
    "ieee_cis": "IEEE-CIS",
}
METRIC_TITLES = {"pr_auc": "Test PR-AUC", "f1": "Test F1 (frozen threshold)"}


def comparisons_from_metrics(metrics: pd.DataFrame, *, reference: str = "cleaned") -> pd.DataFrame:
    """The evaluation tool's comparison table for a training agent metrics table."""
    return compare_arms(metrics, reference=reference).assign(reference=reference)


def table_from_comparisons(
    comparisons: pd.DataFrame, *, reference: str = "cleaned", metric: str = "pr_auc"
) -> pd.DataFrame:
    """Keep one test-score row per dataset, arm and fill level, with the reliable-gain verdict."""
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


def cross_dataset_table(
    run_dirs: Iterable[Path], *, reference: str = "cleaned", metric: str = "pr_auc"
) -> pd.DataFrame:
    """Stack the runs' comparisons and keep one test-score row per dataset, arm and fill level."""
    comparisons = pd.concat(
        [pd.read_csv(Path(run) / "report" / "comparisons.csv") for run in run_dirs],
        ignore_index=True,
    )
    return table_from_comparisons(comparisons, reference=reference, metric=metric)


def plot_cross_dataset(
    comparisons: pd.DataFrame, path: Path, *, reference: str = "cleaned"
) -> None:
    """Test PR-AUC (top) and F1 (bottom) against fill level, one panel per dataset.

    The original and cleaned arms are single points at 0%; SMOTENC and CTGAN are lines over the
    fill levels, with the seed-to-seed standard deviation as bars (none for a one-seed dataset).
    A dotted line marks the reference arm, which the synthetic rows are added to.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    test = comparisons[comparisons.split.eq("test")]
    datasets = list(dict.fromkeys(test.dataset))
    fig, axes = plt.subplots(2, len(datasets), figsize=(3.7 * len(datasets), 6.6), squeeze=False)
    for row, metric in enumerate(METRIC_TITLES):
        for col, dataset in enumerate(datasets):
            ax = axes[row][col]
            rows = test[test.dataset.eq(dataset) & test.metric.eq(metric)]
            base = rows[rows.arm.eq(reference)]
            if not base.empty:
                ax.axhline(base["mean"].iloc[0], color=ARM_COLORS[reference], ls=":", lw=1.2)
            for arm in ARM_ORDER:
                points = rows[rows.arm.eq(arm)].sort_values("fill_level")
                if points.empty:
                    continue
                single = arm in ("original", "cleaned")
                ax.errorbar(
                    [0.0] * len(points) if single else points.fill_level * 100,
                    points["mean"],
                    yerr=points["sd"].fillna(0),
                    color=ARM_COLORS[arm],
                    marker=ARM_MARKERS[arm],
                    ms=7 if single else 6,
                    lw=0 if single else 1.8,
                    capsize=3,
                    elinewidth=1.5,
                    mec="white",
                    mew=0.8,
                )
            ax.set_xticks([0, 10, 25, 50], ["0%", "10%", "25%", "50%"])
            ax.grid(axis="y", color="#d0d0d0", lw=0.5)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if row == 0:
                seeds = int(rows["n_seeds"].max()) if not rows.empty else 0
                name = DATASET_NAMES.get(dataset, dataset)
                ax.set_title(f"{name} ({seeds} seed{'s' if seeds != 1 else ''})", fontsize=11)
            if col == 0:
                ax.set_ylabel(METRIC_TITLES[metric])
            if row == 1:
                ax.set_xlabel("Fill level (synthetic rows added)")
    present = [arm for arm in ARM_ORDER if test.arm.eq(arm).any()]
    handles = [
        Line2D(
            [],
            [],
            color=ARM_COLORS[a],
            marker=ARM_MARKERS[a],
            ls="-" if a in ("smotenc", "ctgan") else "",
            ms=7,
            label=a,
        )
        for a in present
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False)
    note = (
        f"Mean over seeds, bars = standard deviation; dotted line = the {reference} arm that the "
        "synthetic rows are added to. Each panel has its own scale."
    )
    fig.text(0.5, 0.005, note, ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dirs", nargs="*", type=Path, help="finished run folders")
    parser.add_argument("--metrics", type=Path, help="a training agent metrics table instead")
    parser.add_argument("--reference", default="cleaned", help="arm the lifts are measured against")
    parser.add_argument("--out", type=Path, help="write the table here as CSV")
    parser.add_argument("--plot", type=Path, help="write the chart here as PNG (needs --metrics)")
    args = parser.parse_args()
    if bool(args.metrics) == bool(args.run_dirs):
        parser.error("give either run folders or --metrics")
    if args.plot and not args.metrics:
        parser.error("--plot needs --metrics")
    if args.metrics:
        comparisons = comparisons_from_metrics(pd.read_csv(args.metrics), reference=args.reference)
        table = table_from_comparisons(comparisons, reference=args.reference)
    else:
        table = cross_dataset_table(args.run_dirs, reference=args.reference)
    print(table.round(4).to_string(index=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
    if args.plot:
        plot_cross_dataset(comparisons, args.plot, reference=args.reference)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
