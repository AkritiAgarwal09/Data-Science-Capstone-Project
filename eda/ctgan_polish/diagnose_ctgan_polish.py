"""Compare Polish CTGAN minority rows with real training rows where bankrupt is 1.

Writes a loss plot, four numeric distribution figures (16 columns each), a cohort
plot, a chart of the largest correlation differences, and diagnostics.txt.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from agentic_gan.tools.ctgan_fill import fill_paths, range_lines, selection_lines
from agentic_gan.utils.datasets import DATASETS

ROOT = Path(__file__).resolve().parents[2]
CONFIG = DATASETS["polish"]
HORIZONS = tuple(name.removesuffix(".arff") for name in CONFIG.cohorts)
SOURCE_ROW = "source_row"
PAGE_SIZE = 16
KS_STRUGGLE = 0.15
TVD_STRUGGLE = 0.10
SHARE_GAP = 0.10


def _as_level(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _shares(series: pd.Series) -> dict:
    return series.map(_as_level).value_counts(normalize=True).to_dict()


def total_variation(real: pd.Series, synthetic: pd.Series) -> float:
    levels = set(_shares(real)) | set(_shares(synthetic))
    real_share = _shares(real)
    synthetic_share = _shares(synthetic)
    gaps = (abs(real_share.get(level, 0.0) - synthetic_share.get(level, 0.0)) for level in levels)
    return 0.5 * sum(gaps)


def plot_loss(losses: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(losses["Epoch"], losses["Generator Loss"], color="#1f4e79")
    axes[1].plot(losses["Epoch"], losses["Discriminator Loss"], color="#b85c38")
    axes[0].set_ylabel("Generator loss")
    axes[1].set_ylabel("Discriminator loss")
    axes[1].set_xlabel("Epoch")
    axes[0].set_title("Polish CTGAN training loss")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def plot_numeric_pages(
    real: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str], artifacts: Path
) -> None:
    pages = math.ceil(len(columns) / PAGE_SIZE)
    for page in range(pages):
        chunk = columns[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]
        rows, cols = 4, 4
        figure, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        flat = np.atleast_1d(axes).ravel()
        for axis, column in zip(flat, chunk, strict=False):
            low, high = real[column].quantile([0.01, 0.99])
            if low == high:
                low, high = float(real[column].min()), float(real[column].max())
            axis.hist(
                real[column].clip(low, high), bins=30, density=True, alpha=0.55,
                label="real bankrupt", color="#1f4e79",
            )
            axis.hist(
                synthetic[column].clip(low, high), bins=30, density=True, alpha=0.55,
                label="synthetic", color="#b85c38",
            )
            axis.set_title(column, fontsize=10)
        for axis in flat[len(chunk):]:
            axis.axis("off")
        flat[0].legend(fontsize=8)
        figure.suptitle(
            f"Polish numeric columns {chunk[0]}-{chunk[-1]} "
            "(display clipped to the real 1st-99th percentile)",
            y=1.01,
        )
        figure.tight_layout()
        figure.savefig(artifacts / f"dist_numeric_{page + 1}.png", dpi=120, bbox_inches="tight")
        plt.close(figure)


def plot_cohort(real: pd.Series, synthetic: pd.Series, path: Path) -> None:
    levels = sorted(set(real.dropna()) | set(synthetic.dropna()))
    real_share = _shares(real)
    synthetic_share = _shares(synthetic)
    position = np.arange(len(levels))
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(
        position - 0.18, [real_share.get(level, 0) for level in levels],
        width=0.36, label="real bankrupt", color="#1f4e79",
    )
    axis.bar(
        position + 0.18, [synthetic_share.get(level, 0) for level in levels],
        width=0.36, label="synthetic", color="#b85c38",
    )
    axis.set_xticks(position, levels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Share")
    axis.set_title("Cohort share among bankrupt rows")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def correlation_report(
    real: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str], path: Path
) -> tuple[float, list[tuple[str, str, float]]]:
    difference = synthetic[columns].corr() - real[columns].corr()
    mean_abs = float(difference.abs().mean().mean())
    pairs = []
    for left in columns:
        for right in columns:
            if left >= right:
                continue
            pairs.append((left, right, float(difference.loc[left, right])))
    pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    top = pairs[:20]
    figure, axis = plt.subplots(figsize=(10, 6))
    labels = [f"{left}-{right}" for left, right, _ in top]
    values = [gap for _, _, gap in top]
    axis.barh(labels[::-1], values[::-1], color="#1f4e79")
    axis.axvline(0, color="black", linewidth=0.6)
    axis.set_xlabel("Synthetic correlation minus real bankrupt correlation")
    axis.set_title(f"Largest correlation differences, mean absolute {mean_abs:.3f}")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return mean_abs, top


def _numeric_note(name: str, real: pd.Series, synthetic: pd.Series, ks: float) -> str:
    parts = [f"{name}: KS {ks:.3f}"]
    real_std = float(real.std(ddof=0))
    mean_gap = float(synthetic.mean() - real.mean())
    if real_std > 0 and abs(mean_gap) > 0.25 * real_std:
        direction = "higher" if mean_gap > 0 else "lower"
        parts.append(f"centre is {direction} ({synthetic.mean():.4g} vs {real.mean():.4g})")
    synth_std = float(synthetic.std(ddof=0))
    if real_std > 0 and synth_std < 0.7 * real_std:
        parts.append(f"spread is compressed (sd {synth_std:.4g} vs {real_std:.4g})")
    elif real_std > 0 and synth_std > 1.3 * real_std:
        parts.append(f"spread is wider (sd {synth_std:.4g} vs {real_std:.4g})")
    real_zero = float((real == 0).mean())
    synth_zero = float((synthetic == 0).mean())
    if abs(synth_zero - real_zero) >= SHARE_GAP and max(real_zero, synth_zero) >= SHARE_GAP:
        parts.append(f"zero share {synth_zero:.1%} synthetic vs {real_zero:.1%} real")
    real_neg = float((real < 0).mean())
    synth_neg = float((synthetic < 0).mean())
    if abs(synth_neg - real_neg) >= SHARE_GAP and max(real_neg, synth_neg) >= 0.05:
        parts.append(f"negative share {synth_neg:.1%} synthetic vs {real_neg:.1%} real")
    real_p95 = float(real.quantile(0.95))
    synth_p95 = float(synthetic.quantile(0.95))
    if abs(real_p95) > abs(float(real.median())) and abs(synth_p95) < 0.5 * abs(real_p95):
        parts.append(f"upper tail is short (95th {synth_p95:.4g} vs {real_p95:.4g})")
    return ". ".join(parts) + "."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", choices=["v1", "v2"], default="v1")
    parser.add_argument("--horizon", choices=HORIZONS, default=None)
    args = parser.parse_args()
    paths = fill_paths("polish", args.settings, args.horizon)
    artifacts = paths.artifacts
    synthetic_path = paths.fill_file("fill50")
    meta_path = artifacts / "run_meta.json"
    if not meta_path.exists() or not synthetic_path.exists():
        raise SystemExit("run the Polish CTGAN script before diagnostics")
    meta = json.loads(meta_path.read_text())
    target = CONFIG.target
    raw = pd.read_csv(paths.train_file).drop(columns=[SOURCE_ROW])
    if args.horizon is not None:
        raw = raw.loc[raw["cohort"] == args.horizon].reset_index(drop=True)
    real = raw.loc[raw[target] == 1].reset_index(drop=True)
    synthetic = pd.read_csv(synthetic_path)
    numeric = [f"A{index}" for index in range(1, 65)]
    missing = [column for column in numeric if column not in raw.columns]
    if missing or "cohort" not in raw.columns:
        raise SystemExit(f"Polish training columns are missing: {missing}")

    plot_loss(pd.read_csv(artifacts / "ctgan_losses.csv"), artifacts / "loss_curves.png")
    plot_numeric_pages(real, synthetic, numeric, artifacts)
    plot_cohort(real["cohort"], synthetic["cohort"], artifacts / "dist_cohort.png")
    mean_corr, top_pairs = correlation_report(
        real, synthetic, numeric, artifacts / "corr_difference.png"
    )

    numeric_rows = []
    for column in numeric:
        ks = float(ks_2samp(real[column], synthetic[column]).statistic)
        numeric_rows.append({
            "column": column,
            "real_mean": float(real[column].mean()),
            "synthetic_mean": float(synthetic[column].mean()),
            "real_std": float(real[column].std(ddof=0)),
            "synthetic_std": float(synthetic[column].std(ddof=0)),
            "real_median": float(real[column].median()),
            "synthetic_median": float(synthetic[column].median()),
            "ks": ks,
        })
    cohort_tvd = float(total_variation(real["cohort"], synthetic["cohort"]))
    real_cohort = _shares(real["cohort"])
    synth_cohort = _shares(synthetic["cohort"])

    horizon_note = (
        f"horizon: {args.horizon} only (rule PB-1); {meta.get('imputation_fit_on', '')}"
        if args.horizon is not None
        else "horizon: all five cohorts pooled, as the cleaned table provides them"
    )
    lines = [
        "Polish CTGAN diagnostics",
        f"settings: {meta.get('settings_version', 'v1')}, epochs {meta['epochs']}, "
        f"batch {meta['batch_size']}, pac {meta['pac']}, "
        f"discriminator steps {meta['discriminator_steps']}, seed {meta['seed']}, "
        f"device {meta['device']}",
        horizon_note,
        "sampling: " + meta.get("sampling", "conditional on bankrupt=1"),
        *selection_lines(meta),
        f"training seconds: {meta['train_seconds']}",
        "comparison: synthetic fill50 vs real training rows with bankrupt=1",
        f"real bankrupt rows: {len(real):,}",
        f"synthetic rows: {len(synthetic):,}",
        "numeric plots show the real 1st to 99th percentile so the body of each column is visible",
        "KS, means, and medians below use every row",
        "",
        "Numeric columns",
    ]
    for row in numeric_rows:
        lines.append(
            f"  {row['column']}: mean {row['real_mean']:.4g} vs {row['synthetic_mean']:.4g}; "
            f"sd {row['real_std']:.4g} vs {row['synthetic_std']:.4g}; "
            f"median {row['real_median']:.4g} vs {row['synthetic_median']:.4g}; "
            f"KS {row['ks']:.3f}"
        )
    lines.extend(["", "Cohort shares among bankrupt rows"])
    for level in sorted(set(real_cohort) | set(synth_cohort)):
        lines.append(
            f"  {level}: real {real_cohort.get(level, 0):.1%}, "
            f"synthetic {synth_cohort.get(level, 0):.1%}"
        )
    lines.append(f"  cohort TVD: {cohort_tvd:.3f}")
    far = [
        level for level in set(real_cohort) | set(synth_cohort)
        if abs(synth_cohort.get(level, 0) - real_cohort.get(level, 0)) >= SHARE_GAP
    ]
    if far:
        lines.append(
            "Cohort shares far from the real bankrupt rows: " + ", ".join(sorted(far)) + "."
        )
    else:
        lines.append("No synthetic cohort share is 10 percentage points away from the real share.")
    lines.extend([
        "",
        f"Mean absolute correlation difference: {mean_corr:.3f}",
        "A 64 by 64 heatmap is not a useful check. The largest off-diagonal gaps are:",
    ])
    for left, right, gap in top_pairs:
        lines.append(f"  {left}, {right}: {gap:+.3f}")
    lines.extend(["", *range_lines(meta), ""])

    ranked = sorted(numeric_rows, key=lambda row: row["ks"], reverse=True)
    struggled = [row for row in ranked if row["ks"] >= KS_STRUGGLE]
    lines.append("Columns that struggled")
    if meta.get("rake_columns"):
        lines.append(
            "Raked columns: " + ", ".join(meta["rake_columns"]) + ". "
            "A small cohort TVD after raking is the reweighting. Numeric columns were not raked."
        )
    lines.append(
        f"Numeric columns are ranked by KS. A column is marked as struggling at KS "
        f"{KS_STRUGGLE:.2f} or higher. Cohort is marked at TVD {TVD_STRUGGLE:.2f} or higher."
    )
    if not struggled:
        lines.append("No numeric column reached the KS line. The largest gaps are still listed.")
    for row in ranked:
        mark = " struggled." if row in struggled else ""
        note = _numeric_note(
            row["column"], real[row["column"]], synthetic[row["column"]], row["ks"]
        )
        lines.append(f"  {note}{mark}")
    cohort_mark = " struggled." if cohort_tvd >= TVD_STRUGGLE else ""
    lines.append(f"  cohort: TVD {cohort_tvd:.3f}.{cohort_mark}")
    (artifacts / "diagnostics.txt").write_text("\n".join(lines) + "\n")
    worst = ", ".join(f"{row['column']} {row['ks']:.3f}" for row in ranked[:5])
    print(f"wrote {artifacts.relative_to(ROOT) / 'diagnostics.txt'}")
    print("worst numeric:", worst)
    print(f"cohort TVD: {cohort_tvd:.3f}")


if __name__ == "__main__":
    main()
