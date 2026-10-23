"""Compare Default CTGAN minority rows with real training rows where y is 1.

Writes loss curves, per-column distribution plots, a correlation-difference
heatmap and diagnostics.txt. The last section names the columns that struggled.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from agentic_gan.utils.datasets import DATASETS

ROOT = Path(__file__).resolve().parents[2]
CONFIG = DATASETS["default"]
TRAIN_FILE = ROOT / "data" / "interim" / CONFIG.name / "train_cleaned.csv"
ARTIFACTS = ROOT / "eda" / "ctgan_default" / "artifacts"
SYNTHETIC = ROOT / "data" / "interim" / "default" / "synthetic" / "default_ctgan_fill50.csv"
SOURCE_ROW = "source_row"
PAY_COLUMNS = [f"PAY_{month}" for month in range(1, 7)]
KS_STRUGGLE = 0.15
TVD_STRUGGLE = 0.10


def total_variation(real: pd.Series, synthetic: pd.Series) -> float:
    levels = sorted(set(_levels(real)) | set(_levels(synthetic)), key=str)
    real_share = _shares(real)
    synthetic_share = _shares(synthetic)
    gaps = (abs(real_share.get(level, 0.0) - synthetic_share.get(level, 0.0)) for level in levels)
    return 0.5 * sum(gaps)


def _levels(series: pd.Series) -> list:
    return [_as_level(value) for value in series.dropna().unique()]


def _as_level(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _shares(series: pd.Series) -> dict:
    mapped = series.map(_as_level)
    return mapped.value_counts(normalize=True).to_dict()


def _grid(count: int, columns: int = 4) -> tuple[int, int]:
    return math.ceil(count / columns), columns


def plot_loss(losses: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(losses["Epoch"], losses["Generator Loss"], color="#1f4e79")
    axes[1].plot(losses["Epoch"], losses["Discriminator Loss"], color="#b85c38")
    axes[0].set_ylabel("Generator loss")
    axes[1].set_ylabel("Discriminator loss")
    axes[1].set_xlabel("Epoch")
    axes[0].set_title("Default CTGAN training loss")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def plot_numeric(
    real: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str], path: Path
) -> None:
    rows, cols = _grid(len(columns))
    figure, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    flat = np.atleast_1d(axes).ravel()
    for axis, column in zip(flat, columns, strict=False):
        axis.hist(
            real[column], bins=30, density=True, alpha=0.55,
            label="real minority", color="#1f4e79",
        )
        axis.hist(
            synthetic[column], bins=30, density=True, alpha=0.55,
            label="synthetic", color="#b85c38",
        )
        axis.set_title(column, fontsize=10)
    for axis in flat[len(columns):]:
        axis.axis("off")
    flat[0].legend(fontsize=8)
    figure.suptitle("Default numeric columns: real minority vs synthetic", y=1.01)
    figure.tight_layout()
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def plot_categorical(
    real: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str], path: Path
) -> None:
    rows, cols = _grid(len(columns), columns=3)
    figure, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.4 * rows))
    flat = np.atleast_1d(axes).ravel()
    for axis, column in zip(flat, columns, strict=False):
        levels = sorted(set(_levels(real[column])) | set(_levels(synthetic[column])), key=str)
        real_share = _shares(real[column])
        synthetic_share = _shares(synthetic[column])
        position = np.arange(len(levels))
        axis.bar(
            position - 0.18,
            [real_share.get(level, 0) for level in levels],
            width=0.36, label="real minority", color="#1f4e79",
        )
        axis.bar(
            position + 0.18,
            [synthetic_share.get(level, 0) for level in levels],
            width=0.36, label="synthetic", color="#b85c38",
        )
        labels = [str(level) for level in levels]
        axis.set_xticks(position, labels, rotation=45, ha="right", fontsize=7)
        axis.set_title(column, fontsize=10)
        axis.set_ylim(0, 1)
    for axis in flat[len(columns):]:
        axis.axis("off")
    flat[0].legend(fontsize=8)
    figure.suptitle("Default discrete columns: real minority vs synthetic", y=1.01)
    figure.tight_layout()
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def plot_correlation(
    real: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str], path: Path
) -> float:
    difference = synthetic[columns].corr() - real[columns].corr()
    mean_abs = float(difference.abs().mean().mean())
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(difference, cmap="coolwarm", vmin=-0.5, vmax=0.5)
    axis.set_xticks(range(len(columns)), columns, rotation=90, fontsize=7)
    axis.set_yticks(range(len(columns)), columns, fontsize=7)
    figure.colorbar(image, ax=axis, fraction=0.046)
    axis.set_title(f"Correlation difference, mean absolute {mean_abs:.3f}")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return mean_abs


def _numeric_note(name: str, real: pd.Series, synthetic: pd.Series, ks: float) -> str:
    parts = [f"{name}: KS {ks:.3f}"]
    real_std = float(real.std(ddof=0))
    mean_gap = float(synthetic.mean() - real.mean())
    if real_std > 0 and abs(mean_gap) > 0.25 * real_std:
        direction = "higher" if mean_gap > 0 else "lower"
        parts.append(
            f"synthetic mean is {direction} ({synthetic.mean():.4g} vs real {real.mean():.4g})"
        )
    synth_std = float(synthetic.std(ddof=0))
    if real_std > 0 and synth_std < 0.7 * real_std:
        parts.append(
            f"spread is compressed (synthetic sd {synth_std:.4g}, real sd {real_std:.4g})"
        )
    elif real_std > 0 and synth_std > 1.3 * real_std:
        parts.append(f"spread is wider (synthetic sd {synth_std:.4g}, real sd {real_std:.4g})")
    real_zero = float((real == 0).mean())
    synth_zero = float((synthetic == 0).mean())
    if abs(synth_zero - real_zero) >= 0.10 and max(real_zero, synth_zero) >= 0.10:
        parts.append(
            f"share at zero is {synth_zero:.1%} synthetic vs {real_zero:.1%} real minority"
        )
    real_p95 = float(real.quantile(0.95))
    synth_p95 = float(synthetic.quantile(0.95))
    if abs(real_p95) > abs(float(real.median())) and abs(synth_p95) < 0.5 * abs(real_p95):
        parts.append(
            f"upper tail is short (synthetic 95th {synth_p95:.4g}, real 95th {real_p95:.4g})"
        )
    on_edge = float(((synthetic == real.max()) | (synthetic == real.min())).mean())
    if on_edge >= 0.05:
        parts.append(f"{on_edge:.1%} of synthetic values sit on the real min or max")
    real_neg = float((real < 0).mean())
    synth_neg = float((synthetic < 0).mean())
    if abs(synth_neg - real_neg) >= 0.10 and max(real_neg, synth_neg) >= 0.05:
        parts.append(
            f"negative share is {synth_neg:.1%} synthetic vs {real_neg:.1%} real minority"
        )
    return ". ".join(parts) + "."


def _discrete_note(name: str, real: pd.Series, synthetic: pd.Series, tvd: float) -> str:
    real_share = _shares(real)
    synth_share = _shares(synthetic)
    gaps = []
    for level in set(real_share) | set(synth_share):
        gaps.append((abs(synth_share.get(level, 0.0) - real_share.get(level, 0.0)), level))
    gaps.sort(reverse=True)
    _, worst = gaps[0]
    return (
        f"{name}: TVD {tvd:.3f}. Largest share gap is code {worst}: "
        f"synthetic {_shares(synthetic).get(worst, 0):.1%}, "
        f"real minority {real_share.get(worst, 0):.1%}."
    )


def _payment_note(real: pd.DataFrame, synthetic: pd.DataFrame) -> str:
    lines = ["Payment status, share of rows with a delay code of 1 or more:"]
    for column in PAY_COLUMNS:
        real_share = float((real[column] >= 1).mean())
        synth_share = float((synthetic[column] >= 1).mean())
        lines.append(f"  {column}: real minority {real_share:.1%}, synthetic {synth_share:.1%}")
    real_recent = float((real["PAY_1"] >= 1).mean())
    synth_recent = float((synthetic["PAY_1"] >= 1).mean())
    if abs(synth_recent - real_recent) >= 0.10:
        lines.append(
            "The most recent status, PAY_1, does not match the real default rows. "
            "That payment-history pattern is unrealistic."
        )
    else:
        lines.append(
            "The most recent status, PAY_1, is close to the real default rows. "
            "Earlier months are listed above so a flat or drifting history is visible."
        )
    return "\n".join(lines)


def main() -> None:
    meta_path = ARTIFACTS / "run_meta.json"
    if not meta_path.exists() or not SYNTHETIC.exists():
        raise SystemExit("run the Default CTGAN script before diagnostics")
    meta = json.loads(meta_path.read_text())
    target = CONFIG.target
    discrete = list(CONFIG.discrete)
    raw = pd.read_csv(TRAIN_FILE).drop(columns=[SOURCE_ROW])
    real = raw.loc[raw[target] == 1].reset_index(drop=True)
    synthetic = pd.read_csv(SYNTHETIC)
    numeric = [
        column for column in raw.columns if column not in discrete and column != target
    ]
    if len(numeric) != 14 or len(discrete) != 9:
        raise SystemExit(
            f"expected 14 numeric and 9 discrete columns, found {len(numeric)} and {len(discrete)}"
        )

    plot_loss(pd.read_csv(ARTIFACTS / "ctgan_losses.csv"), ARTIFACTS / "loss_curves.png")
    plot_numeric(real, synthetic, numeric, ARTIFACTS / "dist_numeric.png")
    plot_categorical(real, synthetic, discrete, ARTIFACTS / "dist_categorical.png")
    mean_corr = plot_correlation(real, synthetic, numeric, ARTIFACTS / "corr_difference.png")

    numeric_rows = []
    for column in numeric:
        ks = float(ks_2samp(real[column], synthetic[column]).statistic)
        numeric_rows.append(
            {
                "column": column,
                "real_mean": float(real[column].mean()),
                "synthetic_mean": float(synthetic[column].mean()),
                "real_std": float(real[column].std(ddof=0)),
                "synthetic_std": float(synthetic[column].std(ddof=0)),
                "real_median": float(real[column].median()),
                "synthetic_median": float(synthetic[column].median()),
                "ks": ks,
            }
        )
    discrete_rows = []
    for column in discrete:
        tvd = float(total_variation(real[column], synthetic[column]))
        discrete_rows.append({"column": column, "tvd": tvd})

    lines = [
        "Default CTGAN diagnostics",
        f"settings: {meta.get('settings_version', 'v1')}, epochs {meta['epochs']}, "
        f"batch {meta['batch_size']}, pac {meta['pac']}, "
        f"discriminator steps {meta['discriminator_steps']}, seed {meta['seed']}, "
        f"device {meta['device']}",
        "sampling: " + meta.get("sampling", "conditional on the minority class"),
        f"training seconds: {meta['train_seconds']}",
        "comparison: synthetic fill50 vs real training rows with y=1",
        f"real minority rows: {len(real):,}",
        f"synthetic rows: {len(synthetic):,}",
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
    lines.extend(["", "Discrete columns (total variation distance)"])
    for row in discrete_rows:
        lines.append(f"  {row['column']}: TVD {row['tvd']:.3f}")
    lines.extend(["", f"Mean absolute correlation difference: {mean_corr:.3f}", ""])

    invalid = []
    for column in discrete:
        extra = set(_levels(synthetic[column])) - set(_levels(raw[column]))
        if extra:
            invalid.append(f"{column} {sorted(extra)}")
    if invalid:
        lines.append("Invalid category codes: " + "; ".join(invalid))
    else:
        lines.append(
            "Invalid category codes: none. Every discrete code appears in the real training data."
        )
    lines.extend(["", _payment_note(real, synthetic), ""])

    ranked_numeric = sorted(numeric_rows, key=lambda row: row["ks"], reverse=True)
    ranked_discrete = sorted(discrete_rows, key=lambda row: row["tvd"], reverse=True)
    struggled_numeric = [row for row in ranked_numeric if row["ks"] >= KS_STRUGGLE]
    struggled_discrete = [row for row in ranked_discrete if row["tvd"] >= TVD_STRUGGLE]

    lines.append("Columns that struggled")
    if meta.get("rake_columns"):
        raked = ", ".join(meta["rake_columns"])
        lines.append(
            "These discrete columns were raked to the real minority shares after sampling: "
            f"{raked}. A small TVD on a raked column is the reweighting, not evidence that "
            "CTGAN learned that column. Numeric columns were not raked."
        )
    lines.append(
        "Numeric columns are ranked by KS and discrete columns by TVD. "
        f"A numeric column is marked as struggling at KS {KS_STRUGGLE:.2f} or higher, "
        f"and a discrete column at TVD {TVD_STRUGGLE:.2f} or higher. "
        "If nothing crosses the line, the largest gap is still listed without the struggled mark."
    )
    if not struggled_numeric:
        lines.append("No numeric column reached the KS line.")
    if not struggled_discrete:
        lines.append("No discrete column reached the TVD line.")
    lines.append("Numeric, worst first:")
    for row in ranked_numeric:
        mark = " struggled." if row in struggled_numeric else ""
        note = _numeric_note(
            row["column"], real[row["column"]], synthetic[row["column"]], row["ks"]
        )
        lines.append(f"  {note}{mark}")
    lines.append("Discrete, worst first:")
    for row in ranked_discrete:
        mark = " struggled." if row in struggled_discrete else ""
        note = _discrete_note(
            row["column"], real[row["column"]], synthetic[row["column"]], row["tvd"]
        )
        lines.append(f"  {note}{mark}")
    (ARTIFACTS / "diagnostics.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {ARTIFACTS.relative_to(ROOT) / 'diagnostics.txt'}")
    worst_numeric = ", ".join(f"{row['column']} {row['ks']:.3f}" for row in ranked_numeric[:3])
    worst_discrete = ", ".join(f"{row['column']} {row['tvd']:.3f}" for row in ranked_discrete[:3])
    print("worst numeric:", worst_numeric)
    print("worst discrete:", worst_discrete)


if __name__ == "__main__":
    main()
