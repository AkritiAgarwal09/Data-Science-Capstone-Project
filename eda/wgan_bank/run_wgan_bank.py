"""Train WGAN-GP on Bank Marketing positive rows and score the synthetic output.

Reads the cleaned training split (data/interim/bank_marketing/train_cleaned.csv).
The validation split is used ONLY for the real-vs-real fidelity baseline and the
distinguishability scorecard; it is never seen during training or generation.

Run:
    uv run --no-sync python eda/wgan_bank/run_wgan_bank.py [--epochs 300] [--seed 42]
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.figure  # noqa: E402
import pandas as pd

from agentic_gan.schemas import fill_rows
from agentic_gan.tools.ctgan_sampling import rule_row_filter
from agentic_gan.tools.quality_fidelity import (
    RULES,
    fidelity_checks,
    real_vs_real,
    scorecard,
    suggest_thresholds,
)
from agentic_gan.tools.quality_privacy import distinguishability_check
from agentic_gan.tools.wgan_gp import sample_wgan_gp, save_model, train_wgan_gp
from agentic_gan.utils.datasets import DATASETS

ROOT = Path(__file__).resolve().parents[2]
CONFIG = DATASETS["bank_marketing"]
TRAIN_FILE = ROOT / "data" / "interim" / CONFIG.name / "train_cleaned.csv"
VAL_FILE = ROOT / "data" / "interim" / CONFIG.name / "validation_cleaned.csv"
ARTIFACT_DIR = ROOT / "eda" / "wgan_bank" / "artifacts"
ID_COLUMN = "source_row"
FILL_LEVELS = (0.10, 0.25, 0.50)


def _lib_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Load data ---
    train = pd.read_csv(TRAIN_FILE)
    if ID_COLUMN in train.columns:
        train = train.drop(columns=ID_COLUMN)

    # Validation is used for scoring only — never for training or generation
    val = pd.read_csv(VAL_FILE)
    if ID_COLUMN in val.columns:
        val = val.drop(columns=ID_COLUMN)

    target = CONFIG.target
    positive = 1
    discrete = list(CONFIG.discrete)

    n_positives = int((train[target] == positive).sum())
    n_negatives = int((train[target] != positive).sum())
    log.info(
        "Training split: %d rows (%d positive, %d negative)",
        len(train), n_positives, n_negatives,
    )

    # --- 2. Train ---
    loss_path = ARTIFACT_DIR / "wgan_losses.csv"
    t0 = time.perf_counter()
    model = train_wgan_gp(
        train,
        target_col=target,
        positive=positive,
        discrete_columns=discrete,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        log_path=loss_path,
    )
    wall_time = time.perf_counter() - t0
    save_model(model, ARTIFACT_DIR / "wgan_bank.pkl")
    log.info("Training wall time: %.1f s", wall_time)

    # --- 3. Sample nested pools for three fill levels ---
    counts = {level: fill_rows(level, n_positives, n_negatives) for level in FILL_LEVELS}
    max_count = max(counts.values())

    bank_rules = RULES.get(CONFIG.name, ())
    row_filter = rule_row_filter(bank_rules) if bank_rules else None

    log.info("Sampling pool of %d rows (max fill level)…", max_count)
    pool = sample_wgan_gp(
        model,
        max_count,
        seed=args.seed,
        row_filter=row_filter,
        drop_duplicates_of=train,
    )

    synth_sets: dict[int, pd.DataFrame] = {}
    for level, n in counts.items():
        level_pct = int(level * 100)
        synth_sets[level_pct] = pool.head(n).reset_index(drop=True)

    # --- 4. Scorecard ---
    real_pos_train = train[train[target] == positive].reset_index(drop=True)
    rvr_report = real_vs_real(
        real_pos_train, val, target=target, positive=positive,
        discrete=CONFIG.discrete, rules=bank_rules,
    )
    thresholds = suggest_thresholds(rvr_report.values, margin=2.0)

    # Collect per-level reports for the unified scorecard table
    fid_reports: dict[str, object] = {}
    disc_results: dict[str, tuple] = {}
    extra_metrics: dict[str, dict] = {}
    real_keys = {tuple(r) for r in real_pos_train[model.columns].itertuples(index=False)}
    job_col = "job" if "job" in real_pos_train.columns else None

    for level_pct, synth in synth_sets.items():
        label = f"fill_{level_pct}pct"
        fid_reports[label] = fidelity_checks(
            real_pos_train,
            synth,
            target=target,
            positive=positive,
            discrete=CONFIG.discrete,
            rules=bank_rules,
            thresholds=thresholds,
        )
        disc_check, disc_detail = distinguishability_check(
            train, val, synth,
            columns=[c for c in synth.columns if c not in {target, "is_synthetic"}],
            target=target,
            positive_class=positive,
            seed=args.seed,
        )
        disc_results[label] = (disc_check, disc_detail)
        n_uniq = synth.drop(columns=["is_synthetic"]).drop_duplicates().shape[0]
        unique_ratio = n_uniq / max(len(synth), 1)
        exact_copies = sum(
            1 for row in synth[model.columns].itertuples(index=False)
            if tuple(row) in real_keys
        )
        extra_metrics[label] = {
            "unique_ratio": unique_ratio,
            "exact_copies": exact_copies,
            "tech_real": float((real_pos_train["job"] == "technician").mean()) if job_col else None,
            "tech_synth": float((synth["job"] == "technician").mean()) if job_col else None,
            "n_rows": len(synth),
        }

    # Build scorecard text
    sc_table = scorecard(rvr_report, fid_reports, thresholds)
    scorecard_lines: list[str] = []
    scorecard_lines.append("=" * 70)
    scorecard_lines.append("WGAN-GP Bank Marketing Scorecard")
    scorecard_lines.append("=" * 70)
    scorecard_lines.append(sc_table)

    for label, (disc_check, disc_detail) in disc_results.items():
        level_pct = label.replace("fill_", "").replace("pct", "")
        em = extra_metrics[label]
        scorecard_lines.append(f"\n--- Fill {level_pct}% ({em['n_rows']} rows) ---")
        scorecard_lines.append(
            f"  real_vs_synthetic_auc: {disc_check.value:.4f}  "
            f"real_vs_real_auc: {disc_detail['real_vs_real_auc']:.4f}"
        )
        scorecard_lines.append(
            f"  unique_ratio: {em['unique_ratio']:.4f}  "
            f"exact_train_copies: {em['exact_copies']}"
        )
        if job_col:
            scorecard_lines.append(
                f"  technician_share real={em['tech_real']:.4f}  "
                f"synthetic={em['tech_synth']:.4f}"
            )

    scorecard_text = "\n".join(scorecard_lines)
    print(scorecard_text)
    (ARTIFACT_DIR / "scorecard.txt").write_text(scorecard_text)

    # --- 5. Loss curve ---
    loss_df = pd.read_csv(loss_path)
    fig = matplotlib.figure.Figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(loss_df["Epoch"], loss_df["Generator Loss"], label="Generator Loss")
    ax.plot(loss_df["Epoch"], loss_df["Discriminator Loss"], label="Discriminator Loss")
    ax2 = ax.twinx()
    ax2.plot(loss_df["Epoch"], loss_df["Gradient Penalty"], color="gray", linestyle="--",
             alpha=0.5, label="Gradient Penalty")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax2.set_ylabel("Gradient Penalty")
    ax.set_title("WGAN-GP Bank Marketing: Training Losses")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    loss_png = ARTIFACT_DIR / "loss_curves.png"
    fig.savefig(loss_png, dpi=120)
    log.info("Loss curve saved to %s", loss_png)

    # --- 6. Sanity gates ---
    final_crit = loss_df["Discriminator Loss"].iloc[-1]
    first10_median = loss_df["Discriminator Loss"].iloc[:10].median()
    diverged = abs(final_crit) > 10 * abs(first10_median) if first10_median != 0 else False

    gates: list[str] = []
    has_nan = pool.isna().any().any()
    n_uniq_pool = pool.drop(columns=["is_synthetic"]).drop_duplicates().shape[0]
    unique_ratio_full = n_uniq_pool / max(len(pool), 1)
    real_keys_pool = {
        tuple(r) for r in real_pos_train[model.columns].itertuples(index=False)
    }
    exact_copy_count = sum(
        1 for row in pool[model.columns].itertuples(index=False)
        if tuple(row) in real_keys_pool
    )
    gates.append(f"{'PASS' if not has_nan else 'FLAG'} no NaN in pool")
    gates.append(
        f"{'PASS' if unique_ratio_full >= 0.95 else 'FLAG'} "
        f"unique_ratio={unique_ratio_full:.4f} (threshold 0.95)"
    )
    gates.append(
        f"{'PASS' if exact_copy_count == 0 else 'FLAG'} exact_train_copies={exact_copy_count}"
    )
    gates.append(
        f"{'PASS' if not diverged else 'FLAG'} "
        f"loss_convergence final_D={final_crit:.3f} 10ep_median={first10_median:.3f}"
    )

    print("\n--- Sanity gates ---")
    for g in gates:
        print(g)

    # --- 7. run_meta.json ---
    meta = {
        "settings": {"epochs": args.epochs, "seed": args.seed, "device": args.device},
        "note": "validation_cleaned.csv used for scorecard only, never for training or generation",
        "wall_time_seconds": round(wall_time, 1),
        "n_train_rows": len(train),
        "n_positive_rows": n_positives,
        "n_negative_rows": n_negatives,
        "fill_counts": {str(int(k * 100)): v for k, v in counts.items()},
        "pool_rows_sampled": len(pool),
        "library_versions": {
            "torch": _lib_version("torch"),
            "ctgan": _lib_version("ctgan"),
            "pandas": _lib_version("pandas"),
        },
        "loss_final": {
            "generator": float(loss_df["Generator Loss"].iloc[-1]),
            "discriminator": float(loss_df["Discriminator Loss"].iloc[-1]),
            "gradient_penalty": float(loss_df["Gradient Penalty"].iloc[-1]),
        },
        "sanity_gates": gates,
    }
    meta_path = ARTIFACT_DIR / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("run_meta.json written to %s", meta_path)
    print(f"\nWall time: {wall_time:.1f}s")
    print(f"Artifacts: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
