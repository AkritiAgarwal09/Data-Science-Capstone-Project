"""Augmented-training comparison: real training rows plus synthetic rows, at several fill levels.

For each generator, fill level and seed, the synthetic rows are added to the real training
rows only, encoded with the preprocessing fitted on the real training rows, and scored with
``run_models`` on the untouched real validation and test tables. Splits, models, threshold and
PR-AUC method match Noor's baseline (``tools/baselines.py``); only the model seed varies.

    uv run python -m agentic_gan.tools.augmented --generator smotenc
    uv run python -m agentic_gan.tools.augmented --generator ctgan --arm-label ctgan_v1
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from agentic_gan.tools.baselines import build_models, load_protocol, run_models
from agentic_gan.tools.preprocessing import Preprocessor
from agentic_gan.tools.smotenc import smotenc_rows
from agentic_gan.utils.datasets import DATASETS, DatasetConfig
from agentic_gan.utils.synthetic import IS_SYNTHETIC, fill_counts

ROOT = Path(__file__).resolve().parents[3]
ID_COLUMN = "source_row"
SEEDS = (42,)
FILL_LEVELS = (0.10, 0.25, 0.50)
# Fixed Sprint 2 suite, shared with the baseline runner.
DEFAULT_MODELS = ("Gradient boosting", "LightGBM", "CatBoost", "Stacking ensemble")
REAL_ONLY = "real_only"
RESULT_COLUMNS = [
    "dataset",
    "arm",
    "fill_level",
    "n_synthetic",
    "seed",
    "cv_folds",
    "model",
    "status",
    "selected",
    "protocol_id",
    "threshold",
    "threshold_source",
    "val_pr_auc",
    "val_f1",
    "test_pr_auc",
    "test_f1",
    "test_precision",
    "test_recall",
    "test_roc_auc",
    "test_accuracy",
    "fit_seconds",
]
KEYS = ["dataset", "arm", "fill_level", "seed"]
DEFAULT_OUT = ROOT / "results" / "augmented_bank_marketing_frozen.csv"
BASELINES_FILE = ROOT / "results" / "baselines.csv"
# Colorblind-safe (Okabe-Ito): one hue per arm, fixed order.
ARM_COLORS = {"smotenc": "#0072B2", "ctgan_v1": "#E69F00", "ctgan_v2": "#009E73"}


def _protocol_path(protocol_dir, dataset, seed):
    """Prefer the canonical CV3 protocol, or discover a unique alternate protocol."""
    root = Path(protocol_dir)
    canonical = root / f"baseline_protocol_{dataset}_{seed}_cv3.json"
    if canonical.is_file():
        return canonical
    candidates = sorted(root.glob(f"baseline_protocol_{dataset}_{seed}_cv*.json"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple frozen baseline protocols found for {dataset} seed {seed}; "
            "remove ambiguity or pass a protocol directory containing one"
        )
    return canonical


def _protocol_cv_folds(path):
    try:
        folds = json.loads(Path(path).read_text()).get("cv_folds")
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read the frozen protocol CV setting: {path}") from exc
    if isinstance(folds, bool) or not isinstance(folds, int) or (folds != 0 and folds < 2):
        raise ValueError(f"Frozen protocol has an invalid cv_folds value: {folds!r}")
    return folds


def load_cleaned(config: DatasetConfig, root: Path = ROOT) -> dict[str, pd.DataFrame]:
    """Read Noor's cleaned train, validation and test tables (row IDs kept for the check)."""
    folder = root / "data" / "interim" / config.name
    return {s: pd.read_csv(folder / f"{s}_cleaned.csv") for s in ("train", "validation", "test")}


def check_holdouts(tables: dict[str, pd.DataFrame], config: DatasetConfig) -> None:
    """Fail unless validation and test are real rows that are disjoint from training."""
    for split in ("validation", "test"):
        if IS_SYNTHETIC in tables[split].columns:
            raise ValueError(f"{split} contains an {IS_SYNTHETIC} column")
        if not tables[split][config.target].isin([0, 1]).all():
            raise ValueError(f"{split} target is not binary")
    ids = {s: set(t[ID_COLUMN]) for s, t in tables.items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if ids[a] & ids[b]:
            raise ValueError(f"{a} and {b} share source rows")


def _encode(pre: Preprocessor, df: pd.DataFrame, target: str):
    encoded = pre.transform_cleaned(df.reset_index(drop=True))
    return encoded.drop(columns=target), encoded[target]


def run_experiment(
    tables: dict[str, pd.DataFrame],
    config: DatasetConfig,
    generator: str,
    arm_label: str,
    pool: pd.DataFrame | None = None,
    seeds: Sequence[int] = SEEDS,
    fill_levels: Sequence[float] = FILL_LEVELS,
    model_names: Sequence[str] = DEFAULT_MODELS,
    include_real_only: bool = True,
    protocol_dir: Path = ROOT / "results",
) -> pd.DataFrame:
    """Return one row per fill level, seed and model.

    ``generator`` is "smotenc" (regenerated for every seed) or "ctgan" (a fixed ``pool`` of
    synthetic rows whose first n rows are used, so the fill levels are nested).
    """
    if generator not in ("smotenc", "ctgan"):
        raise ValueError("generator must be 'smotenc' or 'ctgan'")
    if generator == "ctgan" and pool is None:
        raise ValueError("ctgan needs a pool of synthetic rows")
    check_holdouts(tables, config)
    target = config.target
    real = {s: t.drop(columns=ID_COLUMN) for s, t in tables.items()}
    discrete = [c for c in config.discrete if c in real["train"].columns]

    pre = Preprocessor(config).fit(real["train"])  # real training rows only
    X_train, y_train = _encode(pre, real["train"], target)
    X_val, y_val = _encode(pre, real["validation"], target)
    X_test, y_test = _encode(pre, real["test"], target)
    counts = fill_counts(real["train"][target], fill_levels)

    # Validate every requested protocol before training or generating synthetic rows.
    protocols = {}
    protocol_cv_folds = {}
    for seed in seeds:
        if set(model_names) != set(build_models(seed)) or len(model_names) != len(set(model_names)):
            raise ValueError("Augmented arms must use the complete frozen baseline suite")
        path = _protocol_path(protocol_dir, config.name, seed)
        protocol_cv_folds[seed] = _protocol_cv_folds(path)
        protocols[seed] = load_protocol(
            path,
            seed=seed,
            cv_folds=protocol_cv_folds[seed],
            y_val=y_val,
            y_test=y_test,
        )
    frames = []
    for seed in seeds:
        levels = [(0.0, 0)] if include_real_only else []
        levels += [(f, counts[f]) for f in fill_levels]
        synthetic_pool = None
        if generator == "smotenc":
            synthetic_pool = smotenc_rows(
                real["train"], target, max(counts.values()), discrete_columns=discrete, seed=seed
            )
        elif pool is not None:
            synthetic_pool = pool
        for fill, n in levels:
            if n == 0:
                X, y, arm = X_train, y_train, REAL_ONLY
            else:
                extra = synthetic_pool.head(n)
                if len(extra) < n:
                    raise ValueError(f"pool has {len(extra)} rows, need {n}")
                X_syn, y_syn = _encode(pre, extra, target)
                X = pd.concat([X_train, X_syn], ignore_index=True)
                y = pd.concat([y_train.reset_index(drop=True), y_syn], ignore_index=True)
                arm = arm_label
            print(f"[{arm}] fill={fill:.2f} seed={seed} rows={len(X)}", flush=True)
            table = run_models(
                X,
                y,
                X_val,
                y_val,
                X_test,
                y_test,
                seed=seed,
                cv_folds=protocol_cv_folds[seed],
                model_names=model_names,
                thresholds=protocols[seed]["thresholds"],
            )
            table["protocol_id"] = protocols[seed]["protocol_id"]
            table["dataset"], table["arm"] = config.name, arm
            table["fill_level"], table["n_synthetic"] = fill, n
            frames.append(table)
    return pd.concat(frames, ignore_index=True).reindex(columns=RESULT_COLUMNS)


def save_results(table: pd.DataFrame, path: Path) -> None:
    """Upsert by dataset, arm, fill level and seed, keeping other arms' rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        if "protocol_id" not in old or old.protocol_id.isna().any():
            raise ValueError("Legacy 0.5-threshold results require a separate output file")
        for (dataset, seed), incoming_rows in table.groupby(["dataset", "seed"]):
            existing = old[(old.dataset == dataset) & (old.seed == seed)]
            if set(existing.protocol_id) - set(incoming_rows.protocol_id):
                raise ValueError("Cannot combine different frozen protocols in one comparison")
        incoming = set(map(tuple, table[KEYS].drop_duplicates().to_numpy()))
        keep = [tuple(row) not in incoming for row in old[KEYS].to_numpy()]
        table = pd.concat([old.loc[keep], table], ignore_index=True)
    table.reindex(columns=RESULT_COLUMNS).to_csv(path, index=False)


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over seeds, per arm and fill level.

    ``val_*`` are averaged over the models; ``test_*`` come from each seed's validation winner
    (the only model scored on test), as in the baseline.
    """
    keys = ["arm", "fill_level"]
    ok = table[table.status == "success"]
    val = ok.groupby([*keys, "seed"])[["val_pr_auc", "val_f1"]].mean().reset_index()
    val = val.groupby(keys)[["val_pr_auc", "val_f1"]].agg(["mean", "std"])
    test = ok[ok.selected].groupby(keys)[["test_pr_auc", "test_f1"]].agg(["mean", "std"])
    out = val.join(test)
    out.columns = ["_".join(c) for c in out.columns]
    return out.reset_index()


def noor_baseline(
    dataset: str,
    path: Path = BASELINES_FILE,
    protocol_dir: Path = ROOT / "results",
) -> dict | None:
    """Noor's seed-42 test PR-AUC and F1 for the validation winner of the cleaned arm."""
    if not Path(path).exists():
        return None
    table = pd.read_csv(path)
    protocol_path = _protocol_path(protocol_dir, dataset, 42)
    if not protocol_path.is_file():
        return None
    protocol = json.loads(protocol_path.read_text())
    row = table[
        (table.dataset == dataset)
        & (table.arm == "cleaned")
        & (table.seed == 42)
        & (table.protocol_id == protocol["protocol_id"])
        & table.selected
    ]
    if row.empty:
        return None
    return {"test_pr_auc": float(row.test_pr_auc.iloc[0]), "test_f1": float(row.test_f1.iloc[0])}


def plot_summary(summary: pd.DataFrame, path: Path, baseline: dict | None = None) -> None:
    """Test PR-AUC and F1 against fill level; each arm starts from the real-only point."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real = summary[summary.arm == REAL_ONLY].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = [
        ("test_pr_auc", "Test PR-AUC (higher is better)"),
        ("test_f1", "Test F1 at frozen validation thresholds"),
    ]
    for ax, (metric, title) in zip(axes, panels, strict=True):
        for arm in [a for a in summary.arm.unique() if a != REAL_ONLY]:
            rows = summary[summary.arm == arm].sort_values("fill_level")
            x = [0.0, *(rows.fill_level * 100)]
            y = [real[f"{metric}_mean"], *rows[f"{metric}_mean"]]
            err = [real[f"{metric}_std"], *rows[f"{metric}_std"]]
            ax.errorbar(
                x, y, yerr=err, marker="o", lw=1.8, capsize=3, label=arm, color=ARM_COLORS.get(arm)
            )
        if baseline is not None:
            ax.axhline(
                baseline[metric], color="#555555", ls=":", lw=1.2, label="Noor baseline (seed 42)"
            )
        ax.set_xticks([0, 10, 25, 50])
        ax.set_xlabel("Fill level (% of class gap added as synthetic subscribers)")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", color="#d0d0d0", lw=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Bank Marketing: real-only (0%) vs real + synthetic rows; mean ± SD over seeds", fontsize=11
    )
    note = "Thresholds are frozen from original-arm validation and reused at every fill level."
    fig.text(0.99, 0.005, note, ha="right", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="bank_marketing")
    parser.add_argument("--generator", choices=["smotenc", "ctgan"], required=True)
    parser.add_argument("--arm-label", help="Name of the arm in the results (default: generator)")
    parser.add_argument("--pool", type=Path, help="CTGAN file with the largest fill level")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--no-real-only", action="store_true")
    parser.add_argument("--protocol-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    config = DATASETS[args.dataset]
    pool = None
    if args.generator == "ctgan":
        path = args.pool or ROOT / "data" / "interim" / config.name / "synthetic" / (
            "bank_ctgan_fill50.csv"
        )
        pool = pd.read_csv(path)
    table = run_experiment(
        load_cleaned(config),
        config,
        args.generator,
        args.arm_label or args.generator,
        pool,
        seeds=args.seeds,
        model_names=args.models,
        include_real_only=not args.no_real_only,
        protocol_dir=args.protocol_dir,
    )
    save_results(table, args.out)
    combined = pd.read_csv(args.out)
    summary = summarize(combined)
    print(summary.round(4).to_string(index=False))
    plot_summary(
        summary,
        args.out.with_suffix(".png"),
        noor_baseline(config.name, protocol_dir=args.protocol_dir),
    )


if __name__ == "__main__":
    main()
