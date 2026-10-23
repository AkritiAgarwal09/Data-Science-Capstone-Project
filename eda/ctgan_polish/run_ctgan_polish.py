"""Train CTGAN on the cleaned Polish training split and write minority fill files.

The main run uses the Bank v1 settings: 300 epochs, batch size 500, pac 10,
discriminator steps 1, generator and discriminator widths 256, CPU, seed 42.
Sampling is conditional on bankrupt=1. The 10, 25 and 50 percent files are
nested prefixes of one pool. The cleaned table is used as provided, including
its pooled cohort column.

v2 settings are not in the repo yet. When they are chosen, save them as
eda/ctgan_v2_settings.json and rerun with --settings v2.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_gan.tools.ctgan import sample_synthetic, save_model, train_ctgan
from agentic_gan.utils.ctgan_settings import load_ctgan_settings
from agentic_gan.utils.datasets import DATASETS
from agentic_gan.utils.synthetic import IS_SYNTHETIC, fill_counts

ROOT = Path(__file__).resolve().parents[2]
CONFIG = DATASETS["polish"]
EDA = ROOT / "eda" / "ctgan_polish"
ARTIFACTS = EDA / "artifacts"
TRAIN_FILE = ROOT / "data" / "interim" / CONFIG.name / "train_cleaned.csv"
SYNTHETIC = ROOT / "data" / "interim" / CONFIG.name / "synthetic"
SOURCE_ROW = "source_row"
FILL_LEVELS = (0.10, 0.25, 0.50)
FILL_NAMES = {0.10: "fill10", 0.25: "fill25", 0.50: "fill50"}
EXPECTED_ROWS = {"fill10": 2353, "fill25": 5883, "fill50": 11766}
MINORITY = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", choices=["v1", "v2"], default="v1")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"])
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_training_frame() -> pd.DataFrame:
    """Read the cleaned training split only. Validation and test are not opened."""
    frame = pd.read_csv(TRAIN_FILE)
    target = CONFIG.target
    if target not in frame.columns:
        raise SystemExit(f"cleaned training file has no {target} column")
    if SOURCE_ROW not in frame.columns:
        raise SystemExit("cleaned training file has no source_row column")
    model = frame.drop(columns=[SOURCE_ROW])
    missing = [column for column in model.columns if model[column].isna().any()]
    if missing:
        raise SystemExit(f"model columns still have missing values: {missing}")
    missing_discrete = [
        column for column in CONFIG.discrete if column not in model.columns
    ]
    if missing_discrete:
        raise SystemExit(f"discrete columns missing from the training file: {missing_discrete}")
    counts = model[target].value_counts()
    if counts.idxmin() != MINORITY:
        raise SystemExit(f"expected minority class {MINORITY}, found {counts.idxmin()}")
    return model


def rake_weights(pool: pd.DataFrame, real: pd.DataFrame, columns: list[str], iterations: int = 50):
    """Row weights that match ``real`` category shares on each column, one margin at a time."""
    weights = np.ones(len(pool))
    for _ in range(iterations):
        for column in columns:
            target = real[column].value_counts(normalize=True)
            current = pd.Series(weights, index=pool.index).groupby(
                pool[column], observed=True
            ).sum()
            current = current / current.sum()
            ratio = (target / current).replace([np.inf, -np.inf], np.nan).fillna(1.0)
            weights = weights * pool[column].map(ratio).fillna(1.0).to_numpy()
    total = weights.sum()
    if total <= 0:
        raise SystemExit("raking weights summed to zero")
    return weights / total


def raked_sample(
    pool: pd.DataFrame, reference: pd.DataFrame, n: int, seed: int, columns: list[str]
) -> pd.DataFrame:
    """Pick ``n`` pool rows, without replacement, so the discrete shares match ``reference``."""
    weights = rake_weights(pool, reference, columns)
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(pool), size=n, replace=False, p=weights)
    return pool.iloc[rng.permutation(picked)].reset_index(drop=True)


def _codes(series: pd.Series) -> set:
    values = []
    for value in series.dropna().unique():
        if isinstance(value, float) and value.is_integer():
            values.append(int(value))
        else:
            values.append(value)
    return set(values)


def _is_prefix(smaller: pd.DataFrame, larger: pd.DataFrame) -> bool:
    head = larger.head(len(smaller)).reset_index(drop=True)
    return smaller.reset_index(drop=True).equals(head)


def _copies_training_rows(model: pd.DataFrame, features: pd.DataFrame) -> bool:
    marked = features.copy()
    marked["_synthetic_row"] = True
    hit = marked.merge(model.drop_duplicates(), on=list(model.columns), how="inner")
    return not hit.empty


def validate(
    model: pd.DataFrame,
    pool: pd.DataFrame,
    files: dict[str, pd.DataFrame],
    epochs: int,
    loss_path: Path,
) -> None:
    """Stop before diagnostics if the written files do not match the Bank format."""
    target = CONFIG.target
    expected_columns = [*model.columns, IS_SYNTHETIC]
    problems: list[str] = []
    ordered = [files[FILL_NAMES[level]] for level in FILL_LEVELS]
    if not _is_prefix(ordered[0], ordered[1]):
        problems.append("fill10 is not the first rows of fill25")
    if not _is_prefix(ordered[1], ordered[2]):
        problems.append("fill25 is not the first rows of fill50")
    features = pool.drop(columns=[IS_SYNTHETIC])
    if features.duplicated().any():
        problems.append("the synthetic pool has exact duplicate rows")
    if _copies_training_rows(model, features):
        problems.append("the synthetic pool copies real training rows")
    for column in CONFIG.discrete:
        extra = _codes(pool[column]) - _codes(model[column])
        if extra:
            problems.append(f"{column} has codes outside the training data: {sorted(extra)}")
    losses = pd.read_csv(loss_path)
    if len(losses) != epochs:
        problems.append(f"loss file has {len(losses)} rows, expected {epochs}")
    for name, frame in files.items():
        if len(frame) != EXPECTED_ROWS[name]:
            problems.append(f"{name} has {len(frame)} rows, expected {EXPECTED_ROWS[name]}")
        if list(frame.columns) != expected_columns:
            problems.append(f"{name} columns are {list(frame.columns)}")
        if not (frame[target] == MINORITY).all():
            problems.append(f"{name} has rows whose target is not {MINORITY}")
        if not (frame[IS_SYNTHETIC] == 1).all():
            problems.append(f"{name} has rows whose is_synthetic is not 1")
    if problems:
        detail = "\n- ".join(problems)
        raise SystemExit(f"Polish synthetic files failed the format checks:\n- {detail}")


def main() -> None:
    args = parse_args()
    settings = load_ctgan_settings(
        args.settings, epochs=args.epochs, device=args.device, seed=args.seed
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    SYNTHETIC.mkdir(parents=True, exist_ok=True)
    model = load_training_frame()
    target = CONFIG.target
    counts = fill_counts(model[target], FILL_LEVELS)
    named_counts = {FILL_NAMES[level]: counts[level] for level in FILL_LEVELS}
    if named_counts != EXPECTED_ROWS:
        raise SystemExit(f"fill counts {named_counts} do not match {EXPECTED_ROWS}")
    largest = counts[FILL_LEVELS[-1]]
    print(
        f"training rows {len(model):,} positives {(model[target] == MINORITY).sum():,} "
        f"fill rows { {FILL_NAMES[level]: counts[level] for level in FILL_LEVELS} }",
        flush=True,
    )

    loss_path = ARTIFACTS / "ctgan_losses.csv"
    trained = train_ctgan(
        model,
        target,
        discrete_columns=list(CONFIG.discrete),
        epochs=settings.epochs,
        batch_size=settings.batch_size,
        pac=settings.pac,
        discriminator_steps=settings.discriminator_steps,
        generator_dim=settings.generator_dim,
        discriminator_dim=settings.discriminator_dim,
        device=settings.device,
        seed=settings.seed,
        log_path=loss_path,
    )
    model_path = ARTIFACTS / "ctgan_polish.pkl"
    save_model(trained, model_path)

    sample_n = settings.pool_rows if settings.rake else largest
    if settings.pool_rows is not None and sample_n < largest:
        raise SystemExit(f"pool of {sample_n} rows is smaller than the 50 percent file ({largest})")
    pool = sample_synthetic(
        trained,
        sample_n,
        target_value=MINORITY,
        seed=settings.seed,
        drop_duplicates_of=model,
        conditional=settings.conditional,
        max_rounds=20 if settings.conditional else 100,
    )
    if len(pool) < sample_n:
        raise SystemExit(
            f"sampling returned {len(pool)} minority rows; the pool needs {sample_n}"
        )
    pool = pool.iloc[:sample_n].reset_index(drop=True)
    rake_columns = list(CONFIG.discrete) if settings.rake else []
    if settings.rake:
        reference = model.loc[model[target] == MINORITY]
        pool = raked_sample(pool, reference, largest, settings.seed, rake_columns)
    if settings.conditional:
        sampling = f"conditional on {target}={MINORITY}"
    else:
        sampling = f"unconditional, rows with {target}={MINORITY} kept"
    if settings.rake:
        sampling += ", discrete columns raked to the real minority shares"

    written: dict[str, str] = {}
    loaded: dict[str, pd.DataFrame] = {}
    for level in FILL_LEVELS:
        name = f"polish_ctgan_{FILL_NAMES[level]}.csv"
        path = SYNTHETIC / name
        frame = pool.head(counts[level])
        frame.to_csv(path, index=False)
        written[FILL_NAMES[level]] = str(path.relative_to(ROOT))
        loaded[FILL_NAMES[level]] = pd.read_csv(path)
        print(f"wrote {path.relative_to(ROOT)} ({len(frame):,} rows)", flush=True)

    validate(model, pool, loaded, settings.epochs, loss_path)
    meta = {
        "dataset": "polish",
        "settings_version": settings.version,
        "settings_source": settings.source,
        "input": str(TRAIN_FILE.relative_to(ROOT)),
        "rows": len(model),
        "positive_count": int((model[target] == MINORITY).sum()),
        "negative_count": int((model[target] != MINORITY).sum()),
        "class_gap": int((model[target] != MINORITY).sum() - (model[target] == MINORITY).sum()),
        "target": target,
        "target_value": MINORITY,
        "discrete": list(CONFIG.discrete),
        "epochs": settings.epochs,
        "batch_size": settings.batch_size,
        "pac": settings.pac,
        "discriminator_steps": settings.discriminator_steps,
        "generator_dim": list(settings.generator_dim),
        "discriminator_dim": list(settings.discriminator_dim),
        "device": settings.device,
        "seed": settings.seed,
        "conditional": settings.conditional,
        "rake_columns": rake_columns,
        "sampling": sampling,
        "train_seconds": round(trained.train_seconds, 1),
        "sampled_rows": sample_n,
        "outputs": written,
        "model": str(model_path.relative_to(ROOT)),
        "losses": str(loss_path.relative_to(ROOT)),
        "v2_settings": "not available" if settings.version == "v1" else settings.source,
        "validation_and_test_read": False,
    }
    (ARTIFACTS / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"trained in {trained.train_seconds:.1f}s on {settings.device}", flush=True)
    print("Polish synthetic files passed the format checks.", flush=True)


if __name__ == "__main__":
    main()
