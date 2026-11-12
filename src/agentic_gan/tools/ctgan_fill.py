"""Train CTGAN on one dataset's cleaned training split and write nested minority fill files.

This is the shared body of ``eda/ctgan_default/run_ctgan_default.py`` and
``eda/ctgan_polish/run_ctgan_polish.py``. It trains on the training split only, writes the
10, 25 and 50 percent fills as prefixes of one pool, and records in ``run_meta.json``:

* the generator the written selection rules would have chosen for this table, every rule
  step, and an ``override`` line saying why CTGAN was trained anyway (rules G-1, G-5, G-9);
* the share of draws that fell outside the training range before clipping (rule PB-7);
* the dataset's documented consistency rules that were applied while sampling.

``horizon`` restricts Polish to one cohort file, which is what rule PB-1 requires. The
imputation in the cleaned table was still fitted on the pooled training split; the meta
says so.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_gan.agents.model_selection import measure_training_frame
from agentic_gan.schemas import fill_rows
from agentic_gan.tools.ctgan import sample_synthetic, save_model, train_ctgan
from agentic_gan.tools.quality_fidelity import RULES
from agentic_gan.tools.selection import SelectionResult, select_generator
from agentic_gan.utils.ctgan_settings import CTGANRunSettings
from agentic_gan.utils.datasets import DATASETS
from agentic_gan.utils.synthetic import IS_SYNTHETIC

ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROW = "source_row"
HORIZON_LABEL = "cohort"
FILL_LEVELS = (0.10, 0.25, 0.50)
FILL_NAMES = {0.10: "fill10", 0.25: "fill25", 0.50: "fill50"}
MINORITY = 1
RANGE_FLAG = 0.01  # a column is listed in diagnostics when over 1% of draws left the range

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FillPaths:
    """Where one run reads and writes. ``horizon`` and ``version`` pick the suffixes."""

    dataset: str
    version: str
    horizon: str | None
    train_file: Path
    synthetic_dir: Path
    artifacts: Path
    file_stem: str
    model_name: str

    def fill_file(self, name: str) -> Path:
        return self.synthetic_dir / f"{self.file_stem}_{name}.csv"


def fill_paths(dataset: str, version: str, horizon: str | None = None) -> FillPaths:
    if dataset not in DATASETS:
        raise SystemExit(f"unknown dataset {dataset!r}")
    if version not in {"v1", "v2"}:
        raise SystemExit("version must be v1 or v2")
    suffix = "_v2" if version == "v2" else ""
    folder = f"artifacts{suffix}"
    name = dataset if horizon is None else f"{dataset}_{horizon}"
    stem = f"{name}_ctgan{suffix}"
    model = f"ctgan_{name}{suffix}.pkl"
    if horizon is not None:
        folder = f"{folder}_{horizon}"
    return FillPaths(
        dataset=dataset,
        version=version,
        horizon=horizon,
        train_file=ROOT / "data" / "interim" / dataset / "train_cleaned.csv",
        synthetic_dir=ROOT / "data" / "interim" / dataset / "synthetic",
        artifacts=ROOT / "eda" / f"ctgan_{dataset}" / folder,
        file_stem=stem,
        model_name=model,
    )


# --- data -------------------------------------------------------------------------------------


def load_training_frame(paths: FillPaths) -> pd.DataFrame:
    """Read the cleaned training split only. Validation and test are not opened."""
    config = DATASETS[paths.dataset]
    frame = pd.read_csv(paths.train_file)
    target = config.target
    if target not in frame.columns:
        raise SystemExit(f"cleaned training file has no {target} column")
    if SOURCE_ROW not in frame.columns:
        raise SystemExit("cleaned training file has no source_row column")
    model = frame.drop(columns=[SOURCE_ROW])
    if paths.horizon is not None:
        if HORIZON_LABEL not in model.columns:
            raise SystemExit(f"{paths.dataset} has no {HORIZON_LABEL} column to pin a horizon")
        model = model.loc[model[HORIZON_LABEL] == paths.horizon].reset_index(drop=True)
        if model.empty:
            raise SystemExit(f"no training rows have {HORIZON_LABEL} == {paths.horizon!r}")
    missing = [column for column in model.columns if model[column].isna().any()]
    if missing:
        raise SystemExit(f"model columns still have missing values: {missing}")
    missing_discrete = [column for column in config.discrete if column not in model.columns]
    if missing_discrete:
        raise SystemExit(f"discrete columns missing from the training file: {missing_discrete}")
    counts = model[target].value_counts()
    if counts.idxmin() != MINORITY:
        raise SystemExit(f"expected minority class {MINORITY}, found {counts.idxmin()}")
    return model


def fill_sizes(model: pd.DataFrame, target: str) -> dict[float, int]:
    """Rows per fill level from ``fill_rows``, the one copy of the class-gap formula."""
    positives = int((model[target] == MINORITY).sum())
    negatives = int((model[target] != MINORITY).sum())
    return {level: fill_rows(level, positives, negatives) for level in FILL_LEVELS}


# --- raking -----------------------------------------------------------------------------------


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


# --- rules and selection ------------------------------------------------------------------------


def rule_mask(dataset: str):
    """A row filter: True where a row breaks none of the documented rules for ``dataset``."""
    rules = RULES.get(dataset, ())

    def mask(frame: pd.DataFrame) -> pd.Series:
        broken = pd.Series(False, index=frame.index)
        for rule in rules:
            broken |= rule.violated(frame).fillna(False)
        return ~broken

    return mask


def selection_record(model: pd.DataFrame, dataset: str, target: str) -> dict:
    """Run the written selection rules on this table and say why CTGAN was trained anyway."""
    described = measure_training_frame(model, dataset, target, MINORITY)
    result = select_generator(described)
    return {
        "status": result.status,
        "generator": result.generator,
        "trainable": result.trainable,
        "fallback": result.fallback,
        "comparator": result.comparator,
        "scope": result.scope,
        "rationale": result.rationale,
        "trained": "ctgan",
        "override": _override(result),
        "steps": [
            {
                "rule": step.rule,
                "outcome": step.outcome,
                "rationale": step.rationale,
                "inputs": _jsonable(step.inputs),
            }
            for step in result.steps
        ],
    }


def _override(result: SelectionResult) -> str:
    if result.status == "skipped":
        return (
            "CTGAN was trained on a table the selection rule refused: "
            f"{result.rationale} This is an EDA experiment, not an agent run."
        )
    if result.generator == "ctgan" and result.trainable:
        return "none: the rule selected CTGAN."
    if result.generator == "timegan":
        return (
            "CTGAN was trained as the documented fallback (rule DC-4, CTGAN on the flattened "
            "wide table) because TimeGAN is not implemented. The rule's choice is TimeGAN for "
            f"{result.scope or 'the temporal block'}."
        )
    if result.generator == "wgan_gp":
        return (
            "CTGAN was trained as the EDA comparator (rule PB-3) alongside WGAN-GP, the "
            "rule's choice for an all-numeric table."
        )
    return f"CTGAN was trained although the rule selected {result.generator}."


def _relative(path: Path) -> str:
    """Repo-relative when the file is inside the repo, absolute otherwise (tests use tmp dirs)."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


# --- format checks ------------------------------------------------------------------------------


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
    dataset: str,
    model: pd.DataFrame,
    pool: pd.DataFrame,
    files: dict[str, pd.DataFrame],
    expected_rows: dict[str, int],
    epochs: int,
    loss_path: Path,
) -> None:
    """Stop before diagnostics if the written files do not match the Bank format."""
    config = DATASETS[dataset]
    target = config.target
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
    for column in config.discrete:
        extra = _codes(pool[column]) - _codes(model[column])
        if extra:
            problems.append(f"{column} has codes outside the training data: {sorted(extra)}")
    losses = pd.read_csv(loss_path)
    if len(losses) != epochs:
        problems.append(f"loss file has {len(losses)} rows, expected {epochs}")
    for name, frame in files.items():
        if len(frame) != expected_rows[name]:
            problems.append(f"{name} has {len(frame)} rows, expected {expected_rows[name]}")
        if list(frame.columns) != expected_columns:
            problems.append(f"{name} columns are {list(frame.columns)}")
        if not (frame[target] == MINORITY).all():
            problems.append(f"{name} has rows whose target is not {MINORITY}")
        if not (frame[IS_SYNTHETIC] == 1).all():
            problems.append(f"{name} has rows whose is_synthetic is not 1")
    if problems:
        detail = "\n- ".join(problems)
        raise SystemExit(f"{dataset} synthetic files failed the format checks:\n- {detail}")


# --- the run ------------------------------------------------------------------------------------


def run_fill(paths: FillPaths, settings: CTGANRunSettings) -> dict:
    """Train, sample, write the fills, check them, and write ``run_meta.json``."""
    config = DATASETS[paths.dataset]
    target = config.target
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    paths.synthetic_dir.mkdir(parents=True, exist_ok=True)
    model = load_training_frame(paths)
    counts = fill_sizes(model, target)
    named_counts = {FILL_NAMES[level]: counts[level] for level in FILL_LEVELS}
    if any(count < 1 for count in counts.values()):
        raise SystemExit(f"a fill level rounds to no rows: {named_counts}")
    largest = counts[FILL_LEVELS[-1]]
    selection = selection_record(model, paths.dataset, target)
    print(
        f"training rows {len(model):,} positives {(model[target] == MINORITY).sum():,} "
        f"fill rows {named_counts}",
        flush=True,
    )
    print(f"selection rule: {selection['rationale']}", flush=True)
    print(f"override: {selection['override']}", flush=True)

    loss_path = paths.artifacts / "ctgan_losses.csv"
    trained = train_ctgan(
        model,
        target,
        discrete_columns=list(config.discrete),
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
    model_path = paths.artifacts / paths.model_name
    save_model(trained, model_path)

    sample_n = settings.pool_rows if settings.rake else largest
    if settings.pool_rows is not None and sample_n < largest:
        raise SystemExit(f"pool of {sample_n} rows is smaller than the 50 percent file ({largest})")
    rules = RULES.get(paths.dataset, ())
    mask = rule_mask(paths.dataset)
    if rules:
        failed = int((~mask(model)).sum())
        if failed:
            raise SystemExit(f"{failed} real training rows break a documented consistency rule")
    range_report: dict[str, dict[str, float]] = {}
    pool = sample_synthetic(
        trained,
        sample_n,
        target_value=MINORITY,
        seed=settings.seed,
        drop_duplicates_of=model,
        conditional=settings.conditional,
        row_filter=mask if rules else None,
        max_rounds=20 if settings.conditional else 100,
        range_report=range_report,
    )
    if len(pool) < sample_n:
        raise SystemExit(f"sampling returned {len(pool)} minority rows; the pool needs {sample_n}")
    pool = pool.iloc[:sample_n].reset_index(drop=True)
    rake_columns = list(config.discrete) if settings.rake else []
    if settings.rake:
        reference = model.loc[model[target] == MINORITY]
        pool = raked_sample(pool, reference, largest, settings.seed, rake_columns)
    sampling = (
        f"conditional on {target}={MINORITY}"
        if settings.conditional
        else f"unconditional, rows with {target}={MINORITY} kept"
    )
    if settings.rake:
        sampling += ", discrete columns raked to the real minority shares"
    sampling += (
        f", {len(rules)} documented consistency rules applied"
        if rules
        else ", documented consistency rules applied: none"
    )

    written: dict[str, str] = {}
    loaded: dict[str, pd.DataFrame] = {}
    for level in FILL_LEVELS:
        path = paths.fill_file(FILL_NAMES[level])
        frame = pool.head(counts[level])
        frame.to_csv(path, index=False)
        written[FILL_NAMES[level]] = _relative(path)
        loaded[FILL_NAMES[level]] = pd.read_csv(path)
        print(f"wrote {_relative(path)} ({len(frame):,} rows)", flush=True)

    validate(paths.dataset, model, pool, loaded, named_counts, settings.epochs, loss_path)
    meta = {
        "dataset": paths.dataset,
        "horizon": paths.horizon,
        "imputation_fit_on": (
            "pooled training split; this run only filtered rows afterwards"
            if paths.horizon is not None
            else "training split"
        ),
        "settings_version": settings.version,
        "settings_source": settings.source,
        "input": _relative(paths.train_file),
        "rows": len(model),
        "positive_count": int((model[target] == MINORITY).sum()),
        "negative_count": int((model[target] != MINORITY).sum()),
        "class_gap": int((model[target] != MINORITY).sum() - (model[target] == MINORITY).sum()),
        "target": target,
        "target_value": MINORITY,
        "discrete": list(config.discrete),
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
        "rules_applied": [rule.name for rule in rules],
        "sampling": sampling,
        "train_seconds": round(trained.train_seconds, 1),
        "sampled_rows": sample_n,
        "outputs": written,
        "model": _relative(model_path),
        "losses": _relative(loss_path),
        "recipe": settings.source,
        "validation_and_test_read": False,
        "selection": selection,
        "out_of_range_before_clip": range_report,
    }
    (paths.artifacts / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"trained in {trained.train_seconds:.1f}s on {settings.device}", flush=True)
    print(f"{paths.dataset} synthetic files passed the format checks.", flush=True)
    return meta


# --- diagnostics text shared by the diagnose scripts -------------------------------------------


def selection_lines(meta: dict) -> list[str]:
    """Lines for diagnostics.txt: the rule's choice and why CTGAN ran instead."""
    selection = meta.get("selection")
    if not selection:
        return ["selection rule: not recorded for this run"]
    fired = [
        step for step in selection["steps"]
        if step["outcome"] not in {"NOT_REACHED", "PASS", "EXCLUDED"}
        or step["rule"] in {"temporal_test", "type_routing"}
    ]
    lines = [
        f"selection rule: {selection['status']}, generator {selection['generator']}, "
        f"trainable {selection['trainable']}, fallback {selection['fallback']}",
        f"  {selection['rationale']}",
        f"  override: {selection['override']}",
    ]
    if fired:
        lines.append("  deciding steps: " + "; ".join(
            f"{step['rule']} {step['outcome']}" for step in fired
        ))
    return lines


def range_lines(meta: dict, flag: float = RANGE_FLAG) -> list[str]:
    """Lines for diagnostics.txt: columns whose draws left the training range before clipping."""
    report = meta.get("out_of_range_before_clip")
    if not report:
        return [
            "Values outside the training range before clipping",
            "  not recorded for this run (sampled before the range report existed)",
        ]
    rows = next(iter(report.values()))["rows"]
    lines = [
        "Values outside the training range before clipping",
        f"  share of {rows:,} on-target draws per column; the written files are clipped, "
        "so this is the only place the overshoot is visible (rule PB-7)",
    ]
    if meta.get("out_of_range_source"):
        lines.append(f"  source: {meta['out_of_range_source']}")
    ranked = sorted(report.items(), key=lambda item: -item[1]["outside"])
    flagged = [(name, stats) for name, stats in ranked if stats["outside"] >= flag]
    if not flagged:
        best = ranked[0]
        lines.append(
            f"  no column had {flag:.0%} or more of its draws outside the range "
            f"(largest {best[0]} at {best[1]['outside']:.2%})"
        )
        return lines
    for name, stats in flagged:
        lines.append(
            f"  {name}: {stats['outside']:.1%} outside "
            f"(below {stats['below']:.1%}, above {stats['above']:.1%})"
        )
    return lines
