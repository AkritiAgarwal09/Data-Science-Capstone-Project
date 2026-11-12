"""Export Bank Marketing deciles from a saved run of the training agent.

uv run python -m agentic_gan.tools.bank_deciles runs/<run_id> --seed 42
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from agentic_gan.agents.training import model_artifact_stem
from agentic_gan.orchestrator.manifest import RunDir
from agentic_gan.schemas import Arm, Status
from agentic_gan.tools.deciles import decile_table
from agentic_gan.tools.preprocessing import load_dataset
from agentic_gan.utils.datasets import DATASETS
from agentic_gan.utils.synthetic import IS_SYNTHETIC

DATASET = "bank_marketing"
SPLIT_LABELS = {"train": "Train (in-sample)", "validation": "Validation", "test": "Test"}
FEATURE_COLUMNS = ("age", "balance", "job", "marital", "education", "housing", "poutcome")
OUTPUT_STEM = "decile_bank_marketing"


def _real_rows(path):
    rows = pd.read_csv(path, float_precision="round_trip")
    ids = rows.source_row
    if (
        rows.empty
        or not pd.api.types.is_integer_dtype(ids)
        or ids.duplicated().any()
        or ids.lt(0).any()
    ):
        raise ValueError(f"{path} must contain real rows with unique nonnegative integer IDs")
    if IS_SYNTHETIC in rows and not rows[IS_SYNTHETIC].eq(0).all():
        raise ValueError(f"{path} contains synthetic rows or invalid provenance")
    return rows.set_index("source_row").sort_index()


def _cleaned_split(run, split):
    folder = run.subdir("data")
    path = folder / f"baseline_{split}.csv"
    if split == "validation" and not path.is_file():
        path = folder / "baseline_val.csv"
    return _real_rows(path)


def _selected_model(models, seed):
    metrics = pd.read_csv(models / "metrics.csv", float_precision="round_trip")
    selected = metrics[
        metrics.dataset.eq(DATASET)
        & metrics.arm.eq("cleaned")
        & metrics.fill_level.eq(0)
        & metrics.split.eq("validation")
        & metrics.status.eq(Status.SUCCESS.value)
        & metrics.selected.eq(True)
    ]
    if seed is not None:
        selected = selected[selected.seed.eq(seed)]
    if len(selected) != 1:
        raise ValueError(
            "Expected one selected cleaned Bank model. Specify --seed for multi-seed runs."
        )
    return selected.iloc[0]


def _train_scores(run, config, rows, artifact):
    source = rows
    if (run.subdir("profile") / "treatment_plan.json").is_file():
        # This estimator cleans real inputs inside its fitted encoder. Score exactly the
        # retained cleaned training IDs, using source values as the encoder expects.
        raw_split = run.subdir("data") / "train.csv"
        raw = (
            _real_rows(raw_split)
            if raw_split.is_file()
            else load_dataset(DATASETS[DATASET], config.dataset_path)
        )
        source = raw.loc[rows.index]
    X = source.assign(**{IS_SYNTHETIC: 0})[artifact["features"]]
    probability = artifact["estimator"].predict_proba(X)[:, 1]
    return probability, probability >= artifact["threshold"]


def _holdout_scores(path, rows, target, threshold):
    saved = _real_rows(path)
    if not saved.index.equals(rows.index) or not np.array_equal(saved.y_true, rows[target]):
        raise ValueError(f"{path} differs from the cleaned split's row IDs or labels")
    if not saved.threshold.eq(threshold).all():
        raise ValueError(f"{path} disagrees with the model's frozen threshold")
    expected = saved.probability.ge(threshold).astype(int)
    if not saved.prediction.eq(expected).all():
        raise ValueError(f"{path} has classifications inconsistent with the frozen threshold")
    return saved.probability, saved.prediction


def _write_chart(table, split, upper, path):
    fig = Figure(figsize=(7.4, 4.3), layout="constrained")
    ax = fig.subplots()
    ax.plot(table.decile, table.actual_rate, "o-", color="#12686A", label="Actual response")
    ax.plot(
        table.decile,
        table.mean_predicted_probability,
        "s--",
        color="#A34B18",
        label="Predicted response",
    )
    ax.set(
        title=f"Bank Marketing · {SPLIT_LABELS[split]}",
        xlabel="Decile (1 = highest predicted probability)",
        ylabel="Subscription rate",
        xticks=table.decile,
        ylim=(0, upper),
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=160)


def _write_notes(tables, run, model, path):
    lines = [
        "# Bank Marketing deciles",
        "",
        f"{model.model} was selected by validation PR-AUC for the cleaned arm at seed "
        f"{model.seed}. The frozen classification threshold was {model.threshold:.6f}. "
        f"Source run `{run.run_id}` used protocol `{model.protocol_id}`.",
        "",
    ]
    for split, table in tables.items():
        top = table.iloc[:3]
        contact_share = top.n.sum() / table.n.sum()
        subscriber_share = top.positives.sum() / table.positives.sum()
        lines.append(
            f"- **{SPLIT_LABELS[split]}**. Contacting the highest-scored {contact_share:.0%} "
            f"of customers would reach {subscriber_share:.1%} of subscribers in this split."
        )
    lines += ["", "## Held-out response-rate checks", ""]
    for split in ("validation", "test"):
        table = tables[split].set_index("decile")
        increases = table.index[table.actual_rate.diff().gt(0)]
        for decile in increases:
            before, after = table.actual_rate.loc[[decile - 1, decile]]
            lines.append(
                f"- {SPLIT_LABELS[split]} rose from {before:.2%} in decile {decile - 1} "
                f"to {after:.2%} in decile {decile}."
            )
        if increases.empty:
            lines.append(f"- {SPLIT_LABELS[split]} had no increases between consecutive deciles.")
    lines += [
        "",
        "Customers with equal scores are ranked by source-row ID. The train split is in-sample.",
        "",
        "| Split | Customers | Subscribers | Table | Chart |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for split, table in tables.items():
        stem = f"{OUTPUT_STEM}_{split}"
        lines.append(
            f"| {SPLIT_LABELS[split]} | {table.n.sum():,} | {table.positives.sum():,} "
            f"| [CSV]({stem}.csv) | [Chart]({stem}.png) |"
        )
    path.write_text("\n".join(lines) + "\n")


def export_bank_deciles(run_dir: Path, output_dir: Path, *, seed: int | None = None) -> None:
    """Write three tables, three charts and the summary from a trusted Bank run folder."""
    run = RunDir(run_dir)
    config = run.load_config()
    if config.dataset_name != DATASET:
        raise ValueError("Expected a Bank Marketing run")
    models = run.subdir("models")
    model = _selected_model(models, seed)
    stem = model_artifact_stem(Arm.BASELINE, model.fill_level, model.seed, model.model)
    artifact = joblib.load(models / f"{stem}.joblib")
    threshold = artifact["threshold"]
    if threshold != model.threshold or artifact["protocol_id"] != model.protocol_id:
        raise ValueError("The model artifact disagrees with the selected metrics row")

    tables = {}
    for split in SPLIT_LABELS:
        rows = _cleaned_split(run, split)
        if split == "train":
            if len(rows) != model.training_rows:
                raise ValueError("The cleaned train split does not match the model's training size")
            probability, prediction = _train_scores(run, config, rows, artifact)
        else:
            probability, prediction = _holdout_scores(
                models / f"{stem}_{split}.csv", rows, config.target, threshold
            )
        tables[split] = decile_table(
            probability, rows[config.target], prediction, rows, FEATURE_COLUMNS
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    rate_columns = ["actual_rate", "mean_predicted_probability"]
    highest = max(table[rate_columns].max().max() for table in tables.values())
    upper = min(1, max(0.1, np.ceil(highest * 10) / 10 + 0.05))
    for split, table in tables.items():
        prefix = output_dir / f"{OUTPUT_STEM}_{split}"
        table.to_csv(prefix.with_suffix(".csv"), index=False)
        _write_chart(table, split, upper, prefix.with_suffix(".png"))
    _write_notes(tables, run, model, output_dir / f"{OUTPUT_STEM}_notes.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--seed", type=int, help="Select the training agent's winner for this seed")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    export_bank_deciles(args.run_dir, args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
