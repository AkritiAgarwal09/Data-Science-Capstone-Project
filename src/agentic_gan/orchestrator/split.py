"""Step S0: check the dataset against the config, then split it into train, validation and test.

The split is made once, before any cleaning (architecture.md section 2). It follows the recipe
already used for the EDA row-ID files and the baselines: two stratified steps with a fixed seed,
so every script and the orchestrator put the same rows in each part.

The shared dataset loader handles the four configured source formats.
"""

from dataclasses import replace

import pandas as pd
from sklearn.model_selection import train_test_split

from agentic_gan.orchestrator.manifest import RunDir
from agentic_gan.schemas import RunConfig, SplitOutput, Status
from agentic_gan.tools.preprocessing import load_dataset, split_dataset
from agentic_gan.utils.datasets import DATASETS

# Each class needs a row in all three parts, so three is the least that can be split.
MIN_CLASS_ROWS = 3
# Below this many positives a part gives noisy metrics (one missed case moves recall by 1 point).
WARN_BELOW_POSITIVES = 100


class SchemaError(ValueError):
    """The dataset does not match the config. The message lists every problem found."""


def validate_schema(config: RunConfig, df: pd.DataFrame) -> None:
    """Raise SchemaError unless the data can be split as the config describes."""
    problems: list[str] = []
    if len(df) == 0:
        raise SchemaError("the dataset has no rows")

    duplicated = df.columns[df.columns.duplicated()].unique().tolist()
    if duplicated:
        problems.append(f"duplicate column names: {duplicated}")
    empty = df.columns[df.isna().all()].tolist() if config.dataset_name != "ieee_cis" else []
    if empty:
        problems.append(f"columns with no values: {empty}")

    excluded = config.excluded_features
    if config.target in excluded:
        problems.append("the target cannot be an excluded feature")
    unknown = [c for c in excluded if c not in df.columns]
    if unknown:
        problems.append(f"excluded features not in the data: {unknown}")
    features = [c for c in df.columns if c != config.target and c not in excluded]
    if not features:
        problems.append("no feature columns are left besides the target")

    if config.target not in df.columns:
        problems.append(f"target column {config.target!r} not found")
    else:
        problems += _target_problems(config, df[config.target])

    if problems:
        raise SchemaError("; ".join(problems))


def _target_problems(config: RunConfig, target: pd.Series) -> list[str]:
    problems = []
    if target.isna().any():
        problems.append(f"target has {int(target.isna().sum())} missing values")
    values = target.dropna().unique().tolist()
    if len(values) != 2:
        problems.append(f"target must have exactly 2 values, found {len(values)}")
    if config.positive_class not in values:
        problems.append(f"positive class {config.positive_class!r} not among {values[:5]}")
    else:
        positives = int((target == config.positive_class).sum())
        negatives = int(target.notna().sum()) - positives
        if min(positives, negatives) < MIN_CLASS_ROWS:
            problems.append(
                f"each class needs at least {MIN_CLASS_ROWS} rows, found {positives} positive "
                f"and {negatives} negative"
            )
    return problems


def split_row_ids(config: RunConfig, df: pd.DataFrame) -> dict[str, pd.Index]:
    """Row labels for train, val and test: two stratified steps with config.split_seed.

    The first step holds out validation plus test. The second divides the hold-out between them.
    The data is not touched, so this can be run on any dataset to list the rows in each part.
    """
    is_positive = df[config.target] == config.positive_class
    ratios = config.split_ratios
    holdout = ratios.val + ratios.test
    train, hold = train_test_split(
        df.index, test_size=holdout, stratify=is_positive, random_state=config.split_seed
    )
    val, test = train_test_split(
        hold,
        test_size=ratios.test / holdout,
        stratify=is_positive.loc[hold],
        random_state=config.split_seed,
    )
    return {"train": train, "val": val, "test": test}


def split_data(config: RunConfig, run: RunDir) -> SplitOutput:
    """Validate the dataset, split it, and write train.csv, val.csv and test.csv to data/.

    Rows keep every source column and a source_row provenance column. Nothing is written
    if the dataset fails validation.
    """
    source = DATASETS.get(config.dataset_name)
    if source and source.split.get("strategy") == "time":
        source = replace(source, sample={**source.sample, "seed": config.split_seed})
    df = load_dataset(source, config.dataset_path) if source else pd.read_csv(config.dataset_path)
    validate_schema(config, df)
    if source and source.split.get("strategy") == "time":
        ids = split_dataset(df, source)
        ids = {"train": ids["train"], "val": ids["validation"], "test": ids["test"]}
    else:
        ids = split_row_ids(config, df)

    paths, warnings = {}, []
    for part, index in ids.items():
        paths[part] = run.subdir("data") / f"{part}.csv"
        df.loc[index].to_csv(paths[part], index_label="source_row")
        positives = int((df.loc[index, config.target] == config.positive_class).sum())
        if positives < WARN_BELOW_POSITIVES:
            warnings.append(f"{part} has only {positives} positives, so its metrics will be noisy")

    train_positive = df.loc[ids["train"], config.target] == config.positive_class
    return SplitOutput(
        status=Status.SUCCESS,
        warnings=warnings,
        train_path=paths["train"],
        val_path=paths["val"],
        test_path=paths["test"],
        target=config.target,
        seed=config.split_seed,
        train_positives=int(train_positive.sum()),
        train_negatives=int((~train_positive).sum()),
    )


def split_for_training(config: RunConfig, run: RunDir) -> SplitOutput:
    """Dataset-aware split with stable source IDs and the frozen baseline recipe.

    Retains the source features, including leakage columns, for auditing. The next
    stage excludes them. IEEE uses the configured time cuts and training sample.
    """
    from dataclasses import replace

    from agentic_gan.tools.preprocessing import load_dataset, normalize, split_dataset
    from agentic_gan.utils.datasets import load_configs

    dataset = load_configs()[config.dataset_name]
    if config.target != dataset.target:
        raise SchemaError("Run target differs from the dataset configuration")
    dataset = replace(
        dataset,
        split={
            **dataset.split,
            "seed": config.split_seed,
            "train": config.split_ratios.train,
            "validation": config.split_ratios.val,
            "test": config.split_ratios.test,
        },
    )
    data = load_dataset(dataset, config.dataset_path)
    if "source_row" in data or "is_synthetic" in data:
        raise SchemaError("Source data contains reserved provenance columns")
    normalized = normalize(data, dataset, dataset.include_duration)
    if normalized[config.target].nunique() != 2:
        raise SchemaError("Source target must contain both classes")
    ids = split_dataset(data if dataset.name == "ieee_cis" else normalized, dataset)
    paths, warnings = {}, []
    for part, rows in ids.items():
        name = "val" if part == "validation" else part
        path = run.subdir("data") / f"{name}.csv"
        data.loc[rows].to_csv(path, index_label="source_row")
        paths[f"{name}_path"] = path
        positives = int(normalized.loc[rows, config.target].sum())
        if positives < WARN_BELOW_POSITIVES:
            warnings.append(f"{name} has only {positives} positives, so its metrics will be noisy")
    positives = int(normalized.loc[ids["train"], config.target].sum())
    return SplitOutput(
        status=Status.SUCCESS,
        warnings=warnings,
        **paths,
        target=config.target,
        seed=config.split_seed,
        train_positives=positives,
        train_negatives=len(ids["train"]) - positives,
    )
