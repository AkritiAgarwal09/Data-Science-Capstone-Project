"""Load the shared, editable dataset configuration."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    file: str
    target: str
    discrete: tuple[str, ...]
    split: dict
    imputation: str = "median"
    include_duration: bool = False
    audit_file: str | None = None
    cohorts: tuple[str, ...] = ()
    # Optional settings for a dataset that is more than one flat file. None for the others.
    # `split` may also carry strategy "time" with time_column and cut_points.
    id_column: str | None = None  # row key, never a feature
    merge: dict | None = None  # second table joined onto `file`: file, on, how
    duplicates: dict | None = None  # drop, ignore_columns, keep, before_split
    sample: dict | None = None  # part, rows, stratify, seed, max_missing


def load_configs(path=None):
    path = Path(path) if path else Path(__file__).with_name("datasets.json")
    settings = json.loads(path.read_text())
    return {
        name: DatasetConfig(
            name=name,
            **{
                **values,
                "discrete": tuple(values["discrete"]),
                "cohorts": tuple(values.get("cohorts", [])),
            },
        )
        for name, values in settings.items()
    }


DATASETS = load_configs()
