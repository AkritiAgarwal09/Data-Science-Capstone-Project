"""Reproducible cleaning and encoding for the shared tabular datasets.

IEEE-CIS writes cleaned tables and train-sample-fitted sparse model matrices.
"""

import argparse
import hashlib
import io
import json
import pickle
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import arff
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from agentic_gan.utils.datasets import DATASETS, DatasetConfig, load_configs

IEEE_CATEGORY_LIMIT = 50
MISSING_CATEGORY = "__missing__"
OTHER_CATEGORY = "__other__"


def _category_string(value):
    if isinstance(value, (int, float, np.number)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def load_dataset(config: DatasetConfig, path: Path, *, identity_path=None) -> pd.DataFrame:
    """Load cached data without fetching or silently changing source order."""
    if config.name == "ieee_cis":
        path = Path(path)
        identity_path = (
            Path(identity_path)
            if identity_path
            else path.parent / Path(config.merge["file"]).name
        )
        transaction = pd.read_csv(path)
        identity = pd.read_csv(identity_path)
        key = config.id_column
        if config.merge["on"] != key or config.merge["how"] != "left":
            raise ValueError("IEEE-CIS requires a one-to-one left join on its row key")
        for name, frame in (("transaction", transaction), ("identity", identity)):
            if key not in frame or frame[key].isna().any() or not frame[key].is_unique:
                raise ValueError(f"{name} table needs unique, nonmissing {key}")
            if {"has_identity", "source_row"} & set(frame.columns):
                raise ValueError(f"{name} table contains a reserved provenance column")
        if config.target not in transaction or config.target in identity:
            raise ValueError("The observed target must come only from transactions")
        overlap = (set(transaction) & set(identity)) - {key}
        if overlap:
            raise ValueError(f"Overlapping transaction/identity columns: {sorted(overlap)}")
        if not identity[key].isin(transaction[key]).all():
            raise ValueError(
                "Identity keys are absent from the transaction table; check the file pair"
            )
        original_keys = transaction[key].copy()
        data = transaction.merge(
            identity, on=key, how="left", validate="one_to_one", sort=False, indicator=True
        )
        if not data[key].reset_index(drop=True).equals(original_keys.reset_index(drop=True)):
            raise ValueError("The left join changed transaction row order")
        data["has_identity"] = data.pop("_merge").eq("both").astype("int8")
    elif config.name == "polish":

        def unpack(blob, depth=0):
            if depth > 4:
                raise ValueError("Unexpected archive nesting")
            found = {}
            with zipfile.ZipFile(io.BytesIO(blob)) as archive:
                for name in archive.namelist():
                    if name.endswith("/"):
                        continue
                    if name.lower().endswith(".zip"):
                        found.update(unpack(archive.read(name), depth + 1))
                    elif name.lower().endswith(".arff"):
                        key = Path(name).name.lower()
                        if key in found:
                            raise ValueError(f"Duplicate cohort: {key}")
                        found[key] = archive.read(name)
            return found

        files = unpack(path.read_bytes())
        frames = []
        for name in config.cohorts:
            records, _ = arff.loadarff(io.StringIO(files[name].decode("utf-8-sig")))
            frame = pd.DataFrame(records)
            frame.columns = [str(c).lower() for c in frame]
            frame = frame.rename(
                columns={"class": config.target, **{f"attr{i}": f"A{i}" for i in range(1, 65)}}
            )
            required = [f"A{i}" for i in range(1, 65)] + [config.target]
            frame = frame[required].apply(pd.to_numeric, errors="raise")
            frame.insert(0, "cohort", name.removesuffix(".arff"))
            frames.append(frame)
        data = pd.concat(frames, ignore_index=True)
    else:
        data = pd.read_csv(path, sep=None, engine="python")
        if config.name == "default":
            data = data.drop(columns=["ID"], errors="ignore").rename(
                columns={"default payment next month": "y", "Y": "y", "PAY_0": "PAY_1"}
            )
    if not data.columns.is_unique:
        raise ValueError("Duplicate source columns")
    data.index.name = "source_row"
    return data


def normalize(
    data: pd.DataFrame, config: DatasetConfig, include_duration: bool = False
) -> pd.DataFrame:
    """Apply fixed semantic rules; never learn statistics here."""
    data = data.copy()
    if config.name == "ieee_cis":
        data = data.drop(
            columns=[config.id_column, config.split["time_column"], "source_row", "is_synthetic"],
            errors="ignore",
        )
    if config.name == "bank_marketing":
        data = data.rename(columns={"day_of_week": "day_of_month", "day": "day_of_month"})
        if not data.columns.is_unique:
            raise ValueError("Ambiguous day column aliases")
        if not data["day_of_month"].between(1, 31).all():
            raise ValueError("Expected Bank day-of-month values 1..31")
        for c in ("job", "education", "contact", "poutcome"):
            data[c] = data[c].fillna("unknown")
        if not include_duration:
            data = data.drop(columns="duration", errors="ignore")
        data["previously_contacted"] = (data["pdays"] != -1).astype(int)
        if data["pdays"].isna().any():
            raise ValueError("Unexpected missing pdays; expected -1 for no previous contact")
        if config.target in data and not pd.api.types.is_numeric_dtype(data[config.target]):
            data[config.target] = data[config.target].map({"no": 0, "yes": 1})
    if config.target in data:
        if data[config.target].isna().any() or not data[config.target].isin([0, 1]).all():
            raise ValueError("Target must be observed and binary")
        data[config.target] = data[config.target].astype(int)
    return data.replace([np.inf, -np.inf], np.nan)


def split_dataset(
    data: pd.DataFrame, config: DatasetConfig, *, sample=True
) -> dict[str, list[int]]:
    """Recreate stratified splits from config; no separate row-ID files."""
    settings = config.split
    if settings.get("strategy") == "time":
        splits = _time_split(data, config)
        return _sample_train(data, config, splits) if sample else splits
    if settings.get("strategy", "stratified") != "stratified":
        raise ValueError("Unknown split strategy")
    ratios = [settings[s] for s in ("train", "validation", "test")]
    if not all(np.isfinite(v) and 0 < v < 1 for v in ratios) or not np.isclose(
        sum(ratios), 1.0, rtol=0, atol=1e-9
    ):
        raise ValueError("Split ratios must be positive and sum to 1")
    holdout = settings["validation"] + settings["test"]
    train, hold = train_test_split(
        data.index,
        test_size=holdout,
        stratify=data[config.target],
        random_state=settings["seed"],
    )
    validation, test = train_test_split(
        hold,
        test_size=settings["test"] / holdout,
        stratify=data.loc[hold, config.target],
        random_state=settings["seed"],
    )
    return {
        name: ids.tolist()
        for name, ids in zip(
            ("train", "validation", "test"), (train, validation, test), strict=True
        )
    }


def _time_split(data, config):
    if not data.index.is_unique or data.index.hasnans:
        raise ValueError("Source row indices must be unique and nonmissing")
    if data[config.target].isna().any() or not data[config.target].isin([0, 1]).all():
        raise ValueError("Time splitting needs observed binary labels")
    clock = pd.to_numeric(data[config.split["time_column"]], errors="raise")
    cuts = np.asarray(config.split["cut_points"], dtype=float)
    if len(cuts) != 2 or not np.isfinite(cuts).all() or cuts[0] >= cuts[1]:
        raise ValueError("Expected two increasing finite time cutoffs")
    if not np.isfinite(clock).all():
        raise ValueError("Time values must be observed and finite")
    policy = config.duplicates
    if (
        not policy
        or policy["drop"] != "exact"
        or not policy["before_split"]
        or policy["keep"] != "first"
    ):
        raise ValueError("IEEE-CIS requires exact deduplication before splitting, keeping first")
    # Follow #50 exactly: equality excludes TransactionID and includes TransactionDT.
    columns = [c for c in data if c not in {*policy["ignore_columns"], "has_identity"}]
    hashes = pd.util.hash_pandas_object(data[columns], index=False)
    candidates = data.loc[hashes.duplicated(keep=False)]
    duplicates = candidates.index[candidates.duplicated(subset=columns, keep="first")]
    kept = clock.loc[~data.index.isin(duplicates)]
    masks = (kept < cuts[0], (kept >= cuts[0]) & (kept < cuts[1]), kept >= cuts[1])
    splits = {
        part: kept.index[mask].tolist()
        for part, mask in zip(("train", "validation", "test"), masks, strict=True)
    }
    if any(not ids for ids in splits.values()):
        raise ValueError("Fixed time cutoffs must leave all three parts nonempty")
    return splits


def _sample_train(data, config, splits):
    if config.sample is None:
        return splits
    settings = config.sample
    if settings["part"] != "train" or settings["stratify"] != config.target:
        raise ValueError("Sample must be stratified on the target within training only")
    ids = splits["train"]
    count = settings["rows"]
    if isinstance(count, bool) or not isinstance(count, int) or not 0 < count <= len(ids):
        raise ValueError(f"Training sample needs 1..{len(ids)} rows, got {count}")
    if count == len(ids):
        return splits
    chosen, _ = train_test_split(
        ids,
        train_size=count,
        stratify=data.loc[ids, config.target],
        random_state=settings["seed"],
    )
    selected = set(chosen)
    # #50 returns the stratified sample in the original file order.
    return {**splits, "train": [row for row in ids if row in selected]}


def profile_dataframe(data: pd.DataFrame) -> dict:
    """Profile only the supplied training rows."""
    return {
        "rows": len(data),
        "duplicate_rows": int(data.duplicated().sum()),
        "columns": {
            c: {
                "dtype": str(data[c].dtype),
                "missing": int(data[c].isna().sum()),
                "unique": int(data[c].nunique()),
            }
            for c in data
        },
    }


class Preprocessor:
    """Training-fitted cleaning and encoding, reusable after serialization."""

    def __init__(
        self,
        config: DatasetConfig,
        imputation="median",
        include_duration=False,
        from_cleaned=False,
    ):
        if imputation not in ("median", "mean"):
            raise ValueError("imputation must be median or mean")
        self.config = config
        self.imputation = imputation
        self.include_duration = include_duration
        # True when fit() gets an already-cleaned table (IEEE-CIS), which holds the reserved
        # missing/other buckets; they are then left out of the category counts.
        self.from_cleaned = from_cleaned

    def fit(self, raw_train: pd.DataFrame):
        if self.config.name == "ieee_cis":
            return self._fit_ieee(raw_train)
        train = normalize(raw_train, self.config, self.include_duration).drop(
            columns=list(getattr(self, "excluded", ())), errors="ignore"
        )
        self.columns = train.columns.drop(self.config.target).tolist()
        self.discrete = [c for c in self.config.discrete if c in self.columns]
        self.numeric = [c for c in self.columns if c not in self.discrete]
        self.fill_values = {}
        for c in self.columns:
            values = train[c]
            if c in self.discrete:
                modes = values.mode()
                self.fill_values[c] = modes.iloc[0] if len(modes) else "__missing__"
            else:
                values = pd.to_numeric(values, errors="raise")
                value = values.median() if self.imputation == "median" else values.mean()
                self.fill_values[c] = 0.0 if pd.isna(value) else float(value)
        self.training_rows = len(train)
        self.encoder = ColumnTransformer(
            [
                ("numeric", StandardScaler(), self.numeric),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.discrete,
                ),
            ],
            remainder="drop",
        )
        self.encoder.fit(self.clean(raw_train)[self.columns])
        self.feature_names = self.encoder.get_feature_names_out().tolist()
        return self

    def _fit_ieee(self, raw_train):
        if "is_synthetic" in raw_train and not raw_train["is_synthetic"].eq(0).all():
            raise ValueError("Preprocessing must fit on real training rows only")
        train = normalize(raw_train, self.config).drop(
            columns=list(getattr(self, "excluded", ())), errors="ignore"
        )
        if self.config.target not in train or train.empty:
            raise ValueError("Fit requires nonempty labeled training rows")
        features = train.drop(columns=self.config.target)
        self.training_rows = len(train)
        self.missing_rates = features.isna().mean().to_dict()
        maximum = self.config.sample["max_missing"]
        if not 0 <= maximum < 1:
            raise ValueError("max_missing must be in [0, 1)")
        self.dropped_columns = [c for c in features if self.missing_rates[c] > maximum]
        self.value_columns = [c for c in features if c not in self.dropped_columns]
        if not self.value_columns:
            raise ValueError("No usable features remain after the training missingness filter")
        # Preserve informative missingness even when the mostly-empty value is dropped.
        self.indicators = {
            c: f"{c}__missing" for c in features if 0 < self.missing_rates[c] < 1
        }
        if set(self.indicators.values()) & set(features):
            raise ValueError("Missing-indicator names collide with source columns")
        self.columns = [*self.value_columns, *self.indicators.values()]
        self.discrete = [c for c in self.config.discrete if c in self.value_columns]
        self.gan_discrete = [*self.discrete, *self.indicators.values()]
        if "has_identity" in self.value_columns:
            self.gan_discrete.append("has_identity")
        self.numeric = [c for c in self.columns if c not in self.discrete]
        self.fill_values, self.category_levels = {}, {}
        for c in self.value_columns:
            if c in self.discrete:
                values = features[c].dropna().map(_category_string)
                reserved = values.isin([MISSING_CATEGORY, OTHER_CATEGORY])
                if reserved.any() and not self.from_cleaned:
                    raise ValueError(f"Reserved categorical bucket appears in source column {c}")
                counts = values[~reserved].value_counts()
                self.category_levels[c] = sorted(counts.index, key=lambda v: (-counts[v], v))[
                    :IEEE_CATEGORY_LIMIT
                ]
                self.fill_values[c] = MISSING_CATEGORY
            else:
                values = pd.to_numeric(features[c], errors="raise")
                statistic = values.median() if self.imputation == "median" else values.mean()
                self.fill_values[c] = float(statistic)
        self.fill_values.update({column: 0.0 for column in self.indicators.values()})
        categories = [
            [*self.category_levels[c], MISSING_CATEGORY, OTHER_CATEGORY] for c in self.discrete
        ]
        self.encoder = ColumnTransformer(
            [
                ("numeric", StandardScaler(), self.numeric),
                (
                    "categorical",
                    OneHotEncoder(
                        categories=categories,
                        handle_unknown="ignore",
                        sparse_output=True,
                    ),
                    self.discrete,
                ),
            ],
            remainder="drop",
            sparse_threshold=1.0,
        )
        self.encoder.fit(self.clean(raw_train)[self.columns])
        self.feature_names = self.encoder.get_feature_names_out().tolist()
        return self

    def _fill(self, data):
        missing = set(self.columns) - set(data.columns)
        if missing:
            raise ValueError(f"Missing features: {sorted(missing)}")
        values = {}
        for c in self.columns:
            if c in self.discrete:
                if self.config.name == "ieee_cis":
                    column = data[c].astype(object).where(data[c].notna(), MISSING_CATEGORY)
                    column = column.map(_category_string)
                    allowed = {*self.category_levels[c], MISSING_CATEGORY, OTHER_CATEGORY}
                    values[c] = column.where(column.isin(allowed), OTHER_CATEGORY)
                    continue
                column = data[c].fillna(self.fill_values[c])
                # Stable strings for numeric category codes, including synthetic float codes.
                values[c] = column.map(
                    lambda v: (
                        str(int(v))
                        if isinstance(v, (int, float, np.number)) and float(v).is_integer()
                        else str(v)
                    )
                )
            else:
                values[c] = (
                    pd.to_numeric(data[c], errors="raise")
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(self.fill_values[c])
                )
        if self.config.target in data:
            values[self.config.target] = data[self.config.target]
        return pd.DataFrame(values, index=data.index)

    def clean(self, raw: pd.DataFrame) -> pd.DataFrame:
        data = normalize(raw, self.config, self.include_duration)
        if self.config.name == "ieee_cis":
            missing = set(self.value_columns) - set(data)
            if missing:
                raise ValueError(f"Missing features: {sorted(missing)}")
            flags = pd.DataFrame(
                {
                    flag: data[source].isna().astype("int8")
                    for source, flag in self.indicators.items()
                },
                index=data.index,
            )
            data = pd.concat([data, flags], axis=1)
        return self._fill(data)

    def transform_features(self, cleaned):
        """Encode feature rows; keep IEEE-CIS sparse to avoid a dense multi-gigabyte table."""
        cleaned = self._fill(cleaned)
        values = self.encoder.transform(cleaned[self.columns])
        if self.config.name == "ieee_cis":
            result = sparse.csr_matrix(values, dtype=np.float32)
            if not np.isfinite(result.data).all():
                raise ValueError("Encoded features must be finite")
            return result
        return values

    def transform_cleaned(self, cleaned: pd.DataFrame) -> pd.DataFrame:
        """Encode real or synthetic CLEANED rows, without refitting or scaling labels/IDs."""
        cleaned = self._fill(cleaned)
        values = self.transform_features(cleaned)
        result = pd.DataFrame(
            values.toarray() if sparse.issparse(values) else values,
            columns=self.feature_names,
            index=cleaned.index,
        )
        if self.config.target in cleaned:
            if not cleaned[self.config.target].isin([0, 1]).all():
                raise ValueError("Synthetic target must be binary and nonmissing")
            result[self.config.target] = cleaned[self.config.target].astype(int)
        return result

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        return self.transform_cleaned(self.clean(raw))


def write_treatment_log(raw_train, processor, path):
    if processor.config.name == "ieee_cis":
        return _write_ieee_treatments(raw_train, processor, path)
    normalized = normalize(raw_train, processor.config, processor.include_duration)
    rows = []
    for c in dict.fromkeys([*raw_train.columns, *normalized.columns]):
        issue, action, reason = "No special issue", "Retain", "Preserve source information"
        new = "day_of_month" if c in ("day", "day_of_week") else c
        if new != c:
            issue, action, reason = "Mislabeled day", "Rename to day_of_month", "Values are 1..31"
        elif c == "duration" and not processor.include_duration:
            issue, action, reason = "Post-call leakage", "Drop", "Pre-call prediction"
        elif c == "duration":
            issue, action, reason = "Post-call feature", "Retain by switch", "Post-call diagnostic"
        elif c in ("job", "education", "contact", "poutcome"):
            issue, action, reason = (
                "Unknown category",
                "Preserve/fill unknown",
                "Informative category",
            )
        elif c == "pdays":
            issue, action, reason = (
                "-1 sentinel",
                "Keep -1; add previously_contacted",
                "No prior contact",
            )
        elif c == "previously_contacted":
            issue, action, reason = (
                "Derived feature",
                "1 if pdays != -1",
                "Separate contact history",
            )
        elif processor.config.name == "default" and c in processor.discrete:
            issue = "Category codes, including undocumented values"
            action = "Keep each code as a separate category"
            reason = "EDA flags EDUCATION 0/5/6, MARRIAGE 0, PAY -2/0; avoid invented semantics"
        if new in processor.fill_values:
            count = int(normalized[new].isna().sum())
            if count:
                issue += f"; {count} training gaps"
                action += f"; fill {processor.fill_values[new]!r}"
                method = processor.imputation if new in processor.numeric else "mode"
                reason += f"; training-only {method}"
                if normalized[new].isna().all():
                    reason += "; all-missing fallback"
            action += "; one-hot encode" if new in processor.discrete else "; standard scale"
        rows.append({"column": c, "issue": issue, "treatment": action, "reason": reason})
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_ieee_treatments(raw_train, processor, path):
    config = processor.config
    excluded = {config.id_column, config.split["time_column"], "source_row", "is_synthetic"}
    flag_sources = {flag: source for source, flag in processor.indicators.items()}
    rows = []
    for column in dict.fromkeys([*raw_train.columns, *processor.columns]):
        rate = processor.missing_rates.get(flag_sources.get(column, column))
        fill = processor.fill_values.get(column)
        if column == config.target:
            treatment = "Keep binary target; never encode, scale, or impute"
            reason = "Observed outcome label"
        elif column in excluded:
            treatment = "Exclude from features"
            reason = "Join/row key or split clock; prevent ID/time shortcut"
        elif column in flag_sources:
            treatment = "Add binary missingness flag; scale for models; mark discrete for GAN"
            reason = f"Training gaps vary for {flag_sources[column]}; preserve before imputation"
        elif column in processor.dropped_columns:
            treatment = "Drop value column"
            if column in processor.indicators:
                treatment += f"; retain {processor.indicators[column]}"
            reason = (
                "Value missingness exceeds the training-only "
                f"{config.sample['max_missing']:.0%} limit"
            )
        elif column == "has_identity":
            treatment = "Keep binary join-presence flag; scale for models; mark discrete for GAN"
            reason = "Identity-table presence is informative even when sparse fields are dropped"
        elif column in processor.discrete:
            treatment = (
                f"Keep up to {IEEE_CATEGORY_LIMIT} training categories plus missing/other buckets; "
                "one-hot encode"
            )
            reason = "Bound feature width; category vocabulary learned from train_sample only"
        else:
            treatment = f"Fill training {processor.imputation}; standard scale for models"
            reason = "Retain finite numeric values; fill and scaling learned from train_sample only"
        if column in processor.indicators and column not in flag_sources:
            treatment += f"; add {processor.indicators[column]}"
        rows.append(
            {
                "column": column,
                "training_missing_rate": rate,
                "fill_value": fill,
                "treatment": treatment,
                "reason": reason,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_counts(raw, full_splits, splits, config):
    result = {}
    parts = {**full_splits, "train_sample": splits["train"]}
    for part, ids in parts.items():
        subset = raw.loc[ids]
        times = pd.to_numeric(subset[config.split["time_column"]], errors="raise")
        result[part] = {
            "rows": len(subset),
            "fraud_rows": int(subset[config.target].sum()),
            "transaction_dt_from": int(times.min()),
            "transaction_dt_to": int(times.max()),
        }
    return result


def _verify_reference_counts(counts, path):
    if not path.exists():
        return False
    expected = pd.read_csv(path).set_index("part")
    if set(expected.index) != set(counts):
        raise ValueError("IEEE-CIS reference split parts differ")
    for part, observed in counts.items():
        for name, value in observed.items():
            if value != expected.loc[part, name]:
                raise ValueError(
                    f"IEEE-CIS split mismatch: {part} {name} is {value}, "
                    f"expected {expected.loc[part, name]}"
                )
    return True


def _save_ieee(raw, config, root, source, identity_path, imputation):
    full_splits = split_dataset(raw, config, sample=False)
    splits = _sample_train(raw, config, full_splits)
    counts = _split_counts(raw, full_splits, splits, config)
    reference_checked = _verify_reference_counts(
        counts, root / "splits" / "ieee_cis_time_split.csv"
    )
    train = raw.loc[splits["train"]]
    processor = Preprocessor(config, imputation).fit(train)
    output = root / "data" / "interim" / config.name
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").unlink(missing_ok=True)

    encoded_files = []
    for part, ids in splits.items():
        blocks = []
        metadata = []
        for offset in range(0, len(ids), 10_000):
            batch = raw.loc[ids[offset : offset + 10_000]]
            cleaned = processor.clean(batch)
            cleaned.to_csv(
                output / f"{part}_cleaned.csv",
                index_label="source_row",
                mode="w" if offset == 0 else "a",
                header=offset == 0,
            )
            blocks.append(processor.transform_features(cleaned))
            metadata.append(
                pd.DataFrame(
                    {"source_row": cleaned.index, config.target: cleaned[config.target].to_numpy()}
                )
            )
        encoded_path = output / f"{part}_encoded.npz"
        sparse.save_npz(encoded_path, sparse.vstack(blocks, format="csr"))
        metadata_path = output / f"{part}_encoded_rows.csv"
        pd.concat(metadata, ignore_index=True).to_csv(metadata_path, index=False)
        encoded_files.extend((encoded_path.name, metadata_path.name))

    with (output / "preprocessor.pkl").open("wb") as stream:
        pickle.dump(processor, stream, protocol=pickle.HIGHEST_PROTOCOL)
    scaler = processor.encoder.named_transformers_["numeric"]
    one_hot = processor.encoder.named_transformers_["categorical"]
    settings = {
        "format_version": 1,
        "fit_part": "train_sample",
        "training_rows": processor.training_rows,
        "missingness_limit": config.sample["max_missing"],
        "category_limit": IEEE_CATEGORY_LIMIT,
        "category_levels": processor.category_levels,
        "missing_category": MISSING_CATEGORY,
        "other_category": OTHER_CATEGORY,
        "fill_values": processor.fill_values,
        "missing_rates": processor.missing_rates,
        "dropped_columns": processor.dropped_columns,
        "missing_indicators": processor.indicators,
        "cleaned_features": processor.columns,
        "discrete_columns": processor.discrete,
        "gan_discrete_columns": processor.gan_discrete,
        "numeric_columns": processor.numeric,
        "feature_names": processor.feature_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "one_hot_categories": {
            column: levels.tolist()
            for column, levels in zip(processor.discrete, one_hot.categories_, strict=True)
        },
        "encoded_format": (
            "SciPy CSR float32; matching *_encoded_rows.csv gives source_row and target"
        ),
    }
    settings_path = output / "encoder_settings.json"
    settings_path.write_text(json.dumps(settings, indent=2, allow_nan=False) + "\n")
    treatment_path = output / "treatment_log.csv"
    write_treatment_log(train, processor, treatment_path)
    manifest = {
        "dataset": config.name,
        "config": asdict(config),
        "imputation": imputation,
        "source_sha256": {
            "transaction": _file_hash(source),
            "identity": _file_hash(identity_path),
        },
        "split_counts": counts,
        "reference_counts_verified": reference_checked,
        "duplicates_dropped": len(raw) - sum(len(ids) for ids in full_splits.values()),
        "training_rows": len(train),
        "cleaned_feature_count": len(processor.columns),
        "encoded_feature_count": len(processor.feature_names),
        "output_sha256": {
            name: _file_hash(output / name)
            for name in (
                "preprocessor.pkl",
                settings_path.name,
                treatment_path.name,
                *[f"{part}_cleaned.csv" for part in splits],
                *encoded_files,
            )
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return output


def load_preprocessed(output):
    """Load local cleaned IEEE-CIS splits and the fitted train-sample encoder."""
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    for name, digest in manifest["output_sha256"].items():
        if _file_hash(output / name) != digest:
            raise ValueError(f"Preprocessing artifact changed: {name}")
    with (output / "preprocessor.pkl").open("rb") as stream:
        processor = pickle.load(stream)
    tables = {
        part: processor._fill(pd.read_csv(output / f"{part}_cleaned.csv", index_col="source_row"))
        for part in ("train", "validation", "test")
    }
    return tables, processor


def load_encoded(output, part):
    """Load one sparse encoded split as (features, target, source_row) in matching row order."""
    if part not in ("train", "validation", "test"):
        raise ValueError("part must be train, validation, or test")
    output = Path(output)
    matrix = sparse.load_npz(output / f"{part}_encoded.npz")
    rows = pd.read_csv(output / f"{part}_encoded_rows.csv")
    if matrix.shape[0] != len(rows) or rows["source_row"].duplicated().any():
        raise ValueError("Encoded matrix and row metadata are not aligned")
    return matrix, rows["isFraud"].to_numpy(dtype=int), rows["source_row"].to_numpy()


def run_preprocessing(
    dataset,
    root=None,
    input_path=None,
    imputation=None,
    include_duration=None,
    config_path=None,
    identity_path=None,
):
    root = Path(root) if root else Path(__file__).resolve().parents[3]
    config = load_configs(config_path)[dataset]
    imputation = config.imputation if imputation is None else imputation
    include_duration = config.include_duration if include_duration is None else include_duration
    source = Path(input_path) if input_path else root / config.file
    if config.name == "ieee_cis":
        if identity_path:
            identity = Path(identity_path)
        elif input_path:
            identity = source.parent / Path(config.merge["file"]).name
        else:
            identity = root / config.merge["file"]
        raw = load_dataset(config, source, identity_path=identity)
        return _save_ieee(raw, config, root, source, identity, imputation)
    if identity_path:
        raise ValueError("identity_path applies only to IEEE-CIS")
    raw = load_dataset(config, source)
    normalized = normalize(raw, config, include_duration)
    splits = split_dataset(normalized, config)
    train = raw.loc[splits["train"]]
    processor = Preprocessor(config, imputation, include_duration).fit(train)
    output = root / "data" / "interim" / dataset
    output.mkdir(parents=True, exist_ok=True)
    for split, ids in splits.items():
        cleaned = processor.clean(raw.loc[ids])
        cleaned.to_csv(output / f"{split}_cleaned.csv", index_label="source_row")
    # Remove only artifacts generated by older versions of this pipeline.
    legacy_files = [
        "preprocessor.pkl",
        "profile.json",
        "treatment_log.csv",
        "manifest.json",
        "default_code_audit.csv",
        "eda_default_code_audit.csv",
        *[f"{split}_{kind}.csv" for split in splits for kind in ("encoded", "missing_mask")],
    ]
    for name in legacy_files:
        (output / name).unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--input", dest="input_path", type=Path)
    parser.add_argument("--identity-input", dest="identity_path", type=Path)
    parser.add_argument("--imputation", choices=["mean", "median"])
    parser.add_argument("--include-duration", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--config", dest="config_path", type=Path)
    args = vars(parser.parse_args())
    from agentic_gan.tools.preprocessing import run_preprocessing as run

    print(run(**args))


if __name__ == "__main__":
    main()


class TreatmentPreprocessor(Preprocessor):
    """Apply validated per-column repairs before the existing training-only encoder.

    Forests use numeric predictors only, never labels or excluded columns. A training
    median/mode remains the deterministic fallback when no predictive fit is possible.
    """

    def __init__(self, config, treatments, excluded=()):
        super().__init__(config, config.imputation, config.include_duration)
        self.treatments = dict(treatments)
        self.excluded = tuple(excluded)

    def _source(self, raw):
        return normalize(raw, self.config, self.include_duration).drop(
            columns=list(self.excluded), errors="ignore"
        )

    def fit(self, raw_train):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline

        source = self._source(raw_train)
        self.repairs = {}
        self.models = {}
        self.repair_flags = {}
        for column, action in self.treatments.items():
            if column not in source or action not in ("median", "mode", "model_missing"):
                continue
            values = source[column]
            categorical = column in self.config.discrete
            if categorical:
                mode = values.mode()
                fill = mode.iloc[0] if len(mode) else MISSING_CATEGORY
            else:
                fill = values.median()
                fill = 0.0 if pd.isna(fill) else float(fill)
            self.repairs[column] = fill
            if action != "model_missing":
                continue
            flag = f"{column}__missing"
            if flag in source:
                raise ValueError(f"Missing-indicator name collision: {flag}")
            # IEEE already creates its own flags from the original missingness.
            if self.config.name != "ieee_cis":
                self.repair_flags[column] = flag
            predictors = [
                c for c in source.select_dtypes(include="number")
                if c not in (column, self.config.target) and source[c].notna().any()
            ]
            observed = values.notna()
            if predictors and observed.sum() >= 2:
                estimator = RandomForestClassifier if categorical else RandomForestRegressor
                model = make_pipeline(
                    SimpleImputer(strategy="median", keep_empty_features=True),
                    estimator(n_estimators=32, max_depth=8, random_state=42, n_jobs=1),
                )
                labels = values.loc[observed].astype(str) if categorical else values.loc[observed]
                model.fit(source.loc[observed, predictors], labels)
                self.models[column] = (predictors, model)
        # Base fit must see original IEEE gaps so its 20% rule and flags stay intact.
        self._fitting = True
        try:
            return super().fit(self._prepare(raw_train, repair=self.config.name != "ieee_cis"))
        finally:
            self._fitting = False

    def _prepare(self, raw, *, repair=True):
        data = raw.copy()
        source = self._source(raw)
        for column, flag in self.repair_flags.items():
            data[flag] = source[column].isna().astype("int8")
        if repair:
            for column, fill in self.repairs.items():
                missing = source[column].isna()
                values = (
                    source[column].astype(object)
                    if column in self.config.discrete else source[column]
                )
                values = values.copy()
                if column in self.models and missing.any():
                    predictors, model = self.models[column]
                    values.loc[missing] = model.predict(source.loc[missing, predictors])
                else:
                    values = values.fillna(fill)
                data[column] = values
        return data

    def clean(self, raw):
        if getattr(self, "_fitting", False):
            # Base fit passes prepared rows for encoder fitting.
            if self.config.name != "ieee_cis":
                return Preprocessor.clean(self, raw)
        if self.config.name == "ieee_cis":
            cleaned = Preprocessor.clean(self, raw)
            repaired = self._prepare(raw)
            for column in self.repairs:
                if column in self.value_columns and column not in self.discrete:
                    cleaned[column] = repaired[column]
            return cleaned
        return Preprocessor.clean(self, self._prepare(raw))
