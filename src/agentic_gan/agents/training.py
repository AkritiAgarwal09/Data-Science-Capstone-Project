"""Train the frozen Sprint 2 suite and return audited, typed artifacts.

The file-based agent consumes unencoded raw/cleaned splits with source_row IDs.
Only training rows may be augmented. Thresholds are never tuned by this agent.
"""

import hashlib
import json
import re
import warnings
from dataclasses import asdict, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from threadpoolctl import threadpool_limits

from agentic_gan.orchestrator.decisions import log_decision
from agentic_gan.orchestrator.manifest import RunDir
from agentic_gan.schemas import (
    Arm,
    ModelRecord,
    Status,
    TrainingOutput,
    TrainingRequest,
    TreatmentPlan,
    fill_rows,
)
from agentic_gan.tools import baselines as baselines
from agentic_gan.tools.baselines import load_protocol, run_models
from agentic_gan.tools.preprocessing import (
    Preprocessor,
    TreatmentPreprocessor,
    load_dataset,
    normalize,
)
from agentic_gan.tools.smotenc import smotenc_rows
from agentic_gan.utils.datasets import load_configs
from agentic_gan.utils.synthetic import IS_SYNTHETIC

__all__ = ["run_models", "run_training", "run_training_arms"]

ROOT = Path(__file__).resolve().parents[3]
PARTS = ("train", "validation", "test")
METRICS = ("pr_auc", "f1", "precision", "recall", "roc_auc", "accuracy")
ARM_LABELS = {Arm.RAW: "original", Arm.BASELINE: "cleaned", Arm.SMOTE: "smotenc", Arm.GAN: "ctgan"}


class ArmEncoder(TransformerMixin, BaseEstimator):
    """Cloneable encoder that excludes synthetic rows even inside stacking folds."""

    def __init__(self, config, original=False, plan=None):
        self.config = config
        self.original = original
        self.plan = plan

    def fit(self, X, y):
        real = X[IS_SYNTHETIC].eq(0).to_numpy()
        if not real.any():
            raise ValueError("Encoder fit needs real training rows")
        data = X.drop(columns=IS_SYNTHETIC).loc[real].copy()
        self.fit_rows_ = len(data)
        if self.original:
            self.encoder_ = baselines._minimal_transformer(
                data,
                self.config.discrete,
                self.config.imputation,
                sparse_output=self.config.name == "ieee_cis",
            ).fit(data)
        elif self.plan is not None:
            # Synthetic rows may contribute extra cleaned columns to the union schema.
            columns = [c for c in self.plan.source_columns if c in data]
            data = data[columns].copy()
            data[self.config.target] = np.asarray(y)[real]
            self.encoder_ = TreatmentPreprocessor(
                self.config, self.plan.treatments, self.plan.excluded
            ).fit(data)
        else:
            data[self.config.target] = np.asarray(y)[real]
            self.encoder_ = Preprocessor(
                self.config, self.config.imputation, self.config.include_duration
            ).fit(data)
        return self

    def transform(self, X):
        data = X.drop(columns=IS_SYNTHETIC)
        if self.original:
            return self.encoder_.transform(data)
        if self.plan is not None:
            blocks, positions = [], []
            for synthetic in (False, True):
                mask = X[IS_SYNTHETIC].eq(int(synthetic)).to_numpy()
                if not mask.any():
                    continue
                frame = data.iloc[np.flatnonzero(mask)]
                if not synthetic:
                    frame = frame[[c for c in self.plan.source_columns if c in frame]]
                    frame = self.encoder_.clean(frame)
                blocks.append(self.encoder_.transform_features(frame))
                positions.extend(np.flatnonzero(mask))
            combined = sparse.vstack(blocks) if sparse.issparse(blocks[0]) else np.vstack(blocks)
            return combined[np.argsort(positions)]
        return self.encoder_.transform_features(data)


def _read_real(files, config, run):
    tables = {}
    for part, field in zip(PARTS, ("train_path", "val_path", "test_path"), strict=True):
        data = pd.read_csv(run.root / getattr(files, field))
        if IS_SYNTHETIC in data:
            if not data[IS_SYNTHETIC].eq(0).all():
                raise ValueError(f"{part} contains synthetic rows or invalid provenance")
            data = data.drop(columns=IS_SYNTHETIC)
        if "source_row" not in data:
            raise ValueError(f"{part} needs source_row IDs to verify the frozen split")
        ids = data.pop("source_row")
        if ids.isna().any() or ids.duplicated().any() or not pd.api.types.is_integer_dtype(ids):
            raise ValueError(f"{part} has invalid or duplicate source_row IDs")
        if (ids < 0).any():
            raise ValueError("Real source_row IDs must be nonnegative")
        data.index = pd.Index(ids, name="source_row")
        if config.target not in data or data.empty:
            raise ValueError(f"{part} must be nonempty and include the target")
        # Mapping labels is fixed semantics, not a fitted transformation.
        data[config.target] = normalize(data, config, config.include_duration)[config.target]
        if set(data[config.target]) != {0, 1}:
            raise ValueError(f"{part} must contain both binary classes")
        tables[part] = data
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if not tables[a].index.intersection(tables[b].index).empty:
            raise ValueError(f"{a} and {b} share source rows")
    return tables


def _same_table(actual, expected, label):
    try:
        pd.testing.assert_frame_equal(
            actual, expected, check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-12
        )
    except AssertionError as exc:
        raise ValueError(f"{label} differs from the frozen source/training-only cleaning") from exc


def _protocol_cv_folds(protocol_path):
    """Read the fold setting, then let load_protocol verify its checksum and drift."""
    try:
        folds = json.loads(Path(protocol_path).read_text()).get("cv_folds")
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read the frozen protocol's CV setting: {protocol_path}") from exc
    if isinstance(folds, bool) or not isinstance(folds, int) or (folds != 0 and folds < 2):
        raise ValueError(f"Frozen protocol has an invalid cv_folds value: {folds!r}")
    return folds


def _find_training_protocol(cfg, protocol_path=None):
    if protocol_path or cfg.training_protocol_path:
        return Path(protocol_path or cfg.training_protocol_path)
    results = ROOT / "results"
    canonical = results / f"baseline_protocol_{cfg.dataset_name}_{cfg.split_seed}_cv3.json"
    if canonical.is_file():
        return canonical
    candidates = sorted(
        results.glob(f"baseline_protocol_{cfg.dataset_name}_{cfg.split_seed}_cv*.json")
    )
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            "Multiple frozen baseline protocols exist; set training_protocol_path to the "
            "protocol for this experiment"
        )
    return canonical


def _load_plan(request, run):
    if request.treatment_plan_path is None:
        if request.treatment_plan_sha256 is not None:
            raise ValueError("Treatment plan checksum requires a plan path")
        return None
    path = run.root / request.treatment_plan_path
    run.relative(path)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != request.treatment_plan_sha256:
        raise ValueError("Treatment plan checksum mismatch")
    plan = TreatmentPlan.model_validate_json(payload)
    current = asdict(load_configs()[request.config.dataset_name])
    if json.dumps(plan.dataset_config, sort_keys=True) != json.dumps(current, sort_keys=True):
        raise ValueError("Treatment plan dataset rules changed")
    excluded = sorted({"source_row", "is_synthetic", *request.config.excluded_features})
    if plan.excluded != excluded or request.config.target in plan.treatments:
        raise ValueError("Treatment plan exclusions/target differ from the run")
    if set(plan.treatments) - set(plan.source_columns):
        raise ValueError("Treatment plan contains unknown columns")
    log_decision(
        "training",
        "treatment_plan",
        {"sha256": request.treatment_plan_sha256},
        "VALIDATED",
        "Checked recipe integrity and dataset rules; refit on training rows only",
        run=run,
    )
    return plan


def _raw_features(data, config, plan):
    if plan is None:
        return data
    excluded = set(plan.excluded)
    if config.name == "bank_marketing" and not config.include_duration:
        excluded.add("duration")
    if config.name == "ieee_cis":
        excluded.update((config.id_column, config.split["time_column"]))
    return data.drop(columns=list(excluded), errors="ignore")


def _cleaner(config, plan, train):
    if plan is None:
        return Preprocessor(config, config.imputation, config.include_duration).fit(train)
    return TreatmentPreprocessor(config, plan.treatments, plan.excluded).fit(train)


def _inputs(request, run, protocol_path):
    cfg = request.config
    plan = _load_plan(request, run)
    if cfg.calibrate or cfg.excluded_features:
        raise ValueError("Calibration or new feature exclusions require a new frozen protocol")
    config = load_configs()[cfg.dataset_name]
    config = replace(config, split={**config.split, "seed": cfg.split_seed})
    expected_positive = "yes" if cfg.dataset_name == "bank_marketing" else 1
    if cfg.positive_class not in (expected_positive, 1):
        raise ValueError("positive_class differs from the frozen target mapping")
    if cfg.target != config.target or cfg.split_ratios.model_dump() != {
        "train": config.split["train"],
        "val": config.split["validation"],
        "test": config.split["test"],
    }:
        raise ValueError("Run target/split ratios differ from the frozen dataset configuration")
    raw = _read_real(request.raw, config, run)
    cleaned = _read_real(request.baseline, config, run)
    metadata = {
        "source_sha256": baselines.source_hash_for(config, cfg.dataset_path),
        "split_hashes": {part: baselines._digest(t.index.tolist()) for part, t in raw.items()},
        "config": asdict(config),
    }
    # The protocol's seed fixes the SPLIT and reference thresholds. Model seeds vary below.
    protocol = load_protocol(
        protocol_path,
        seed=cfg.split_seed,
        cv_folds=_protocol_cv_folds(protocol_path),
        metadata=metadata,
        y_val=raw["validation"][cfg.target],
        y_test=raw["test"][cfg.target],
    )
    if protocol["dataset"] != cfg.dataset_name:
        raise ValueError("Frozen protocol belongs to a different dataset")
    if set(cfg.candidate_models) != set(protocol["models"]):
        raise ValueError("candidate_models must contain exactly the frozen four-model suite")
    source = load_dataset(config, cfg.dataset_path)
    source[cfg.target] = normalize(source, config, config.include_duration)[cfg.target]
    for part, data in raw.items():
        if not data.index.isin(source.index).all():
            raise ValueError(f"{part} has source_row IDs outside the source data")
        _same_table(data, _raw_features(source.loc[data.index], config, plan), f"raw {part}")
    # Restore audited source columns for duplicate detection and fold-local fitting.
    raw = {part: source.loc[data.index].copy() for part, data in raw.items()}
    fitting = raw["train"].drop_duplicates() if plan is not None else raw["train"]
    cleaner = _cleaner(config, plan, fitting)
    for part in PARTS:
        expected = cleaner.clean(fitting if part == "train" else raw[part])
        # CSV inference can turn string category codes back into numeric values.
        actual = cleaned[part].copy()
        for column in cleaner.discrete:
            actual[column] = actual[column].map(
                lambda v: (
                    str(int(v)) if isinstance(v, (int, float)) and float(v).is_integer() else str(v)
                )
            )
        _same_table(actual, expected, f"cleaned {part}")
        cleaned[part] = actual
    return config, raw, cleaned, protocol, plan


def _synthetic(data, real, target, count):
    if len(data) != count:
        raise ValueError(f"Synthetic row count {len(data)} does not match fill level ({count})")
    if set(data.columns) != {*real.columns, IS_SYNTHETIC}:
        raise ValueError("Synthetic schema must match cleaned features/target plus is_synthetic")
    if not data[IS_SYNTHETIC].eq(1).all() or not data[target].eq(1).all():
        raise ValueError("Synthetic rows must be flagged and belong to positive class 1")
    if data.isna().any().any():
        raise ValueError("Synthetic data must be cleaned and nonmissing")
    data = data.drop(columns=IS_SYNTHETIC)[real.columns].copy()
    for col in real.drop(columns=target):
        if not pd.api.types.is_numeric_dtype(real[col]):
            data[col] = data[col].map(
                lambda v: (
                    str(int(v)) if isinstance(v, (int, float)) and float(v).is_integer() else str(v)
                )
            )
        elif not np.isfinite(pd.to_numeric(data[col], errors="raise")).all():
            raise ValueError(f"Synthetic column {col} has nonfinite values")
    if data.isna().any().any():
        raise ValueError("Synthetic data must be cleaned and nonmissing")
    return data


def _model_data(table, config, *, original=False, synthetic=None):
    data = table.copy()
    data[IS_SYNTHETIC] = 0
    if synthetic is not None:
        data = pd.concat([data, synthetic.assign(**{IS_SYNTHETIC: 1})], ignore_index=True)
    if original and config.name == "bank_marketing" and not config.include_duration:
        data = data.drop(columns="duration", errors="ignore")
    if config.name == "ieee_cis":
        data = data.drop(columns=[config.id_column, config.split["time_column"]], errors="ignore")
    return data.drop(columns=config.target), data[config.target]


def _arm_sets(config, raw, cleaned, levels, accepted, seed):
    yield Arm.RAW, None, raw, None, None
    yield Arm.BASELINE, None, cleaned, None, None
    train = cleaned["train"]
    positives = int(train[config.target].sum())
    counts = {lv: fill_rows(lv, positives, len(train) - positives) for lv in levels}
    maximum = max(counts.values())
    pool, error = None, None
    if maximum:
        try:
            pool = smotenc_rows(
                train.reset_index(drop=True),
                config.target,
                maximum,
                discrete_columns=[
                    c
                    for c in train
                    if c in config.discrete or c.endswith(("__missing", "_indicator"))
                ],
                seed=seed,
                k_neighbors=min(5, positives - 1),
            )
        except Exception as exc:
            error = f"SMOTENC generation failed: {type(exc).__name__}: {exc}"
    for level, count in counts.items():
        if not count:
            yield Arm.SMOTE, level, cleaned, None, "SKIPPED: fill level adds zero rows"
        elif error:
            yield Arm.SMOTE, level, cleaned, None, error
        else:
            yield (
                Arm.SMOTE,
                level,
                cleaned,
                _synthetic(pool.head(count), train, config.target, count),
                None,
            )
    for level, extra in accepted.items():
        yield Arm.GAN, level, cleaned, extra, None


def _prediction_table(table, target, probabilities, threshold):
    return pd.DataFrame(
        {
            "source_row": table.index,
            "y_true": table[target].to_numpy(),
            "probability": probabilities,
            "prediction": (probabilities >= threshold).astype(int),
            "threshold": threshold,
            IS_SYNTHETIC: 0,
        }
    )


def _fit_record(
    arm,
    level,
    tables,
    extra,
    error,
    name,
    estimator,
    seed,
    config,
    protocol,
    folder,
    plan=None,
    raw=None,
):
    fields = dict(
        arm=arm,
        fill_level=level,
        model=name,
        seed=seed,
        threshold=protocol["thresholds"][name],
        protocol_id=protocol["protocol_id"],
        hyperparameters=json.loads(json.dumps(estimator.get_params(deep=True), default=str)),
        training_rows=len(tables["train"]) + (0 if extra is None else len(extra)),
        synthetic_rows=0 if extra is None else len(extra),
    )
    if error:
        status = Status.SKIPPED if error.startswith("SKIPPED:") else Status.FAILED
        return ModelRecord(status=status, reason=error, **fields)
    caught = []
    try:
        original = arm is Arm.RAW
        if plan is not None and not original:
            tables = {part: raw[part].loc[frame.index] for part, frame in tables.items()}
        X, y = _model_data(tables["train"], config, original=original, synthetic=extra)
        estimator = baselines._with_preprocessing(estimator, ArmEncoder(config, original, plan))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(X, y)
            predictions, metrics = {}, {}
            for part in ("validation", "test"):
                features, labels = _model_data(tables[part], config, original=original)
                probabilities = estimator.predict_proba(features)[:, 1]
                metrics[part] = baselines.metric_row(
                    labels, probabilities, threshold=fields["threshold"]
                )
                predictions[part] = _prediction_table(
                    tables[part], config.target, probabilities, fields["threshold"]
                )
        stem = f"{arm.value}_{level or 0:g}_{seed}_{re.sub('[^a-z0-9]+', '_', name.lower())}"
        model_path = folder / f"{stem}.joblib"
        paths = {part: folder / f"{stem}_{part}.csv" for part in predictions}
        joblib.dump(
            {
                "estimator": estimator,
                "threshold": fields["threshold"],
                "protocol_id": fields["protocol_id"],
                "features": list(X.columns),
            },
            model_path,
        )
        for part, data in predictions.items():
            data.to_csv(paths[part], index=False)
        score_range = float(np.ptp(predictions["validation"]["probability"].to_numpy()))
        notices = {str(w.message) for w in caught}
        if score_range == 0:
            notices.add("Constant validation probabilities; excluded from model selection")
        return ModelRecord(
            status=Status.SUCCESS,
            validation_probability_range=score_range,
            model_path=model_path,
            val_predictions_path=paths["validation"],
            test_predictions_path=paths["test"],
            val_metrics=metrics["validation"],
            test_metrics=metrics["test"],
            warnings=sorted(notices),
            **fields,
        )
    except Exception as exc:
        return ModelRecord(
            status=Status.FAILED,
            reason=f"{type(exc).__name__}: {exc}",
            warnings=sorted({str(w.message) for w in caught}),
            **fields,
        )


def _select_group(records):
    """Exclude non-ranking predictors without changing frozen metric definitions.

    Trapezoidal PR-AUC can reward constant probabilities through its endpoint
    interpolation. Selection eligibility uses only validation score variation.
    """
    eligible = [
        r for r in records if r.status is Status.SUCCESS and r.validation_probability_range != 0
    ]
    winner = max(eligible, key=lambda r: r.val_metrics["pr_auc"]) if eligible else None
    return [r.updated(selected=winner is not None and r.model == winner.model) for r in records]


def _save_metrics(records, dataset, folder):
    rows = []
    for record in records:
        for split, scores in (("validation", record.val_metrics), ("test", record.test_metrics)):
            rows.append(
                {
                    "dataset": dataset,
                    "arm": ARM_LABELS[record.arm],
                    "fill_level": record.fill_level or 0.0,
                    "seed": record.seed,
                    "model": record.model,
                    "split": split,
                    "status": record.status.value,
                    "reason": record.reason,
                    "selected": record.selected,
                    "validation_probability_range": record.validation_probability_range,
                    "selection_eligible": record.status is Status.SUCCESS
                    and record.validation_probability_range != 0,
                    "training_rows": record.training_rows,
                    "n_synthetic": record.synthetic_rows,
                    "protocol_id": record.protocol_id,
                    "threshold": record.threshold,
                    **{metric: scores.get(metric, np.nan) for metric in METRICS},
                }
            )
    table = pd.DataFrame(rows)
    path = folder / "metrics.csv"
    table.to_csv(path, index=False)
    valid = table[(table.split == "validation") & (table.status == Status.SUCCESS.value)]
    summary = valid.groupby(["dataset", "arm", "fill_level", "model"], sort=False)[
        list(METRICS)
    ].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary_path = folder / "validation_summary.csv"
    summary.reset_index().to_csv(summary_path, index=False)
    return path, summary_path


def _repeated_cv(request, config, raw, protocol, folder, cv_gan_factory, plan=None):
    """Repeated real-row CV; only threshold-independent PR/ROC AUC are estimated.

    The frozen cutoff already used the original validation labels, so reporting
    CV F1 on train+validation would reuse calibration labels. Test is never read here.
    A CTGAN factory must refit AND quality-check using only the supplied fold training data.
    Without it, CTGAN CV is explicitly skipped; full-split GAN evaluation still runs.
    """
    cfg = request.config
    combined = pd.concat([raw["train"], raw["validation"]])
    if combined[config.target].value_counts().min() < cfg.training_cv_folds:
        raise ValueError("Too few rows per class for repeated CV")
    splitter = RepeatedStratifiedKFold(
        n_splits=cfg.training_cv_folds,
        n_repeats=cfg.training_cv_repeats,
        random_state=cfg.split_seed,
    )
    scores, predictions = [], []
    for index, (fit_ids, eval_ids) in enumerate(splitter.split(combined, combined[config.target])):
        repeat, fold = divmod(index, cfg.training_cv_folds)
        fold_raw = {"train": combined.iloc[fit_ids], "validation": combined.iloc[eval_ids]}
        fitting = fold_raw["train"].drop_duplicates() if plan is not None else fold_raw["train"]
        cleaner = _cleaner(config, plan, fitting)
        fold_cleaned = {
            part: cleaner.clean(fitting if part == "train" else data)
            for part, data in fold_raw.items()
        }
        for seed in cfg.seeds:
            # Separate reproducible generator randomness from the fixed row partition.
            generation_seed = int(np.random.SeedSequence([seed, repeat, fold]).generate_state(1)[0])
            accepted = {}
            if cv_gan_factory is not None:
                train = fold_cleaned["train"]
                positives = int(train[config.target].sum())
                for item in request.accepted_synthetic:
                    n = fill_rows(item.fill_level, positives, len(train) - positives)
                    if n:
                        extra = cv_gan_factory(train.copy(), n, generation_seed)
                        accepted[item.fill_level] = _synthetic(extra, train, config.target, n)
            arms = list(
                _arm_sets(
                    config, fold_raw, fold_cleaned, cfg.fill_levels, accepted, generation_seed
                )
            )
            arms += [
                (
                    Arm.GAN,
                    item.fill_level,
                    fold_cleaned,
                    None,
                    "CTGAN CV needs a fold-fitted, quality-checked cv_gan_factory"
                    if cv_gan_factory is None
                    else "SKIPPED: fill level adds zero rows",
                )
                for item in request.accepted_synthetic
                if item.fill_level not in accepted
            ]
            for arm, level, tables, extra, error in arms:
                for name, model in baselines.build_models(seed).items():
                    row = dict(
                        arm=ARM_LABELS[arm],
                        fill_level=level or 0.0,
                        model=name,
                        seed=seed,
                        repeat=repeat,
                        fold=fold,
                        protocol_id=protocol["protocol_id"],
                        status=Status.SUCCESS.value,
                        reason="",
                        pr_auc=np.nan,
                        roc_auc=np.nan,
                    )
                    if error:
                        row.update(
                            status=Status.SKIPPED.value
                            if arm is Arm.GAN or error.startswith("SKIPPED:")
                            else Status.FAILED.value,
                            reason=error,
                        )
                    else:
                        try:
                            if plan is not None and arm is not Arm.RAW:
                                tables = {
                                    part: fold_raw[part].loc[frame.index]
                                    for part, frame in tables.items()
                                }
                            X, y = _model_data(
                                tables["train"], config, original=arm is Arm.RAW, synthetic=extra
                            )
                            val, labels = _model_data(
                                tables["validation"], config, original=arm is Arm.RAW
                            )
                            fitted = baselines._with_preprocessing(
                                model, ArmEncoder(config, arm is Arm.RAW, plan)
                            )
                            fitted.fit(X, y)
                            probabilities = fitted.predict_proba(val)[:, 1]
                            probabilities = baselines._probabilities(labels, probabilities)
                            row.update(
                                pr_auc=baselines.pr_auc_score(labels, probabilities),
                                roc_auc=roc_auc_score(labels, probabilities),
                            )
                            predictions.append(
                                pd.DataFrame(
                                    {
                                        **{
                                            k: v
                                            for k, v in row.items()
                                            if k not in ("pr_auc", "roc_auc")
                                        },
                                        "source_row": tables["validation"].index,
                                        "y_true": labels.to_numpy(),
                                        "probability": probabilities,
                                        IS_SYNTHETIC: 0,
                                    }
                                )
                            )
                        except Exception as exc:
                            row.update(
                                status=Status.FAILED.value, reason=f"{type(exc).__name__}: {exc}"
                            )
                    scores.append(row)
    score_path, pred_path = folder / "cv_metrics.csv", folder / "cv_predictions.csv"
    result = pd.DataFrame(scores)
    result.to_csv(score_path, index=False)
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(pred_path, index=False)
    else:
        pd.DataFrame(columns=["source_row", "y_true", "probability", IS_SYNTHETIC]).to_csv(
            pred_path, index=False
        )
    return score_path, pred_path, result


def run_training(request: TrainingRequest, run: RunDir, *, protocol_path=None, cv_gan_factory=None):
    """Orchestrator callable, after the quality gate; see docs/training.md.

    No accepted CTGAN sets means only original, cleaned and SMOTENC run.
    The reference protocol's split seed stays fixed while config.seeds varies model randomness.
    """
    cfg = request.config
    path = _find_training_protocol(cfg, protocol_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Frozen baseline protocol not found: {path}. Freeze this dataset's original-arm "
            f"thresholds first with `python -m agentic_gan.tools.baselines --dataset "
            f"{cfg.dataset_name} --arm both --cv-folds 3`."
        )
    config, raw, cleaned, protocol, plan = _inputs(request, run, path)
    accepted = {}
    train = cleaned["train"]
    positives = int(train[config.target].sum())
    for item in request.accepted_synthetic:
        count = fill_rows(item.fill_level, positives, len(train) - positives)
        if item.rows != count:
            raise ValueError("Accepted synthetic row count differs from the configured fill level")
        accepted[item.fill_level] = _synthetic(
            pd.read_csv(run.root / item.path), train, config.target, count
        )
    folder = run.subdir("models").resolve()
    folder.mkdir(parents=True, exist_ok=True)
    saved_protocol = folder / "training_protocol.json"
    saved_protocol.write_text(json.dumps(protocol, indent=2) + "\n")
    records = []
    with threadpool_limits(limits=2):
        for seed in cfg.seeds:
            for arm, level, tables, extra, error in _arm_sets(
                config, raw, cleaned, cfg.fill_levels, accepted, seed
            ):
                group = [
                    _fit_record(
                        arm,
                        level,
                        tables,
                        extra,
                        error,
                        name,
                        estimator,
                        seed,
                        config,
                        protocol,
                        folder,
                        plan=plan,
                        raw=raw,
                    )
                    for name, estimator in baselines.build_models(seed).items()
                ]
                group = _select_group(group)
                excluded = [r.model for r in group if r.validation_probability_range == 0]
                if excluded:
                    log_decision(
                        "training",
                        "constant_validation_scores",
                        {"arm": arm.value, "fill_level": level, "seed": seed, "models": excluded},
                        "EXCLUDED_FROM_SELECTION",
                        "Retained fits and metrics; constant validation scores cannot rank rows",
                        run=run,
                    )
                records.extend(group)
                log_decision(
                    "training",
                    "fit_arm",
                    {
                        "arm": arm.value,
                        "fill_level": level,
                        "seed": seed,
                        "protocol_id": protocol["protocol_id"],
                    },
                    "FAILED" if any(r.status is Status.FAILED for r in group) else "SUCCESS",
                    "Frozen suite and thresholds; selection uses real validation PR-AUC",
                    run=run,
                )
        cv_paths, cv_result = (None, None), None
        if cfg.training_cv_folds:
            cv_metrics, cv_predictions, cv_result = _repeated_cv(
                request, config, raw, protocol, folder, cv_gan_factory, plan
            )
            cv_paths = (cv_metrics, cv_predictions)
    metric_path, summary_path = _save_metrics(records, cfg.dataset_name, folder)
    failures = sum(r.status is Status.FAILED for r in records)
    if cv_result is not None:
        failures += int(cv_result.status.eq(Status.FAILED.value).sum())
    notices = []
    constant = sum(r.validation_probability_range == 0 for r in records)
    if constant:
        notices.append(f"{constant} fits had constant validation scores and were not selectable")
    if cfg.training_cv_folds and request.accepted_synthetic and cv_gan_factory is None:
        notices.append("CTGAN repeated CV skipped: no fold-fitted, quality-checked cv_gan_factory")
    return TrainingOutput(
        status=Status.FAILED if failures else Status.SUCCESS,
        reason=f"{failures} model fits failed; inspect records/CV metrics" if failures else None,
        warnings=notices,
        records=records,
        metrics_path=metric_path,
        validation_summary_path=summary_path,
        protocol_path=saved_protocol,
        cv_metrics_path=cv_paths[0],
        cv_predictions_path=cv_paths[1],
    )


def run_training_arms(
    training_sets,
    X_val,
    y_val,
    X_test,
    y_test,
    seed=42,
    *,
    protocol_path,
    preprocessor=None,
    cv_folds=None,
):
    """Run each {arm: (X_train, y_train)} with identical holdouts and settings.

    Accepts original, cleaned, GAN-augmented, SMOTE, or other caller-named arms.
    Synthetic rows must already be generated and belong only in training_sets.
    For encoded inputs, encode synthetic rows with the real-training-fitted setup.
    protocol_path is the frozen original-arm protocol for this dataset and seed.
    Every arm uses its full model suite and identical numerical thresholds.
    """
    if not training_sets:
        raise ValueError("Supply at least one training arm")
    if cv_folds is None:
        cv_folds = _protocol_cv_folds(protocol_path)
    protocol = load_protocol(
        protocol_path, seed=seed, cv_folds=cv_folds, y_val=y_val, y_test=y_test
    )
    tables = []
    for arm, (X_train, y_train) in training_sets.items():
        table = run_models(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            seed=seed,
            preprocessor=preprocessor,
            cv_folds=cv_folds,
            thresholds=protocol["thresholds"],
        )
        table.insert(0, "arm", arm)
        table["protocol_id"] = protocol["protocol_id"]
        tables.append(table)
    return pd.concat(tables, ignore_index=True)
