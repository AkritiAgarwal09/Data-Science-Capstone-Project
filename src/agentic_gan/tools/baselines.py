"""Four frozen binary baselines with validation-selected, shared arm thresholds."""

import argparse
import hashlib
import json
import platform
import time
import warnings
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from agentic_gan.tools.preprocessing import Preprocessor, load_dataset, normalize, split_dataset
from agentic_gan.utils.datasets import load_configs
from agentic_gan.utils.files import sha256_file

METRICS = ("pr_auc", "f1", "roc_auc", "precision", "recall", "accuracy")
SUITE_VERSION = "sprint2-four-models-v3"


RESULT_COLUMNS = [
    "dataset",
    "arm",
    "model",
    "seed",
    "cv_folds",
    "status",
    "selected",
    "protocol_id",
    "threshold",
    "threshold_source",
    "cv_pr_auc_mean",
    "cv_pr_auc_sd",
    "val_pr_auc",
    "val_f1",
    "val_f1_at_0_5",
    "test_pr_auc",
    "test_f1",
    "test_roc_auc",
    "test_precision",
    "test_recall",
    "test_accuracy",
]


def build_models(seed=42):
    """Fixed suite; keep original settings, including the stack's three base learners."""
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier

    models = {
        # Shallow, stochastic trees bound exact split-search cost on IEEE-CIS.
        "Gradient boosting": GradientBoostingClassifier(
            n_estimators=50,
            max_depth=2,
            max_features="sqrt",
            subsample=0.2,
            random_state=seed,
        ),
        "LightGBM": LGBMClassifier(n_estimators=150, verbosity=-1, n_jobs=2, random_state=seed),
        "CatBoost": CatBoostClassifier(
            iterations=200, thread_count=2, verbose=0, random_seed=seed, allow_writing_files=False
        ),
    }
    models["Stacking ensemble"] = StackingClassifier(
        estimators=[
            ("logistic", LogisticRegression(max_iter=100, tol=0.05, random_state=seed)),
            ("forest", RandomForestClassifier(
                n_estimators=10, max_depth=6, n_jobs=1, random_state=seed
            )),
        ],
        final_estimator=LogisticRegression(max_iter=100, tol=0.05, random_state=seed),
        stack_method="predict_proba",
        cv=2,
        n_jobs=1,
    )
    return models


def pr_auc_score(y, probabilities):
    """Trapezoidal PR-AUC, matching the notebook (not average precision)."""
    precision, recall, _ = precision_recall_curve(y, probabilities)
    return float(auc(recall, precision))


def _probabilities(y, probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or len(probabilities) != len(y) or not len(y):
        raise ValueError("Expected one probability per label")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and between zero and one")
    if np.asarray(y).ndim != 1 or set(np.unique(y)) != {0, 1}:
        raise ValueError("Threshold/metric labels must contain both binary classes")
    return probabilities


def select_f1_threshold(y_validation, probabilities):
    """Maximize validation F1 over observed scores; ties choose the highest cutoff."""
    probabilities = _probabilities(y_validation, probabilities)
    precision, recall, thresholds = precision_recall_curve(y_validation, probabilities)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def metric_row(y, probabilities, *, threshold):
    probabilities = _probabilities(y, probabilities)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and between zero and one")
    prediction = (probabilities >= threshold).astype(int)
    return {
        "pr_auc": pr_auc_score(y, probabilities),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "accuracy": float(accuracy_score(y, prediction)),
    }


def _pr_scorer(estimator, X, y):
    return pr_auc_score(y, estimator.predict_proba(X)[:, 1])


def _check_inputs(X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, cv_folds):
    if cv_folds != 0 and cv_folds < 2:
        raise ValueError("cv_folds must be zero (disabled) or at least two")
    if len({X.shape[1] for X in (X_train, X_val, X_test)}) != 1:
        raise ValueError("Feature counts differ across splits")
    frames = [X for X in (X_train, X_val, X_test) if isinstance(X, pd.DataFrame)]
    reference_columns = list(frames[0].columns) if frames else None
    for X, y in ((X_train, y_train), (X_val, y_val), (X_test, y_test)):
        if len(X) != len(y) or not len(X):
            raise ValueError("Each feature table must have matching nonempty labels")
        if np.asarray(y).ndim != 1 or set(np.unique(y)) != {0, 1}:
            raise ValueError("Each split must contain both binary labels 0 and 1")
        if isinstance(X, pd.DataFrame):
            if not X.columns.is_unique or list(X.columns) != reference_columns:
                raise ValueError("Feature columns/order differ across splits")
            if "source_row" in X.columns:
                raise ValueError("Remove source_row from model features")
            if isinstance(y, pd.Series) and not X.index.equals(y.index):
                raise ValueError("Feature and label indices are misaligned")
        if preprocessor is None and not np.isfinite(np.asarray(X, dtype=float)).all():
            raise ValueError("Encoded inputs must be finite; supply a preprocessor for raw data")
    if np.bincount(np.asarray(y_train, dtype=int)).min() < max(3, cv_folds):
        raise ValueError("Too few training examples per class for stacking/CV")


def _with_preprocessing(estimator, preprocessor):
    """Fit preprocessing only on the rows used by each estimator or stacking fold."""
    estimator = clone(estimator)
    if preprocessor is None:
        return estimator
    if isinstance(estimator, StackingClassifier):
        estimator.set_params(
            estimators=[
                (name, make_pipeline(clone(preprocessor), clone(base)))
                for name, base in estimator.estimators
            ]
        )
        return estimator
    return make_pipeline(clone(preprocessor), estimator)


def run_models(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    seed=42,
    *,
    preprocessor=None,
    cv_folds=3,
    thresholds=None,
    model_names=None,
) -> pd.DataFrame:
    """Return one row per model; select on validation PR-AUC, test only the winner.

    Six positional inputs may be encoded arrays/DataFrames. For raw data, pass
    an unfitted sklearn-compatible preprocessor, cloned inside each CV fold.
    Synthetic rows belong in X_train/y_train only. Use identical seed, feature
    schema, preprocessing, CV settings and real holdouts for every arm.
    With already-encoded inputs, CV describes training-fitted preprocessing;
    pass raw data + preprocessor for outer-fold-local preprocessing.
    Stacking base models each own their preprocessing, including inside inner folds.
    Test metric cells are empty for models not selected. Failures and convergence
    warnings are recorded; failed models cannot win. No estimator is fitted on test.
    With thresholds=None, calibrate on validation (original reference arm only).
    All comparison arms must pass the reference's frozen model-to-threshold map.
    Training CV reports only threshold-independent PR-AUC and ROC-AUC.
    """
    _check_inputs(X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, cv_folds)
    models = build_models(seed)
    if model_names is not None:
        if not model_names or len(set(model_names)) != len(model_names):
            raise ValueError("Choose at least one model, without duplicates")
        unknown = sorted(set(model_names) - set(models))
        if unknown:
            raise ValueError(f"Unknown models: {unknown}")
        models = {name: models[name] for name in model_names}
    if thresholds is not None:
        _validate_thresholds(thresholds, models)
    rows, fitted = [], {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed) if cv_folds else None
    scoring = {
        "pr_auc": _pr_scorer,
        "roc_auc": "roc_auc",
    }
    with threadpool_limits(limits=2):
        for name, estimator in models.items():
            started = time.perf_counter()
            row = {
                "model": name,
                "seed": seed,
                "cv_folds": cv_folds,
                "status": "success",
                "selected": False,
                "error": "",
                "threshold": thresholds[name] if thresholds is not None else np.nan,
                "threshold_source": (
                    "frozen_original_validation_f1"
                    if thresholds is not None
                    else "original_validation_f1"
                ),
                **{f"test_{m}": np.nan for m in METRICS},
            }
            model = _with_preprocessing(estimator, preprocessor)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    if cv is not None:
                        scores = cross_validate(
                            model,
                            X_train,
                            y_train,
                            cv=cv,
                            scoring=scoring,
                            n_jobs=1,
                            error_score="raise",
                        )
                        for metric in scoring:
                            row[f"cv_{metric}_mean"] = float(scores[f"test_{metric}"].mean())
                            row[f"cv_{metric}_sd"] = float(scores[f"test_{metric}"].std(ddof=1))
                    model.fit(X_train, y_train)
                    probabilities = model.predict_proba(X_val)[:, 1]
                    if thresholds is None:
                        row["threshold"] = select_f1_threshold(y_val, probabilities)
                    row["val_f1_at_0_5"] = metric_row(y_val, probabilities, threshold=0.5)["f1"]
                    row.update(
                        {
                            f"val_{k}": v
                            for k, v in metric_row(
                                y_val, probabilities, threshold=row["threshold"]
                            ).items()
                        }
                    )
                    fitted[name] = model
                except Exception as exc:
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                row["warnings"] = " | ".join(sorted({str(w.message) for w in caught}))
            row["fit_seconds"] = time.perf_counter() - started
            rows.append(row)
            print(
                f"{name}: {row['status']}; "
                f"validation PR-AUC={row.get('val_pr_auc', float('nan')):.6f}",
                flush=True,
            )
        successful = [r for r in rows if r["status"] == "success"]
        if not successful:
            raise RuntimeError("All models failed: " + json.dumps(rows, default=str))
        winner = max(successful, key=lambda r: r["val_pr_auc"])
        winner["selected"] = True
        winner.update(
            {
                f"test_{k}": v
                for k, v in metric_row(
                    y_test,
                    fitted[winner["model"]].predict_proba(X_test)[:, 1],
                    threshold=winner["threshold"],
                ).items()
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("val_pr_auc", ascending=False, kind="stable", na_position="last")
        .reset_index(drop=True)
    )


def _validate_thresholds(thresholds, models):
    if set(thresholds) != set(models):
        raise ValueError("Frozen thresholds must contain exactly the fixed model suite")
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in thresholds.values()):
        raise ValueError("Frozen thresholds must be finite and between zero and one")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _model_settings(seed):
    return json.loads(
        json.dumps(
            {name: model.get_params(deep=True) for name, model in build_models(seed).items()},
            sort_keys=True,
            default=str,
        )
    )


def _label_hash(y):
    # Labels and order must match even when callers supply already-encoded features.
    return _digest(np.asarray(y, dtype=int).tolist())


def freeze_protocol(path, table, metadata, *, dataset, seed, cv_folds, y_val, y_test):
    """Write a new immutable reference protocol only after every model succeeds."""
    if (table.status != "success").any():
        raise ValueError("Cannot freeze an incomplete model suite")
    thresholds = dict(zip(table.model, table.threshold, strict=True))
    _validate_thresholds(thresholds, build_models(seed))
    protocol = {
        "suite_version": SUITE_VERSION,
        "dataset": dataset,
        "seed": seed,
        "cv_folds": cv_folds,
        "reference_arm": "original",
        "objective": "maximize validation F1; highest threshold breaks exact ties; p >= t",
        "thresholds": thresholds,
        "models": _model_settings(seed),
        "validation_labels_sha256": _label_hash(y_val),
        "test_labels_sha256": _label_hash(y_test),
        **{key: metadata[key] for key in ("source_sha256", "split_hashes", "config", "versions")},
    }
    protocol["protocol_id"] = _digest(protocol)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(protocol, stream, indent=2)
    return protocol


def load_protocol(path, *, seed=42, cv_folds=3, metadata=None, y_val=None, y_test=None):
    """Refuse settings/data drift; never silently recalibrate an existing experiment."""
    protocol = json.loads(Path(path).read_text())
    payload = {k: v for k, v in protocol.items() if k != "protocol_id"}
    if protocol.get("protocol_id") != _digest(payload):
        raise ValueError("Frozen protocol checksum mismatch")
    expected = {
        "suite_version": SUITE_VERSION,
        "seed": seed,
        "cv_folds": cv_folds,
        "models": _model_settings(seed),
        "versions": _versions(),
    }
    if metadata is not None:
        expected.update({k: metadata[k] for k in ("source_sha256", "split_hashes", "config")})
    for key, value in expected.items():
        if _digest(protocol.get(key)) != _digest(value):
            raise ValueError(f"Frozen protocol mismatch: {key}; create a separate experiment")
    for split, y in (("validation", y_val), ("test", y_test)):
        if y is not None and protocol[f"{split}_labels_sha256"] != _label_hash(y):
            raise ValueError(f"Frozen {split} labels/order do not match")
    _validate_thresholds(protocol["thresholds"], build_models(seed))
    return protocol


def _versions():
    return {
        p: version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy", "lightgbm", "catboost")
    }


class SharedPreprocessor(TransformerMixin, BaseEstimator):
    """Make the shared cleaner cloneable, so each training CV fold learns its own fills."""

    def __init__(self, config):
        self.config = config

    def fit(self, X, y):
        train = X.copy()
        train[self.config.target] = np.asarray(y)
        self.fitted_ = Preprocessor(
            self.config, self.config.imputation, self.config.include_duration
        ).fit(train)
        return self

    def transform(self, X):
        return self.fitted_.transform(X).to_numpy()


def _minimal_transformer(X, discrete, strategy="median", *, sparse_output=False):
    categorical = [c for c in X if c in discrete or not pd.api.types.is_numeric_dtype(X[c])]
    numeric = [c for c in X if c not in categorical]
    return ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(strategy=strategy, keep_empty_features=True), StandardScaler()
                ),
                numeric,
            ),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output),
                ),
                categorical,
            ),
        ],
        sparse_threshold=1.0 if sparse_output else 0.3,
    )


def prepare_inputs(dataset, arm, root, seed, config_path=None):
    """Split raw data and return an unfitted preprocessing pipeline for the chosen arm."""
    if arm not in ("original", "cleaned"):
        raise ValueError(f"Unknown arm: {arm}; choose original or cleaned")
    config = load_configs(config_path)[dataset]
    config = replace(config, split={**config.split, "seed": seed})
    raw = load_dataset(config, root / config.file)
    # Normalize IEEE features before model fitting: this removes the transaction key
    # and split clock while retaining the raw table for the configured time split.
    data = normalize(raw, config, config.include_duration)
    split_source = raw if config.name == "ieee_cis" else data
    if arm == "cleaned":
        transformer = SharedPreprocessor(config)
    else:
        transformer = _minimal_transformer(
            data.drop(columns=config.target),
            config.discrete,
            config.imputation,
            sparse_output=config.name == "ieee_cis",
        )
    ids = split_dataset(split_source, config)
    frames = [data.loc[ids[name]] for name in ("train", "validation", "test")]
    source_hash = source_hash_for(config, root / config.file)
    inputs = tuple(
        item
        for frame in frames
        for item in (frame.drop(columns=config.target), frame[config.target])
    )
    metadata = {
        "source_sha256": source_hash,
        "input_hashes": {"raw": source_hash},
        "input_kind": "raw",
        "split_origin": "config and seed",
        "config": asdict(config),
        "include_duration": config.include_duration,
        "split_hashes": {
            s: hashlib.sha256(json.dumps(v).encode()).hexdigest() for s, v in ids.items()
        },
        "versions": _versions(),
        "python": platform.python_version(),
    }
    return inputs, transformer, metadata


def source_hash_for(config, source_path):
    """Hash every source file that defines a dataset, including IEEE identity rows."""
    source_path = Path(source_path)
    if config.name != "ieee_cis":
        return sha256_file(source_path)
    identity_path = source_path.parent / Path(config.merge["file"]).name
    if not identity_path.is_file():
        raise FileNotFoundError(f"IEEE-CIS identity table not found: {identity_path}")
    digest = hashlib.sha256()
    for label, path in (("transaction", source_path), ("identity", identity_path)):
        digest.update(label.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def save_results(table, path):
    """Upsert by dataset/arm/seed/CV, accepting any arm name and retaining other runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["dataset", "arm", "seed", "cv_folds"]
    if path.exists():
        old = pd.read_csv(path)
        incoming = set(map(tuple, table[keys].drop_duplicates().to_numpy()))
        keep = [tuple(row) not in incoming for row in old[keys].to_numpy()]
        table = pd.concat([old.loc[keep], table], ignore_index=True)
    temporary = path.with_suffix(".csv.tmp")
    table.reindex(columns=RESULT_COLUMNS).to_csv(temporary, index=False)
    temporary.replace(path)


def run_dataset(dataset, *, root=None, arm="cleaned", seed=42, cv_folds=3, config_path=None):
    root = Path(root) if root else Path(__file__).resolve().parents[3]
    inputs, transformer, metadata = prepare_inputs(dataset, arm, root, seed, config_path)
    metadata["preprocessing"] = (
        "Raw inputs; imputation, encoding and scaling fitted inside each outer CV fold; "
        "stacking base pipelines also refit preprocessing inside inner folds. "
        "Final fitting uses training rows only; validation/test are transform-only."
    )
    protocol_path = root / "results" / f"baseline_protocol_{dataset}_{seed}_cv{cv_folds}.json"
    protocol = None
    if protocol_path.exists():
        protocol = load_protocol(
            protocol_path,
            seed=seed,
            cv_folds=cv_folds,
            metadata=metadata,
            y_val=inputs[3],
            y_test=inputs[5],
        )
    elif arm != "original":
        raise ValueError("Run --arm original or --arm both first to freeze validation thresholds")
    table = run_models(
        *inputs,
        seed=seed,
        preprocessor=transformer,
        cv_folds=cv_folds,
        thresholds=protocol["thresholds"] if protocol else None,
    )
    if protocol is None:
        protocol = freeze_protocol(
            protocol_path,
            table,
            metadata,
            dataset=dataset,
            seed=seed,
            cv_folds=cv_folds,
            y_val=inputs[3],
            y_test=inputs[5],
        )
    table["protocol_id"] = protocol["protocol_id"]
    for key, value in [
        ("dataset", dataset),
        ("arm", arm),
        ("pr_auc_method", "trapezoidal"),
        ("include_duration", metadata["include_duration"]),
        ("source_sha256", metadata["source_sha256"]),
    ]:
        table[key] = value
    for split, y in zip(("train", "validation", "test"), inputs[1::2], strict=True):
        table[f"{split}_rows"] = len(y)
        table[f"{split}_positive_rate"] = float(np.mean(y))
    result_path = root / "results" / "baselines.csv"
    save_results(table, result_path)
    metadata.update(
        dataset=dataset,
        arm=arm,
        seed=seed,
        cv_folds=cv_folds,
        results=json.loads(table.to_json(orient="records")),
        models=_model_settings(seed),
        protocol_id=protocol["protocol_id"],
        selection=(
            "highest validation trapezoidal PR-AUC; "
            "frozen original-validation F1 thresholds; test winner only"
        ),
    )
    (result_path.parent / f"{dataset}_{arm}_{seed}_cv{cv_folds}.json").write_text(
        json.dumps(metadata, indent=2, default=str)
    )
    print(
        table.loc[table.selected, ["model", *[f"test_{m}" for m in METRICS]]].to_string(
            index=False
        ),
        flush=True,
    )
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Key in shared datasets.json")
    parser.add_argument("--arm", choices=["original", "cleaned", "both"], default="cleaned")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--config", dest="config_path", type=Path)
    args = vars(parser.parse_args())
    arms = ["original", "cleaned"] if args["arm"] == "both" else [args["arm"]]
    for arm in arms:
        table = run_dataset(**{**args, "arm": arm})
        if (table.status != "success").any():
            raise SystemExit(
                "Some models failed; see errors in the run JSON beside results/baselines.csv"
            )


if __name__ == "__main__":
    main()
