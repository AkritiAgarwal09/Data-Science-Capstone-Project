"""Turn fidelity, privacy, and utility checks into the pipeline quality verdict.

The verdict is always decided by the checks. When it is FAIL, an LLM step explains the failure and
suggests what to change; its answer is stored in the report and never changes the verdict.

The fidelity and rule limits are calibrated per dataset from the real data (``margin`` times the
real-vs-real values). The two augmentation-gain checks are reported but never decide the verdict.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import get_args

import pandas as pd
from matplotlib.figure import Figure

from agentic_gan.llm import LLMStepOptions, run_llm_step
from agentic_gan.orchestrator.decisions import log_decision, log_progress
from agentic_gan.orchestrator.manifest import RunDir
from agentic_gan.schemas import (
    Adjustment,
    LevelReport,
    QualityDiagnosis,
    QualityInput,
    QualityReport,
    RunConfig,
    Stage,
    Status,
    Verdict,
)
from agentic_gan.tools.quality_fidelity import (
    RULES,
    fidelity_checks,
    real_vs_real,
    schema_thresholds,
    suggest_thresholds,
)
from agentic_gan.tools.quality_privacy import PRIVACY_CHECKS, quality_privacy_checks
from agentic_gan.utils.datasets import DATASETS
from agentic_gan.utils.files import write_text_atomic

logger = logging.getLogger(__name__)

# Below this many positive validation rows the real-vs-real baseline is too noisy to calibrate on.
MIN_VALIDATION_POSITIVES = 20
CALIBRATION_FILE = "calibrated_thresholds.json"

DIAGNOSIS_PROMPT = (
    "You explain why synthetic tabular data failed a quality gate, for the person tuning the "
    "generator. Use only the check results given: each failed check lists its value and the "
    "limit it missed. Name only checks and columns that appear in them. Quote the classifier AUC "
    "(real_vs_synthetic_auc) with its real_vs_real_auc baseline. Say in plain language what the "
    "failures suggest went wrong. Begin `likely_cause` with one or two sentences that say what "
    "the checks measure (how closely synthetic rows match real positive-class rows, whether they "
    "copy real rows, whether they help a classifier) and why that matters for this dataset, "
    "naming the dataset; then give the cause. Choose adjustments only from allowed_adjustments. "
    "You never change a verdict, threshold or rule."
)


def _positive_class(request: QualityInput, train: pd.DataFrame) -> int | str:
    positive = request.config.positive_class
    if train[request.config.target].eq(positive).any():
        return positive
    # Bank preprocessing maps the raw label "yes" to 1 in every cleaned split.
    if positive == "yes" and train[request.config.target].eq(1).any():
        return 1
    raise ValueError(f"positive class {positive!r} is absent from cleaned training data")


def _rules(request: QualityInput):
    available = {rule.name: rule for rule in RULES.get(request.config.dataset_name, ())}
    chosen = [name.removeprefix("rule:") for name in request.config.sanity_rules]
    unknown = set(chosen) - available.keys()
    if unknown:
        raise ValueError(f"Unknown sanity rules: {', '.join(sorted(unknown))}")
    return [available[name] for name in chosen]


def _calibrate(config: RunConfig, run: RunDir, train, val, positive, rules, discrete) -> dict:
    """The limits for this run's fidelity and rule checks, and where they came from.

    Each limit is ``margin`` times the real-vs-real value of its check (``suggest_thresholds``):
    about as close as two real samples are, with room for noise. It is computed once per run,
    before any level is scored, and kept in quality/calibrated_thresholds.json so a retry is judged
    by the same limits. The fixed numbers in ``QualityThresholds`` are the fallback when
    calibration is off or the validation split has too few positive rows for a stable baseline.
    """
    gate = config.quality_gate
    saved = run.subdir("quality") / CALIBRATION_FILE
    if saved.exists():
        return json.loads(saved.read_text(encoding="utf-8"))
    positives = int(val[config.target].eq(positive).sum())
    if not gate.calibrate or positives < MIN_VALIDATION_POSITIVES:
        limits = schema_thresholds(gate, rules)
        reason = "Calibration is off, so the fixed limits are used."
        if gate.calibrate:
            reason = (
                f"Only {positives} positive validation rows; at least {MIN_VALIDATION_POSITIVES} "
                "are needed for a stable baseline, so the fixed limits are used."
            )
            logger.warning(reason)
        log_decision(
            Stage.QUALITY_CHECK.value,
            "thresholds_calibrated",
            {"validation_positives": positives, "limits": limits},
            "FIXED",
            reason,
            run=run,
        )
        return {"source": "fixed", "margin": None, "limits": limits}
    baseline = real_vs_real(
        train, val, target=config.target, positive=positive, discrete=discrete, rules=rules
    )
    limits = suggest_thresholds(baseline.values, margin=gate.margin)
    calibration = {
        "source": "calibrated",
        "margin": gate.margin,
        "validation_positives": positives,
        "baseline": baseline.values,
        "limits": limits,
    }
    write_text_atomic(saved, json.dumps(calibration, indent=2) + "\n")
    log_decision(
        Stage.QUALITY_CHECK.value,
        "thresholds_calibrated",
        {"margin": gate.margin, "baseline": baseline.values, "limits": limits},
        "CALIBRATED",
        f"Each limit is {gate.margin:g} times the real-vs-real value of its check",
        run=run,
    )
    return calibration


def _plot(path, fidelity, diagnostics, level: float) -> None:
    distances = fidelity.columns["distance"].sort_values(ascending=False).head(10)
    figure = Figure(figsize=(10, 9))
    top, middle, bottom = figure.subplots(3, 1)
    top.barh(distances.index[::-1], distances.iloc[::-1])
    top.set_xlabel("Column distance from real train")
    top.set_title(f"Fidelity at {level:.0%} fill")
    middle.bar(
        ["Real validation", "Synthetic"],
        [diagnostics["validation_median"], diagnostics["synthetic_median"]],
    )
    middle.set_ylabel("Median distance to closest real train row")
    bottom.bar(
        ["PR-AUC real", "PR-AUC augmented", "F1 real", "F1 augmented"],
        [
            diagnostics["real_only_pr_auc"],
            diagnostics["augmented_pr_auc"],
            diagnostics["real_only_best_f1"],
            diagnostics["augmented_best_f1"],
        ],
    )
    bottom.set_ylim(0, 1)
    bottom.set_title(
        f"Real vs synthetic AUC: {diagnostics['real_vs_synthetic_auc']:.3f}; "
        f"revealing: {', '.join(diagnostics['revealing_columns']) or 'none'}"
    )
    figure.tight_layout()
    figure.savefig(path)


def _rule_hash(config: RunConfig) -> str:
    """Fingerprint of the rules and thresholds that the diagnosis must not change."""
    locked = {
        "quality_gate": config.quality_gate.model_dump(),
        "sanity_rules": config.sanity_rules,
        "gan_bounds": config.gan_bounds,
    }
    return hashlib.sha256(json.dumps(locked, sort_keys=True).encode()).hexdigest()


def _default_diagnosis(results) -> QualityDiagnosis:
    """Facts-only diagnosis for when the LLM step fails: what failed and one safe next step.

    Like model selection's own retry, it asks for more epochs unless a privacy check failed.
    """
    failed = [c for level, _, _ in results for c in level.checks if not c.passed]
    names = list(dict.fromkeys(c.name for c in sorted(failed, key=lambda c: not c.hard)))
    hard = {c.name for c in failed if c.hard}
    privacy = bool(hard & PRIVACY_CHECKS)
    columns = []
    for _, details, distances in results:
        columns += [*map(str, distances.nlargest(3).index), *details["revealing_columns"]]
    columns = list(dict.fromkeys(columns))[:5]
    auc = max(details["real_vs_synthetic_auc"] for _, details, _ in results)
    floor = results[0][1]["real_vs_real_auc"]
    adjustments: list[Adjustment] = ["change_sampling"] if privacy else ["more_epochs"]
    if not privacy and any(name.startswith("rule:") for name in hard):
        adjustments.append("change_sampling")
    return QualityDiagnosis(
        failed_checks=names,
        columns=columns,
        likely_cause=(
            f"Failed checks: {', '.join(names[:5])}. A classifier tells real from synthetic rows "
            f"with AUC {auc:.2f} (two real samples score {floor:.2f}), relying mostly on "
            f"{', '.join(columns[:3]) or 'no single column'}."
        ),
        adjustments=adjustments,
    )


def _diagnosis_context(dataset: str, results, calibration) -> str:
    def brief(check):
        return {
            "name": check.name,
            "value": round(check.value, 4),
            "limit": round(check.threshold, 4),
        }

    levels = []
    for level, details, distances in results:
        failed = [c for c in level.checks if not c.passed]
        levels.append(
            {
                "fill_level": level.fill_level,
                "passed": level.passed,
                "failed_hard_checks": [brief(c) for c in failed if c.hard],
                "failed_soft_checks": [brief(c) for c in failed if not c.hard],
                "largest_column_distances": {
                    str(c): round(float(v), 3) for c, v in distances.nlargest(5).items()
                },
                "revealing_columns": details["revealing_columns"],
                "real_vs_synthetic_auc": round(float(details["real_vs_synthetic_auc"]), 3),
                "real_vs_real_auc": round(float(details["real_vs_real_auc"]), 3),
            }
        )
    return json.dumps(
        {
            "dataset": dataset,
            "limits": {
                "source": calibration["source"],
                "margin": calibration["margin"],
                "values": calibration["limits"],
            },
            "levels": levels,
            "allowed_adjustments": get_args(Adjustment),
        }
    )


def _diagnose(request: QualityInput, run: RunDir, results, client, calibration) -> QualityDiagnosis:
    """Ask the LLM to explain a FAIL; keep its answer only if it matches the results."""
    fallback = _default_diagnosis(results)
    options = LLMStepOptions(
        fallback=fallback,
        expected_rule_hash=_rule_hash(request.config),
        current_rule_hash=lambda: _rule_hash(run.load_config()),
    )
    answer = run_llm_step(
        Stage.QUALITY_CHECK.value,
        DIAGNOSIS_PROMPT,
        _diagnosis_context(request.config.dataset_name, results, calibration),
        QualityDiagnosis,
        options,
        run,
        client=client,
    )
    failed = {c.name for level, _, _ in results for c in level.checks if not c.passed}
    known = {str(c) for _, _, distances in results for c in distances.index}
    unknown = sorted((set(answer.failed_checks) - failed) | (set(answer.columns) - known))
    if unknown:
        log_decision(
            Stage.QUALITY_CHECK.value,
            "llm_step",
            {"not_in_results": unknown},
            "LLM_FALLBACK",
            "Diagnosis named checks or columns that are not in the results",
            run=run,
        )
        return fallback
    return answer


def run_quality(request: QualityInput, run: RunDir, *, client=None) -> QualityReport:
    """Score each fill level, persist the report and plots, and log the decisions.

    A FAIL verdict also gets an LLM diagnosis (`client` is the LLM client; tests pass a FakeLLM).
    """
    config = request.config
    target = config.target
    train = pd.read_csv(request.preprocess.baseline.train_path).drop(
        columns="source_row", errors="ignore"
    )
    val = pd.read_csv(request.preprocess.baseline.val_path).drop(
        columns="source_row", errors="ignore"
    )
    positive = _positive_class(request, train)
    rules = _rules(request)
    discrete = DATASETS[config.dataset_name].discrete if config.dataset_name in DATASETS else ()
    calibration = _calibrate(config, run, train, val, positive, rules, discrete)
    thresholds = calibration["limits"]
    reference = train.loc[train[target].eq(positive)]
    attempt = max(run.load_state().attempts(Stage.QUALITY_CHECK), 1)
    levels = []
    plots = []
    decisions = []

    for synthetic_set in request.generator.synthetic_sets:
        log_progress(
            "quality_check",
            f"score fill={synthetic_set.fill_level:g} attempt={attempt}",
            tool="fidelity_checks + quality_privacy_checks",
            next_action="save quality plot and check verdict",
        )
        synthetic = pd.read_csv(synthetic_set.path)
        if len(synthetic) != synthetic_set.rows:
            raise ValueError(
                f"Synthetic row count differs from declared count at {synthetic_set.fill_level}"
            )
        fidelity = fidelity_checks(
            reference,
            synthetic,
            target=target,
            positive=positive,
            discrete=discrete,
            rules=rules,
            thresholds=thresholds,
        )
        privacy, details = quality_privacy_checks(
            train,
            val,
            synthetic,
            target,
            positive,
            thresholds=config.quality_gate,
            seed=config.split_seed,
        )
        checks = [*fidelity.checks, *privacy]
        level = LevelReport(
            fill_level=synthetic_set.fill_level,
            checks=checks,
            passed=all(check.passed for check in checks if check.hard),
        )
        levels.append(level)
        details["real_vs_synthetic_auc"] = next(
            check.value for check in privacy if check.name == "real_vs_synthetic_auc"
        )
        plot_path = run.subdir("quality") / (
            f"fill_{str(level.fill_level).replace('.', 'p')}_attempt_{attempt}.png"
        )
        _plot(plot_path, fidelity, details, level.fill_level)
        plots.append(plot_path.resolve())
        decisions.append((level, details, fidelity.columns["distance"]))

    if not levels:
        raise ValueError("Generator supplied no synthetic fill levels")
    passing = [level.fill_level for level in levels if level.passed]
    verdict = Verdict.PASS if passing else Verdict.FAIL

    for level, details, _ in decisions:
        for check in level.checks:
            log_decision(
                Stage.QUALITY_CHECK.value,
                check.name,
                {
                    "fill_level": level.fill_level,
                    "value": check.value,
                    "threshold": check.threshold,
                    "hard": check.hard,
                },
                "PASS" if check.passed else "FAIL",
                "Quality check result",
                run=run,
            )
        log_decision(
            Stage.QUALITY_CHECK.value,
            "fill_level",
            {
                "fill_level": level.fill_level,
                "revealing_columns": details["revealing_columns"],
                "real_vs_real_auc": details["real_vs_real_auc"],
            },
            "PASS" if level.passed else "FAIL",
            "Hard checks determine the level verdict",
            run=run,
        )
    log_decision(
        Stage.QUALITY_CHECK.value,
        "quality_verdict",
        {"attempt": attempt, "passing_levels": passing},
        verdict.value,
        "At least one fill level must pass",
        run=run,
    )
    # Only a failure needs explaining: the diagnosis goes to model selection with the retry.
    diagnosis = (
        _diagnose(request, run, decisions, client, calibration) if verdict is Verdict.FAIL else None
    )
    report_path = run.subdir("quality") / f"quality_report_attempt_{attempt}.json"
    report = QualityReport(
        status=Status.SUCCESS,
        verdict=verdict,
        levels=levels,
        report_path=report_path.resolve(),
        plot_paths=plots,
        diagnosis=diagnosis,
    )
    write_text_atomic(report_path, report.model_dump_json(indent=2) + "\n")
    return report
