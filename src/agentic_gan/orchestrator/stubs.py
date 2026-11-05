"""Typed placeholder agents. Replace each callable without changing graph.py."""

from pathlib import Path

from agentic_gan.schemas import (
    STAGE_ORDER,
    Arm,
    ArmFiles,
    CheckResult,
    EvaluationOutput,
    GeneratorOutput,
    LevelReport,
    ModelRecord,
    PreprocessOutput,
    QualityReport,
    SplitOutput,
    Stage,
    Status,
    SyntheticSet,
    TrainingOutput,
    Verdict,
)

from .decisions import log_decision
from .manifest import RunDir
from .state import RunState


def _file(run: RunDir, folder: str, name: str, content: str = "stub\n") -> Path:
    path = run.subdir(folder) / name
    path.write_text(content, encoding="utf-8")
    return path.resolve()


def _stub_table(target: str, positive_class: str | int) -> str:
    if isinstance(positive_class, str):
        positive = positive_class
        negative = "no" if positive_class.lower() == "yes" else "yes"
    else:
        positive = str(positive_class)
        negative = "0" if positive_class != 0 else "1"
    return f"feature,{target}\n1,{positive}\n0,{negative}\n"


class StubAgent:
    """failures=0 succeeds; positive counts fail first N calls; -1 fails forever."""

    def __init__(self, stage: Stage, *, failures: int = 0, failure_mode: str = "error"):
        if stage not in STAGE_ORDER or failures < -1 or failure_mode not in ("error", "quality"):
            raise ValueError("invalid stub configuration")
        if failure_mode == "quality" and stage is not Stage.QUALITY_CHECK:
            raise ValueError("quality failure mode only applies to quality_check")
        self.stage = stage
        self.failures = failures
        self.failure_mode = failure_mode
        self.calls = 0

    def __call__(self, request, run: RunDir):
        self.calls += 1
        fail = self.failures == -1 or self.calls <= self.failures
        if fail and self.failure_mode == "error":
            raise RuntimeError(f"stub {self.stage.value} failed on call {self.calls}")

        if self.stage is Stage.SPLIT:
            paths = {
                part: _file(
                    run, "data", f"{part}.csv",
                    _stub_table(request.target, request.positive_class),
                )
                for part in ("train", "val", "test")
            }
            output = SplitOutput(status=Status.SUCCESS, train_path=paths["train"],
                                 val_path=paths["val"], test_path=paths["test"],
                                 target=request.target, seed=request.split_seed,
                                 train_positives=1, train_negatives=1)
        elif self.stage is Stage.PREPROCESSING:
            table = _stub_table(request.config.target, request.config.positive_class)
            raw = ArmFiles(**{
                f"{part}_path": _file(run, "data", f"raw_{part}.csv", table)
                for part in ("train", "val", "test")
            })
            baseline = ArmFiles(**{
                f"{part}_path": _file(run, "data", f"baseline_{part}.csv", table)
                for part in ("train", "val", "test")
            })
            output = PreprocessOutput(
                status=Status.SUCCESS,
                raw=raw,
                baseline=baseline,
                profile_path=_file(run, "profile", "profile.json", "{}\n"),
                treatment_log_path=_file(run, "profile", "treatments.json", "[]\n"),
                transformer_path=_file(run, "profile", "transformer.txt"),
                transformer_fit_rows=2, baseline_train_positives=1,
                baseline_train_negatives=1,
            )
        elif self.stage is Stage.MODEL_SELECTION:
            level = request.config.fill_levels[0]
            output = GeneratorOutput(
                status=Status.SUCCESS, generator="stub", rationale="Placeholder generator",
                checkpoint_path=_file(run, "synthetic", "checkpoint.txt"),
                loss_history_path=_file(run, "synthetic", "loss.csv", "loss\n0\n"),
                synthetic_sets=[SyntheticSet(
                    fill_level=level, rows=1,
                    path=_file(run, "synthetic", "sample.csv", "feature,y,is_synthetic\n1,yes,1\n"),
                )],
            )
        elif self.stage is Stage.QUALITY_CHECK:
            level = request.generator.synthetic_sets[0].fill_level
            passed = not fail
            output = QualityReport(
                status=Status.SUCCESS, verdict=Verdict.PASS if passed else Verdict.FAIL,
                levels=[LevelReport(fill_level=level, passed=passed, checks=[CheckResult(
                    name="stub_gate", value=float(passed), threshold=1.0, passed=passed,
                )])],
                report_path=_file(
                    run, "quality", "quality.json", f'{{"passed": {str(passed).lower()}}}\n'
                ),
            )
        elif self.stage is Stage.TRAINING:
            records = []
            arms = [(Arm.RAW, None), (Arm.BASELINE, None)]
            arms += [(Arm.GAN, item.fill_level) for item in request.accepted_synthetic]
            for arm, level in arms:
                name = arm.value if level is None else f"{arm.value}_{level}"
                records.append(ModelRecord(
                    status=Status.SUCCESS, arm=arm, model="stub", seed=request.config.seeds[0],
                    fill_level=level, model_path=_file(run, "models", f"{name}.txt"),
                    val_predictions_path=_file(
                        run, "models", f"{name}_val.csv", "prediction\n0.5\n"
                    ),
                    test_predictions_path=_file(
                        run, "models", f"{name}_test.csv", "prediction\n0.5\n"
                    ),
                ))
            output = TrainingOutput(status=Status.SUCCESS, records=records)
        else:
            state = RunState.load(request.state_path)
            availability = "AVAILABLE" if state.gan_available else "NOT_AVAILABLE"
            output = EvaluationOutput(
                status=Status.SUCCESS,
                metrics_table_paths=[_file(run, "report", "metrics.csv", "arm,metric\nstub,0\n")],
                report_path=_file(
                    run, "report", "final_report.md",
                    f"# Stub report\n\nSynthetic data: {availability}\n",
                ),
                numbers_verified=True,
            )
        decision = (
            output.verdict.value if self.stage is Stage.QUALITY_CHECK else output.status.value
        )
        log_decision(self.stage.value, "stub_choice", {"call": self.calls}, decision,
                     "Configured placeholder result", run=run)
        return output


def default_agents():
    return {stage: StubAgent(stage) for stage in STAGE_ORDER}
