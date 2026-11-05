"""LangGraph pipeline with deterministic routes and typed agent handoffs."""

from pathlib import Path
from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from agentic_gan.agents.evaluation import run_evaluation
from agentic_gan.agents.model_selection import run_model_selection
from agentic_gan.agents.preprocessing import preprocess
from agentic_gan.agents.quality import run_quality
from agentic_gan.agents.training import run_training
from agentic_gan.schemas import (
    STAGE_IO,
    EvaluationInput,
    GeneratorOutput,
    ModelSelectionInput,
    PreprocessInput,
    PreprocessOutput,
    QualityInput,
    QualityReport,
    RunConfig,
    SplitOutput,
    Stage,
    Status,
    TrainingOutput,
    TrainingRequest,
    Verdict,
)
from agentic_gan.utils.datasets import DATASETS

from .decisions import log_decision
from .manifest import Manifest, RunDir, create_run
from .split import split_for_training
from .state import RunState
from .stubs import StubAgent, default_agents


class PipelineState(TypedDict, total=False):
    config: RunConfig
    run: RunDir
    run_state: RunState
    manifest: Manifest
    split: SplitOutput
    preprocessing: PreprocessOutput
    model_selection: GeneratorOutput
    quality_check: QualityReport
    training: TrainingOutput
    selection_attempt: int


def config_for_dataset(dataset: str) -> RunConfig:
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {', '.join(DATASETS)}")
    source = DATASETS[dataset]
    return RunConfig(
        dataset_name=dataset, dataset_path=Path(source.file).resolve(),
        target=source.target, positive_class="yes" if dataset == "bank_marketing" else 1,
    )


def build_graph(agents=None):
    """Compile the visible stage graph; agents accept (typed request, RunDir)."""
    if agents is None:
        agents = default_agents()
        agents[Stage.SPLIT] = split_for_training
        agents[Stage.PREPROCESSING] = preprocess
        agents[Stage.QUALITY_CHECK] = run_quality
        agents[Stage.TRAINING] = run_training
        agents[Stage.EVALUATION] = run_evaluation
        # Default to the real selector while preserving explicitly supplied test agents.
        if isinstance(agents[Stage.MODEL_SELECTION], StubAgent):
            agents[Stage.MODEL_SELECTION] = run_model_selection
    if set(agents) != set(Stage):
        raise ValueError("provide one agent for every stage")

    def execute(stage, request, state, *, accept_skipped=False):
        run = state["run"]
        run_state = state["run_state"]
        manifest = state["manifest"]
        for retry in range(2):
            run_state = run_state.start(stage)
            run_state.save(run.state_path)
            try:
                result = STAGE_IO[stage][1].model_validate(agents[stage](request, run))
                if result.status is Status.SKIPPED and accept_skipped:
                    if stage is not Stage.MODEL_SELECTION:
                        raise ValueError("only model selection may return SKIPPED")
                    next_manifest = manifest.register(run, stage, result)
                    next_manifest.save(run)
                    run_state = run_state.finish(stage, Status.SKIPPED, result.reason)
                    run_state = run_state.skip(Stage.QUALITY_CHECK, result.reason)
                    run_state = run_state.with_gan(False, result.reason)
                    run_state.save(run.state_path)
                    log_decision(
                        "orchestrator", "selection_skipped",
                        {"stage": stage.value, "attempt": run_state.attempts(stage)},
                        "SKIPPED", result.reason, run=run,
                    )
                    return {
                        "run_state": run_state, "manifest": next_manifest, stage.value: result,
                    }
                if result.status is not Status.SUCCESS:
                    raise ValueError(f"{stage.value} returned {result.status}: {result.reason}")
                next_manifest = manifest.register(run, stage, result)
                next_manifest.save(run)
                manifest = next_manifest
                run_state = run_state.finish(stage, Status.SUCCESS)
                run_state.save(run.state_path)
                log_decision("orchestrator", "stage_completed",
                             {"stage": stage.value, "attempt": run_state.attempts(stage)},
                             "SUCCESS", "Valid output and registered artifacts", run=run)
                return {"run_state": run_state, "manifest": manifest, stage.value: result}
            except Exception as exc:
                outcome = "RETRY" if retry == 0 else "FAILED"
                log_decision("orchestrator", "stage_error",
                             {"stage": stage.value, "attempt": run_state.attempts(stage)},
                             outcome, str(exc), run=run)
                run_state = run_state.finish(
                    stage, Status.RETRY if retry == 0 else Status.FAILED, str(exc)
                )
                run_state.save(run.state_path)
        return {"run_state": run_state, "manifest": manifest}

    def split(state):
        return execute(Stage.SPLIT, state["config"], state)

    def preprocessing(state):
        return execute(Stage.PREPROCESSING, PreprocessInput(
            config=state["config"], split=state["split"]
        ), state)

    def model_selection(state):
        attempt = state.get("selection_attempt", 0) + 1
        previous = state.get("quality_check") if attempt > 1 else None
        result = execute(Stage.MODEL_SELECTION, ModelSelectionInput(
            config=state["config"], preprocess=state["preprocessing"],
            attempt=attempt, previous_report=previous,
        ), state, accept_skipped=True)
        return {**result, "selection_attempt": attempt}

    def quality_check(state):
        result = execute(Stage.QUALITY_CHECK, QualityInput(
            config=state["config"], preprocess=state["preprocessing"],
            generator=state["model_selection"],
        ), state)
        quality = result.get("quality_check")
        if quality is None:
            return result
        attempt = state["selection_attempt"]
        if quality.verdict is Verdict.PASS:
            run_state = result["run_state"].with_gan(True)
            outcome, reason = "ACCEPT", "At least one synthetic level passed"
        else:
            exhausted = attempt >= min(state["config"].max_attempts, 3)
            run_state = result["run_state"]
            if exhausted:
                run_state = run_state.with_gan(False, "quality gate failed after all attempts")
            outcome = "NOT_AVAILABLE" if exhausted else "RETRY_MODEL_SELECTION"
            reason = "No synthetic level passed"
        run_state.save(state["run"].state_path)
        log_decision("orchestrator", "quality_gate", {"attempt": attempt}, outcome,
                     reason, run=state["run"])
        return {**result, "run_state": run_state}

    def training(state):
        quality = state.get("quality_check")
        accepted = []
        if quality is not None and quality.verdict is Verdict.PASS:
            accepted = [item for item in state["model_selection"].synthetic_sets
                        if item.fill_level in quality.passing_levels]
        return execute(Stage.TRAINING, TrainingRequest(
            config=state["config"], raw=state["preprocessing"].raw,
            baseline=state["preprocessing"].baseline, accepted_synthetic=accepted,
            treatment_plan_path=state["preprocessing"].treatment_plan_path,
            treatment_plan_sha256=state["preprocessing"].treatment_plan_sha256,
        ), state)

    def evaluation(state):
        run = state["run"]
        return execute(Stage.EVALUATION, EvaluationInput(
            config=state["config"], training=state["training"],
            quality=state.get("quality_check"),
            treatment_log_path=state["preprocessing"].treatment_log_path,
            decision_log_path=run.decisions_path, manifest_path=run.manifest_path,
            state_path=run.state_path,
        ), state)

    def partial_report(state):
        run_state = state["run_state"]
        failed = run_state.failed_stage
        report = state["run"].subdir("report") / "partial_report.md"
        report.write_text(f"# Partial run report\n\nFailed stage: {failed.value}\n"
                          f"Reason: {run_state.stages[failed].reason}\n\n"
                          "Decision log: ../decisions.jsonl\n", encoding="utf-8")
        return {}

    def route(next_stage):
        return lambda state: "partial_report" if state["run_state"].failed_stage else next_stage

    def after_model_selection(state):
        if state["run_state"].failed_stage:
            return "partial_report"
        if state["run_state"].status(Stage.MODEL_SELECTION) is Status.SKIPPED:
            return "training"
        return "quality_check"

    def after_quality(state):
        if state["run_state"].failed_stage:
            return "partial_report"
        if state["quality_check"].verdict is Verdict.PASS:
            return "training"
        if state["selection_attempt"] < min(state["config"].max_attempts, 3):
            return "model_selection"
        return "training"

    graph = StateGraph(PipelineState)
    for name, node in (
        ("split", split), ("preprocessing", preprocessing),
        ("model_selection", model_selection), ("quality_check", quality_check),
        ("training", training), ("evaluation", evaluation),
        ("partial_report", partial_report),
    ):
        graph.add_node(name, node)
    graph.add_edge(START, "split")
    for current, following in (
        ("split", "preprocessing"),
        ("preprocessing", "model_selection"),
        ("training", "evaluation"),
        ("evaluation", END),
    ):
        graph.add_conditional_edges(current, route(following), [following, "partial_report"])
    graph.add_conditional_edges(
        "model_selection", after_model_selection,
        ["quality_check", "training", "partial_report"],
    )
    graph.add_conditional_edges(
        "quality_check", after_quality,
        ["training", "model_selection", "partial_report"],
    )
    graph.add_edge("partial_report", END)
    return graph.compile()


def run_pipeline(config: RunConfig, *, runs_dir: Path = Path("runs"), agents=None):
    """Run the compiled graph and return (RunState, RunDir)."""
    graph = build_graph(agents)
    run = create_run(config, run_id=f"{config.dataset_name}_{uuid4().hex[:12]}", runs_dir=runs_dir)
    result = graph.invoke({
        "config": config, "run": run, "run_state": run.load_state(),
        "manifest": run.load_manifest(), "selection_attempt": 0,
    })
    return result["run_state"], run
