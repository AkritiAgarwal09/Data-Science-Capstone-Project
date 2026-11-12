"""The pipeline finishes on deterministic defaults when the LLM does not behave (issue #86).

The real agents and the real `VertexClient` run. Only Google's credentials and the OpenAI client
inside `VertexClient` are replaced, so a failure takes the same path as in a live run. Each failure
must end in the agent's deterministic default, an `LLM_FALLBACK` line in `decisions.jsonl`, and a
run that completes.

Show it live with `uv run pytest tests/test_llm_failures.py -v`.
"""

import json
from types import SimpleNamespace

import httpx
import openai
import pandas as pd
import pytest
from google.auth.exceptions import DefaultCredentialsError, RefreshError
from pydantic import BaseModel
from test_quality_agent import V1_DRIVERS, v1  # noqa: F401  (v1 is a fixture)
from test_selection import (  # noqa: F401
    _bank_frame,
    _config,
    _request,
    _retry_lines,
    _retry_request,
    gan,
)
from test_training_agent import experiment, make_experiment, planning, tiny_suite  # noqa: F401

import agentic_gan.agents.quality as quality_module
from agentic_gan.agents import training
from agentic_gan.agents.model_selection import CTGAN_DEFAULTS, run_model_selection
from agentic_gan.llm import VertexClient
from agentic_gan.orchestrator import graph
from agentic_gan.orchestrator.decisions import read_decisions
from agentic_gan.schemas import Status, Verdict
from agentic_gan.tools.ctgan_sampling import recipe_for


def raise_timeout(request):
    raise openai.APITimeoutError(request=httpx.Request("POST", "https://aiplatform.googleapis.com"))


# Valid JSON for the step that asked, but a value that step never allowed.
OUTSIDE_THE_ALLOWED_OPTIONS = {
    "TreatmentChoice": {"treatment": "drop_every_column", "reason": "Because I said so"},
    "AllowedTrainingPlan": {
        "smote_levels": [0.99],
        "gan_levels": [0.99],
        "seeds": [999],
        "repeated_cv": False,
        "reason": "More is better",
    },
    "GroundedRunSummary": {"summary": "The ctgan arm reached a test F1 of 0.987654."},
    "VerifiedNarrative": {
        "summary": "PR-AUC rose to 0.987654.",
        "improvements": "Everything improved.",
        "segments": "Every segment improved.",
        "negative_results": "None.",
        "limitations": "None.",
    },
    "QualityDiagnosis": {
        "failed_checks": ["not_a_check"],
        "columns": ["not_a_column"],
        "likely_cause": "Unknown.",
        "adjustments": ["retrain_everything"],
    },
    "SelectionExplanation": {
        "generator": "wgan_gp",
        "explanation": "Always use something else.",
    },
    "RetrySettings": {
        "epochs": 1000000,
        "pac": 3,
        "discriminator_steps": 99,
        "batch_size": 7,
        "reason": "Because I said so",
    },
}


def outside_the_allowed_options(request):
    name = request["response_format"]["json_schema"]["name"]
    return json.dumps(OUTSIDE_THE_ALLOWED_OPTIONS.get(name, {}))


SCENARIOS = {
    "invalid_output": {"reply": lambda request: "Sorry, I can't answer in JSON today."},
    "unexpected_fields": {"reply": lambda request: json.dumps({"verdict": "PASS"})},
    "timeout": {"reply": raise_timeout},
    "expired_credential": {
        "refresh_error": RefreshError("Token has expired and could not refresh")
    },
    "no_credentials": {"no_credentials": True},
    "outside_the_allowed_options": {"reply": outside_the_allowed_options},
}


@pytest.fixture
def vertex(monkeypatch):
    """Let `VertexClient` build itself from fake settings, then misbehave as the test says.

    `attempts` counts clients built, one per call; `requests` counts calls that reached the model.
    """
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    monkeypatch.setenv("VERTEX_MODEL", "google/gemini-test")
    seen = SimpleNamespace(attempts=0, requests=0)

    def install(*, reply=None, refresh_error=None, no_credentials=False):
        def refresh(_):
            if refresh_error:
                raise refresh_error

        def default(scopes=None):
            seen.attempts += 1
            if no_credentials:
                raise DefaultCredentialsError("no application default credentials")
            return SimpleNamespace(token="token", refresh=refresh), "test-project"

        def create(**request):
            seen.requests += 1
            content = reply(request)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

        class FakeOpenAI:
            def __init__(self, **kwargs):
                seen.client_kwargs = kwargs
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        monkeypatch.setattr("agentic_gan.llm.vertex.google.auth.default", default)
        monkeypatch.setattr("agentic_gan.llm.vertex.OpenAI", FakeOpenAI)
        return seen

    return install


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_pipeline_completes_on_defaults_when_every_llm_call_fails(
    tmp_path, monkeypatch, vertex, scenario
):
    monkeypatch.setattr(training.baselines, "build_models", tiny_suite)
    request, *_ = make_experiment(tmp_path, missing=True)
    config = request.config.updated(training_cv_folds=2, training_cv_repeats=1, seeds=[42])
    seen = vertex(**SCENARIOS[scenario])
    monkeypatch.chdir(tmp_path)

    state, run = graph.run_pipeline(config, runs_dir="pipeline_runs")

    assert state.is_complete, run.decisions_path.read_text()
    decisions = read_decisions(run)
    steps = decisions.query("rule == 'llm_step'")
    # The tiny Default fixture has no GAN route, so there is no quality or training LLM step here;
    # the tests below cover those two. Other agents may add steps (such as a run narrative).
    assert set(steps.agent) >= {"preprocessing", "evaluation"}
    assert (steps.outcome == "LLM_FALLBACK").all()
    # One try and one retry per step, then the default.
    assert seen.attempts == 2 * len(steps)
    # Preprocessing used the first allowed treatment of every column: the rule's own default.
    allowed = None
    for row in decisions.itertuples():
        if row.agent == "preprocessing" and row.rule == "llm_step":
            allowed = row.inputs["allowed_options"]["treatment"]
        elif row.rule == "column_treatment":
            assert row.outcome == allowed[0]
    report = (run.subdir("report") / "final_report.md").read_text()
    assert "Negative or inconclusive results" in report
    assert run.load_manifest().verify(run) == []


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_quality_diagnosis_falls_back_to_the_facts_and_keeps_the_verdict(v1, vertex, scenario):  # noqa: F811
    request, run = v1
    seen = vertex(**SCENARIOS[scenario])

    report = quality_module.run_quality(request, run)

    assert report.verdict is Verdict.FAIL
    assert set(V1_DRIVERS) <= set(report.diagnosis.columns)
    assert "AUC 0.86" in report.diagnosis.likely_cause
    assert read_decisions(run).query("rule == 'llm_step'").outcome.tolist() == ["LLM_FALLBACK"]
    assert seen.attempts == 2


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_model_selection_explanation_falls_back_and_keeps_the_rule(
    tmp_path, gan, vertex, scenario,  # noqa: F811
):
    seen = vertex(**SCENARIOS[scenario])
    request, run, _path = _request(tmp_path, _config(tmp_path), _bank_frame(), 500, 1500)
    output = run_model_selection(request, run)
    assert output.status is Status.SUCCESS
    assert output.generator == "ctgan"
    # the failed explanation changes nothing: attempt 1 keeps the dataset's recipe (Bank: V2)
    recipe = recipe_for("bank_marketing")
    assert {name: output.hyperparameters[name] for name in CTGAN_DEFAULTS} == {
        name: getattr(recipe, name) for name in CTGAN_DEFAULTS
    }
    steps = read_decisions(run).query("rule == 'llm_step'")
    assert steps.outcome.tolist() == ["LLM_FALLBACK"]
    assert set(steps.agent) == {"model_selection"}
    assert seen.attempts == 2


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_model_selection_retry_falls_back_to_the_rule_epochs(
    tmp_path, gan, vertex, scenario,  # noqa: F811
):
    seen = vertex(**SCENARIOS[scenario])
    request, run, _path = _retry_request(tmp_path, {"epochs": (100, 600)}, "column_distance")
    output = run_model_selection(request, run)
    assert output.status is Status.SUCCESS
    assert gan.trains[-1][1]["epochs"] == 400
    retry = _retry_lines(run)
    assert len(retry) == 1
    assert retry.iloc[0].inputs["source"] == "ladder"
    steps = read_decisions(run).query("rule == 'llm_step'")
    assert steps.outcome.tolist() == ["LLM_FALLBACK"]
    assert set(steps.agent) == {"model_selection"}
    assert seen.attempts == 2


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_training_falls_back_and_never_uses_a_failed_level(planning, vertex, scenario):  # noqa: F811
    request, run, quality = planning  # level 0.25 failed the gate, 0.5 passed
    seen = vertex(**SCENARIOS[scenario])

    result = training.run_training(request, run, quality=quality)

    assert result.status is Status.SUCCESS
    steps = read_decisions(run).query("rule == 'llm_step'")
    assert steps.agent.tolist() == ["training", "training"]  # the plan, then the summary
    assert (steps.outcome == "LLM_FALLBACK").all()
    assert seen.attempts == 4
    metrics = pd.read_csv(result.metrics_path)
    assert set(metrics.query("arm == 'ctgan'").fill_level) == {0.5}
    assert set(metrics.query("arm == 'smotenc'").fill_level) == {0.25, 0.5}
    assert (run.subdir("models") / "run_summary.md").read_text().startswith("default: ")


def test_a_stalled_call_is_cut_off_after_a_bounded_time(vertex, monkeypatch):
    seen = vertex(reply=lambda request: "{}")

    class Answer(BaseModel):
        text: str = ""

    VertexClient().complete("system", "context", Answer)
    assert seen.client_kwargs["timeout"] == 90
    monkeypatch.setenv("VERTEX_TIMEOUT_SECONDS", "15")
    VertexClient().complete("system", "context", Answer)
    assert seen.client_kwargs["timeout"] == 15
