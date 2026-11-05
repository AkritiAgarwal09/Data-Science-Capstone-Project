"""No-network contract tests for the shared LLM step."""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from agentic_gan.llm import FakeLLM, LLMStepOptions, RuleOverrideError, VertexClient, run_llm_step
from agentic_gan.orchestrator.decisions import read_decisions
from agentic_gan.orchestrator.manifest import create_run
from agentic_gan.schemas import RunConfig


class Choice(BaseModel):
    action: str
    fraction: float


@pytest.fixture
def run(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("x,y\n0,no\n1,yes\n")
    config = RunConfig(
        dataset_name="bank_marketing", dataset_path=source, target="y", positive_class="yes"
    )
    return create_run(config, runs_dir=tmp_path / "runs")


def options(current=lambda: "locked"):
    return LLMStepOptions(
        fallback=Choice(action="keep", fraction=0.5),
        expected_rule_hash="locked",
        current_rule_hash=current,
        allowed_options={"action": ("keep", "trim")},
        numeric_bounds={"fraction": (0, 1)},
    )


def test_valid_answer_is_typed_and_logged(run):
    fake = FakeLLM([{"action": "trim", "fraction": 0.25}])
    result = run_llm_step("preprocessing", "system", "context", Choice, options(), run, client=fake)
    assert result == Choice(action="trim", fraction=0.25)
    assert len(fake.calls) == 1
    decision = read_decisions(run).iloc[0]
    assert decision.inputs["context"] == "context"
    assert decision.outcome == {"action": "trim", "fraction": 0.25}


@pytest.mark.parametrize(
    "answer",
    [
        {"action": "replace_rule", "fraction": 0.5},
        {"action": "keep", "fraction": 1.5},
        {"action": "keep", "fraction": 0.5, "thresholds": {"privacy": 999}},
    ],
)
def test_out_of_bounds_or_extra_fields_fall_back_after_one_retry(run, answer):
    fake = FakeLLM([answer, answer])
    result = run_llm_step("preprocessing", "system", "context", Choice, options(), run, client=fake)
    assert result == Choice(action="keep", fraction=0.5)
    assert len(fake.calls) == 2
    assert read_decisions(run).outcome.tolist() == ["LLM_FALLBACK"]


def test_changed_rule_hash_is_rejected_and_logged(run):
    current = ["locked"]

    class MutatingLLM:
        def complete(self, *args):
            current[0] = "changed"
            return {"action": "keep", "fraction": 0.5}

    with pytest.raises(RuleOverrideError):
        run_llm_step(
            "preprocessing",
            "system",
            "context",
            Choice,
            options(lambda: current[0]),
            run,
            client=MutatingLLM(),
        )
    assert read_decisions(run).outcome.tolist() == ["RULE_OVERRIDE_REJECTED"]


def test_changed_rule_hash_is_rejected_even_when_provider_fails(run):
    current = ["locked"]

    class MutatingLLM:
        def complete(self, *args):
            current[0] = "changed"
            raise RuntimeError("provider failed")

    with pytest.raises(RuleOverrideError):
        run_llm_step("preprocessing", "system", "context", Choice,
                     options(lambda: current[0]), run, client=MutatingLLM())
    assert read_decisions(run).outcome.tolist() == ["RULE_OVERRIDE_REJECTED"]


def test_vertex_refreshes_adc_before_each_structured_call(monkeypatch):
    import agentic_gan.llm.vertex as vertex

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "team-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    monkeypatch.setenv("VERTEX_MODEL", "google/gemini-test")
    credentials = SimpleNamespace(token=None, refresh=lambda request: None)
    refreshes, clients, requests = [], [], []

    def refresh(request):
        refreshes.append(request)
        credentials.token = f"token-{len(refreshes)}"

    credentials.refresh = refresh
    monkeypatch.setattr(vertex.google.auth, "default", lambda **kwargs: (credentials, None))
    monkeypatch.setattr(vertex, "Request", lambda: object())

    def openai(**kwargs):
        clients.append(kwargs)

        def create(**request):
            requests.append(request)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"action":"keep","fraction":0.5}')
                    )
                ]
            )

        class Client:
            chat = SimpleNamespace(completions=SimpleNamespace(create=create))

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        return Client()

    monkeypatch.setattr(vertex, "OpenAI", openai)
    client = VertexClient()
    assert client.complete("system", "context", Choice)["action"] == "keep"
    assert client.complete("system", "context", Choice)["action"] == "keep"
    assert [item["api_key"] for item in clients] == ["token-1", "token-2"]
    assert all(item["base_url"].endswith("/locations/global/endpoints/openapi") for item in clients)
    assert all(request["temperature"] == 0 for request in requests)
    assert all(request["response_format"]["type"] == "json_schema" for request in requests)
