# Shared LLM runtime

`run_llm_step` is the one entry point for agent LLM choices. It requests Pydantic JSON
from Gemini on Vertex, validates allowed values and numeric bounds, checks the run's
rule hash before and after the call, retries once, and logs either the answer, a
fallback, or a rejected rule change to `decisions.jsonl`. The graph still decides
stage routes and quality gates in Python.

```python
from agentic_gan.llm import LLMStepOptions, run_llm_step

answer = run_llm_step(
    role="preprocessing",
    system_prompt="Choose a permitted imputation setting and explain why.",
    context="Profile of training rows only: ...",
    output_model=MyPydanticModel,
    options=LLMStepOptions(
        fallback=MyPydanticModel(method="median"),
        allowed_options={"method": ("median", "mean")},
        numeric_bounds={},
        expected_rule_hash=locked_hash,
        current_rule_hash=lambda: rule_hash(run.load_config()),
    ),
    run=run,
)
```

Pass a `FakeLLM([...])` as `client=` in tests. Each scripted answer is a mapping or
exception. No credentials or network are used. Extra output fields, choices outside
the allowed set, and values outside bounds trigger one retry, then the fixed fallback.
A changed rule hash raises `RuleOverrideError` and is logged; it never becomes an
accepted answer.

Mark a test that calls the real model with `@pytest.mark.live`. Live tests are deselected by
default (`addopts` in `pyproject.toml`); run them with `uv run pytest -m live` once the Vertex
setup below works. Use at most one per agent.

## Vertex setup

From the repo root, install the locked dependencies with `uv sync`. Export the team
project, supported Vertex location, and model (the spike used the values below):

```sh
export GOOGLE_CLOUD_PROJECT='<team-project-id>'
export GOOGLE_CLOUD_LOCATION='global'
export VERTEX_MODEL='google/gemini-3.5-flash'
gcloud auth application-default login
uv run python -m agentic_gan.llm
```

The final command makes one structured live call and prints its confirmation. It
uses Application Default Credentials and refreshes the access token before each
model call. Do not commit credentials or keys. Each teammate should run this check
with their own Vertex access and report the result on issue #73.
