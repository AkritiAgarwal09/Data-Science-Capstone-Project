# Issue #82: Model-selection agent (LLM) — change log

Branch: `arrico/model-selection-agent` (created from `origin/main` at `b66b2b2`, PR #116).

This file records every change made on the branch for issue #82. It is append-only: each
work session adds a new dated entry at the bottom. Earlier entries are never edited or removed,
even if a later entry reverses them. To find the current state of something, read from the
bottom up.

## How to read this file

- **Entry** = one block of work, with the date and the files it touched.
- **Why** explains the reason in plain language; **What** lists the concrete changes.
- **Verified** lists the commands run and their results at the time of the entry.
- Nothing here is committed unless an entry says so.

## The issue in one paragraph

The rule that picks a generator (`tools/selection.py`) already existed and is read-only. This
issue wraps two bounded LLM steps around it inside `agents/model_selection.py`: one that
explains the rule's outcome in plain language, and one that proposes new CTGAN settings for a
retry after the quality gate fails. The LLM never chooses the generator, never changes a rule or
a threshold, and may only pick values inside `config.gan_bounds`. Every LLM answer is validated,
logged with `log_decision`, and replaced by a deterministic default (logged as `LLM_FALLBACK`)
when it is invalid or the call fails.

---

## Entry 1 — 2026-11-05 — Branch setup

**Files:** none (git only)

**Why:** The local `main` was stale (at PR #65) and did not yet contain `tools/selection.py`,
the shared LLM runtime (`llm/runtime.py`, issue #73), or the quality diagnosis (issue #76) that
this issue builds on.

**What:**
- Created branch `arrico/model-selection-agent`.
- Reset it to `origin/main` (`b66b2b2`) and moved local `main` to the same commit.

**Verified:**
- `uv run --no-sync pytest tests/test_selection.py -q` → 29 passed (baseline before any code
  change).

---

## Entry 2 — 2026-11-05 — Explanation step, retry proposer, skipped-route logging

**Files:**
- `src/agentic_gan/agents/model_selection.py` (modified)
- `tests/test_selection.py` (tests appended; no existing test edited)

**Not touched (hard constraints of the issue):** `src/agentic_gan/schemas.py`,
`src/agentic_gan/orchestrator/graph.py`, `src/agentic_gan/tools/selection.py`,
`src/agentic_gan/tools/ctgan.py`.

### Why

The agent already selected a generator, logged each rule step, trained CTGAN, and had a
rule-based retry (+100 epochs inside `gan_bounds` unless a privacy check failed). It had no LLM
involvement and did not say, per route, why TimeGAN or WGAN-GP were not used.

### What — `agents/model_selection.py`

**Entry point**
- `run_model_selection(request, run)` → `run_model_selection(request, run, *, client=None)`.
  The graph still calls it as `agent(request, run)`; with no client both LLM steps fall back
  and log `LLM_FALLBACK`. Tests pass a `FakeLLM`.
- Module docstring expanded to describe the two LLM steps and what they may not do.

**New constants**
- `ROLE = Stage.MODEL_SELECTION.value` (the agent name in the decision log).
- `TUNABLE = ("epochs", "batch_size", "pac", "discriminator_steps")`.
- `ROUTES = ("timegan", "wgan_gp", "ctgan")`.
- `EXPLAIN_PROMPT`, `PROPOSAL_PROMPT` — system prompts for the two steps.

**New Pydantic output models** (kept here because `schemas.py` is read-only for this issue)
- `SelectionExplanation(summary: str, decisive_rule: str)` — both non-empty.
- `CTGANProposal(epochs, batch_size, pac, discriminator_steps: int ≥ 1, rationale: str)` with a
  validator requiring `batch_size % pac == 0`, and a `.settings()` helper returning the four
  numbers as a dict.

**(A) Explanation step**
- `decisive_rule(result)` — public helper: the rule step that settled the outcome (the failing
  gate, a `PASS` temporal test, or the `SELECTED` type-routing step).
- `_default_explanation(result)` — fallback: the rationale of every reached step, joined.
- `_explain_context(result, request)` — JSON of dataset, attempt, status, generator, trainable,
  fallback, the ordered steps (rule, outcome, rationale, and only scalar inputs — lists and dicts
  such as column names are dropped), and the skipped routes. No table rows ever go in.
- `_explain(result, request, run, client)` — calls `run_llm_step` with
  `allowed_options={"decisive_rule": <rules whose outcome is not NOT_REACHED>}`. Logs one
  decision with rule `explanation`, outcome `EXPLAINED`, the summary as rationale, and inputs
  `{attempt, generator, decisive_rule, rule_text_used}`.
- `GeneratorOutput.rationale` is unchanged (still the rule's own text), so the code remains the
  source of truth for the choice.

**(B) Retry proposer** (attempt 2 only, as before)
- `_rule_retry(settings, failed, bounds)` — the old deterministic logic, now silent (no logging)
  and returning `(settings, rationale)`. Used as the LLM fallback.
- `_proposal_bounds(settings, bounds, privacy)` — the numeric bounds the LLM must respect:
  `gan_bounds[name]` rounded to ints; a setting with no config bound is pinned to
  `(current, current)`; after a privacy hard-check failure `epochs` may not rise above its
  current value.
- `_proposal_context(...)` — JSON of dataset, attempt, failed hard checks, the
  `QualityDiagnosis` from the previous report (or `null`), previous settings, allowed ranges.
- `_retry_settings(settings, request, run, client)` — rewritten: builds the fallback from
  `_rule_retry`, calls `run_llm_step` with `numeric_bounds` from `_proposal_bounds`, then runs
  `_clamp` and a range/divisibility check as a second guard (logs `LLM_FALLBACK` if it trips),
  and logs the final settings once with the existing `hyperparameter_retry` (ADJUSTED) or
  `retry_unchanged` (UNCHANGED) rule.
- `_log_retry(...)` — inputs now also include `previous_settings`, `settings`, `rule_settings`
  and all `bounds`; the old `previous_epochs` / `epochs` / `bound` keys are kept.

**Skipped routes**
- `skipped_routes(result)` — public helper: for each of `timegan`, `wgan_gp`, `ctgan` that will
  not be trained, the reason from the rule (gate reason; temporal-test reason; type-routing
  reason; or "not implemented; documented fallback is …").
- `_log_skipped_routes(result, attempt, run)` — one decision per route with rule
  `route_skipped`, outcome `SKIPPED`, inputs `{attempt, generator, selected}`.

**Rule lock**
- `_rule_hash(config)` — local fingerprint of `gan_bounds`, `min_rows`, `max_attempts`,
  `fill_levels`. Passed to `LLMStepOptions` as `expected_rule_hash`, with
  `current_rule_hash=lambda: _rule_hash(run.load_config())`. A change during a call raises
  `RuleOverrideError`, which is never caught here.
- `_options(fallback, request, run, **limits)` — small builder for `LLMStepOptions`.

**Decision-log lines added per run** (agent `model_selection`)
- `route_skipped` — one per unused route.
- `llm_step` — from the runtime, for each LLM call (validated answer or `LLM_FALLBACK`).
- `explanation` — always, once.
- On attempt 2: `hyperparameter_retry` or `retry_unchanged` (already existed; richer inputs).

### What — `tests/test_selection.py`

New imports: `json`, `os`, `CTGAN_DEFAULTS`, `decisive_rule`, `skipped_routes`, `FakeLLM`,
`RuleOverrideError`, `QualityDiagnosis`. 25 tests appended at the end of the file.

Precedence (pure rule, no I/O):
- `test_the_first_failing_gate_decides_and_the_rest_are_not_reached` (4 parametrized cases
  where several gates would fail at once; exactly one `FAIL`, everything after `NOT_REACHED`).
- `test_minority_gate_records_its_outcome_but_does_not_override_the_temporal_test`.
- `test_type_routing_cannot_run_once_the_temporal_test_selected_timegan`.

Skipped routes:
- `test_every_unused_route_has_a_reason`.
- `test_skipped_routes_are_logged_when_ctgan_trains`.
- `test_skipped_routes_are_logged_when_timegan_is_documented_only`.

Explanation step (FakeLLM):
- `test_explanation_comes_from_the_fake_llm_and_sees_only_summaries` (also checks the prompt
  has the step order and contains no row values).
- `test_explanation_falls_back_to_the_rule_text_when_the_llm_is_down`.
- `test_explanation_outside_the_logged_steps_is_rejected_and_the_choice_stands` (4 cases:
  NOT_REACHED rule, non-rule, extra `generator` field, empty summary).
- `test_explanation_runs_when_a_gate_stops_selection`.

Retry proposer (FakeLLM):
- `test_proposal_inside_bounds_is_used_and_logged`.
- `test_proposal_sees_the_diagnosis_and_the_ranges_but_no_rows`.
- `test_proposal_outside_bounds_is_rejected_and_the_rule_step_is_used` (the required
  out-of-bounds case: `epochs=5000` → `LLM_FALLBACK`, rule's 400 is trained).
- `test_a_setting_without_a_config_bound_cannot_move`.
- `test_batch_size_not_divisible_by_pac_is_rejected`.
- `test_after_a_privacy_failure_the_llm_cannot_raise_epochs`.
- `test_after_a_privacy_failure_other_settings_may_still_move`.
- `test_retry_without_a_diagnosis_still_asks_and_falls_back_cleanly`.
- `test_widening_gan_bounds_during_the_llm_call_is_rejected` (`RuleOverrideError`).

Live:
- `test_live_gemini_explains_the_bank_selection` — `@pytest.mark.live`, skipped unless
  `RUN_LIVE_LLM=1`.

Every FakeLLM test asserts `len(fake.calls)` so an exhausted answer list cannot pass silently.

### Verified

| Command | Result |
| --- | --- |
| `uv run --no-sync ruff check src/agentic_gan/agents/model_selection.py tests/test_selection.py` | All checks passed |
| `uv run --no-sync pytest tests/test_selection.py -q` | 54 passed, 1 skipped (live) |
| `uv run --no-sync pytest -q --ignore=tests/tools` | 533 passed, 3 skipped |
| `uv run --no-sync pytest tests/tools/test_augmented.py` | 6 passed |
| `uv run --no-sync pytest tests/tools/test_ctgan.py` | 11 passed |
| `uv run --no-sync pytest tests/tools/test_smotenc.py` | 4 passed |

### Known, pre-existing, not caused by this entry

- Running all of `tests/tools` in one pytest process segfaults. Confirmed identical on
  unmodified `main` (torch/OpenMP collision). Each file passes on its own.
- `ruff format --check` already failed on `model_selection.py` before this entry (compact
  `log_decision(...)` calls). New lines were formatted; pre-existing lines were left alone to
  keep the diff focused. There is no CI workflow enforcing format.
- `uv run pytest` without `--no-sync` tried to rewrite the editable `.pth` file and failed in
  the sandbox used for this work; `--no-sync` was used throughout.

### Status at end of entry

Uncommitted, on the working tree of `arrico/model-selection-agent`.

---

## Entry 3 — 2026-11-05 — This change log

**Files:** `docs/notes/issue-82-model-selection-agent-changelog.md` (new)

**What:** Added this file. Rule for future edits: append a new entry at the bottom after each
change; never edit or delete earlier entries.

**Status:** Uncommitted.

---

## Entry 4 — 2026-11-05 — Bug fix: `exact_train_copies` was not treated as a privacy failure

**Commit:** `a5c9cfe` — "Treat exact_train_copies as a privacy failure in the epoch retry"

**Files:**
- `src/agentic_gan/tools/quality_privacy.py` (modified)
- `src/agentic_gan/agents/quality.py` (modified)
- `src/agentic_gan/agents/model_selection.py` (modified)
- `tests/test_selection.py` (tests added)

### Why

The retry rule in `model_selection.py` decided "a privacy check failed" by substring matching
against `PRIVACY_TOKENS = ("dcr", "duplicate", "privacy")`. The real hard privacy checks emitted
by `tools/quality_privacy.py` are `exact_train_copies`, `synthetic_duplicates`, `dcr_ratio`.
`exact_train_copies` matched no token, so a report failing both `column_distance` and
`exact_train_copies` still got **more** epochs — the opposite of the rule's intent (more training
means more copying of real rows).

### What

- `tools/quality_privacy.py`: new module-level constant
  `PRIVACY_CHECKS = {"exact_train_copies", "synthetic_duplicates", "dcr_ratio"}` with a one-line
  comment. This is now the single source of truth for the names.
- `agents/quality.py`: deleted its local `PRIVACY_CHECKS` and imports it from the tool. No
  behavior change.
- `agents/model_selection.py`: imports `PRIVACY_CHECKS` from the tool; removed
  `PRIVACY_TOKENS`; added `_privacy_failed(failed) -> bool` =
  `bool(set(failed) & PRIVACY_CHECKS)`, used wherever the retry asked "did privacy fail"
  (the deterministic rule and, in the uncommitted Entry 2 work, the LLM epoch cap).
  `_matches` / `FIDELITY_TOKENS` are unchanged and still used for fidelity.
- `tests/test_selection.py`: `test_exact_train_copies_does_not_add_epochs` and
  `test_every_privacy_check_name_blocks_more_epochs` (parametrized over all three names).
  Both assert trained epochs stay at 300 and `retry_unchanged == "UNCHANGED"`. Checked that the
  `exact_train_copies` cases fail against the old substring logic and pass with the fix.

### How it was committed

The working tree also held the uncommitted Entry 2 work in the same two files. To keep this
commit a standalone fix, the current files were copied aside, tracked files reverted to `HEAD`,
the fix re-applied on the clean files, tested, and committed; then the full files were copied
back. The working tree is therefore `a5c9cfe` + the Entry 2 work, and the Entry 2 diff already
contains this fix.

### Verified

| Command | Result |
| --- | --- |
| `python -c "import agentic_gan.agents.model_selection, agentic_gan.agents.quality, agentic_gan.orchestrator.graph"` | imports ok (no circular import) |
| `uv run --no-sync ruff check src tests` | All checks passed |
| `uv run --no-sync pytest tests/test_selection.py tests/test_quality_agent.py tests/test_quality_privacy.py -q` (fix only, before restoring Entry 2) | 53 passed, 1 skipped |
| same command after restoring Entry 2 work | 78 passed, 2 skipped |

### Status at end of entry

- Committed: `a5c9cfe` (this fix only).
- Still uncommitted: Entry 2 (LLM steps) and Entry 3 (this file).

---

## Entry 5 — 2026-11-05 — Plan change: Entry 2 draft parked; stepwise build begins

**Files:** none (git only)

**Why:** The work is now being done as a sequence of small, individually specified and committed
steps (Step 1 = Entry 4, Step 2 = Entry 6, Steps 3 and 4 to follow). Step 2 requires "no behavior
change yet", which the uncommitted Entry 2 draft contradicted: it already wired both LLM steps
and defined a `SelectionExplanation` with different fields.

**What:**
- The Entry 2 draft (`model_selection.py` and `test_selection.py` changes only) was moved to a
  git stash so nothing is lost:
  `stash@{0}: issue-82 draft: LLM explain/proposer first pass (superseded by stepwise plan)`.
  Recover with `git stash show -p stash@{0}` (read) or `git stash apply stash@{0}` (apply).
- The Entry 2 description above stays as written; it documents the draft, not the final design.
  Where the stepwise design differs, the later entries win.

**Useful pieces in the stash for Steps 3 and 4:** `skipped_routes()`, `decisive_rule()`, the
`_proposal_bounds()` logic (pin unbounded settings, cap epochs after a privacy failure), the
post-LLM `_clamp` + divisibility guard, and the FakeLLM tests (in-bounds, out-of-bounds,
pinned setting, non-divisible batch, privacy cap, missing diagnosis, bounds edited mid-call).

---

## Entry 6 — 2026-11-05 — Step 2: output models, rule hash, prompts (no behavior change)

**Commit:** `1ed4270` — "Add LLM output models and rule hash for model selection"

**Files:**
- `src/agentic_gan/agents/model_selection.py` (additions only)
- `tests/test_selection.py` (tests appended; import line expanded)

### Why

Steps 3 and 4 will add the explanation step and the retry proposer. This step lands the pieces
they share, so each later step is small: the two Pydantic answer shapes, the fingerprint of what
the LLM may not change, and the two system prompts. Nothing calls any of it yet.

### What — `agents/model_selection.py`

- New imports: `hashlib`, `inspect`, `json`, `pydantic.Field`, `RunConfig` and `Schema` from
  `agentic_gan.schemas`, `STEP_ORDER` from `tools.selection`. (The spec also listed
  `dataclasses.asdict`, `LLMStepOptions`, `run_llm_step`, `Stage`; these are unused until Steps 3
  and 4 and would fail `ruff` F401 now, so they are deferred.)
- `SETTING_NAMES = ("epochs", "pac", "discriminator_steps", "batch_size")`.
- `EXPLAIN_PROMPT` and `RETRY_PROMPT` — verbatim from the step spec.
- `class SelectionExplanation(Schema)`: `generator: str` (must echo the rule's generator, or
  `"none"`), `explanation: str` (min length 1).
- `class RetrySettings(Schema)`: `epochs`, `pac`, `discriminator_steps`, `batch_size: int`,
  `reason: str` (min length 1).
  Both inherit `Schema`, so unknown fields are rejected and instances are frozen.
- `_rule_hash(config) -> str`: SHA-256 over `min_rows`, `gan_bounds`, `STEP_ORDER`,
  `CTGAN_DEFAULTS`, `EPOCH_STEP`, `sorted(PRIVACY_CHECKS)`, and the `minority_min` /
  `min_steps` defaults read from `select_generator`'s signature with `inspect`. JSON-dumped with
  sorted keys.

### What — `tests/test_selection.py` (4 tests appended)

- `test_rule_hash_is_stable_for_equal_configs_and_moves_with_the_locked_fields` — equal
  configs hash the same; changing `min_rows`, changing a bound, or adding a bound changes it.
- `test_rule_hash_survives_the_config_yaml_round_trip` — `_rule_hash(run.load_config())`
  equals `_rule_hash(config)` for a run made by `create_run`. Passed as written; no change to the
  hashing input was needed (tuples become JSON lists on both sides).
- `test_selection_explanation_rejects_unknown_fields_and_empty_text`.
- `test_retry_settings_reject_unknown_fields_and_an_empty_reason`.

### Verified

| Command | Result |
| --- | --- |
| `uv run --no-sync ruff check src tests` | All checks passed |
| `uv run --no-sync pytest tests/test_selection.py -q` | 37 passed |
| `uv run --no-sync pytest tests/test_orchestrator.py tests/test_quality_agent.py -q` | 26 passed, 1 skipped |
| `git diff tests/test_selection.py` removed lines | only the old one-line import (expanded to a block) |

### Status at end of entry

- Committed: `a5c9cfe`, `1ed4270`.
- Uncommitted: this file. Draft work parked in `stash@{0}` (see Entry 5).

---

## Entry 7 — 2026-11-05 — Step 3: the EXPLAIN step

**Commit:** `c30c51c` — "Add LLM explanation of the generator rule"

**Files:**
- `src/agentic_gan/agents/model_selection.py` (modified)
- `tests/test_selection.py` (one test appended; one import added)

### Why

First of the two LLM steps from the issue: after the rule has chosen (or declined to choose) a
generator and logged its steps, the LLM explains that outcome in plain language. The answer is
for the decision log only; the generator choice and `GeneratorOutput` are untouched.

### What — `agents/model_selection.py`

- Signature: `run_model_selection(request, run)` → `run_model_selection(request, run, *, client=None)`.
  The graph still calls it positionally, so nothing upstream changes. With no client the runtime
  logs `LLM_FALLBACK` and uses the rule's own rationale.
- New imports: `dataclasses.asdict`, `LLMStepOptions` / `run_llm_step` from `agentic_gan.llm`,
  `Stage` from `agentic_gan.schemas`.
- New `_explain(result, request, run, client) -> SelectionExplanation`:
  - context = JSON of `dataset`, `status`, `generator` (the rule's, or `"none"`), and every
    `DecisionStep` via `asdict` (rule, inputs, outcome, rationale). No table rows.
  - `allowed_options={"generator": (expected,)}` so any answer naming a different generator is
    rejected by the runtime.
  - fallback = `SelectionExplanation(generator=expected, explanation=result.rationale)`;
    verified that `select_generator` sets a non-empty `rationale` on every path.
  - rule lock via `_rule_hash` (expected from `request.config`, current from
    `run.load_config()`). `RuleOverrideError` is not caught.
- Call site: directly after `_log_steps(...)`, before the early `skipped` return, and only when
  `request.attempt == 1`. Return value ignored; the runtime already logged it under rule
  `llm_step`, agent `model_selection`.

### What — `tests/test_selection.py`

- Import `FakeLLM`.
- `test_first_attempt_asks_the_llm_to_explain_and_logs_the_answer`: attempt 1 with a FakeLLM
  answer; asserts one call, output generator still `ctgan`, and exactly one `llm_step` decision
  from agent `model_selection` whose outcome is the validated answer.
- Existing tests pass no client and now produce an `LLM_FALLBACK` line each; all still pass.
  The full explanation test suite is scheduled for Step 6.

### Verified

| Command | Result |
| --- | --- |
| `uv run --no-sync ruff check src tests` | All checks passed |
| `uv run --no-sync pytest tests/test_selection.py tests/test_orchestrator.py tests/test_llm_failures.py -q` | 70 passed |
| `git diff tests/test_selection.py` removed lines | none |

### Status at end of entry

- Committed: `a5c9cfe`, `1ed4270`, `c30c51c`.
- Uncommitted: this file. Draft work still parked in `stash@{0}`.

---

## Entry 8 — 2026-11-05 — Step 4A: split the rule retry from its logging (refactor only)

**Commit:** `fd9d439` — "Split the rule retry from its logging"

**Files:**
- `src/agentic_gan/agents/model_selection.py` (modified, +27/−14)

### Why

Step 4B will add the LLM retry proposer and needs the deterministic retry as its fallback. Until
now `_retry_settings` both computed the new settings and wrote the decision log, so it could not
be reused without logging twice. This step separates the two with no change in behaviour or in
what is logged.

### What

- New pure function `_rule_retry(settings, request) -> (new_settings, rule_name, rationale)`.
  Same logic as before: more epochs by `EPOCH_STEP` inside `gan_bounds["epochs"]` when a
  fidelity check failed, unless a privacy check failed or epochs have no bound / are at the
  limit. `rule_name` is `"hyperparameter_retry"` when epochs change, else `"retry_unchanged"`.
  The three rationale strings are the ones that existed before, verbatim. It does not log.
- `_retry_settings(settings, request, run)` is now a thin wrapper: calls `_rule_retry`, then the
  unchanged `_log_retry` with the same arguments (attempt, previous epochs, new epochs,
  bounds, failed checks, rationale) and so the same inputs keys (`attempt`, `previous_epochs`,
  `epochs`, `bound`, `failed_checks`) and outcomes (`ADJUSTED` / `UNCHANGED`).
- `run_model_selection` still calls `_retry_settings`; `_log_retry` untouched.

### Verified

| Command | Result |
| --- | --- |
| `uv run --no-sync ruff check src tests` | All checks passed |
| `uv run --no-sync pytest tests/test_selection.py -q -k "retry or privacy or exact_train_copies"` | 8 passed (the three original retry tests + the Step 1 tests) |
| `uv run --no-sync pytest tests/test_selection.py tests/test_orchestrator.py -q` | 51 passed |

### Status at end of entry

- Committed: `a5c9cfe`, `1ed4270`, `c30c51c`, `fd9d439`.
- Uncommitted: this file. Draft work still parked in `stash@{0}`.

---

## Entry 9 — 2026-11-05 — Step 4B: retry limits helper (pure function, not called yet)

**Commit:** `f9f0943` — "Add retry limits helper"

**Files:**
- `src/agentic_gan/agents/model_selection.py` (+28, additions only)
- `tests/test_selection.py` (+49, tests appended; two names added to the import block)

### Why

The retry proposer (Step 4C) must hand the LLM runtime an inclusive integer range per CTGAN
setting so that the runtime, not the prompt, enforces `gan_bounds`. This helper computes those
ranges. Nothing calls it yet.

### What — `agents/model_selection.py`

- `import math`.
- `_retry_limits(settings, gan_bounds, privacy) -> dict[str, tuple[int, int]] | None`, for each
  name in `SETTING_NAMES`:
  - bounded in `gan_bounds` → `(ceil(low), floor(high))`; if that leaves `low > high`, pin;
  - not bounded → pinned to `(current, current)`;
  - `privacy=True` → `epochs` high is capped at the current value (never raise epochs after a
    privacy failure); if that inverts the range, pin;
  - if the current value falls outside the range (should not happen after `_clamp`, but not
    assumed), pin;
  - returns `None` when every setting is pinned (nothing is tunable); all values are ints.

### What — `tests/test_selection.py` (6 unit tests, no GAN, no run)

- empty `gan_bounds` → `None` (with and without privacy);
- only `epochs` bounded `(100, 600)` → `epochs (100, 600)`, others pinned;
- same with `privacy=True` → `epochs (100, 300)`;
- fractional edges `(99.5, 600.4)` → `(100, 600)`, ints;
- all four bounded → all four ranges returned unchanged;
- a bound holding no integer (`pac (10.2, 10.8)`) and a bound excluding the current value
  (`epochs (400, 600)` with current 300) are both pinned, while a normal bound alongside stays open.

### Verified

| Command | Result |
| --- | --- |
| `uv run --no-sync ruff check src tests` | All checks passed |
| `uv run --no-sync pytest tests/test_selection.py -q` | 44 passed |

### Status at end of entry

- Committed: `a5c9cfe`, `1ed4270`, `c30c51c`, `fd9d439`, `f9f0943`.
- Uncommitted: this file. Draft work still parked in `stash@{0}`.

---

## Entry 10 — 2026-11-05 — Step 4C: the retry proposer and the normalizer, wired for attempt ≥ 2

**Commit:** `eea3718` — "Add LLM retry proposer with normalization"

**Files:**
- `src/agentic_gan/agents/model_selection.py` (+114/−10)
- `tests/test_selection.py` (+55, tests appended; one name added to the import block)

### Why

Second LLM step from the issue. After a quality-gate failure the LLM reads the failed checks and
the diagnosis and proposes the four CTGAN settings for the retry. The runtime enforces the
ranges from Step 4B; the code then makes the answer a legal CTGAN setting or rejects it; the
rule retry from Step 4A is the fallback throughout.

### What — `agents/model_selection.py`

- `_normalize(answer, limits) -> dict | None`: clips each setting into its limit (safety net),
  returns `None` if any value is below 1, and if `batch_size % pac != 0` snaps the batch down
  to a multiple of `pac` (or up to the first multiple at or above the lower bound); `None` if the
  snapped value leaves the range.
- `_failed_check_summaries(request)`: `[{fill_level, name, value, limit}]` for every hard
  check in the previous report that did not pass. Summaries only, no rows.
- `_propose_retry(settings, request, run, client) -> dict`:
  1. `rule_settings, _, rule_reason = _rule_retry(...)`; `failed`; `privacy` via
     `PRIVACY_CHECKS`.
  2. `limits = _retry_limits(...)`. If `None` (nothing tunable): **no LLM call**; the rule
     settings are used.
  3. Otherwise context = JSON of `attempt`, `dataset`, `failed_hard_checks`, `diagnosis`
     (`model_dump()` or `null`), `current_settings`, `limits` as `[low, high]`, `privacy_failed`.
  4. `fallback = RetrySettings(**rule_settings, reason="Rule-based default: " + rule_reason)`.
     The fixed prefix is how the code tells the fallback from an LLM answer
     (`source = "rule" if answer.reason == fallback.reason else "llm"`).
  5. `run_llm_step(... RetrySettings, LLMStepOptions(fallback, numeric_bounds=limits,
     expected_rule_hash=_rule_hash(config), current_rule_hash=... run.load_config()))`.
     `RuleOverrideError` is not caught.
  6. If the answer came from the LLM: `_normalize`; if that returns `None`, a `retry_rejected`
     decision (outcome `REJECTED`, inputs: proposed values and limits, rationale "could not be
     made a valid CTGAN setting") is logged and the rule settings are used.
  7. **Bridge (see note):** ends with the existing `_log_retry(...)` call — rule name
     `hyperparameter_retry` if epochs changed else `retry_unchanged`, same inputs keys as before,
     rationale = the LLM's `reason` or the rule's reason.
- `run_model_selection`: `_retry_settings(...)` → `_propose_retry(settings, request, run, client)`.
- `_retry_settings` wrapper deleted (nothing else used it).

**Note on the bridge.** The step said not to log the final settings here because Step 4D will.
But the acceptance criterion ("all of `tests/test_selection.py` passes") requires the existing
retry tests, which assert on `hyperparameter_retry` / `retry_unchanged`, to keep passing. So the
old one-line `_log_retry` is kept at the end of `_propose_retry`, marked with a comment for
Step 4D to replace with a log of all four settings and their source.

### What — `tests/test_selection.py`

- `test_rule_retry_always_lies_inside_the_retry_limits` (5 parametrized cases: fidelity only,
  fidelity + privacy, epochs at the bound, bound closer than one step, two bounds). Proves the
  fallback always validates against `numeric_bounds`.
- `test_llm_proposal_inside_the_limits_is_trained`: attempt 2, `epochs (100, 600)`, FakeLLM
  answers epochs 500 → trained epochs 500, one LLM call (no explanation call on attempt 2),
  `hyperparameter_retry == ADJUSTED`.
- `test_llm_proposal_outside_the_limits_falls_back_to_the_rule`: answer epochs 900 twice →
  two calls, `LLM_FALLBACK` logged, trained epochs 400 (the rule's), `ADJUSTED`.

### Verified

| Command | Result |
| --- | --- |
| `uv run --no-sync ruff check src tests` | All checks passed |
| `uv run --no-sync pytest tests/test_selection.py -q` | 51 passed |
| `uv run --no-sync pytest tests/test_orchestrator.py tests/test_llm_failures.py -q` | 32 passed |
| `git diff tests/test_selection.py` removed lines | none |

### Status at end of entry

- Committed: `a5c9cfe`, `1ed4270`, `c30c51c`, `fd9d439`, `f9f0943`, `eea3718`.
- Uncommitted: this file. Draft work still parked in `stash@{0}`.

---

## Entry 11 — 2026-11-05 — Step 4D: one retry log line with its source; unsupported adjustments

**Commit:** `184bd07` — "Log retry source and unsupported adjustments"

**Files:**
- `src/agentic_gan/agents/model_selection.py` (+73/−28)
- `tests/test_selection.py` (+82, tests appended; `QualityDiagnosis` added to the import block)

### Why

Step 4C left the old one-setting `_log_retry` line as a bridge. A reader of `decisions.jsonl`
now needs to see all four settings, whether they came from the LLM or the rule, and which
diagnosis adjustments the agent could not act on.

### What — `agents/model_selection.py`

- The bridge `_log_retry(...)` call and the `_log_retry` function are gone. `_propose_retry`
  now writes **one** decision line per retry, on every path including the no-LLM path:
  - rule `hyperparameter_retry` / outcome `ADJUSTED` if any of the four settings changed from
    the pre-retry settings, else `retry_unchanged` / `UNCHANGED` (names the existing tests rely on);
  - inputs: `attempt`, `source` (`"llm"` or `"rule"`), `previous_settings`, `proposed` (the raw
    LLM values, or `null` when the rule was used), `applied`, `limits` (`{name: [low, high]}` or
    `null`), `failed_checks`, `privacy_failed`, plus the old keys `previous_epochs`, `epochs`,
    `bound` so earlier readers keep working;
  - rationale: the LLM's `reason` when `source == "llm"`, else the rule's rationale string.
- `UNSUPPORTED_ADJUSTMENTS = ("rebalance_categories", "change_sampling")` and
  `_log_unsupported_adjustments(request, run)`: when the previous report has a diagnosis, one
  `unsupported_adjustment` line per distinct such adjustment (outcome `NOT_APPLIED`, rationale
  "No CTGAN setting implements this adjustment, so it is reported only."), written before the
  retry line. `more_epochs` and `change_pac` are applied through settings and log nothing here.
- The `hyperparameters` / `SET` line from `_train_ctgan` is unchanged.

### What — `tests/test_selection.py` (4 tests appended)

- `test_an_accepted_proposal_logs_one_retry_line_from_the_llm`: exactly one retry line,
  `source == "llm"`, `proposed` / `applied` / `previous_settings` / `limits` / `privacy_failed`
  / `failed_checks` as expected, rationale is the LLM reason, old keys present.
- `test_a_fallback_logs_one_retry_line_from_the_rule`: out-of-bounds answer → one line,
  `source == "rule"`, `proposed is None`, applied epochs 400, rule rationale.
- `test_the_no_llm_path_also_logs_exactly_one_retry_line`: empty `gan_bounds` → no LLM call
  (`FakeLLM([])` has zero calls), one `retry_unchanged` line, `limits is None`.
- `test_unsupported_adjustments_are_reported_once_and_not_applied`: a diagnosis with
  `["more_epochs", "rebalance_categories"]` produces exactly one `unsupported_adjustment` line,
  placed before the retry line; `more_epochs` produces none; training still uses the LLM's 500.

### Verified

| Command | Result |
| --- | --- |
| `uv run --no-sync ruff check src tests` | All checks passed |
| `uv run --no-sync pytest tests/test_selection.py -q` | 55 passed |
| `uv run --no-sync pytest tests/test_orchestrator.py tests/test_llm_failures.py -q` | 32 passed |
| `git diff tests/test_selection.py` removed lines | none |

### Status at end of entry

- Committed: `a5c9cfe`, `1ed4270`, `c30c51c`, `fd9d439`, `f9f0943`, `eea3718`, `184bd07`.
- Uncommitted: this file. Draft work still parked in `stash@{0}`.

---

## Entry 12 — 2026-11-05 — Step 5: log why each unchosen generator route lost

**Commit:** (this step) — "Log why unchosen generator routes were skipped"

**Files:**
- `src/agentic_gan/agents/model_selection.py` (+44)
- `tests/test_selection.py` (+88; `_route_reasons` added to the import block)

### Why

The issue asks that skipped routes (TimeGAN, WGAN-GP, and CTGAN when they are not chosen) be
logged with a reason, without editing the read-only rule in `tools/selection.py`. The reasons
are derived from the already-logged `DecisionStep`s.

### What — `agents/model_selection.py`

- `_route_reasons(result) -> dict[str, str]`: a reason for each of `timegan`, `wgan_gp`,
  `ctgan` that is not `result.generator`.
  - `stop_gate` = first of `size_gate`, `horizon_gate`, `missing_gate`, `leakage_gate` with
    outcome `FAIL`; if none failed but `status == "skipped"`, `stop_gate = "dcgan_excluded"`
    (non-tabular data). Every unchosen route then gets
    `"Not evaluated: <stop_gate> stopped selection. <that step's rationale>"`.
  - No stop gate, generator `ctgan`: TimeGAN gets the `temporal_test` rationale; WGAN-GP gets
    the `type_routing` rationale.
  - No stop gate, generator `wgan_gp`: TimeGAN gets `temporal_test`; CTGAN gets `type_routing`.
  - No stop gate, generator `timegan`: CTGAN gets `"TimeGAN was selected; CTGAN is its
    documented fallback"`; WGAN-GP gets `"TimeGAN was selected, so type routing did not run"`.
  Verified against `select_generator`: the four mappings match the code's outcomes and
  rationales (TimeGAN's type_routing step is `NOT_REACHED` with a similar sentence; the
  logged WGAN-GP reason uses the spec's wording, not that step's text).
- `_log_skipped_routes(result, run)`: one `route_skipped` / `NOT_SELECTED` line per unchosen
  route, inputs `{route, attempt: 1}`. Called on attempt 1 only, after `_log_steps` and before
  `_explain`.
- `_explain` context now includes `"routes_not_chosen": _route_reasons(result)`.

### What — `tests/test_selection.py`

- Bank mixed → TimeGAN and WGAN-GP reasons, no CTGAN entry.
- Default complete block → CTGAN and WGAN-GP reasons as specified.
- Polish all numeric → WGAN-GP chosen; CTGAN reason from type routing.
- Too few rows → all three `"Not evaluated: size_gate ..."`.
- Non-tabular → all three `"Not evaluated: dcgan_excluded ..."`.
- Agent-level: attempt 1 logs exactly one `route_skipped` per unchosen route; attempt 2 logs
  none.

### Verified

| Command | Result |
| --- | --- |
| `uv run --no-sync ruff check src tests` | All checks passed |
| `uv run --no-sync pytest tests/test_selection.py -q` | 61 passed |
| `git diff tests/test_selection.py` removed lines | none |

### Status at end of entry

- Committed: `a5c9cfe`, `1ed4270`, `c30c51c`, `fd9d439`, `f9f0943`, `eea3718`, `184bd07`, `264e305`.
- Uncommitted: this file. Draft work still parked in `stash@{0}`.

---

## Entry 13 — 2026-11-05 — Step 6A: complete the explain-step tests

**Commit:** `24e639e` — "Test the model-selection explanation"

**Files:**
- `tests/test_selection.py` (+85; existing tests untouched)

### Why

Step 3 landed one minimal FakeLLM test. The issue wants the full explanation contract: a
validated answer is logged, a downed or cheating LLM falls back, skipped selections are still
explained, retries do not re-explain, and TimeGAN's allowed generator is `"timegan"`.

### What

Six tests added next to `test_first_attempt_asks_the_llm_to_explain_and_logs_the_answer`
(left unchanged). Every FakeLLM test asserts `len(fake.calls)`.

- `test_explanation_does_not_change_the_generator_output`: valid answer, one call, `llm_step`
  outcome is the answer, `generator` / `hyperparameters` / synthetic row counts match a run with
  no client.
- `test_explanation_falls_back_when_the_llm_is_down`: two `RuntimeError`s → two calls,
  `LLM_FALLBACK`, generator still `ctgan`.
- `test_an_explanation_that_names_another_generator_is_rejected`: `"wgan_gp"` twice →
  `LLM_FALLBACK`, generator still `ctgan`.
- `test_a_skipped_selection_is_still_explained`: small table, answer `generator: "none"` → one
  `llm_step`, status `SKIPPED`, no training.
- `test_a_retry_does_not_ask_for_an_explanation`: attempt 2's only call is `RetrySettings`.
- `test_timegan_explanation_must_echo_timegan`: Default complete block, allowed generator
  `"timegan"`, still `SKIPPED` with no training.

### Verified

| Command | Result |
| --- | --- |
| `uv run --no-sync ruff check src tests` | All checks passed |
| `uv run --no-sync pytest tests/test_selection.py -q` | 67 passed |

### Status at end of entry

- Committed: previous plus `24e639e`.
- Uncommitted: this file. Draft work still parked in `stash@{0}`.

---

## Entry 14 — 2026-11-05 — Step 6B: complete the retry-proposer tests

**Commit:** `8ebd8c3` — "Test the model-selection retry proposer"

**Files:**
- `tests/test_selection.py` (+209/−3; existing test bodies untouched)

### Why

The issue requires FakeLLM coverage of the proposer: in-bounds accept, out-of-bounds reject,
pinned settings, batch/pac snapping, the privacy epoch cap (including `exact_train_copies`),
no call when nothing is tunable, and diagnosis adjustments that have no CTGAN setting.

### What

- `_fail_report(*names, diagnosis=None)` now attaches a `QualityDiagnosis`. It asserts every
  built `CheckResult` is hard (`CheckResult.hard` defaults to `True`). Existing callers that
  omit `diagnosis` still work.
- `_retry_request` takes optional `diagnosis` and `run_id`.
- `ANSWER` / `_answer(**overrides)` is the scripted proposer reply (epochs 400 by default).
- New tests, each asserting `len(fake.calls)` and reading settings from `gan.trains[-1][1]`:
  1. epochs 500 inside `(100, 600)` → trained 500, retry `source == "llm"`.
  2. epochs 900 → two calls, `LLM_FALLBACK`, trained 400, `source == "rule"` (the required
     out-of-bounds case).
  3. only epochs bounded, answer `pac=5` → fallback, pac stays 10.
  4. batch/pac snap-down 500/8 → 496; snap-up when 496 is below the bound → 504; no legal
     multiple → `retry_rejected` and the rule's settings.
  5. `column_distance` + `dcr_ratio` or `exact_train_copies`: epochs 500 rejected (300);
     epochs 200 accepted.
  6. empty `gan_bounds` → zero LLM calls, defaults kept, `retry_unchanged`.
  7. attempt 1's call is `SelectionExplanation`; attempt 2's is `RetrySettings`.
  8. diagnosis `["more_epochs", "change_sampling"]` → exactly one `unsupported_adjustment`
     for `change_sampling`.
  9. exactly one retry line per attempt-2 run: `hyperparameter_retry` when settings changed,
     else `retry_unchanged`.

### Verified

| Command | Result |
| --- | --- |
| `uv run --no-sync ruff check src tests` | All checks passed |
| `uv run --no-sync pytest tests/test_selection.py -q` | 82 passed |

### Status at end of entry

- Committed: previous plus `8ebd8c3`.
- Uncommitted: this file. Draft work still parked in `stash@{0}`.

---

## Entry 15 — 2026-11-05 — Step 6C: rule lock, grounding, precedence, live test

**Commit:** `d993b69` — "Test rule lock, grounding and precedence for model selection"

**Files:**
- `tests/test_selection.py` (+160/−1; import block expanded, tests appended)

### Why

The issue still needed: a rule-hash lock the LLM cannot rewrite, a prompt that never carries
table rows, precedence so no gate silently overrides another, and one skipped-by-default live
call.

### What

- Imports: `json`, `os`, `RuleOverrideError`, `SETTING_NAMES`, and `V1_ANSWER` / `V1_DRIVERS`
  from `test_quality_agent` (same style as `tests/test_llm_failures.py`).
- Rule tampering: `_ConfigEditingLLM` rewrites `min_rows` in `config.yaml` before answering.
  Both the explain step (attempt 1) and the proposer (attempt 2) raise `RuleOverrideError` and
  log `RULE_OVERRIDE_REJECTED`; CTGAN is not trained.
- Context grounding: a FAIL report with `mean_column_distance`, `worst_column_distance`,
  `real_vs_synthetic_auc` and a `QualityDiagnosis` from `V1_ANSWER`. The proposer's context
  contains `job` / `month` / `day_of_month`, those check names, and the diagnosis text; it
  does not contain `source_row`; `json.loads(context)` has only the summary keys (no dataset
  rows).
- Multi-gate precedence: one `_data()` that fails size, horizon, missing, and leakage at once
  stops at `size_gate`; later steps are `NOT_REACHED`; each of the four gates appears once.
  Parametrized `cleared=0..3` peels gates in `STEP_ORDER` so the next one becomes the stopper.
- Live: `@pytest.mark.live` + skip unless `RUN_LIVE_LLM=1`. No client. Attempt 1 generator
  equals the rule's (`ctgan`); attempt 2 settings sit inside `LIVE_BOUNDS` and
  `batch_size % pac == 0`.

### Verified

| Command | Result |
| --- | --- |
| `uv run --no-sync ruff check src tests` | All checks passed |
| `uv run --no-sync pytest tests/test_selection.py -q` | 90 passed, 1 skipped (live) |

### Status at end of entry

- Committed: previous plus `d993b69`.
- Uncommitted: this file. Draft work still parked in `stash@{0}`.

---

## Entry 16 — 2026-11-05 — Step 6D: cover model-selection LLM failures

**Commit:** `53a0818` — "Cover model-selection LLM failures"

**Files:**
- `tests/test_llm_failures.py` (+54; pipeline test untouched)

### Why

Issue #86 already fakes Vertex for every LLM failure mode. Model selection now has two LLM
steps, so those same modes must fall back to the rule without changing the generator or the
retry numbers.

### What

- `OUTSIDE_THE_ALLOWED_OPTIONS` gained `SelectionExplanation` (`generator: "wgan_gp"`) and
  `RetrySettings` (epochs 1_000_000, pac 3, discriminator_steps 99, batch_size 7).
- Imports: helpers and the `gan` fixture from `test_selection`, plus `run_model_selection`.
- Two tests parametrized over `SCENARIOS` (same pattern as the quality-diagnosis test):
  - Attempt 1: `SUCCESS`, CTGAN defaults, the only `llm_step` is `LLM_FALLBACK` from
    `model_selection`, `seen.attempts == 2`.
  - Attempt 2 with `gan_bounds={"epochs": (100, 600)}` and a failed report: trained epochs
    400, retry `source == "rule"`, exactly one `llm_step` is `LLM_FALLBACK`,
    `seen.attempts == 2`.

### Verified

| Command | Result |
| --- | --- |
| `uv run --no-sync ruff check src tests` | All checks passed |
| `uv run --no-sync pytest tests/test_llm_failures.py -q` | 31 passed |

### Status at end of entry

- Committed: previous plus `53a0818`.
- Uncommitted: this file. Draft work still parked in `stash@{0}`.

---

## Entry 17 — 2026-11-05 — Step 7: verify client wiring; graph-level FakeLLM test

**Commit:** `f157eb3` — "Test model selection inside the graph"

**Files:**
- `tests/test_selection.py` (+49)
- `graph.py` not changed

### Callers of `run_model_selection`

None pass a third positional argument. `run_model_selection(request, run)` is still valid
(`client` is keyword-only, default `None`).

- `src/agentic_gan/agents/model_selection.py` — definition.
- `src/agentic_gan/orchestrator/graph.py` — `build_graph` assigns
  `agents[Stage.MODEL_SELECTION] = run_model_selection` when the slot is still a `StubAgent`.
  `execute` calls `agents[stage](request, run)`. The graph's `llm=` goes only to
  `write_narrative`, not to this agent.
- `tests/test_selection.py` — `(request, run)` or `(request, run, client=...)`.
- `tests/test_llm_failures.py` — `(request, run)` twice.
- `eda/` — no callers.

### What

One graph-level test in the quality-agent style: `default_agents()`, stub preprocess that
writes a 2000-row bank table, `partial(run_model_selection, client=FakeLLM([EXPLAIN_ANSWER]))`,
stub quality so the pipeline completes on attempt 1. Asserts one `SelectionExplanation` call
and that `llm_step` outcome is the fake answer. The `gan` fixture stubs CTGAN.

A full real-agent graph (split + preprocess + quality + training + eval) is not cheap; this
mirrors `test_quality_agent_in_graph` instead.

### Verified

| Command | Result |
| --- | --- |
| `uv run --no-sync ruff check src tests` | All checks passed |
| `uv run --no-sync pytest tests/test_selection.py::test_model_selection_in_the_graph_uses_the_fake_explanation -q` | 1 passed |
| `uv run --no-sync pytest -q` | segfault in `tests/tools/test_ctgan.py` (pre-existing torch/OpenMP) |
| `uv run --no-sync pytest -q --ignore=tests/tools` | 582 passed, 3 skipped |
| `tests/tools` files run separately | 6 + 11 + 4 passed |

### Status at end of entry

- Committed: previous plus `f157eb3`.
- Uncommitted: this file. Draft work still parked in `stash@{0}`.

---

## Entry 18 — 2026-11-05 — Step 8: document the model-selection LLM steps

**Commit:** `774d9cb` — "Document the model-selection LLM steps"

**Files:**
- `docs/architecture.md`
- `docs/llm_agents.md`
- No code. No diagram.

### What

- Section 6.2 Model Selection status: "Sprint 3 (#82). The rule and the epoch retry are built"
  → "Built (#82)".
- Section 5.2 Decision logic: explanation runs once on attempt 1; the settings proposal runs
  on attempt 2 (default `max_attempts` 3); values must lie in `gan_bounds`; an unbounded
  setting cannot change; epochs are never raised after a privacy failure; `batch_size` is
  made divisible by `pac`; an unusable answer is replaced by the rule's retry and logged.
  `rebalance_categories` and `change_sampling` are logged as `unsupported_adjustment` and
  not applied.
- Section 6.5: model selection hashes `min_rows`, `gan_bounds`, the step order and its
  defaults itself (local `_rule_hash`), because importing `graph.rule_hash` would be circular.
- `docs/llm_agents.md`: agents take `client=None`; tests pass a `FakeLLM` (or
  `functools.partial` in a graph run); the pipeline's `llm=` argument only reaches the run
  narrative.

Each claim was checked against `model_selection.py` before writing.

### Status at end of entry

- Committed: previous plus `774d9cb`.
- Uncommitted: this file. Draft work still parked in `stash@{0}`.
