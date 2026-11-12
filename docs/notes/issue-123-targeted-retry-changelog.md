# Issue #123: Targeted retry after a failed quality gate — change log

Branch: `arrico/targeted-retry-123` (from `origin/main`).

This file records every change made on the branch for issue #123. It is append-only: each
work session adds a new dated entry at the bottom. Earlier entries are never edited or removed,
even if a later entry reverses them. To find the current state of something, read from the
bottom up.

## How to read this file

- **Entry** = one block of work, with the date and the files it touched.
- **Why** explains the reason in plain language; **What** lists the concrete changes.
- **Verified** lists the commands run and their results at the time of the entry.
- Nothing here is committed unless an entry says so.

## The issue in one paragraph

After a quality-gate `FAIL`, the model-selection agent must change more than “+100 epochs”: it
must target the failure with seven CTGAN knobs, a deterministic ladder as the LLM fallback and
safety net, attempt-history so rungs are not repeated, and an attempt-3 path that still trains
CTGAN until WGAN-GP (#124) exists. The LLM may only propose values inside `gan_bounds` /
`RETRY_OPTIONS`; invalid or repeated proposals are rejected and the ladder applies.

## Decisions on open questions Q1–Q4

| Q | Decision taken |
|---|----------------|
| **Q1** (rake set) | Allowed rake columns = discrete feature columns ∪ recipe rake columns that exist in the table, in table order (`allowed_rake_columns`). Bank includes `day_of_month`, excludes the target. |
| **Q2** (ladder vs `gan_bounds`) | Ladder numerics are clamped into `config.gan_bounds` when a bound exists; otherwise pinned to the previous value. Fixed ladder caps (epochs 600, steps 5, batch 1000) still apply as upper limits when building candidates. |
| **Q3** (privacy + fidelity together) | Global invariant: if any privacy check failed, epochs never rise above the previous attempt (even when a fidelity row also matches). |
| **Q4** (rung order) | Candidate rungs = matching table rows in table order, then diagnosis-mapped candidates, then the catch-all (+100 epochs). First candidate whose settings key is not in history wins. |

---

## Entry 1 — 2026-11-12 — Step 0 preflight + Step 1 default gan_bounds

**Files:**
- `src/agentic_gan/orchestrator/graph.py` (modified: `DEFAULT_GAN_BOUNDS`, `config_for_dataset`)
- `tests/test_default_bounds.py` (new)

**Why:** Every dataset must start with legal, shared CTGAN bounds so retries have room to move
without each caller inventing ranges.

**What:**
- Module constant `DEFAULT_GAN_BOUNDS` with epochs (300, 600), pac (5, 10), discriminator_steps
  (1, 5), batch_size (250, 1000).
- `config_for_dataset` copies those bounds into every `RunConfig`.
- Tests cover all four datasets, YAML round-trip, and that `config_for_dataset` is the only
  `graph.py` change.

**Verified:**
- `uv run --no-sync ruff check src tests` → All checks passed.
- `uv run --no-sync pytest tests/test_default_bounds.py -q` → passed (committed as `e73682e`).

**Committed:** `e73682e` Set default GAN bounds for every dataset.

---

## Entry 2 — 2026-11-12 — Step 2 three new knobs

**Files:**
- `src/agentic_gan/agents/model_selection.py` (modified)
- `tests/test_retry.py` (new)
- `tests/test_selection.py` (call sites updated for new RetrySettings fields)

**Why:** Targeted retry needs `sampling_mode`, `rule_filter`, and `rake_columns` alongside the
four numeric settings, with the same validation and rule-hash lock as the numerics.

**What:**
- `RETRY_OPTIONS`, `SEVEN_KNOBS` / knob plumbing through `RetrySettings`, `_retry_limits`,
  `_normalize`, `_rule_hash`.
- Pure unit tests for bounds, options, rake allowed-set, duplicates, and hash movement.

**Verified:**
- Step 2 suite green (committed as `94dc988`).

**Committed:** `94dc988` Add sampling knobs to the retry settings.

---

## Entry 3 — 2026-11-12 — Step 3 attempt history

**Files:**
- `src/agentic_gan/agents/model_selection.py` (modified)
- `tests/test_retry.py` (appended)

**Why:** The ladder and the LLM must not repeat an earlier attempt’s seven-knob settings.

**What:**
- `AttemptRecord`, `settings_key`, `attempt_history(request, run)` reading `hyperparameters`
  rows and quality reports; missing attempt-1 falls back to the Bank/dataset recipe.

**Verified:**
- History tests green (committed as `8cbc82d`).

**Committed:** `8cbc82d` Read earlier attempt settings from the decision log.

---

## Entry 4 — 2026-11-12 — Step 4 `_ladder_retry`

**Files:**
- `src/agentic_gan/agents/model_selection.py` (modified)
- `tests/test_retry.py` (appended)

**Why:** Need a pure, deterministic fallback that maps failed checks to knob moves and skips
settings already in history (Q1–Q4).

**What:**
- `LadderResult`, `_ladder_retry`, `_apply_ladder_invariants`, row-1 rake helper, privacy
  epoch floor, diagnosis candidates before the catch-all.
- `_rule_retry` left in place but unused until Step 5.

**Verified:**
- Ladder unit tests green including privacy+fidelity and exhaustion (committed as `b98102c`).

**Committed:** `b98102c` Add deterministic retry ladder.

---

## Entry 5 — 2026-11-12 — Step 5 wire the ladder into `_propose_retry`

**Files:**
- `src/agentic_gan/agents/model_selection.py` (modified)
- `tests/test_selection.py`, `tests/test_llm_failures.py` (old behaviour assertions updated)

**Why:** Attempt ≥ 2 must start from history (not `CTGAN_DEFAULTS`), call the LLM inside
limits, reject bad/repeat proposals, and fall back to the ladder; `retry_unchanged` only when
exhausted.

**What:**
- `LADDER_PREFIX`; `_propose_retry` rewritten (history → ladder → LLM → normalize/reject →
  re-ladder); `UNSUPPORTED_ADJUSTMENTS = ()` and no-op unsupported logger; diagnosis as
  suggestions + ladder rungs.
- Selection/llm_failures tests expect Bank recipe previous settings and `source=ladder` on
  fallback.

**Verified (with Steps 6–8):** see Entry 8.

---

## Entry 6 — 2026-11-12 — Step 6 RETRY_PROMPT and summary-only context

**Files:**
- `src/agentic_gan/agents/model_selection.py` (modified: `RETRY_PROMPT`, context JSON)
- `tests/test_retry.py` (appended)

**Why:** The LLM must see failed checks, worst columns, diagnosis, limits, options, allowed
rake, history, and privacy — never raw table rows.

**What:**
- New plain-text `RETRY_PROMPT` (role, knobs, limits, example, under 400 words).
- Context keys: `failed_hard_checks`, `worst_columns`, `diagnosis`, `current_settings`,
  `limits`, `options`, `allowed_rake_columns`, `history`, `privacy_failed`.

**Verified:** prompt/context tests in `test_retry.py` (see Entry 8).

---

## Entry 7 — 2026-11-12 — Step 7 attempt 3 trains CTGAN

**Files:**
- `src/agentic_gan/agents/model_selection.py` (modified)
- `tests/test_orchestrator.py` (appended)

**Why:** Until WGAN-GP (#124), attempt 3 must not skip; it trains CTGAN on the next ladder
rung. Finding F3 (SKIPPED after quality already ran → `StateError` in `graph.py`) is documented
with a strict xfail; `graph.py` was not edited.

**What:**
- `IMPLEMENTED_GENERATORS = ("ctgan",)`; `_propose_retry` returns `(settings, exhausted)`;
  attempt ≥ 3 uses propose unless exhausted → `_fallback_attempt` only then.
- `test_attempt_3_trains_and_quality_runs_again_without_state_error` (pass).
- `test_exhausted_attempt_3_skipped_hits_f3_state_error_in_graph` (`xfail(strict=True)` for F3).

**Verified:** orchestrator suite includes 1 xfailed F3 case (see Entry 10).

---

## Entry 8 — 2026-11-12 — Step 8 complete `tests/test_retry.py`

**Files:**
- `tests/test_retry.py` (required a–g + extras)

**Why:** Lock the wired retry behaviour: Bank default bounds → unconditional + rake job;
LLM down → ladder; OOB/repeat rejection; privacy never raises epochs; attempt 3 distinct;
rule-hash / RuleOverride; no silent `retry_unchanged` / `unsupported_adjustment`; no rows in
context.

**What:**
- Wired tests reuse `test_selection` helpers via `pytest_plugins` and the `gan` fixture;
  every FakeLLM test asserts `len(fake.calls)`.
- Mutation check: ignoring history, dropping privacy invariant, or dropping the rake
  allowed-set check each fails the suite; all reverted.

**Verified:**
- `uv run --no-sync pytest tests/test_retry.py tests/test_selection.py -q` → **155 passed**, 1 skipped.
- `uv run --no-sync ruff check src tests` → All checks passed.
- `tests/test_retry.py` ×3 → **57 passed** each run (no flakiness).

---

## Entry 9 — 2026-11-12 — Step 9 audit script + Bank stub run

**Files:**
- `scripts/audit_retry.py` (new, not part of the package)
- `tests/test_audit_retry.py` (new)

**Why:** A read-only checker that a retry run actually moved knobs via ladder/LLM, never logged
unsupported adjustments, and did not SKIP attempt 3.

**What:**
- `audit(run_dir) -> list[str]`; CLI prints attempt/source/changed knobs/trigger; exit 1 on
  violations.
- Unit tests for good / unchanged / unsupported / skipped-3 / stage_error logs.
- Bank demonstration used stub quality + patched CTGAN (real unpatched CLI train is minutes × 3
  on CPU); run `runs/bank_marketing_4ecaaea180f7` audited to **0 violations**.

**Verified:**
- `uv run --no-sync pytest tests/test_audit_retry.py -q` → **9 passed**.
- Audit on stub Bank run → PASS; `llm_health` showed model_selection LLM_FALLBACK ×3 (no Vertex).

---

## Entry 10 — 2026-11-12 — Step 10 docs and PR

**Files:**
- `docs/notes/issue-123-targeted-retry-changelog.md` (this file)
- `docs/architecture.md` (section 5.2 Decision logic paragraph only)

**Why:** Document the targeted ladder for reviewers and keep architecture aligned with code
(no longer “more epochs only” / `unsupported_adjustment`).

**What:**
- Changelog entries for steps 0–10 with Q1–Q4 decisions.
- Architecture 5.2 decision logic updated for seven knobs, ladder, privacy invariant, attempt 3.

**Verified (2026-11-12, after `git pull origin main` — already up to date):**
- `uv run --no-sync ruff check src tests` → All checks passed.
- `uv run --no-sync pytest tests/test_retry.py tests/test_selection.py -q` → **155 passed**, 1 skipped.
- `uv run --no-sync pytest tests/test_llm_failures.py tests/test_orchestrator.py tests/test_quality_agent.py tests/test_schemas.py -q` → **176 passed**, 1 skipped, **1 xfailed** (F3).
- `uv run --no-sync pytest tests/test_default_bounds.py tests/test_audit_retry.py -q` → **16 passed**.
- `uv run --no-sync pytest tests/tools/test_ctgan.py -q` → **11 passed**.

**Committed:** remaining Steps 5–10 (wire-up, prompt, attempt 3, tests, audit, docs) on this branch for the PR.
