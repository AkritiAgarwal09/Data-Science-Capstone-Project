# Issue #124 Step 9: PR description and issue post

## PR title

`Add WGAN-GP backup generator for attempt 3 (issue #124)`

## Branch

`arrico/backup-gan-WGAN-GP-124` → `main`

---

## PR description (copy-paste)

```
## Summary

Implements issue #124: wire WGAN-GP as the backup generator for CTGAN attempt 3,
and as the primary generator for all-numeric tables.

### What changed

- **`tools/wgan_gp.py`** (Steps 1–5): WGAN-GP implementation with Wasserstein loss,
  gradient penalty, DataTransformer integration, `train_wgan_gp`, `sample_wgan_gp`,
  and `save_model`.

- **`eda/wgan_bank/run_wgan_bank.py`** (Step 5): Bank Marketing experiment confirming
  WGAN-GP trains in ~149 s (3173 positive rows), unique_ratio=1.0, 0 exact copies.

- **`tools/selection.py`** (Step 6): `wgan_gp` marked `trainable=True` for all-numeric
  tables; `backup_eligible()` added to gate the attempt-3 fallback.

- **`agents/model_selection.py`** (Step 7):
  - `WGAN_DEFAULTS`, `WGAN_BACKUP_ENABLED` constants.
  - `_train_wgan_gp()` with lazy torch import (prevents macOS OpenMP crash).
  - `_cleanup_wgan_attempt()` removes partial files after a training failure.
  - `run_model_selection` dispatches on `result.generator`: all-numeric tables
    now train WGAN-GP directly; CTGAN attempt 3 falls through to the WGAN-GP
    backup when `backup_eligible` is True.
  - `RuleOverrideError` is re-raised through the backup exception handler.
  - `WGAN_DEFAULTS` included in `_rule_hash` fingerprint.

- **Tests** (Steps 3, 6, 7):
  - `tests/tools/test_wgan_gp.py` (14 tests)
  - `tests/test_selection.py` updated (3 tests changed + 10 new tests for Steps 6–7)
  - `tests/test_wgan_backup.py` (11 new tests: backup success/failure, primary
    dispatch, torch isolation, rule hash stability)
  - `tests/tools/test_ctgan_fill.py` updated (metadata fixtures updated)

- **Pipeline verification** (Step 8): `scripts/run_bank_stub_wgan_backup.py` confirms
  that the 3-attempt pipeline trains CTGAN on attempts 1–2 and WGAN-GP on attempt 3.

### Decision log behaviour

| attempt | generator | rule                 | outcome       |
|---------|-----------|----------------------|---------------|
| 1       | ctgan     | hyperparameters      | SET           |
| 2       | ctgan     | hyperparameters      | SET           |
| 3       | wgan_gp   | hyperparameters      | SET           |

If WGAN-GP raises, `backup_failed` is logged with `FALLBACK_CTGAN` and CTGAN trains.

### Depends on

- Steps 1–6 are prerequisite commits already on this branch.
- PR #123 (`arrico/targeted-retry-123`) is NOT required; Step 7 uses the existing
  `_propose_retry` path for the CTGAN fallback when WGAN-GP is not eligible or fails.

## Test plan

- [ ] `uv run --no-sync pytest tests/test_wgan_backup.py tests/test_selection.py tests/test_orchestrator.py tests/tools/ -q` → 323 passed, 1 skipped
- [ ] `uv run --no-sync ruff check src tests` → All checks passed
- [ ] `uv run --no-sync python scripts/run_bank_stub_wgan_backup.py` → PASS (wgan_trains=1, ctgan_trains=2)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

## Issue #124 comment (post on the issue after merging)

```
## Closed by this PR

WGAN-GP is now:

1. **Implemented** (`tools/wgan_gp.py`): Wasserstein GAN with gradient penalty,
   DataTransformer integration, full train/sample/save API matching the CTGAN interface.

2. **Bank-tested** (`eda/wgan_bank/`): ~149 s on 3173 positive rows, unique_ratio=1.0,
   0 exact copies, 1 scorecard failure on `balance` column distance (documented).

3. **Wired as the primary generator** for all-numeric tables (Polish, etc.).

4. **Wired as the backup** for CTGAN attempt 3 on mixed tables (Bank Marketing):
   - `backup_eligible()` gates on minority count ≥ 500 and no FORBIDDEN_CLASS_ONLY.
   - `WGAN_BACKUP_ENABLED = True` testing flag for easy rollback.
   - Failure path: RuntimeError/ImportError → `backup_failed` logged → CTGAN trains.
   - `RuleOverrideError` propagates (not swallowed).
   - Partial files cleaned up on failure.

5. **Fingerprinted**: `WGAN_DEFAULTS` is part of `_rule_hash`, so changing
   hyperparameters invalidates the LLM lock.

Pipeline verification: 3-attempt Bank stub confirms `generator=wgan_gp` on attempt 3.
```
