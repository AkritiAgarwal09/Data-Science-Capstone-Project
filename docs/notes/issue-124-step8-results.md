# Issue #124 Step 8 results

## Pipeline verification

### Test suite

```
uv run --no-sync pytest tests/test_wgan_backup.py tests/test_selection.py tests/test_orchestrator.py tests/test_llm_failures.py tests/tools/ -q
→ 323 passed, 1 skipped
```

### Bank stub run (`scripts/run_bank_stub_wgan_backup.py`)

Stubs CTGAN and WGAN-GP; forces quality failures on every attempt so
attempt 3 exercises the WGAN-GP backup.

```
COMPLETE=True  failed_stage=None
model_selection_attempts=3
quality_attempts=3
ctgan_trains=2  wgan_trains=1

Hyperparameter rows:
  attempt=1  generator=ctgan  epochs=300
  attempt=2  generator=ctgan  epochs=300
  attempt=3  generator=wgan_gp  epochs=300

PASS: WGAN-GP backup called 1 time(s)
PASS: CTGAN trained 2 time(s) (attempts 1+2)
```

### RuleOverrideError not swallowed

`test_wgan_backup_rule_override_error_propagates` in `tests/test_wgan_backup.py`
verifies that `RuleOverrideError` raised inside `_train_wgan_gp` propagates
through the exception handler (the handler now has an explicit `isinstance` re-raise).

### torch isolation

`test_importing_model_selection_does_not_import_torch` runs a subprocess that
imports `model_selection` and asserts `torch` is not in `sys.modules`.
