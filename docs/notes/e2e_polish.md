# End-to-end Polish run: issue #93

Run date: 2026-11-05 → 2026-11-06.
Run folder: `runs/polish_112a060d4a34`.
Source revision: `1d94c95` (branch `e2e/polish-93`; worktree dirty:
`src/agentic_gan/utils/datasets.json` patched to `["3year.arff"]` and not staged).

## Scope

Dataset: Polish Bankruptcy, cohort `3year.arff` only.
Total rows: 10,503. Split: train 6,270, validation 2,101 (99 positives),
test 2,101 (99 positives). Positives in train: 296.

The `datasets.json` cohort list was patched to `["3year.arff"]` via
`eda/polish_e2e/polish_3year_datasets.patch` before the run. This patch is
committed and re-applicable; `datasets.json` itself is never committed on
this branch.

The pooled (all-five-cohort) path was not run. `horizon_gate` returns FAIL
when more than one horizon is in the table, so model selection is skipped
immediately and CTGAN is never reached. This is documented pipeline behavior.

## Fix list

| Bug | File | Fix | Test |
| --- | --- | --- | --- |
| Synthetic CSVs missing `cohort` column → `schema_match` hard-fail (threshold 0.0) + potential "Synthetic schema must match" crash in training | `agents/model_selection.py` | Capture the constant cohort value before `model_frame` drops it; re-add after sampling in `_train_ctgan` | `test_single_horizon_polish_synthetic_keeps_the_cohort_column` in `tests/test_selection.py` |
| Orchestrator crashed with `StateError` when attempt 3 returned SKIPPED after quality already finished | `orchestrator/graph.py` | Guard `run_state.skip(QUALITY_CHECK)` with precondition `quality.status is None and not quality.running and quality.attempts == 0` | `test_unimplemented_fallback_after_quality_leaves_the_finished_stage` in `tests/test_orchestrator.py` |
| `run_polish.py` called `state.get(...)` on a `RunState` object (no such method) | `eda/polish_e2e/run_polish.py` | Use `state.is_complete` property | Covered by the successful run |
| `audit_run.py` filtered on `decisions["stage"]`; jsonl field is `"agent"` | `eda/polish_e2e/audit_run.py` | Replace all `"stage"` refs with `"agent"` | Audit runs clean on `polish_112a060d4a34` |
| `check_3year.py` had unsorted imports and an unused `pandas` import | `eda/polish_e2e/check_3year.py` | Fix import order; remove unused import | ruff passes |
| No first-class horizon selection — requires an uncommitted `datasets.json` patch | — | Local patch via `eda/polish_e2e/polish_3year_datasets.patch` (committed). A `RunConfig.horizon` field needs team approval (open gap, see below) | Patch file is committed |

## Selection gates (attempt 1)

| Gate | Outcome | Detail |
| --- | --- | --- |
| dcgan_excluded | EXCLUDED | DCGAN is for image data; not applicable to tabular |
| size_gate | PASS | 6,270 training rows ≥ 2,000 min_rows |
| horizon_gate | PASS | Single cohort loaded |
| missing_gate | PASS | No unresolved missing values in generator table |
| leakage_gate | PASS | No listed leakage column present |
| minority_gate | FORBIDDEN_CLASS_ONLY | 296 minority rows < 500 threshold; class-only generator forbidden; conditional selection continues |
| temporal_test | FAIL | Failed on entity_key, time_index, min_steps, repeated_observation; TimeGAN not selected |
| type_routing | SELECTED | Categorical/indicator columns present; CTGAN selected |
| fallback | NOT_REACHED | Fallback (attempt 3) not needed on attempt 1 |

The same gate sequence repeated on attempt 2 with identical outcomes.

## CTGAN hyperparameters

| Attempt | epochs | batch_size | pac | discriminator_steps | seed | Source |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 300 | 500 | 10 | 1 | 42 | default |
| 2 | 400 | 500 | 10 | 1 | 42 | rule (hyperparameter_retry, ADJUSTED) |
| 3 | — | — | — | — | — | SKIPPED (wgan_gp fallback not implemented) |

## Quality gate results

Both attempts returned `RETRY_MODEL_SELECTION`. The hard-check values below
are from `report/evidence.json` for fill_level 0.1 (the first level checked).

| Check | Value | Threshold | Hard | Result |
| --- | --- | --- | --- | --- |
| mean_column_distance | 0.398 | 0.100 | yes | FAIL |
| worst_column_distance | 0.786 | 0.250 | yes | FAIL |
| spearman_diff | 0.248 | 0.100 | yes | FAIL |
| label_balance | 0.000 | 0.010 | yes | PASS |
| schema_match | 0.000 | 0.000 | yes | PASS |
| column_distance:cohort | 0.000 | 0.250 | no | PASS |

`schema_match` passes at 0.0 — the cohort fix works. The column-distance and
correlation checks exceed their thresholds by a wide margin. No threshold was
relaxed.

Attempt 3: model selection returned `SKIPPED` with reason "Attempt 3 uses
fallback wgan_gp, which is not implemented." Quality was not re-run.
`gan_available` is `False`.

## Training arms

7 arms, 5 seeds (42–46), 280 total fits. No GAN arm (not generated due to
quality gate failure). The table shows means over 5 seeds.

| Arm | fill_level | test F1 | test PR-AUC | test precision | test recall |
| --- | --- | --- | --- | --- | --- |
| original | 0.00 | 0.618 | 0.664 | 0.669 | 0.578 |
| cleaned | 0.00 | 0.613 | 0.704 | 0.654 | 0.578 |
| smotenc | 0.10 | 0.576 | 0.683 | 0.506 | 0.671 |
| smotenc | 0.25 | 0.529 | 0.668 | 0.407 | 0.754 |
| smotenc | 0.50 | 0.453 | 0.656 | 0.323 | 0.760 |
| smotenc | 0.75 | 0.412 | 0.652 | 0.281 | 0.770 |
| smotenc | 1.00 | 0.412 | 0.653 | 0.278 | 0.794 |

Test set: 2,101 rows, 99 positives. `small_test: True` for all arms.

## Reliable-gain verdicts (vs original)

| Arm | fill_level | Metric | Verdict |
| --- | --- | --- | --- |
| cleaned | 0.00 | pr_auc | **reliable gain** (val +0.036, test +0.040) |
| cleaned | 0.00 | f1 | inconclusive |
| smotenc | any | f1 | reliable loss |
| smotenc | any | precision | reliable loss |
| smotenc | 0.10–1.00 | recall | reliable gain |
| smotenc | any | pr_auc | inconclusive |

LLM: all LLM steps returned `LLM_FALLBACK` (no live model). Preprocessing
logged 45 fallback calls; model_selection, quality_check, training, and
evaluation each logged 2 or fewer.

## Open gaps (follow-up needed)

1. **WGAN-GP and TimeGAN not implemented.** Attempt 3 returns `SKIPPED`
   instead of running a fallback generator. The pipeline completes without
   GAN augmentation. This needs a separate implementation task.

2. **No first-class horizon selection.** `RunConfig` has no `horizon` field.
   Running 3year-only requires an uncommitted `datasets.json` patch. A
   `RunConfig.horizon` parameter is needed for reproducible horizon-scoped
   runs (Step 11 in the original plan; needs team approval).

3. **Quality thresholds are tight for 296 minority rows.** `mean_column_distance`
   threshold is 0.10; CTGAN achieved 0.40 on both attempts. Whether the
   threshold should be relaxed for small-minority datasets is a team decision.

## Reproduce

```bash
# 1. Apply the 3year cohort patch (never commit)
git apply eda/polish_e2e/polish_3year_datasets.patch

# 2. Run (Vertex variables must be unset for deterministic run)
unset GOOGLE_CLOUD_PROJECT GOOGLE_CLOUD_LOCATION VERTEX_MODEL
uv run python -u eda/polish_e2e/run_polish.py --mode 3year

# 3. Revert the patch when done
git checkout -- src/agentic_gan/utils/datasets.json
```

## Run artifacts

`runs/` is git-ignored. Local reference only.

- `runs/polish_112a060d4a34/config.yaml`
- `runs/polish_112a060d4a34/manifest.json`
- `runs/polish_112a060d4a34/decisions.jsonl`
- `runs/polish_112a060d4a34/state.json`
- `runs/polish_112a060d4a34/report/final_report.md`
- `runs/polish_112a060d4a34/report/evidence.json`
- `runs/polish_112a060d4a34/report/comparisons.csv`
- `runs/polish_112a060d4a34/models/training_plan.json`
