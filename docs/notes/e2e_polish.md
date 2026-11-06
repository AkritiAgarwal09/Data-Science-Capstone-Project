# End-to-end Polish run: issue #93

Run date: 2026-11-05. Source revision: `d269f7e`. The worktree was dirty:
`src/agentic_gan/utils/datasets.json` was modified and not staged, and
`eda/polish_e2e/run_polish.py` was untracked.

Local-only change: cohorts patched to ["3year.arff"] via
`eda/polish_e2e/polish_3year_datasets.patch` (not committed; `datasets.json` on this
branch still carries all five cohorts).

The Run B driver printed `revision: 600df9c*` when it started. That was HEAD before
the audit-script commit, with the same uncommitted cohorts patch.

## Scope and readiness

Three runs were in scope.

- Run A is the pooled run on `main`'s five-cohort Polish table. `/tmp/polish_run_A.txt`
  was never written. The orchestrator that was running for that step is
  `runs/polish_32e49620ae74/`. When Step 9 read it, training was still marked running
  and the process was gone. It is not a finished run.
- Run B is the 3-year deterministic run, `runs/polish_d50da7c39602/`
  (`/tmp/polish_run_B.txt`). Vertex variables were unset. It is the primary run for
  this note. It did not finish.
- Run C, the 3-year live Vertex run, was not done. It was not started, and
  `/tmp/polish_run_C.txt` does not exist.

What "297 minority rows" means: `3year.arff` alone has 10,503 rows and 495 positives;
60% of 495 = 297 in the train split. Older EDA said 294 because it filtered the pooled
split rather than loading `3year.arff` directly.

Why the pooled run cannot reach the minority gate: `horizon_gate` fails when all five
cohorts are loaded. Model selection is skipped immediately. This is pipeline behavior,
not a bug. Run A's `state.json` records that skip, with `gan_available` false and this
reason: "More than one Polish horizon is in the table. One horizon file is required."

## Fix list

| Finding | Change | Regression coverage |
| --- | --- | --- |
| Synthetic CSV missing `cohort` column, which makes `schema_match` a hard fail (threshold 0.0) and can crash training with "Synthetic schema must match..." | Re-add the constant cohort value after sampling in `_train_ctgan` (`agents/model_selection.py`). | `test_single_horizon_polish_synthetic_keeps_the_cohort_column` in `tests/test_selection.py`. |
| No first-class horizon selection, so every 3-year run needs an uncommitted `datasets.json` patch. | Local patch via `eda/polish_e2e/polish_3year_datasets.patch`. A `RunConfig.horizon` field is proposed in Step 11 and needs team approval. | The patch file is committed. Applying it is still a local step and is not committed as `datasets.json`. |
| The pooled protocol is version-pinned to numpy 2.5.3 / scipy 1.18.1. The brief expected the venv to be 1.26.4 / 1.15.3. | A separate 3-year protocol was frozen at `results/e2e_polish_3year/`. The freeze check in this session printed a match for every pinned package, so the expected mismatch was not what this environment showed. The pooled protocol file was not modified. | `results/e2e_polish_3year/baseline_protocol_polish_42_cv0.json` is the committed 3-year protocol. Step 9 did not reprint the version table. |
| `gan_bounds` is empty by default in `config_for_dataset`, so a retry never changes a setting. | The driver sets bounds explicitly. Default bounds in `config_for_dataset` are a follow-up, not part of this note. | `eda/polish_e2e/run_polish.py` is the driver. It was still untracked when this note was written. |
| Attempt 3 logged the unimplemented WGAN-GP fallback, then the orchestrator raised `StateError: model_selection: cannot finish a stage that is not running`. The logged cause was `quality_check: only a stage that has not started can be skipped`. | Not fixed in this note. Run B stops there. `gan_available` stays null in `state.json`. | No regression test for this crash was added here. |
| The audit script reads a `stage` column. Decision lines use `agent`. | Not fixed in this note. `eda/polish_e2e/audit_run.py` stops with `KeyError: 'stage'` after printing stage status. | The script is committed. It does not complete an audit on these run folders. |

## Reproduce

From the repository root:

```bash
# 1. Apply the 3year patch (never commit)
git apply eda/polish_e2e/polish_3year_datasets.patch
# 2. The protocol is already in results/e2e_polish_3year/
#    To re-freeze from scratch:
#    mkdir -p /tmp/polish3_root/results && ln -sfn "$PWD/data" /tmp/polish3_root/data
#    uv run python -m agentic_gan.tools.baselines --dataset polish --arm both \
#      --cv-folds 0 --root /tmp/polish3_root
#    cp /tmp/polish3_root/results/baseline_protocol_polish_42_cv0.json \
#       results/e2e_polish_3year/
# 3. Run (add DYLD prefix if LightGBM requires it on macOS)
uv run python -u eda/polish_e2e/run_polish.py --mode 3year
# 4. Revert the patch
git checkout -- src/agentic_gan/utils/datasets.json
```

On this Mac, a LightGBM fit succeeded without `DYLD_LIBRARY_PATH`. The prefix was not
used for the freeze or for Run B. Add it only if LightGBM cannot load OpenMP.

## Results

Step 9 printed `state.json` only. For both Run A and Run B it did not print a metrics
table, a verdict table, or evidence. `models/metrics.csv`, `report/comparisons.csv`,
and `report/evidence.json` are absent from both folders.

### Run B state

`runs/polish_d50da7c39602/`

| Stage | Status in state.json | Attempts | Running |
| --- | --- | --- | --- |
| split | SUCCESS | 1 | false |
| preprocessing | SUCCESS | 1 | false |
| model_selection | null | 3 | true |
| quality_check | SUCCESS | 2 | false |
| training | null | 0 | false |
| evaluation | null | 0 | false |

`gan_available` is null. `gan_unavailable_reason` is null. `volume_decision` is null.

Split sizes and positive counts for train, validation, and test are not available.
Step 9 reads `state.json`, and that file has no row counts. The report files it would
also read were not written.

Selection-gate names and outcomes are not available from the Step 9 print. That
command does not read `decisions.jsonl`. The stage table above is the recorded
status: model selection has no final status.

Quality hard-check names, values, and thresholds for attempt 1 and attempt 2 are not
available. Step 9 did not emit them, and `report/evidence.json` was not written.

The retry change (what moved, and whether the source was a rule or an LLM) is not
available. `state.json` does not store hyperparameters.

Attempt 3 is not a finished status in `state.json`. Model selection is still running
on attempt 3. `gan_available` is null, not false. The pipeline did not record a
completed fallback stage in `state.json`.

Arm means for test PR-AUC, F1, precision, and recall are not available. Training did
not complete: training attempts are 0, and `models/metrics.csv` does not exist.
Reason: the run raised `StateError` during model selection before training started.

Reliable-gain verdicts are not available. `report/comparisons.csv` does not exist
because evaluation did not run.

The statement "The GAN arm did not exist in this run; the pipeline ran raw, baseline
and smote arms only" is not supported here. No GAN arm was registered
(`gan_available` is null), and the raw, baseline, and smote arms were not run
(training attempts are 0).

The quality gate rejecting CTGAN is expected behavior; no threshold was relaxed.
`quality_check` is SUCCESS in `state.json` with 2 attempts, which means the stage
returned a report. The pass/fail values inside that report are not available from
Step 9.

"Validation and test each have 99 positives; score differences are noisy" cannot be
checked from Step 9. Positive counts and score differences are not available.

### Run A state

`runs/polish_32e49620ae74/`

| Stage | Status in state.json | Attempts | Running |
| --- | --- | --- | --- |
| split | SUCCESS | 1 | false |
| preprocessing | SUCCESS | 1 | false |
| model_selection | SKIPPED | 1 | false |
| quality_check | SKIPPED | 0 | false |
| training | null | 1 | true |
| evaluation | null | 0 | false |

`gan_available` is false. Reason: "More than one Polish horizon is in the table. One
horizon file is required." `volume_decision` is null.

Arm means, verdicts, and evidence are not available. `models/metrics.csv`,
`report/comparisons.csv`, and `report/evidence.json` do not exist. Training has no
status. When Step 9 was collected the orchestrator process was not running, and the
state file still said training was running, so the run did not finish cleanly.

### Run C

Not available. The live Vertex run was not started.

There is no `llm_step` count, no model-selection explanation, and no verdict
comparison with Run B.

## Run artifacts

`runs/` is git-ignored. These paths are local references only.

Run B, `runs/polish_d50da7c39602/`:

- `runs/polish_d50da7c39602/config.yaml`
- `runs/polish_d50da7c39602/manifest.json`
- `runs/polish_d50da7c39602/decisions.jsonl`
- `runs/polish_d50da7c39602/state.json`

Run A, `runs/polish_32e49620ae74/`:

- `runs/polish_32e49620ae74/config.yaml`
- `runs/polish_32e49620ae74/manifest.json`
- `runs/polish_32e49620ae74/decisions.jsonl`
- `runs/polish_32e49620ae74/state.json`

`models/metrics.csv`, `report/comparisons.csv`, and `report/evidence.json` are not
in either folder.

## Remaining gaps

- WGAN-GP and TimeGAN fallbacks are not implemented. Run B did not get far enough to
  set `gan_available` after the unimplemented WGAN-GP fallback.
- Horizon selection is not a first-class `RunConfig` option (see Step 11).
- The pooled protocol's version pins were a follow-up in the brief. This session's
  freeze check matched the installed packages, and the pooled protocol file was left
  untouched. Step 9 did not reprint the version strings.
- 297 vs 294: the older EDA filtered the pooled split; 297 is the correct 3-year-only
  count, as defined in Scope and readiness.
- Run A has no finished training metrics. Run C was not run. Run B has no arm table
  and no reliable-gain verdicts.
