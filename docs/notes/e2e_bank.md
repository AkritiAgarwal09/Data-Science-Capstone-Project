# Bank Marketing end-to-end run

Run date: 2026-11-05. Source revision: `b66b2b2` plus the control-flow and
progress-logging fixes described below. The local dataset was
`data/raw/bank_marketing.csv`; no reduced fixture or relaxed quality gate was used.

## Result

The completed run was **`runs/bank_marketing_acb8977d5416/` — SUCCESS**. Its
local artifacts were cleared at the user's request on 2026-11-05. Split,
preprocessing, model selection, quality checking, training, and evaluation used
their real implementations. The first two CTGAN attempts failed the quality gate.
Attempt three selected the documented `wgan_gp` fallback, which is unavailable,
so the run correctly continued **without accepted synthetic data**. This is a
complete three-arm evaluation, not a passing CTGAN result.

- Split: 27,126 training rows, 9,042 validation rows, 9,043 test rows.
- Training: all 280 saved model metric records succeeded (four models, five seeds,
  seven arm/fill groups, on validation and test). Arms were original, cleaned,
  and SMOTENC at 10%, 25%, 50%, 75%, and 100% fill.
- Mean test PR-AUC: original 0.464096; cleaned 0.477575; SMOTENC 0.465477 at
  10% fill, declining to 0.425665 at 100% fill. The report labels cleaned's
  PR-AUC gain over original reliable under its paired-seed rule.
- The manifest verifies all 467 registered artifacts with no problems. The
  evaluation report verified 1,113 numeric references against result files.
- The decision log has 387 entries, including the initial rule hash. It records
  ten validated live LLM responses and zero `LLM_FALLBACK` outcomes: four
  preprocessing treatments, two quality diagnoses, two training responses,
  one evaluation narrative, and one final run summary. Split and generator
  routing remain deterministic; model selection made no LLM call in this run.
- Regression suite after the fixes: 548 passed, 2 live tests skipped, with the
  native CTGAN test files excluded. Changed Python files pass Ruff; the script
  passes `bash -n`.

The quality reports for both CTGAN attempts fail the same four hard checks at
each of five fill levels: `mean_column_distance`, `worst_column_distance`,
`augmentation_pr_auc_gain`, and `augmentation_best_f1_gain`. No quality
threshold was changed. This pipeline uses the model-selection agent's CTGAN
defaults (300 epochs on the first attempt), so the standalone **CTGAN V2
3-of-15** result is not the result of this run. Wiring that variant into model
selection requires a separate, recorded generator-setting change.

## Fix list

| Finding | Fix or handling | Evidence |
| --- | --- | --- |
| The first full attempt stopped after two failed quality checks. The third generator attempt returned `SKIPPED`, and the graph tried to skip a quality stage that had already run. | Skip the quality stage only when it has not started. The graph now preserves the two completed quality attempts and routes to training without a GAN. | Regression test `test_third_selection_skip_after_failed_quality_checks_continues`; the completed run has no `stage_error` entries. |
| The CLI gave no useful live progress during multi-minute CTGAN fits or model training. | Added opt-in timestamped console progress for stage attempts, tool calls, LLM calls and decisions, and the next routed stage. The Bank script enables it. The append-only decision log remains the audit record. | `test_progress_log_shows_stage_tool_llm_and_next_action`; the completed run's console log shows CTGAN fit, each quality fill, each training group, evaluation, and routing. |
| CTGAN did not pass the hard gate. | Keep the recorded no-GAN result and train only the permitted original, cleaned, and SMOTENC arms. No threshold or test metric was used to rescue synthetic data. | Both quality reports show zero passing fill levels; the training plan has empty `gan_levels`. |

The interrupted first attempt was in `runs/bank_marketing_e7b311507c16/`.
It had a config, manifest, and decision log but no final report; its local
artifacts were also cleared.

## Reproduce

From the repository root, install the locked environment, provide the three
Vertex settings in the ignored `.env` (see `.env.example`), and authenticate
with Application Default Credentials. Then run:

```bash
bash scripts/run_e2e_bank.sh
```

The script enables progress lines on stderr, disables LangSmith tracing for
this run, and points macOS LightGBM at the OpenMP library bundled with the
local scikit-learn installation. Each invocation creates a new run ID. A
progress line looks like:

```text
[15:17:40] agent=model_selection | train CTGAN attempt=1 epochs=300 rows=27126 | tool=train_ctgan | next=save checkpoint and sample fill levels
```

## Local artifacts

`runs/` is ignored by Git and is now empty. This note retains the observed
counts and findings, but the models, predictions, manifest, and decision log
are no longer available locally. Run the script again to create a new run ID.
