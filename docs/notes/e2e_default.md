# End-to-end Default run — issue #88

Run date: 2026-11-04. Source revision: `72dad26` plus the fixes below.
The checkout was clean at the start, and `git pull --ff-only` reported up to date.

## Scope and readiness

The local Default of Credit Card Clients source and frozen CV0 training protocol
are available. This run uses the entire dataset, the configured split, all four
classifiers, all five model seeds, and all five SMOTENC fill levels. No test fixtures,
reduced dataset, relaxed quality thresholds, or replacement model suite are used.
Repeated outer CV is disabled by the default configuration; stacking retains its
inner CV. Existing validation-selected thresholds are reused without recalibration.

The supported pipeline now completes with live Vertex preprocessing and reporting.
One generator capability remains unavailable:

- Default's complete monthly block selects **TimeGAN**. It is not implemented, and
  the current selector documents CTGAN as a fallback but does not execute it. The
  graph records model selection and quality checks as skipped and continues with
  original, cleaned and SMOTENC arms. This is not a four-arm GAN result or evidence
  that generated data passed the quality gate.

The earlier corrected run used deterministic LLM fallbacks because Vertex settings
were unset. After ADC authentication and explicit authorization of the aggregate
payloads, the live run used project `ieor-test-project`, location `global`, and
model `google/gemini-3.5-flash`. No fake client was injected. All thirteen
preprocessing decisions were validated live, and their treatment plan matched the
earlier deterministic plan.

## Fix list

| Finding | Change / handling | Regression coverage |
| --- | --- | --- |
| The CLI's relative `runs/` root made stage output paths get prefixed twice during manifest registration. The first run failed at split. | `create_run` now resolves the run root to an absolute path, making stage artifacts unambiguous while manifest entries remain relative. | Relative-root artifact resolution test and the real default-graph integration test using a relative run directory. |
| Constant stacking probabilities at high SMOTENC fills were selected as winners because trapezoidal PR-AUC rewards endpoint interpolation. | Selection now requires nonconstant validation probabilities. Keep the fits, metrics and predictions, flag exclusions in metrics/decisions/report, and leave the frozen models, thresholds and metric definitions intact. No test metrics are used for eligibility. | Regression test with a constant predictor whose reported PR-AUC exceeds the varying candidate, plus the corrected full-data run. |
| Live report verification rejected the genuine numeric age-band boundary `20.999` because interval labels were stored only as text. | The segment tool now exports `segment_lower_bound` and `segment_upper_bound` as numeric evidence. Verification remains strict; invented boundaries still fail. Missing numeric values are not turned into a string `nan` group. | Two new segment/verifier regressions and successful live report regeneration. |
| LightGBM needs the OpenMP dynamic library in this macOS environment. | Point `DYLD_LIBRARY_PATH` at the existing scikit-learn bundled library for the process. No package/model/protocol changes. | Full model suite used in the actual run. |

The initial failed run is retained at `runs/default_0ce6530d680a/`, including its
`decisions.jsonl`, `state.json`, `manifest.json`, and `report/partial_report.md`.
Its error was `data/train.csv: not a file in the run folder` after the duplicate
run-directory prefix. The first complete diagnostic run is `runs/default_2a0736aba43b/`. It exposed the
selection problem: stacking emitted exactly `0.5` for every validation/test row at
75% and 100% fill, with ROC-AUC `0.5`. Its trapezoidal PR-AUC was `0.610583`, so it
incorrectly appeared best under area-only selection. The frozen stacking meta-model
had zero coefficients and intercept. We did not retune that frozen model; the
corrected run excludes these non-ranking candidates from selection. The earlier
run is retained for diagnosis and must not be used as evidence of augmentation gains.

## Reproduce

From the repository root, with the locked environment installed:

```bash
export GOOGLE_CLOUD_PROJECT="ieor-test-project"
export GOOGLE_CLOUD_LOCATION="global"
export VERTEX_MODEL="google/gemini-3.5-flash"
# Authenticate once using an account with access to the team project:
gcloud auth application-default login

DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/sklearn/.dylibs" \
  .venv/bin/python -u -m agentic_gan.orchestrator --dataset default
```

The library-path setting is the local macOS workaround, not a model setting.
Each invocation creates a new run ID. The run's `config.yaml` preserves the exact
configuration and the manifest records input, artifact and dependency hashes.

## Results

Live run: **`runs/default_64d1723c076b/` — SUCCESS**.

The earlier corrected fallback run, `runs/default_3f5d737cbb55/`, is retained.
The live run's complete training metrics are identical to that run.

The first live narrative was rejected twice for the valid interval bound `20.999`,
then used the required template fallback. After fixing the segment evidence,
evaluation was replayed from verified saved training metrics/predictions. Its live
narrative passed without fallback. No models were retrained or prediction files
changed during replay. The decision log retains the original rejection/fallback
and the successful replay; evaluation has two stage attempts. The original report
and manifest are archived under `report_before_interval_fix/` and registered in
the final manifest.

- All 140 fits succeeded: four classifiers × five seeds × seven arm/fill combinations.
- Split: 18,000 training rows (3,982 positives); 6,000 validation and 6,000 test rows
  (1,327 positives each). No unresolved missing values.
- Baseline training: 17,986 rows, 3,980 positives, after removing 14 exact duplicates.
  Raw training and both holdouts retain their rows.
- SMOTENC adds 1,003 / 2,506 / 5,013 / 7,520 / 10,026 rows at the five fill levels.
- Ten constant stacking fits are retained but excluded from selection. All 35
  arm/seed groups select a nonconstant model using validation only.
- The manifest verifies all 447 registered artifacts (including six archived report/manifest files) with no problems. The final
  report passes number verification for 1,116 numeric tokens.
- The source SHA-256 is `f733b4e6b314bf28fd83d71af8da409b48747c005278d4963bbf53229326c78a`.
  Frozen protocol ID: `f933190def50f56e645d303da823aecbfb574c9367fee527e99f2fad8020289e`.
  The repository protocol file was not modified.
- Regression suite: **481 passed**; changed Python files pass Ruff and `git diff --check`.

Means over the five seeds, using the model selected on validation for each arm/seed:

| Arm | Fill | Test PR-AUC | Test F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| cleaned | 0% | 0.559845 | 0.538306 | 0.463034 | 0.642803 |
| original | 0% | 0.557932 | 0.539336 | 0.464335 | 0.643255 |
| smotenc | 10% | 0.560745 | 0.521495 | 0.418439 | 0.691937 |
| smotenc | 25% | 0.556798 | 0.508542 | 0.396395 | 0.713941 |
| smotenc | 50% | 0.552351 | 0.501414 | 0.391395 | 0.705200 |
| smotenc | 75% | 0.546577 | 0.485578 | 0.363537 | 0.738659 |
| smotenc | 100% | 0.542049 | 0.476843 | 0.349789 | 0.754333 |

Repair's PR-AUC change over raw is inconclusive. SMOTENC at 10% raises mean test
PR-AUC only slightly over cleaned data; that paired gain is inconclusive. Higher
fills trade precision and F1 for recall. The tool labels the PR-AUC comparison
against cleaned data at 50% and 75% as a reliable loss under its seed-spread rule;
100% remains inconclusive under the requirement to agree on validation and test.
None of the SMOTENC levels demonstrates a reliable PR-AUC gain over cleaned data.
These verdicts are descriptive, not statistical significance tests. The test set
was not used to select models or a fill level, and the volume rule did not run.
Rare demographic groups remain flagged in the segment results.

## Run artifacts

- [Configuration](../../runs/default_64d1723c076b/config.yaml)
- [Manifest](../../runs/default_64d1723c076b/manifest.json)
- [Decision log](../../runs/default_64d1723c076b/decisions.jsonl)
- [State](../../runs/default_64d1723c076b/state.json)
- [All model metrics](../../runs/default_64d1723c076b/models/metrics.csv)
- [Verified report](../../runs/default_64d1723c076b/report/final_report.md)
- [Comparison table](../../runs/default_64d1723c076b/report/comparisons.csv)
- [Segment metrics](../../runs/default_64d1723c076b/report/segments.csv)
- [Evidence and selection exclusions](../../runs/default_64d1723c076b/report/evidence.json)

Run folders are local ignored artifacts. This note is the shareable repository record;
sharing the note alone does not upload its linked models or prediction files. The
remaining scope gap is an implemented generator route for Default. Live Vertex access
and both LLM agent roles have now been exercised. Issue #88's run folder, manifest,
decision log, fix list and results note are available; this remains an explicitly
no-GAN experiment, not evidence of a completed four-arm comparison.
