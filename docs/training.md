# Model Training Agent

`agents.training.run_training(request, run)` implements the orchestrator's
`TrainingRequest -> TrainingOutput` handoff after the quality gate. The graph uses
this callable by default. Other stages can be supplied through `build_graph(agents=...)`;
the default graph uses real dataset splitting, preprocessing, model selection and quality
stages, followed by the verified evaluation report (see `docs/evaluation.md`).

## Inputs and frozen settings

The `raw` and `baseline` `ArmFiles` contain train, validation and test CSVs, **before
encoding**, with the target and a `source_row` column. Raw tables match `load_dataset`
output with configured leakage/provenance exclusions when a treatment plan is supplied.
Baseline tables match the supplied treatment plan fitted on deduplicated real training
rows; legacy requests without a plan use the shared deterministic `Preprocessor`.
Bank labels can be `yes`/`no` in raw files and must be 0/1 in cleaned files. Numeric
category codes are normalized on reading CSVs. `source_row` is never a feature.

The agent validates disjoint, unique row IDs; source-file and split hashes; labels
and row order; dataset settings; package versions; model parameters; and the protocol
checksum. It also compares supplied raw/cleaned table contents to the source and
training-only cleaning. Any synthetic flag in a real file must be exactly zero;
missing or ambiguous flags are rejected. This prevents an unflagged replacement
row or a changed holdout from silently entering the comparison.

For IEEE-CIS, the source hash covers both transaction and identity CSVs. The original
and cleaned feature matrices both omit `TransactionID` and `TransactionDT`; the time
column defines the split but never enters model fitting.

The reference is `results/baseline_protocol_<dataset>_<split_seed>_cv<folds>.json` from
issue #52. The Training Agent uses the canonical CV3 file when present, or discovers a
single alternate protocol such as the refreshed CV0 file. It reads the fold setting from
the frozen protocol, allowing a validation-only
protocol (`cv_folds=0`) when full-data cross-validation is too costly. IEEE-CIS uses that
setting for its 100k-row training sample. Override the path with
`RunConfig.training_protocol_path` or the `protocol_path=` keyword.
Each dataset needs its own original-arm reference. IEEE-CIS now has a frozen original-arm
protocol at `results/baseline_protocol_ieee_cis_42_cv0.json`. Run this command to check both
arms against its saved thresholds or to create the protocol in a fresh results folder:

```bash
python -m agentic_gan.tools.baselines --dataset ieee_cis --arm both --cv-folds 0
```

The training agent reports this setup step if the protocol is missing. The IEEE-CIS
original arm keeps one-hot matrices sparse so high-cardinality payment fields fit within
memory; vocabulary fitting still uses training rows only.
The models are gradient boosting, LightGBM, CatBoost, and stacking.
`candidate_models` must match that complete suite. No tuning or probability
calibration occurs here; feature exclusions, changed splits, or changed settings
require a new reference experiment.

The protocol's seed fixes the data split and original-validation F1 thresholds.
`RunConfig.seeds` varies estimator/SMOTENC randomness **without resplitting or
recalibrating**. Thus each model uses the same numerical threshold across every
arm, fill level, and training seed. The stored protocol is copied into the run.
This is a fixed-split seed experiment, distinct from rerunning the baseline CLI
with a different split seed.

## Arms and saved results

| Schema arm | Results label | Training rows |
|---|---|---|
| `raw` | `original` | Raw real train, with training-fitted imputation/encoding/scaling |
| `baseline` | `cleaned` | Cleaned real train |
| `smote` | `smotenc` | Cleaned real train plus SMOTENC rows at each configured fill level |
| `gan` | `ctgan` | Cleaned real train plus each quality-accepted synthetic set |

SMOTENC is regenerated for each seed. Within a seed, the fill levels use prefixes
of one generated pool. Counts use `fill_rows(level, positives, negatives)` from
the real cleaned training split. A level that adds zero rows has explicit skipped
records. CTGAN inputs are already generated and quality-accepted; the Training Agent
does not refit the main generator. The same accepted pool is used across model seeds,
so that seed spread does not measure CTGAN training randomness. No accepted sets means
the other three arms still run. Synthetic tables must contain exactly the cleaned
columns plus `is_synthetic=1`, have the declared count, and have positive-class labels.

`TrainingRequest.treatment_plan_path` and `treatment_plan_sha256` carry the preprocessing
agent's JSON recipe. Training verifies its checksum and dataset settings, and compares
cleaned tables against a fresh fit of that recipe on the audited raw training rows.
The plan is logged; no pickle is needed for this validation. Both fields are optional
for legacy deterministic requests.

With a plan, `ArmEncoder` fits repair models, encoding and scaling on real raw rows,
including when cloned inside stacking folds. Synthetic rows stay cleaned and do not
set repair statistics, category vocabularies or numeric scaling. Real holdout rows
are repaired using that fit's transformer. Repeated outer CV rebuilds cleaned training
and evaluation tables from the raw fold rows before generating augmentation. The
chosen treatment actions stay fixed across folds; all learned values are refitted.
Derived missing flags are included in SMOTENC's discrete columns.

Each `ModelRecord` carries status, hyperparameters, threshold, protocol ID, row
counts, validation/test metrics, a model artifact, and both prediction paths.
Failures have a reason and make the overall `TrainingOutput` fail; successful fits
are retained for diagnosis. Warnings such as non-convergence are recorded per model.

Artifacts live under the run's `models/` directory and are returned for manifest
registration:

- One `.joblib` dictionary per arm/level/model/seed: `estimator` (including its
  encoder), `features` (input column order), `threshold`, and `protocol_id`.
  Inference inputs omit target/row IDs and include `is_synthetic=0` for real rows.
- Validation and test prediction CSVs: `source_row`, `y_true`, `probability`,
  `prediction`, `threshold`, and `is_synthetic=0`. The decision rule is `p >= threshold`.
- `metrics.csv`: one row per split and model combination, with PR-AUC, F1,
  precision, recall, ROC-AUC, and accuracy last. PR-AUC uses the same trapezoidal
  precision-recall integration as #52, not average precision.
- `validation_summary.csv`: mean, sample standard deviation, and count over seeds,
  grouped by arm/fill/model. A single seed has an undefined standard deviation,
  not an artificial zero. This is the validation-only input for the future #36
  volume rule; this agent does not choose a volume.
- `training_protocol.json`: the exact frozen reference.

The selected model per arm/fill/seed maximizes real validation PR-AUC among fits
with nonconstant validation probabilities; ties retain suite order. A constant
predictor cannot rank rows, but trapezoidal PR-AUC can still give it a high area
because of endpoint interpolation. Its fit, scores and predictions are retained
with `selected=False`, a warning, and an audit decision. `metrics.csv` records
`validation_probability_range` and `selection_eligible`. If every successful fit
is constant, no model is selected for that group. Eligibility uses validation only;
the frozen PR-AUC definition, estimator settings and thresholds are unchanged. Test predictions and metrics are saved for **all** fitted models as
requested by the agent contract, unlike the winner-only baseline CLI. Test metrics
are descriptive and never choose models, thresholds, fill levels, or seeds.

## Optional repeated cross-validation

Set `training_cv_folds` to at least 2 and `training_cv_repeats` to a positive integer
(default 3). Zero folds, the default, disables it. Repeated stratified folds combine
real train and validation rows and never include test. Row partitions are shared
across arms and model seeds. Each outer fold refits cleaning from raw fold-training
rows, refits encoding on real rows, and regenerates SMOTENC only from fold training.
This option applies to the currently supported stratified datasets; a future
chronological fraud protocol needs a corresponding time-aware evaluation design.

The agent saves `cv_metrics.csv` and `cv_predictions.csv` with repeat/fold identifiers
and real row IDs. These report threshold-independent PR-AUC and ROC-AUC. They do not
report CV F1: the frozen threshold already used original validation labels, so F1
on pooled train+validation would reuse calibration labels. CV is a supplementary
stability analysis, not a replacement test set or a set of independent replicates.

The accepted full-training CTGAN rows **cannot** be reused inside CV because some
of their generator's training rows would lie in the CV holdout. For CTGAN CV, supply
`cv_gan_factory(cleaned_fold_train, row_count, seed)`. The factory must fit a fresh
generator, apply its quality checks using only the supplied fold training data,
and return a flagged synthetic DataFrame. It receives no holdout/test rows. Without
this factory, CTGAN CV gets explicit `SKIPPED` rows and an output warning; the main
quality-accepted CTGAN experiment still runs normally.

```python
from agentic_gan.agents.training import run_training

# request is a validated TrainingRequest; run is the orchestrator's RunDir.
request = request.updated(config=request.config.updated(
    seeds=[42, 43, 44],
    training_cv_folds=3,
    training_cv_repeats=3,
))
output = run_training(request, run)
```

Tests in `tests/test_training_agent.py` cover all arm/level/seed combinations,
frozen thresholds, serialization and manifest registration, provenance rejection,
failure propagation, deterministic reruns, fold-local generation, a real four-model
smoke run, and the orchestrator's filtering of quality-rejected fill levels.

## Running every arm on every dataset

`tools/training_grid.py` builds a run folder and a `TrainingRequest` from the files that
already exist (the raw source, the cleaned splits in `data/interim/<dataset>/`, and optionally
CTGAN files), calls `run_training`, and updates two tables in `results/`. It stops with an
error if any saved validation or test prediction file contains a synthetic row.

```bash
uv run python -m agentic_gan.tools.training_grid --dataset default \
    --ctgan-prefix data/interim/default/synthetic/default_ctgan_v2_fill
uv run python -m agentic_gan.tools.training_grid --dataset ieee_cis --seeds 42 \
    --ctgan-prefix data/interim/ieee_cis/synthetic/ieee_ctgan_v1_fill
```

`--ctgan-prefix` plus the fill level as a whole percent plus `.csv` names each CTGAN file
(`..._fill10.csv`, `_fill25`, `_fill50`). Without it the CTGAN arm is not run. Each run
replaces its own dataset's rows, so a dataset can be rerun without touching the others (extra
seeds cannot be added later; run them together with `--seeds`). Use `--no-save` to leave
`results/` alone. IEEE-CIS needs its local files in `data/external/` (not in git) and the cleaned
tables from `uv run python -m agentic_gan.tools.preprocessing --dataset ieee_cis`.

Outputs: `results/training_all_arms.csv` (the `metrics.csv` rows of every run) and
`results/training_run_times.csv`. Run times are on a laptop CPU and include fitting every model
for every arm.

| Dataset | Seeds | Model fits | Run time |
|---|---|---|---|
| Bank Marketing | 5 | 160 | 157 s |
| Default | 5 | 160 | 117 s |
| Polish | 5 | 160 | 238 s |
| IEEE-CIS | 1 | 32 | 1,694 s (28 min) |

Coverage: every arm ran on every dataset, with no failed fits, and no synthetic row reached a
validation or test prediction file (the runner stops if one does).

| Dataset | original | cleaned | SMOTENC | CTGAN files |
|---|---|---|---|---|
| Bank Marketing | run | run | run | v2 (#54) |
| Default | run | run | run | v2 (#55) |
| Polish | run | run | run | v2, pooled cohorts (#55) |
| IEEE-CIS | run | run | run | v1, trained on the fraud rows only (#90) |

Test PR-AUC of each seed's validation winner, mean over the seeds (the models and thresholds
are the frozen ones, so these differ from the #35 comparison, which used a fixed cutoff of
0.5 and its own provisional models):

| Dataset | original | cleaned | SMOTENC 10 / 25 / 50% | CTGAN 10 / 25 / 50% |
|---|---|---|---|---|
| Bank Marketing | 0.464 | 0.478 | 0.459 / 0.449 / 0.438 | 0.476 / 0.466 / 0.448 |
| Default | 0.558 | 0.564 | 0.560 / 0.558 / 0.552 | 0.555 / 0.549 / 0.535 |
| Polish | 0.835 | 0.835 | 0.815 / 0.809 / 0.793 | 0.828 / 0.820 / 0.807 |
| IEEE-CIS (1 seed) | 0.453 | 0.409 | 0.414 / 0.412 / 0.400 | 0.425 / 0.435 / 0.410 |

On Bank, Default and Polish, adding synthetic rows does not beat the cleaned arm they are added
to, and more rows lower PR-AUC. CTGAN loses less than SMOTENC on Bank and Polish; SMOTENC is
slightly better on Default. On Polish the original and cleaned arms give identical results,
because the cleaning changes nothing there. On IEEE-CIS the cleaned arm is itself below the
original (0.409 against 0.453); the synthetic arms recover part of that gap (CTGAN up to 0.435)
without passing the original.

**Caveats**

- **IEEE-CIS has one seed (42), not five**, so it has no seed spread and is not comparable to the
  other rows in that way. Five seeds would be about 160 fits and about 2.5 hours (28 minutes per
  seed, mostly fitting the models and building the SMOTENC pool).
- **IEEE-CIS CTGAN files are untuned.** They are the v1 settings trained on the 3,383 fraud rows
  (`docs/notes/ieee_compute.md`) and have only had a first look, not the quality gate.
- **Polish uses the pooled CTGAN files.** The training agent's fill counts come from the pooled
  training table, so the single-horizon `3year` files (561 / 1,402 / 2,803 rows) do not fit.
  `docs/notes/ctgan_default_polish_rules.md` says pooling the horizons breaks rule PB-1.
- **Polish CTGAN has no seed spread** (SD 0.000): the CTGAN pool is fixed and the same model
  wins every seed.
- **The stacking ensemble** scores near zero on IEEE-CIS (it predicts almost every row as fraud)
  and never wins; the selection rule excludes constant-probability fits.

**IEEE-CIS exposed one bug, now fixed.** The cleaned, SMOTENC and CTGAN arms failed with
`Reserved categorical bucket appears in source column card1`: the encoder is fit on the cleaned
table, which already holds IEEE-CIS's `__missing__` and `__other__` buckets, and
`Preprocessor` rejected them. `Preprocessor(..., from_cleaned=True)` now accepts them (leaving
them out of the category counts), `ArmEncoder` uses it, and the encoder is identical to the one
the preprocessing run builds. The guard against those values in raw data is unchanged
(`tests/test_ieee_preprocessing.py`).

## LLM run plan and summary

`run_training` can plan the run and summarize it with an LLM (issue #80), built on the
shared runtime in [llm_agents.md](llm_agents.md). The models, frozen thresholds and
metrics stay in code. The LLM never fits, scores or chooses a model.

It runs when the run has a quality report: pass `quality=` to `run_training`, or let it read
the newest `quality/quality_report_attempt_*.json` in the run folder (the quality agent writes
it, so the graph needs no change). Without a report there is no LLM call: the default plan
runs and the summary is a template, both logged.

**The plan.** `plan_training` sends the LLM the quality report (which levels passed and which
hard checks failed), the dataset sizes and any small-evaluation-set warning (fewer than 100
positives in validation or test). It returns a `TrainingPlan`:

| Field | Allowed values |
|---|---|
| `smote_levels` | levels in the config's fill-level grid (may be empty) |
| `gan_levels` | only levels with an accepted synthetic set that **passed** the quality gate (may be empty) |
| `seeds` | seeds from the config, at least one |
| `repeated_cv` | yes or no; it cannot be switched off if the config asks for CV |
| `reason` | one line, up to 300 characters |

The original and cleaned arms always run. A plan with a value outside these (a failed level,
an unknown seed, a repeat, an extra field such as a new threshold) is retried once and then
replaced by the default plan: every allowed level, the configured seeds and CV setting. That
fallback is logged as `LLM_FALLBACK`. The plan used is saved as `models/training_plan.json`.

**The summary.** After training, `models/run_summary.md` holds a short summary written from the
result table: mean test PR-AUC and F1 of each seed's validation winner for each arm and fill
level, the plan, the failure count and the warning. Every number in the LLM's text must appear
in that table, or the text is rejected, retried once and replaced by a template written from
the same table (logged as `LLM_FALLBACK`). Test scores are descriptive and never choose
anything.

**Locked rules.** Before and after each LLM call the agent hashes the run config and the frozen
protocol (thresholds, models). If either changed, the call raises `RuleOverrideError` and the
decision is logged as `RULE_OVERRIDE_REJECTED`. Every LLM answer, fallback and rejection is in
`decisions.jsonl`.

Tests use the fake LLM (`FakeLLM`) and no network. One live test calls Vertex and is skipped
unless you opt in; set up Vertex access as in [llm_agents.md](llm_agents.md), then:

```bash
RUN_LIVE_LLM=1 uv run pytest -m live
```
