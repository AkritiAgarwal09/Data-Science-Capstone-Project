# CTGAN on Default and Polish: rule check and gate results

The Default and Polish CTGAN runs (v1 recipe and Kyle's Bank v2 recipe) were done as EDA
experiments before the generator-selection rules were wired into the runners. This note records
what the rule check found, what was changed to close the gaps, and what is left as a team decision.
Rule IDs follow `generator-selection-rules-explained` (G = general, DC = Default of Credit Card
Clients, PB = Polish Bankruptcy). The implementation plan is `docs/model-selection-generator-rules-plan.pdf`.

Files: `data/interim/default/synthetic/default_ctgan[_v2]_fill{10,25,50}.csv`,
`data/interim/polish/synthetic/polish_ctgan[_v2]_fill{10,25,50}.csv`, and the single-horizon
`data/interim/polish/synthetic/polish_3year_ctgan[_v2]_fill{10,25,50}.csv`. Code:
`src/agentic_gan/tools/ctgan_fill.py` (shared runner body), `eda/ctgan_default/`, `eda/ctgan_polish/`.
Each `artifacts*/` folder holds `run_meta.json`, `diagnostics.txt`, `quality_v*.json`, the loss
curve, and the distribution plots.

## What changed

| Gap | Rule | Change |
|---|---|---|
| Fills were never scored | G-8 | `quality_fidelity` scorecard run on every fill file; results in `artifacts*/quality_v*.json` (tables below) |
| No Default or Polish sanity rules | DC-9, PB-9 | `DEFAULT_RULES` (PAY_1..6 in -2..8, payment amounts non-negative, SEX/EDUCATION/MARRIAGE code sets) and `POLISH_RULES` (ratios numeric, target binary, cohort label) registered in `quality_fidelity.RULES`; the runner applies them when writing files |
| Clipping hid out-of-range draws | PB-7 | `sample_synthetic` counts draws below/above the training range before clipping; written to `run_meta.json["out_of_range_before_clip"]` and listed in `diagnostics.txt` when above 1% |
| Runners did not consult the rule | G-1, G-5, G-9 | Both runners call `select_generator`; `run_meta.json["selection"]` records the rule steps, the selected generator, and an `override` line explaining why CTGAN was trained anyway |
| Polish trained on the pooled table | PB-1 | `run_ctgan_polish.py --horizon 3year` trains on one cohort; the pooled runs are kept as the comparison |

The four existing `run_meta.json` files were backfilled: the selection record was computed from the
saved training frame, and the range report was re-sampled (20,000 on-target rows) from the saved
model. The written CSVs are from the original sampling pass; `out_of_range_source` says so.

## Rule status after this work

Rules that were already followed before this work (G-2 training split only, G-3 gates in effect,
G-4 DCGAN excluded, DC-5, DC-7, PB-2, PB-4) are unchanged and not repeated here.

| Rule | Before | After | Note |
|---|---|---|---|
| G-1 coded selection, never LLM | not followed | followed | both runners call `select_generator` on the training frame; the result is in `run_meta.json["selection"]` |
| G-3 size, minority, repair, leakage gates | in effect | recorded | Default 18,000 rows / 3,982 positives; Polish 26,043 / 1,255 pooled, 6,194 / 294 for 3year; every gate step and its outcome is in `selection.steps` |
| G-5 decision order and a comparator | not followed | half | the order is run and logged (temporal test, type routing); the comparator run is still missing because the rule's primary generator is not implemented |
| G-6 conditional sampling, 20/35/50 ladder | partly | amended | v1 samples conditionally, v2 unconditionally with a target filter (kept, it is what fixed Bank's `job` skew); the ladder is amended below |
| G-7 retries and fallback | not applicable | not applicable | the gate now runs but the runners do not retry; a retry loop belongs to the agent, not an EDA script |
| G-8 one quality gate | not followed | fidelity half | `quality_fidelity` scorecard with pass/fail against suggested thresholds on every file; TSTR (downstream) and DCR (privacy) are not run here |
| G-9 logging | partly | followed | `run_meta.json` now also holds the rule that fired, the excluded generators, the override reason, the horizon, `imputation_fit_on`, and the range report |
| DC-1 TimeGAN primary on the monthly block | not followed | not followed, recorded | TimeGAN is not implemented; `selection.override` says CTGAN is the DC-4 fallback |
| DC-4 CTGAN on the flattened wide table | followed, out of order | followed, labelled | recorded as the fallback it is, not as the first choice |
| DC-9 sanity rules | followed by clipping | followed by check | `DEFAULT_RULES` registered; 0 violations in real and synthetic files |
| PB-1 pin one horizon, never pool | not followed | followed for 3year | pooled runs kept as the comparison and marked `status: skipped` with the refusal text |
| PB-3 WGAN-GP primary, CTGAN comparator | half | half, labelled | WGAN-GP not implemented; the CTGAN run is recorded as the PB-3 comparator |
| PB-7 count out-of-range values | not followed | followed | per-column counts before clipping in `run_meta.json`, worst columns in `diagnostics.txt` |
| PB-8 20/35/50 ladder | not followed | amended | see below |
| PB-9 ratio and target sanity rules | followed | followed by check | `POLISH_RULES` registered; 0 violations |

## Quality gate (G-8)

Thresholds are 2x the real-vs-real distance (`--thresholds suggested`). The reference is the real
minority rows of the training split; for the 3year files it is the 3year cohort only (294 rows), so
its noise floor and limits are higher. Numbers are the 50% fill; the 10% and 25% files are within
0.005 of it on every check because they are prefixes of the same sample.

### Default (positive = default next month, 22%)

| Check | Limit | v1 | v2 |
|---|---|---|---|
| Mean column distance | 0.065 | 0.155 | 0.083 |
| Worst column distance | 0.085 | 0.355 (`SEX`) | 0.250 (`PAY_AMT6`) |
| Spearman difference | 0.055 | 0.095 | 0.059 |
| Cramer's V difference | 0.050 | 0.147 | 0.043 |
| Verdict | | FAIL | FAIL |

v2 is close: it passes Cramer's V, the mean is 0.018 over and Spearman 0.004 over the limit. The
remaining miss is the payment-amount columns, which are heavy-tailed with a spike at zero.

### Polish, pooled table (positive = bankrupt, 4.8%)

| Check | Limit | v1 | v2 |
|---|---|---|---|
| Mean column distance | 0.100 | 0.434 | 0.286 |
| Worst column distance | 0.175 | 0.952 (`A27`) | 0.694 (`A58`) |
| Spearman difference | 0.075 | 0.211 | 0.173 |
| Verdict | | FAIL | FAIL |

### Polish, 3year cohort only (294 real positives)

| Check | Limit | v1 | v2 |
|---|---|---|---|
| Mean column distance | 0.230 | 0.441 | 0.344 |
| Worst column distance | 0.445 | 0.815 (`A60`) | 0.957 (`A52`) |
| Spearman difference | 0.185 | 0.332 | 0.228 |
| Verdict | | FAIL | FAIL |

Pinning the horizon did not help. 3year v1 is no better than pooled v1 (mean 0.441 against 0.434),
and 3year v2 is worse than pooled v2 (0.344 against 0.286) even though its limits are looser.
294 positives is too few for CTGAN: 19 of 64 (v1) and 17 of 64 (v2) ratio columns had more than 1%
of draws outside the training range before clipping, against 5 and 7 for the pooled runs. The v2
recipe still closes a quarter of the gap on the mean and Spearman checks, as it did on the pooled
table, but the worst column gets worse (`A52` 0.957). Training took 5 minutes (v1) and 11 minutes
(v2) against 20 and 55 for the pooled table.

## Out-of-range draws before clipping (PB-7)

Share of on-target draws that fell outside the training min/max of each column, worst columns:

| Run | Columns above 1% | Worst |
|---|---|---|
| Default v1 | 7 of 14 | `PAY_AMT3` 26.9%, `PAY_AMT5` 26.3%, `PAY_AMT2` 12.6% |
| Default v2 | 6 of 14 | `PAY_AMT1` 27.6%, `PAY_AMT4` 27.5%, `PAY_AMT3` 19.8% |
| Polish pooled v1 | 5 of 64 | `A4` 69.4%, `A60` 27.8%, `A47` 15.3% |
| Polish pooled v2 | 7 of 64 | `A17` 26.9%, `A50` 25.2%, `A4` 19.4% |
| Polish 3year v1 | 19 of 64 | `A60` 81.7%, `A37` 57.0%, `A8` 53.7% |
| Polish 3year v2 | 17 of 64 | `A47` 66.7%, `A4` 43.3%, `A37` 43.0% |

For Default the overshoot is almost all below zero on the payment amounts: the generator does not
learn the point mass at zero and clipping turns negative draws into exact zeros, which is why the
zero share in the synthetic files is too high. For Polish the worst columns are ratios with a
handful of extreme real values (`A4`, `A60`), so the training range itself is wide and the clipped
rows sit on its edge. Clipping is kept (the files must stay inside the real range) but the counts
now show how much of each column is clipped rather than generated.

## Team decision: DC-1 and PB-3

The rule picks TimeGAN for Default's monthly block and WGAN-GP for the all-numeric Polish table.
Neither is implemented, so CTGAN was trained as the documented fallback (DC-4) and comparator
(PB-3). Two ways to close this:

- **Option A: add indicator columns in Polish preprocessing** so the type-routing step sees a mixed
  table and the rule picks CTGAN. This makes the current runs the rule's choice rather than the
  fallback. It changes `preprocessing.md` and the cleaned files, so it was not done on this branch
  (39 commits behind `main`); it needs to be a separate change against `main`.
- **Option B: implement WGAN-GP (and TimeGAN) and train them**, keeping the CTGAN runs here as the
  comparator the rule asks for. This is the larger piece of work and the one the rule document
  actually describes.

Option A does nothing for Default, where the rule's choice is TimeGAN regardless of column types.
Recommendation: treat the CTGAN runs as the PB-3 comparator and DC-4 fallback they are labelled as,
and decide A or B when the main branch's model-selection work (`status-issues-55-and-56.pdf`) lands.

## Amendment: G-6 and PB-8 fill ladder

The rule document lists a 20/35/50 target-rate ladder (the minority share of the augmented
training set). The architecture (v0.2, `docs/architecture.md`, "Fill level") and the implementation
plan (section 3.7) define volume as a fill level instead: the share of the training class gap
filled with positive rows, `fill_rows` in `schemas.py`, on a 10/25/50 grid. That is what Bank,
Default, and Polish were all generated at and what the downstream comparison (#35) consumes. The
fill-level grid supersedes the ladder in the rule text; `FILL_LEVELS` in `ctgan_fill.py` is the
single definition in the runners. The rule document should be updated to say so.

## Open items

- Neither Default nor Polish passes the quality gate. Default v2 is within 0.02 on every check and
  a payment-amount fix (model the zero spike separately, or sample more rows and reject clipped
  ones) is the next thing to try.
- Polish needs more than a recipe change: the pooled table violates PB-1 and the single-horizon
  table has too few positives for CTGAN. Pooling horizons as a feature (keep `cohort`, which the
  files already carry) against the rule, or waiting for WGAN-GP, are the two routes.
- The range report is computed from the sampling pass; the four pre-existing runs were re-sampled
  from the saved model, so their counts describe the model, not the exact rows written.
