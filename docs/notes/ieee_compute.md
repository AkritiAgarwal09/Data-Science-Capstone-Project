# CTGAN on IEEE-CIS: compute plan

Issue #90. Can the CTGAN stage finish on IEEE-CIS, and with what sample and time limit? Answer:
yes on a laptop CPU, if CTGAN trains on the fraud rows only (3,383 rows, 300 epochs, about 29
minutes). Training on the full 100,000-row sample would take about 15 hours.

## What IEEE-CIS costs

The cleaned training sample has 100,000 rows and 540 columns (539 features plus `isFraud`), 3.4%
fraud (3,383 rows), with about 1,130 categories in 370 discrete columns. CTGAN's internal width is
1,710 to 1,818 (Bank's is far smaller). Timing runs on CPU, batch size 500, `pac` 10
(`eda/ctgan_ieee/time_trial.py`, results in `eda/ctgan_ieee/artifacts/time_trials.csv`):

| Training rows | Epochs | Seconds |
|---|---|---|
| 5,000 | 1 | 101.8 |
| 5,000 | 3 | 120.3 |
| 20,000 | 2 | 193.2 |
| 3,383 (fraud only) | 3 | 107.0 |
| 3,383 (fraud only) | 300 | 1,731.5 (the full run) |

CTGAN takes a fixed time per training step, about **0.93 s** at batch size 500, and an epoch is
rows / 500 steps. On top of that there is about **2 minutes** of one-off set-up (fitting the numeric
columns) that barely grows with the rows. So:

    minutes = 2 + 0.93 x epochs x rows / 500 / 60

The 5,000 and 20,000-row runs fit this, and it predicted the full fraud-only run (about 29 minutes)
to the minute. The estimates for larger tables are extrapolations; none was run.

| Training data | 100 epochs | 300 epochs |
|---|---|---|
| Fraud rows only (3,383) | about 10 min | about 29 min (measured) |
| 10,000 rows | about 31 min | about 1.6 h |
| 20,000 rows | about 1.1 h | about 3.1 h |
| 100,000 rows (the full sample) | about 5 h | about 15 h |

## Decision

- **Train on the fraud rows only**, 300 epochs, batch 500, `pac` 10, one discriminator step (the
  CTGAN v1 settings). Every generated row is fraud, so no target condition is needed when sampling.
- **Time limit: 60 minutes of CPU per training run.** The measured run took 29 minutes. The v2
  settings (`pac` 5, three discriminator steps) cost about twice as much on Bank, so would be about
  an hour here (not measured).
- **GCP is not needed** for this stage.
- The settings are recorded in `src/agentic_gan/utils/datasets.json` under `ieee_cis.ctgan`
  (`train_on: minority_rows`, `epochs`, `batch_size`, `pac`, `discriminator_steps`,
  `max_minutes`); a test checks they fit the limit.

## The full run

`eda/ctgan_ieee/run_ieee_ctgan.py --label v1`: training 1,731.5 s (28.9 min), sampling 55,950 rows
to keep 46,617 in 21 s. The fill levels follow the usual rule, a share of the gap between the
classes in the 100,000-row sample (93,234 rows): 10% = 9,323, 25% = 23,308, 50% = 46,617 rows,
saved as nested files `ieee_ctgan_v1_fill{10,25,50}.csv` in `data/interim/ieee_cis/synthetic/`
(ignored by git, like all derived IEEE-CIS data). Each has the 539 features, `isFraud` = 1 and
`is_synthetic` = 1, with no missing values.

## First look at the quality (not a verdict)

Against the real fraud rows, with the default fidelity limits and no real-vs-real noise floor
(there are only 3,383 real fraud rows to compare with): mean column distance 0.052, Spearman
difference 0.119, Cramer's V difference 0.070. The worst columns are `C11` (0.73), `V130` (0.70),
`C8` (0.70), `V320` and `V313` (0.65); 27 of 539 columns are above 0.25. These are untuned CTGAN v1
settings, so a quality-gate run and the Bank-style fixes (v2 settings, raking) are the next step.
Memorization with 3,383 rows has not been checked.

## Which table these numbers are for

The timings above are for the cleaned tables from `tools.preprocessing --dataset ieee_cis`. They
keep the 50 most common values of each category column plus `__missing__` and `__other__`
(`IEEE_CATEGORY_LIMIT`), so CTGAN's width is 1,710 to 1,818.

The end-to-end note (`docs/notes/e2e_ieee_cis.md`, #85) measured the table the pipeline's own
preprocessing agent builds: 181 columns and a one-hot width of 8,648, because `card1` alone keeps
7,663 categories. It estimated 35 hours for 100,000 rows. Both are right for their table. That
note's fix, "bucket `card1`, `card2`, `addr1` and the other high-cardinality columns", is what the
cleaned tables above already do.

If CTGAN were trained on the un-bucketed pipeline table, the cost per row per epoch would be the
0.0042 s that #85 measured. The fraud-only run (3,383 rows, 300 epochs) would then take about 71
minutes, over the 60-minute limit. That is an extrapolation, not a run. So the pipeline should train
CTGAN on the cleaned tables above, or apply the same 50-category cap first.

## Not done

- **The pipeline does not use these settings yet.** `agents/model_selection.py::_train_ctgan` trains
  on the whole baseline training table with fixed defaults and reads nothing from
  `DATASETS[...].ctgan`. On IEEE-CIS that is the 15-hour run. It needs to read the `ctgan` settings
  and train on the minority rows, then sample with the single-class model (conditional sampling on a
  model trained on one class has not been tried). This is for the end-to-end IEEE-CIS run (#85).
- One timing run per setting; the machine was otherwise idle.
- The quality gate has not been run on these files.

## Reproduce

```bash
uv run python -m agentic_gan.tools.preprocessing --dataset ieee_cis      # needs data/external/
uv run python eda/ctgan_ieee/time_trial.py --fraud-only --epochs 3        # timing
caffeinate -i uv run python eda/ctgan_ieee/run_ieee_ctgan.py --label v1   # about 30 minutes
```
