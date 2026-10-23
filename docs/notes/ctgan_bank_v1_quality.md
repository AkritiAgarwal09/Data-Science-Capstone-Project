# CTGAN v1 on Bank Marketing: quality check results

Issue #49. The fidelity and rule checks come from `src/agentic_gan/tools/quality_fidelity.py` (issue #46). The privacy, distinguishability and utility checks come from `src/agentic_gan/tools/quality_privacy.py` (issue #47). The script is `eda/ctgan_bank/quality_v1.py` and its outputs are in `eda/ctgan_bank/artifacts/quality_v1/`.

## Summary

- **The gate fails v1, as it should.** Mean column distance is 0.106 against 0.023 real-vs-real, the worst column (`job`) is 0.27 to 0.28 against 0.048, and 1.5% to 1.8% of rows break the `pdays` / `poutcome` rule against 0.06% in real data.
- **Seven of 16 columns are far off** (distance above 0.10): `job`, `day_of_month`, `month`, `pdays`, `marital`, `poutcome`, `housing`. Technician is 42% of synthetic subscribers and 15% of real ones.
- **Easy to tell apart.** A random forest separates v1 rows from real subscribers with an AUC of 0.85 to 0.87 (real-vs-real 0.52). The columns that give them away are `job`, `day_of_month`, `month`, then `balance` and `pdays`.
- **No sign of memorization.** No exact copies, and v1 rows sit farther from real train subscribers than unseen real subscribers do (ratio 1.11).
- **Utility falls as v1 rows are added.** PR-AUC drops by 0.025, 0.052 and 0.074 at the 10%, 25% and 50% fill levels, and best-threshold F1 drops by 0.03, 0.05 and 0.08.
- **The columns that matter most are among the ones it gets wrong.** `month`, `poutcome` and `day_of_month` drive the prediction and are all over the limit. `job`, the biggest miss, barely matters.
- **Fidelity does not depend on the fill level.** The three files are nested samples of one pool (the 10% file is the first 2,078 rows of the 50% file), and the scores are the same (0.106 to 0.108). The generator sets the quality, not the number of rows.
- **Two starting limits need changing:** the closest-record ratio of 1.0 fails real data about half the time, and the rule limit of 0 fails real data too (see "Recommendations").

## What was checked

| | |
|---|---|
| Synthetic | `data/interim/bank_marketing/synthetic/bank_ctgan_fill{10,25,50}.csv`: 2,078, 5,195 and 10,390 subscriber rows |
| Reference | the 3,173 real train subscribers (`y = 1`) |
| Noise floor | the 1,058 real validation subscribers against the same train subscribers (real-vs-real) |
| How v1 was made | CTGAN on all 27,126 train rows (11.7% subscribers), 300 epochs, batch 500, `pac` 10, CPU, seed 42 |
| Seeds | 42 to 46 for the classifier runs. The other checks do not depend on the seed. |

The sampler also did three things on its own, so three checks pass by construction and say nothing about CTGAN: it kept only `y = 1` rows (label balance), it applied the row filter `consistent_rows` (the `previously_contacted`, `previous` and `pdays` rules), and it dropped exact copies of real train rows (exact copies).

## Fidelity and rule checks

A value above the limit fails. Distances are between 0 and 1. The rule rows are the share of rows that break the rule.

| Check | Real-vs-real | fill10 | fill25 | fill50 | Starting limit | Suggested limit |
|---|---:|---:|---:|---:|---:|---:|
| Mean column distance | 0.023 | **0.108** | **0.106** | **0.106** | 0.10 | 0.05 |
| Worst column distance (`job`) | 0.048 | **0.283** | **0.279** | **0.274** | 0.25 | 0.10 |
| Spearman difference (numeric pairs) | 0.020 | 0.067 | 0.066 | 0.066 | 0.10 | 0.045 |
| Cramér's V difference (categorical pairs) | 0.031 | 0.053 | 0.046 | 0.046 | 0.10 | 0.065 |
| Rule: `pdays` is -1 exactly when `poutcome` is unknown | 0.0006 (2 of 3,173; 0 of 1,058 validation) | **1.5%** | **1.6%** | **1.8%** | 0 | 0.5% |
| Rule: `previously_contacted` matches `pdays` | 0 | 0 | 0 | 0 | 0 | 0.5% |
| Rule: `previous` above 0 exactly when `pdays` is not -1 | 0 | 0 | 0 | 0 | 0 | 0.5% |
| Rule: `pdays` is -1 or at least 1 | 0 | 0 | 0 | 0 | 0 | 0.5% |
| Label balance (share of `y = 1` off 100%) | 0 | 0 | 0 | 0 | 0.01 | 0.01 |

Bold values fail the suggested limit. Under the **starting thresholds** (`QualityThresholds` defaults) v1 fails the mean distance (0.106 and 0.108 against 0.10), the worst column (0.27 to 0.28 against 0.25) and the `pdays` rule. Under the **suggested thresholds** it also fails the Spearman difference, and the Cramér's V difference passes. The real-vs-real baseline passes every suggested limit.

The `pdays` rule breaks in both directions: `pdays` is -1 with a known `poutcome` in 18, 37 and 72 rows, and `pdays` is above -1 with an unknown `poutcome` in 13, 46 and 112 rows (fill10, fill25, fill50). The issue's first pass said about 1%; this script counts both directions and gets 1.5% to 1.8%.

### Failing columns

Distances for fill50, which has the most rows and so the least noise. All 16 columns are in `column_distances.csv`.

| Column | Real-vs-real | v1 | What is off |
|---|---:|---:|---|
| `job` | 0.048 | 0.274 | technician 42% vs 15%; admin. 3% vs 12%; services 2% vs 7%; management 20% vs 25% |
| `day_of_month` | 0.016 | 0.230 | later days: median 18 vs 15, upper quartile 27 vs 22 |
| `month` | 0.040 | 0.174 | jul 18% vs 12%, apr 15% vs 11%; feb 4% vs 9%, jun 6% vs 10% |
| `pdays` | 0.029 | 0.134 | longer gaps: upper quartile 186 vs 95 days |
| `marital` | 0.012 | 0.132 | divorced 25% vs 12%; married 43% vs 51% |
| `poutcome` | 0.026 | 0.117 | other 16% vs 6%; success 11% vs 18% |
| `housing` | 0.014 | 0.113 | yes 48% vs 37% |

Close to the limit but under it: `campaign` 0.099 (heavier tail, 95th percentile 8 vs 5), `education` 0.089 to 0.098, `balance` 0.093 (negative balances and a lighter upper tail), `age` 0.078 (fewer older customers). The rest are under 0.06.

## Privacy, distinguishability and utility

From `quality_privacy.py`, mean over 5 seeds (`privacy_utility.csv`). The classifier is a random forest on equal-sized real and synthetic samples. The utility check fits a logistic regression on the real train rows, with and without the synthetic rows added as positives, and scores it on the real validation rows.

| Check | Limit | fill10 | fill25 | fill50 |
|---|---|---:|---:|---:|
| Exact copies of real train rows | 0 | 0 | 0 | 0 |
| Repeated synthetic rows | 0 | 0 | 0 | 0 |
| Closest-record ratio (median distance of synthetic rows to real train subscribers, over the same for real validation subscribers: 1.745, 1.728, 1.731 against 1.563) | at least 1.0 | 1.117 | 1.106 | 1.108 |
| Real-vs-synthetic AUC (real-vs-real: 0.518 ± 0.030) | diagnostic, 0.5 is indistinguishable | 0.854 ± 0.015 | 0.863 ± 0.005 | 0.865 ± 0.004 |
| Utility: PR-AUC, real only 0.405 to real plus synthetic | gain of at least 0 | 0.380 (**-0.025**) | 0.353 (**-0.052**) | 0.331 (**-0.074**) |
| Utility: best-threshold F1, real only 0.448 to real plus synthetic | gain of at least 0 | 0.416 (**-0.031**) | 0.396 (**-0.052**) | 0.371 (**-0.076**) |

- **Memorization.** The copy checks pass, but the first one holds by construction. The closest-record ratio is above 1, so synthetic rows are not closer to real train rows than unseen real rows are. A ratio above 1 here also reflects the drift in the previous section: a better generator will move toward 1.
- **Distinguishability.** v1 is far from indistinguishable. The columns that expose it most often are `job`, `day_of_month` and `month` in every run, then `balance` and `pdays` (`marital`, `poutcome` and `campaign` now and then). They are the columns the fidelity check flags, plus `balance`, which is under its distance limit but still gives the rows away.
- **Utility.** Adding v1 rows lowers both PR-AUC and best-threshold F1 at every fill level, and more at higher levels. In the gate, the two gain checks are the ones that fail. The logistic regression is deterministic, so the seeds do not change these numbers.
- **Against the first pass in the issue.** The first pass (exploratory script, not in the repo) had a classifier AUC of about 0.89 against 0.47, PR-AUC 0.451 to 0.429, 0.401 and 0.365, and best F1 0.494 to 0.471, 0.449 and 0.416. The levels here are lower, but the direction and the size of the drops agree. I cannot say why the levels differ without that script.

### Utility in a full model (Kyle's run)

A second source, from `results/augmented_bank_marketing.csv` (issue #35, made by `src/agentic_gan/tools/augmented.py`, plot in `results/augmented_bank_marketing.png`). It picks the best of the nine models on validation and scores the test set, mean over 5 seeds:

| Arm | Fill | Test PR-AUC | Test F1 at 0.5 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| real only | 0 | 0.488 | 0.359 | 0.672 | 0.245 |
| CTGAN v1 | 10% | 0.468 | 0.429 | 0.584 | 0.338 |
| CTGAN v1 | 25% | 0.438 | 0.444 | 0.490 | 0.406 |
| CTGAN v1 | 50% | 0.405 | 0.437 | 0.408 | 0.471 |
| SMOTENC | 10% | 0.462 | 0.433 | 0.589 | 0.343 |
| SMOTENC | 25% | 0.455 | 0.464 | 0.546 | 0.403 |
| SMOTENC | 50% | 0.441 | 0.461 | 0.441 | 0.482 |

The same pattern: CTGAN v1 lowers PR-AUC at every fill level, and more as the fill level grows. F1 at 0.5 goes up only because recall rises while precision falls, a threshold shift; PR-AUC does not depend on the threshold and falls. SMOTENC also lowers PR-AUC, by less at 25% and 50%.

## More analysis behind the plots

- **Not a class mix-up.** v1 subscribers are closer to real non-subscribers than to real subscribers in only 4 of 16 columns (`education`, `default`, `housing`, `day_of_month`). On `contact`, `previous`, `poutcome` and `pdays` they stay close to subscribers (distance 0.04 to 0.13, against 0.22 to 0.24 from non-subscribers), so the class signal is mostly kept. The damage is distortion, not leakage from the majority class (`class_mixup.csv`).
- **Less diverse.** At equal size, v1 has 1,663 distinct profiles over eight categorical columns against 1,846 for real subscribers, and its ten most common profiles cover 8.4% of rows against 5.1%. `job` drops from 3.13 to 2.57 bits of entropy, which fits the technician pile-up (`diversity.csv`).
- **Broken links.** The biggest changes are between the calendar and the contact history: `month` and `previously_contacted` (Cramér's V 0.20 to 0.39), `month` and `poutcome` (0.12 to 0.25), `day_of_month` and `pdays` (Spearman -0.04 to -0.20), `day_of_month` and `previous` (-0.04 to -0.18). `campaign` and `previous` goes the other way, from -0.11 to 0.02 (`association_pairs.csv`).
- **Distance against importance.** The columns that matter most for predicting a subscription are `month` (importance 0.146), `poutcome` (0.128), `contact` (0.078) and `day_of_month` (0.071), measured as the drop in average precision when the column is shuffled. v1 is over the limit on three of them and fine on `contact`. `job`, the biggest miss, has an importance of 0.002. This is a likely reason utility falls as v1 rows are added (`column_importance.csv`).

## Plots

In `eda/ctgan_bank/artifacts/quality_v1/plots/`, made by `eda/ctgan_bank/quality_v1_plots.py`. 16:9, one finding each, titled with the finding.

| File | Finding |
|---|---|
| `01_gate_scorecard.png` | v1 fails the gate on fidelity and utility: 6 of 15 hard checks fail |
| `02_column_distances.png` | 7 of 16 columns are far from real data, and `job` is the worst |
| `03_category_shifts.png` | technician is 42% of v1 subscribers and 15% of real ones; also `month`, `marital`, `poutcome`, `housing` |
| `04_numeric_drift.png` | v1 runs late: later in the month, longer between campaigns |
| `05_broken_associations.png` | v1 ties the calendar to the contact history too tightly |
| `06_fidelity_vs_importance.png` | the columns v1 gets wrong that matter most: `month`, `poutcome`, `day_of_month` |
| `07_distinguishability.png` | a random forest spots v1 rows with AUC 0.86 against 0.52 for real data |
| `08_closest_record.png` | no memorization, and the starting limit of 1.0 is too strict |
| `09_utility.png` | adding v1 rows lowers PR-AUC; the F1 gain is a threshold shift |

## Recommendations

**Thresholds**

- Start from the suggested limits for the fidelity checks (twice the real-vs-real value, rounded up, with a floor): mean column distance 0.05, worst column 0.10, Spearman difference 0.045, Cramér's V difference 0.065, rule violations 0.5% of rows, label balance 0.01. They pass real-vs-real and fail v1.
- Change `sanity_violations_max` in `QualityThresholds` from a count of 0 to a rate (0.005). Zero tolerance fails real data: 2 of 3,173 real train subscribers and 5 of 27,126 real train rows break the `pdays` rule.
- Tighten the starting fidelity limits. `correlation_diff_max` 0.10 would not catch v1 (0.066, real-vs-real 0.020). The mean and worst-column limits (0.10 and 0.25) only just catch it (0.106 and 0.28, real-vs-real 0.023 and 0.048).
- Lower `dcr_ratio_min` from 1.0 to about 0.95. Real subscribers split at random give a ratio of 1.002 ± 0.015 (100 splits, lowest 0.968), and 52% of them fall below 1.0, so a limit of 1.0 would fail real data about half the time. None of the 100 fell below 0.95. A generator that copies its training rows would be far below that.
- The real-vs-synthetic AUC is diagnostic with a limit of 0.5, which no generator can meet (real-vs-real is 0.49 to 0.56 across the seeds). If it should decide, use a limit of about 0.60.
- The utility gain limit of 0 fails any drop, and a generator that is only as good as real data can do no better than tie. A small tolerance is probably needed. Set it the same way as the ratio, by adding held-out real subscribers in place of the synthetic rows. Not done here.
- These are first values. The fidelity baseline is one real-vs-real comparison of 1,058 against 3,173 rows, and its noise depends on those sizes. Check that a good generator passes once CTGAN v2 exists.
- Score one fill level for the fidelity checks, since they do not change with the number of rows. Per-level scoring only matters for checks that depend on the row count.

**Targets for CTGAN v2** (the scorecard and `quality_privacy_checks` measure them)

- Every column at or below 0.10 (now `job` 0.27, `day_of_month` 0.23, `month` 0.17, `pdays` 0.13, `marital` 0.13, `poutcome` 0.12, `housing` 0.11), mean at or below 0.05, Spearman difference at or below 0.045, and `pdays` rule violations at or below 0.5%.
- Real-vs-synthetic AUC down from 0.86 to 0.60 or lower, and no drop in PR-AUC at the 10% fill level (now -0.025).

**Experiments to try**, cheapest first. These are ideas, none has been run.

1. Add the `pdays` / `poutcome` rule to `consistent_rows` in `run_ctgan_bank.py`. Real data breaks it in 0.02% of rows, so this fixes that check at almost no cost. Re-score afterwards, because rejecting rows can shift other columns.
2. Look at the over-concentration. The `job` distance is the same at every fill level and technician is 42% of the output, which points at the generator, not the sampling, and the `job` entropy falls from 3.13 to 2.57 bits. Levers to try: more epochs, `pac`, batch size, discriminator steps and `log_frequency`.
3. Train on the 3,173 subscribers only, so the generator spends its capacity on that class. Watch for memorization with so few rows, which the closest-record ratio is there to catch.
4. Score the SMOTENC rows with the same two tools to get a baseline for a simple oversampler.

## Reproduce

```bash
# fidelity scorecard for any synthetic file (exit status 1 if a hard check fails)
uv run python -m agentic_gan.tools.quality_fidelity data/interim/bank_marketing/synthetic/*.csv --per-column

# everything in this note, about a minute
uv run python eda/ctgan_bank/quality_v1.py

# the plots and the extra tables (run the line above first)
uv run python eda/ctgan_bank/quality_v1_plots.py
```
