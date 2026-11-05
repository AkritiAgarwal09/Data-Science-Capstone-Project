# Sprint 2 baseline freeze — 23 October 2026

## Fixed model suite

The nine-model comparison is reduced to **four** candidates, retaining the four user-selected
families that can be run consistently across arms. The neural net was dropped to keep the
suite tractable on the large IEEE-CIS sample. The three boosting methods remain because they
use meaningfully different tree construction and regularization; they are not presented as
three separate model families. Logistic regression and a small random forest are components
of the stack, not standalone candidates.

| Model | Why retain it | Frozen settings |
|---|---|---|
| Gradient boosting | A familiar sequential boosting baseline with constrained, shallow trees. | 50 trees, depth 2, `max_features=sqrt`, 20% row subsampling. |
| LightGBM | Histogram splits and leaf-wise growth can isolate useful local interactions, as on Polish Bankruptcy. | 150 trees, 2 threads. |
| CatBoost | Symmetric-tree boosting provides a contrasting tree shape and ranked first on Bank and Default. Categories are one-hot encoded first, so this does not test CatBoost's native categorical handling. | 200 iterations, 2 threads. |
| Stacking ensemble | Combines a linear classifier and a shallow bagged forest through a logistic meta-learner. It tests whether different score patterns help together. | Logistic regression (`max_iter=100`, `tol=0.05`) and 10 depth-6 forest trees; 2 inner folds; logistic final estimator. |

The stack's small component models and the shallow stochastic Gradient Boosting trees bound
fit cost on IEEE-CIS. Seeds and constructor settings are the same across arms. No parameter
search is performed. Each protocol records the exact settings and dependency versions.

## Threshold selection and reuse

For every **dataset × model**, choose the threshold that maximizes positive-class F1 on the
original arm's real validation split. Evaluate each unique predicted probability with
`p >= threshold`; choose the higher threshold when F1 ties. Test labels never select the
threshold. The four thresholds are saved in
`results/baseline_protocol_<dataset>_42_cv0.json` and reused unchanged for the cleaned arm
and later augmented arms.

A threshold of 0.5 is a default probability cutoff, not an F1 optimum. With minority positive
classes and imperfectly calibrated scores, it can miss positives. F1 balances precision and
recall for this detection task; it is not a bank-specific cost function. The CSV retains
validation F1 at both the chosen threshold and 0.5. Validation threshold selection is
optimistic, so the untouched test split estimates the resulting operating point.

The threshold is model-specific because probability scales differ. Test PR-AUC stays
threshold-independent. Models rank by validation trapezoidal PR-AUC; test metrics are reported
for the validation winner only. The CSV orders PR-AUC and F1 first and accuracy last. Training CV was disabled (`cv_folds=0`) for the refreshed baselines. IEEE-CIS uses a
100k-row stratified sample from the time-based training period; validation and test retain
their full chronological windows.

## Measured rerun

The three-dataset refresh contributed **24 successful rows: four models × two arms × three
datasets**. The IEEE-CIS extension below adds eight rows, so `results/baselines.csv` now has
**32 successful rows across four datasets**. The previous five-model results are retained
under `results/archive/v1/` and are not mixed with this comparison.
All runs use seed 42, a 60/20/20 stratified split on Bank, Default, and Polish, and
training-only preprocessing. Original uses the generic training-fold imputer, scaling, and
encoding; cleaned uses the shared training-fitted cleaner. The exact per-model thresholds are:

| Model | Bank | Default | Polish |
|---|---:|---:|---:|
| Gradient boosting | 0.163269 | 0.208274 | 0.095493 |
| LightGBM | 0.214943 | 0.279692 | 0.201069 |
| CatBoost | 0.219666 | 0.218698 | 0.132656 |
| Stacking ensemble | 0.153306 | 0.244148 | 0.076379 |

| Dataset / arm | Validation winner | Threshold | Test PR-AUC | Test F1 | ROC-AUC | Precision | Recall | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Bank / original | CatBoost | 0.219666 | 0.4811 | 0.5081 | 0.8048 | 0.4986 | 0.5180 | 0.8827 |
| Bank / cleaned | CatBoost | 0.219666 | 0.4811 | 0.5081 | 0.8048 | 0.4986 | 0.5180 | 0.8827 |
| Default / original | CatBoost | 0.218698 | 0.5591 | 0.5411 | 0.7833 | 0.4664 | 0.6443 | 0.7583 |
| Default / cleaned | CatBoost | 0.218698 | 0.5576 | 0.5420 | 0.7820 | 0.4666 | 0.6466 | 0.7583 |
| Polish / original | LightGBM | 0.201069 | 0.8354 | 0.7584 | 0.9745 | 0.8295 | 0.6986 | 0.9786 |
| Polish / cleaned | LightGBM | 0.201069 | 0.8354 | 0.7584 | 0.9745 | 0.8295 | 0.6986 | 0.9786 |

Validation F1 at 0.5 versus the chosen threshold was 0.3669 versus 0.4986 for Bank,
0.4406 versus 0.5332 for Default original, and 0.7441 versus 0.7734 for Polish. Cleaned
Default F1 at 0.5 was 0.4489 versus 0.5222 at the frozen threshold. Lower accuracy on
Bank and Default accompanies increased minority-class recall; it does not contradict the
F1 improvement.

## IEEE-CIS time-split extension

IEEE-CIS uses issue #50's fixed `TransactionDT` cutoffs. Its training period contains
354,322 rows, from which the configured 100,000-row sample (3,383 frauds; 3.38%) is selected.
Validation and test keep all 118,107 and 118,108 rows (3.90% and 3.44% fraud). The original
arm keeps the high-cardinality one-hot matrix sparse; the cleaned arm applies the shared
training-only missingness rules, caps categories at 50 plus missing/other buckets, and adds
informative missingness flags. Both omit `TransactionID` and `TransactionDT` from features.

The original-arm validation F1 thresholds were frozen once and reused for the cleaned arm:

| Model | IEEE-CIS threshold |
|---|---:|
| Gradient boosting | 0.099203 |
| LightGBM | 0.179973 |
| CatBoost | 0.165394 |
| Stacking ensemble | 0.069860 |

| Arm | Validation winner | Threshold | Validation PR-AUC | Test PR-AUC | Test F1 | ROC-AUC | Precision | Recall | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | LightGBM | 0.179973 | 0.5744 | 0.4527 | 0.4406 | 0.8785 | 0.4611 | 0.4218 | 0.9631 |
| Cleaned | LightGBM | 0.179973 | 0.5492 | 0.4093 | 0.3954 | 0.8750 | 0.3670 | 0.4286 | 0.9549 |

LightGBM ranked first on validation in both arms, ahead of CatBoost (0.5557 original,
0.5282 cleaned). Its test PR-AUC was lower in the later time window. The different fraud
rates and periods may contribute to that gap; this one split cannot establish the cause.

## Why rankings differ

- **Bank Marketing:** 3,173 of 27,126 training rows are positive (11.70%). Customer
  categories and previous-contact information interact with overlapping numeric features.
  CatBoost's symmetric boosted trees ranked first in this run; LightGBM was close in
  validation PR-AUC. The stack performed poorly, showing that combining a linear score with
  a small forest does not help when their out-of-fold scores add little complementary signal.
  Cleaning leaves these metrics unchanged here.
- **Default:** 3,982 of 18,000 training rows are positive (22.12%), with no missing training
  cells. Payment history, balances, and amounts support nonlinear thresholds. CatBoost ranked
  first, but its validation PR-AUC lead over LightGBM and Gradient Boosting is small; treat it
  as this split's result, not a proven advantage. Cleaning slightly changed the winning test
  metrics but did not change the selected model.
- **Polish Bankruptcy:** 1,255 of 26,043 training rows are positive (4.82%). There are
  24,708 missing training cells; A37 is 43.29% missing and A27 is missing in 34.50% of
  bankrupt rows versus 4.90% of other rows. Ratios have extreme tails (maximum absolute
  training value 11,903,000). LightGBM's histogram splits and best-first leaf growth can
  isolate localized, nonlinear patterns, which is consistent with its strong ranking here.
  This is an interpretation, not a causal finding. Both arms median-impute and retain
  extreme ratios; neither tests native missing-value routing or missingness indicators.
- **IEEE-CIS:** The time-based validation period has a 3.90% fraud rate versus 3.44% in test.
  The 100k training sample includes 7,663 distinct `card1` values, alongside substantial
  missingness across transaction and identity fields. LightGBM's histogram splits and
  leaf-wise growth may capture localized fraud interactions in this wide, sparse feature
  space; CatBoost's symmetric trees split it differently after one-hot encoding. LightGBM's
  validation PR-AUC lead (0.574 versus 0.556 for CatBoost) did not carry over unchanged to
  the later test period. This is consistent with time-related distribution change, but the
  single split does not prove it. Cleaning further groups high-cardinality categories and
  drops values with over 20% training missingness while preserving flags, which can remove
  signal as well as noise.

All three boosting methods partition features by thresholds but differ in tree shape, split
search, and regularization. The stack learns how to combine cross-validated linear and forest
scores, which can fail when those scores are weak or redundant. Dataset size, prevalence,
feature encoding, missingness, interactions, and score calibration all affect their ranking.
PR-AUC also depends on prevalence, so compare models within each dataset. These single-seed
results do not establish statistical significance.

## Reproduction and agent contract

Run these commands sequentially; they share `results/baselines.csv`:

```bash
python -m agentic_gan.tools.baselines --dataset bank_marketing --arm both --cv-folds 0
python -m agentic_gan.tools.baselines --dataset default --arm both --cv-folds 0
python -m agentic_gan.tools.baselines --dataset polish --arm both --cv-folds 0
python -m agentic_gan.tools.baselines --dataset ieee_cis --arm both --cv-folds 0
```

`agents.training` loads the same four models and thresholds from the checked protocol. It does
not recalibrate on cleaned or synthetic arms. IEEE-CIS preprocessing uses issue #50's time
split and writes cleaned unencoded and encoded/scaled outputs; its source hash covers both
transaction and identity files. The IEEE-CIS protocol is frozen at
`results/baseline_protocol_ieee_cis_42_cv0.json` and is ready for the Training Agent.

## References

- [scikit-learn: tuning decision thresholds](https://scikit-learn.org/stable/modules/classification_threshold.html).
- [LightGBM: histogram splitting and leaf-wise growth](https://lightgbm.readthedocs.io/en/stable/Features.html).
- [CatBoost: how training is performed](https://catboost.ai/docs/en/concepts/algorithm-main-stages).
