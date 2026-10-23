# Credit Card Fraud dataset: recommendation

Issue #24. Analysis by `eda/fraud/fraud_analysis.py`, run on the raw ULB file (`creditcard.csv`, 284,807 rows, 492 fraud). Prepared 15 October 2026 for the mentor meeting on 16 October.

**Update, 22 October (issue #50).** The mentor confirmed IEEE-CIS as the main fraud dataset, because ULB has only 284 training fraud rows. ULB stays as the stress case. How IEEE-CIS enters the pipeline (the settings entry, the duplicate policy, the split by `TransactionDT`, the training sample and the data-use rules) is under "Adding IEEE-CIS" in section 4.

## Recommendation: adjust, do not replace

The mentors' question was whether to proceed, adjust or replace this dataset. I recommend **adjust**:

1. **Drop exact duplicate rows before splitting.** The key is every column of the raw row, `Time` included. This removes 1,081 rows (0.38% of the data), of which 19 are fraud (3.9% of the fraud rows).
2. **Use the stratified 60/20/20 split as the primary split**, made after the duplicates are dropped. Report a split by time as a sensitivity check.
3. **Set the fill-level grid per dataset.** For this dataset use 0.01, 0.02, 0.05 and 0.10, not the default grid that goes up to 1.0.
4. **Treat it as the stress case.** With 284 training positives, a result of no gain from synthetic data is expected and is a reportable result.
5. **Use IEEE-CIS as the main fraud dataset, as its own run, under the data-use restrictions in section 4** (confirmed by the mentor on 22 October). Rows cannot be merged into this dataset. IEEE-CIS is the one real, large candidate. The other candidates were set aside (section 4).

## 1. Duplicates

| | Count |
|---|---|
| Extra exact copies, all rows | 1,081 (773 groups, 1,854 rows involved) |
| Extra exact copies, fraud rows | 19 (13 groups, 32 rows involved) |
| Transactions with identical features but both labels | 0 |
| Extra copies when `Time` is ignored (same V1 to V28 and Amount) | 9,144 |

- **Fraud copies look like data artifacts.** In every one of the 13 fraud groups the copies share a timestamp, so they are not repeated frauds.
- **Do not drop duplicates while ignoring `Time`.** 8,063 more rows share all of V1 to V28 and Amount with a row at a different time. Checked on 16 October (`eda/fraud/time_twins.py`): none of them is fraud, none sits in a group with both labels, 58% have an amount of 2 or less (most often 1.00, 1.98 and 0.89), and consecutive repeats are a median of 81 minutes apart (only 6% within 10 seconds). That looks like recurring small payments, not copied records, so they are kept. The key must be the raw row including `Time`, so the step has to see the raw table. A step that runs after `Time` has been dropped as a feature would delete them.
- **What the kept repeats do to the splits.** `Time` is not a model feature, so a kept repeat looks identical to its twin. After the recommended split, 3.4% of validation rows and 3.3% of test rows have such a twin in train, and none of them is fraud (0 of 94 and 0 of 95 fraud rows). On a split by time it is 1.0% and 1.1%. The overlap can only touch the legitimate class, so we report it and do not remove it.
- **`Time` as a feature is not decided.** In a quick check (`eda/fraud/time_feature_check.py`, one untuned fit, three seeds) no time feature, hour of day and raw `Time` scored within the noise of each other (seed spread 0.07 to 0.11 PR-AUC). The proposal is hour of day instead of the raw timestamp, because the fraud rate depends on the hour and raw seconds-since-start cannot carry over beyond these two days. Scores were lower on the time split (about 0.23 to 0.32, against 0.44 to 0.52), partly because its fraud rate is lower (0.10% against 0.17%).
- **Correction to the issue text.** The 19 duplicates are across all 492 fraud rows. In the stratified training part, 7 of the 295 fraud rows are extra copies inside train.

### The stratified split leaks

With the original stratified split, many validation and test rows have an exact copy in train.

| Part | Rows with a copy in train | Distinct rows | Fraud rows |
|---|---|---|---|
| Validation | 297 | 266 | 4 (4.1% of its 98 fraud) |
| Test | 271 | 231 | 5 (5.1% of its 99 fraud) |

The 231 distinct test rows are the "about 230" on the 9 October slide. A model, and especially a GAN, can score well on these by memorising. The current rule in `architecture.md` section 5.1, "drop duplicates only inside the training split", does not fix this: all 568 rows stay leaked.

### Policies compared

| Policy (stratified split) | Fraud train / val / test | Val and test rows still seen in train | Fraud rows lost |
|---|---|---|---|
| None | 295 / 98 / 99 | 568 | 0 |
| Dedupe inside train only (current rule) | 288 / 98 / 99 | 568 | 7 |
| Drop val and test copies, dedupe inside parts | 288 / 93 / 93 | 0 | 18 |
| **Dedupe everything, then split (recommended)** | **284 / 94 / 95** | **0** | **19** |

Deduping first costs 19 fraud rows, one more than dropping only the validation and test copies (18). In return the split is made once, on data that is already clean, so the stratification holds exactly and there is no second pass to document.

## 2. Split by time

The data covers two days. Splitting the rows by `Time` (first 60% train, next 20% validation, last 20% test):

| | Stratified, after dedupe | By time, after dedupe |
|---|---|---|
| Fraud rows, train / val / test | 284 / 94 / 95 | 342 / 57 / 74 |
| Fraud rate, train / val / test | 0.17% / 0.17% / 0.17% | 0.20% / 0.10% / 0.13% |
| Time covered by val and test | the whole two days | 33 to 48 hours |
| Duplicate leakage | none, once deduped | none, even before deduping |
| Recall uncertainty (95% half-width at recall 0.8), validation | about 8 points | about 10 points |

**What a time split gives.** It is the realistic setting (train on the past, test on later transactions). Copies share a timestamp, so it has no duplicate leakage even before deduplication. The training part has more fraud rows (342 against 284).

**What it costs.**
- Validation has only 57 fraud rows, so one missed case moves recall by 1.75 points.
- The fraud rate is not stable over the day: 0.40% to 0.48% between midnight and 08:00, and about 0.13% in the rest of the day. Train contains both nights. Validation and test cover only 09:24 to 24:00 of the second day, so they never contain the high-fraud hours.
- With only two days of data, "the future" is a matter of hours. The result mixes time shift with time of day.

So the time split is worth reporting as a robustness check, but it should not be the primary split.

## 3. Fill levels for this dataset

The class gap after deduplication is 169,667 rows (284 positives). The default grid would add up to 597 times the real positives, from a generator trained on 284 examples.

| Fill level | Rows added | Times the real positives |
|---|---|---|
| 0.01 | 1,697 | 6 |
| 0.02 | 3,393 | 12 |
| 0.05 | 8,483 | 30 |
| 0.10 | 16,967 | 60 |
| 0.25 | 42,417 | 149 |
| 1.00 | 169,667 | 597 |

I propose 0.01, 0.02, 0.05 and 0.10 for this dataset. Larger levels cost more compute and are unlikely to be informative, since the generator has so few real examples to learn from.

## 4. Candidate datasets for more fraud rows

The issue asked for real, non-simulated datasets that could add fraud rows. The sources I could read do not name any public real dataset that uses the ULB features. The V1 to V28 columns are principal components of confidential variables, so components from any other dataset mean something different and rows cannot be pooled. A candidate can therefore only be a second dataset, not an addition to this one.

Checked on 16 October (`eda/fraud/compare_schemas.py`). ULB has 31 columns (`Time`, `V1` to `V28`, `Amount`, `Class`). IEEE-CIS has 394 transaction columns and 40 identity columns, and only an amount and a time line up. IEEE-CIS also has columns named `V1` to `V339`, but they are Vesta-engineered features, not principal components: its `V1` takes two values (0 and 1) and is 47% missing, and 263 of its 339 V columns are integer-valued, while ULB's `V1` is a continuous score with 275,663 distinct values. A ULB model that uses only the amount and the time scores 0.012 validation PR-AUC (one untuned fit, three seeds) against 0.515 with all columns, where chance is 0.0017. Almost all of its signal is in columns that IEEE-CIS does not have.

| Candidate | Real? | Size and fraud | Features | Licence and access | Verdict |
|---|---|---|---|---|---|
| ULB / Worldline (current) | Real, 2 days in Sept 2013 | 284,807 rows, 492 fraud (0.17%) | `Time`, `Amount`, V1 to V28 (PCA) | Database Contents License v1.0 per the benchmark below and a 2024 paper. A Kaggle metadata copy lists ODbL, so check the Kaggle page | Keep, with the adjustments above |
| IEEE-CIS (Vesta) | Real, donated by Vesta | 590,540 transactions, 20,663 fraud (3.50%), measured on `train_transaction.csv`. 24.4% have an identity record | About 390 to 430 masked columns in a transaction and an identity table, many missing values | Kaggle competition rules, read on 16 October: the data may be used for non-commercial purposes, including academic research and education (section 7A), and may not be redistributed (7B). Each user must accept the rules to download. The IEEE DataPort copy needs a subscription | Best real candidate for a second dataset, not a merge |
| Vehicle insurance claims (Kaggle, shivamb) | Source listed as "Oracle Databases", collected by web download. Whether it is real or sample data is not documented | 15,420 rows and 923 fraud (5.99%), measured on the downloaded file. 554 / 185 / 184 fraud in a stratified 60/20/20 split | 32 mixed numeric and categorical columns, `FraudFound_P` target, `fraud_oracle.csv` 3.6 MB | CC0 (public domain), per the Kaggle page | Set aside, kept as a quick dry-run option. 554 training fraud rows, in the mentors' range. A different fraud type. Clean (no duplicates, no missing values apart from placeholders). Boosting scores 0.28 validation PR-AUC against 0.06 chance. Label its real or sample status as unverified |
| Elliptic (Bitcoin transactions) | Real, from the Bitcoin blockchain | 203,769 nodes, of which 4,545 illicit and 42,019 licit are labelled | 166 anonymised numeric features, 49 time steps, a transaction graph | CC BY-NC-ND 4.0 per the Kaggle page. NoDerivatives means synthetic data made from it could not be shared | Not recommended: cryptocurrency, anonymised features, and the licence conflicts with sharing synthetic data |
| Feedzai Bank Account Fraud (NeurIPS 2022) | Synthetic. The Kaggle page says it was built from a real dataset with differential-privacy noise, feature encoding and a CTGAN generator | 1 million rows per variant, 30 features, about 11,000 fraud rows (1.1%) in the base file | Tabular, with built-in bias variants | CC BY-NC-SA 4.0 per the Kaggle page | Exclude: synthetic, and made with CTGAN, so a GAN on it would be circular |
| Fraud ecommerce (Kaggle, vbinh002) | The Kaggle page has no description. Real or synthetic is not documented | 151,112 rows and 14,151 fraud (9.36%), measured. `Fraud_Data.csv` is 16 MB with 10 columns plus a label | User id, device id, browser, IP address, purchase value, sign-up and purchase times, 2015 | Licence listed as "Unknown" | Exclude. Every one of the 7,600 purchases made within 60 seconds of sign-up is fraud (54% of its fraud), which looks like a rule or simulation, and provenance and licence are unknown |
| Credit Card Fraud Detection Dataset 2023 (Kaggle) | Described as European cardholders in 2023. No further provenance | Over 550,000 rows (325 MB). The class balance is not on the page | `id`, V1 to V28 (anonymised again), Amount, Class | "Other (specified in description)", but the description gives no licence | Exclude: no provenance or licence, and anonymised features that cannot be pooled with ULB |
| Sparkov simulated (Kaggle) | Simulated | 1.3 million rows, 5.7% fraud | Synthetic | CC0 1.0 | Exclude: simulated |

### Measured on the downloaded files

Run on 16 October with `eda/fraud/candidates/analyze_candidates.py` and `ieee_reduced_features.py` (the data is not in the repo). Stratified 60/20/20 split, seed 42, the same recipe as the orchestrator. "Signal" is one untuned gradient-boosting fit scored on the validation part, so read it as a rough guide only.

| | IEEE-CIS | Vehicle insurance claims | Fraud ecommerce |
|---|---|---|---|
| Rows, fraud (rate) | 590,540, 20,663 (3.50%) | 15,420, 923 (5.99%) | 151,112, 14,151 (9.36%) |
| Fraud in train / val / test | 12,398 / 4,133 / 4,132 | 554 / 185 / 184 | 8,491 / 2,830 / 2,830 |
| Missing values | 41.1% of cells. 374 of 394 columns have gaps and 174 are over 50% empty | None. Placeholders instead: 320 rows with `Age = 0` (all in the "16 to 17" bin), the string `none` in four columns | None |
| Exact duplicates | 3 | 0 | 0 |
| Val and test rows with a copy in train | 74 and 55 repeat a train row at another time, none fraud (time split, see Adding IEEE-CIS) | 0 and 0 | 0 duplicates |
| Fraud rate by time split (train / val / test) | 3.38% / 3.90% / 3.44% | not applicable (three policy years, 1994 to 1996) | 12.5% / 4.6% / 4.6% |
| Signal, validation PR-AUC (chance) | 0.58 on a 100k subsample (0.035). 0.53 on the 109 columns with at most 5% missing, 0.57 on the 179 with at most 20% | 0.28 (0.06). Logistic regression 0.17 | 0.33 (0.09) on the basic columns, 0.62 with time-to-purchase and hour, 0.72 with device sharing |
| Fill level 1.0 (times real positives) | 329,528 rows (26.6). In the stratified 100k subsample measured then, 55,802 (26.6). The sample to use is under Adding IEEE-CIS | 8,144 rows (14.7) | 73,685 rows (8.7) |

Things worth knowing:

- **IEEE-CIS.** The fraud rate is steady over its 182 days, so a time split is a fair primary split here, unlike ULB. Only 24.4% of rows have an identity record, and those are far likelier to be fraud (7.85% against 2.09%). A 100k sample drawn from the time-split train part keeps 3.4% fraud and gives 3,383 fraud rows (see "Adding IEEE-CIS"; the 2,099 measured on 16 October was for a stratified split).
- **Vehicle insurance claims.** `PolicyNumber` is just the row number (1 to 15,420). The fraud rate varies a lot by policy type (13.8% for Sport - Collision, 0.7% for Sedan - Liability). Nothing in the file shows whether it is real or a sample dataset.
- **Fraud ecommerce.** The 7,600 purchases within a minute of sign-up are all fraud, and devices shared by several users are 52.5% fraud. A GAN test on it would mostly learn those rules.

### Adding IEEE-CIS

**Status.** The mentor confirmed IEEE-CIS as the main fraud dataset on 22 October. The settings below are the `ieee_cis` entry in `src/agentic_gan/utils/datasets.json`. They were measured with `eda/fraud/ieee_cis_split.py` on the files downloaded on 16 October. The preprocessing script does not read the entry yet: the join, the duplicate rule and the time split are the IEEE-CIS preprocessing issue.

**Why.** It has 20,663 fraud rows, so a generator has thousands of real positives to learn from, unlike ULB. Its features are masked raw columns, not PCA components. Its fraud rate is steady over time (3.38%, 3.90% and 3.44% across the time split below). It has real missing values, so it is a second testbed for the data-repair step.

| Setting | Value |
|---|---|
| Files | `data/external/ieee-fraud-detection/train_transaction.csv` and `train_identity.csv`. Each person downloads their own copy; git ignores the folder |
| Join | identity left-joined onto transactions, one to one, on `TransactionID`: 590,540 rows and 434 columns (394 transaction, 40 identity) |
| Target and key | `isFraud`; `TransactionID` is the row key and never a feature |
| Discrete columns | 49, the categorical columns named on the competition's data page: `ProductCD`, `card1` to `card6`, `addr1`, `addr2`, `P_emaildomain`, `R_emaildomain`, `M1` to `M9`, `DeviceType`, `DeviceInfo`, `id_12` to `id_38` |
| Duplicates | exact copies dropped before the split, keyed on every column except `TransactionID`, first copy kept |
| Split | by `TransactionDT`, 60/20/20 by rows, cut points 8,745,772 and 12,192,842 |
| Sample | stratified on `isFraud`, 100,000 rows from the train part, seed 42; columns more than 20% empty in the sample are dropped |

**The join.** 24.4% of transactions (144,233) have an identity record. Those are fraud 7.85% of the time, against 2.09% for the rest, so having a record carries signal. The other rows keep empty identity columns. An indicator column for "has an identity record" is a cheap addition for Pre-processing.

**Duplicate policy.** An exact copy is a row identical in every column except `TransactionID`, `TransactionDT` included. There are 3, none fraud. They are dropped before the split and the first copy is kept, as for ULB. Rows with the same values at a different time are kept: 476 extra copies, 3 of them fraud, which look like genuine repeat payments. Eight groups of rows with identical features (time ignored) carry both labels, and they stay too. Copies share a timestamp, so a time split cannot put one in each part. After the split, 74 validation rows and 55 test rows repeat a train row at another time, and none is fraud.

**Split by time.** Order the rows by `TransactionDT` (182 days from the first transaction to the last), after the 3 copies are dropped. The first 60% is train, the next 20% validation and the last 20% test. A row is in train if `TransactionDT` is below the first cut point, in validation if it is below the second, and in test otherwise. The split is stored as that rule, not as a list of IDs (see "Data-use restrictions"); `splits/ieee_cis_time_split.csv` holds the counts.

| Part | `TransactionDT` | Day (`TransactionDT` over 86,400) | Rows | Fraud rows | Fraud rate |
|---|---|---|---:|---:|---:|
| Train | below 8,745,772 | 1 to 101 | 354,322 | 11,988 | 3.38% |
| Validation | 8,745,772 to below 12,192,842 | 101 to 141 | 118,107 | 4,611 | 3.90% |
| Test | 12,192,842 and above | 141 to 183 | 118,108 | 4,064 | 3.44% |

The fraud rate is steady, so a time split is a fair primary split here, unlike ULB. With 4,611 fraud rows in validation and 4,064 in test, recall is known to about 1 point at recall 0.8, against about 8 for ULB.

**Size and the sample.** The train part is 354,322 rows, and CTGAN's cost grows with rows and columns. Bank Marketing (27,126 rows, 17 columns) took 519 seconds on CPU. So CTGAN should not train on the whole train part. Draw a sample:

- **How it is drawn.** After the duplicates are dropped and the split is made, take 100,000 rows from the train part only (rows in file order), stratified on `isFraud`, seed 42 (`train_test_split(train_rows, train_size=100_000, stratify=isFraud, random_state=42)`). Validation and test stay whole.
- **What it gives.** 100,000 rows, of which 3,383 are fraud (3.38%). The class gap is 93,234, so fill levels 0.10, 0.25, 0.50, 0.75 and 1.00 add 9,323, 23,308, 46,617, 69,926 and 93,234 rows (2.8 to 27.6 times the real fraud rows).
- **Same sample for every arm.** Baseline, SMOTE and GAN all train on it, so they start from the same real rows and the fill levels come from its class gap.
- **Columns.** Dropping columns that are more than 20% empty in the sample leaves 179 of the 431 (109 are at most 5% empty). Only 10 of the 49 discrete columns survive. Six of those have many categories: `card1` (7,663), `card2` (499), `addr1` (195), `card5` (90), `card3` (79) and `P_emaildomain` (59). Cap them to the most frequent categories and fold the rest, or treat them as numbers, before CTGAN, because its one-hot encoding cannot take thousands of categories.
- **Check the time first.** 100,000 rows by 179 columns is about 39 times the cells of the Bank run. Time a short run on a smaller sample before spending GPU credit, and if it is too slow draw a smaller sample the same way. The validation and test parts do not change.

**Data-use restrictions** (Kaggle competition rules, section 7, read on 16 October):

| Rule | What it means for us |
|---|---|
| 7A: use for non-commercial purposes only, including academic research and education | A capstone is education. Whether TD's sponsorship counts as commercial is a question for TD, not something we can settle |
| 7B: no transmitting, duplicating, publishing or redistributing the data to anyone who has not accepted the rules | Never commit it, put it in Drive, Slack or email, or hand it to mentors. Each person downloads their own copy. `data/external/` is ignored by git for this reason |
| Account and acceptance | One Kaggle account per person, and each person accepts the rules before downloading |

Our reading of what the rules do not say, to confirm with the mentors or TD:

- Synthetic rows generated from the data may count as derived from it, so they stay private (inside `runs/` or `data/interim/ieee_cis/`, which are ignored by git) and are not published.
- The report and poster show aggregate results (metrics, plots of distributions), not rows.

**Set aside.** The vehicle insurance claims set (CC0, 554 training fraud rows, moderate signal, real or sample status unknown) stays available as a quick dry-run dataset. Fraud ecommerce, Elliptic, Bank Account Fraud, Sparkov and the 2023 card data are excluded for the reasons in the table above.

**Still open.** Rule 7A allows non-commercial use only. Whether TD's sponsorship counts as commercial use is a question for TD, and nothing in the repository says it has been answered.

**Where the data lives.** The entry points into `data/external/`, which git ignores. Cleaned tables and synthetic rows go to `data/interim/ieee_cis/` or `runs/`, which git ignores too. The split is a rule and counts, not a list of IDs, and `eda/fraud/ieee_cis_split.py` writes aggregates only.

### If we kept ULB on its own

The mentors suggested adding fraud rows to this dataset. That is not possible (the features are PCA components, and all 473 fraud rows after deduplication are under 500), so the question is what it would take to make ULB alone give a usable answer. It would take:

1. **Deduplicate first, then split.** Fraud 284 / 94 / 95, no leak (section 1).
2. **Small fill levels only:** 0.01, 0.02, 0.05 and 0.10, which add 1,697 to 16,967 rows (6 to 60 times the real fraud rows). Level 1.0 would add 597 times.
3. **Pool train and validation** (378 fraud rows) and use repeated stratified cross-validation, so recall is judged on 378 fraud rows (about 4 points at recall 0.8) instead of 94 (about 8). Keep the test part (95 fraud rows) for the end. This needs a cross-validation option in the orchestrator's split step.
4. **Judge gains with paired bootstrap intervals** over the five seeds, and accept "no reliable gain" as a result.
5. **Train the generator on the fraud rows only**, with a small network, early stopping and the distance-to-closest-record check, because 284 rows are easy to memorise.
6. **Report a time split** as a sensitivity check (section 2).

Even then, a result of no reliable gain is the likely outcome, and it is a valid result for the stress case. The measurement noise is why a second dataset helps: with 4,611 fraud rows in the IEEE-CIS validation part (time split), recall is known to about 1 point.

## 5. What this changes in the pipeline

These need team agreement before anyone builds them:

- **Where deduplication lives.** If duplicates are dropped before the split, it belongs in the orchestrator's split step, not in Pre-processing. The doc's principle that the split comes before any cleaning still holds, because exact-row deduplication on the raw table is not modelling-side cleaning, but it is a change to section 4. It needs a config option and a branch in `split_data`.
- **Split strategy.** The time split becomes an option in the split step (`split_strategy`), as the issue said. It is the primary split for IEEE-CIS (`split.strategy` is `time` in its settings entry) and a sensitivity check for ULB.
- **Pre-processing rule.** Section 5.1's row "Exact duplicate rows: drop only inside the training split" should say that exact duplicates are removed before the split, keyed on the full raw row.
- **Row-ID files.** `splits/` now holds the recommended split: stratified, and by time, both after deduplication, plus the list of dropped rows. IEEE-CIS has no ID file: its split is a rule with cut points plus counts, because the data may not be redistributed. See `splits/README.md`.

## Questions for the mentors

1. Is ULB acceptable as the extreme-imbalance stress case, with a no-gain result reported as a result?
2. IEEE-CIS as the main fraud dataset: confirmed by the mentor on 22 October. Still open: does TD need to confirm that a sponsored capstone qualifies as non-commercial use?
3. Do they want the time split reported as a sensitivity check?

## Sources

- [Fraud Dataset Benchmark (Amazon Science)](https://github.com/amazon-science/fraud-dataset-benchmark), the catalogue of public fraud datasets with their sources and licences, and its [paper](https://arxiv.org/pdf/2208.14417)
- [IEEE-CIS Fraud Detection on IEEE DataPort](https://ieee-dataport.org/documents/ieee-cis-fraud-detection), and the [Kaggle competition](https://www.kaggle.com/c/ieee-fraud-detection)
- [Bank Account Fraud (Feedzai)](https://github.com/feedzai/bank-account-fraud) and the [NeurIPS 2022 paper](https://arxiv.org/abs/2211.13358)
- [ULB credit card fraud dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Dataset pages and the IEEE-CIS rules were read in a browser on 16 October. The counts and signal checks come from the files downloaded that day (section 4, "Measured on the downloaded files"). The Kaggle pages do not say whether the vehicle insurance and Fraud ecommerce data are real. Nothing downloaded is in the repository.
