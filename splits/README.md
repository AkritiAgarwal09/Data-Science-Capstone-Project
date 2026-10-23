# Split row IDs

Row-ID files that fix which rows of a raw dataset belong to train, validation and test, so every script uses the same split.

IEEE-CIS is the exception: it cannot be redistributed, so its split is stored as a rule and counts, not as IDs (see the last section).

## Credit Card Fraud (ULB)

Made by `eda/fraud/fraud_analysis.py --write-splits splits` from `creditcard.csv`. The reasoning is in `docs/fraud_dataset_recommendation.md`.

| File | Contents | Rows | Fraud |
|---|---|---|---|
| `credit_card_fraud_train_row_ids.csv` | Stratified split, after dropping duplicates | 170,235 | 284 |
| `credit_card_fraud_validation_row_ids.csv` | same | 56,745 | 94 |
| `credit_card_fraud_test_row_ids.csv` | same | 56,746 | 95 |
| `credit_card_fraud_time_train_row_ids.csv` | Split by `Time`, after dropping duplicates | 170,236 | 342 |
| `credit_card_fraud_time_validation_row_ids.csv` | same | 56,745 | 57 |
| `credit_card_fraud_time_test_row_ids.csv` | same | 56,745 | 74 |
| `credit_card_fraud_dropped_duplicates.csv` | The 1,081 exact duplicate rows that were dropped, and the earlier row each one copies | 1,081 | 19 |

- `source_row` is the 0-based position of the row in the raw CSV, with the header not counted.
- The stratified recipe is two stratified steps, seed 42: first hold out 40%, then split the hold-out in half. It is the same recipe as the baselines and the orchestrator's `split_row_ids`, run on the data left after dropping duplicates. The time split takes the first 60% of rows by `Time` as train, the next 20% as validation and the last 20% as test.
- Duplicates are exact copies of the whole row, `Time` included. The first copy in the file is kept.
- The part is called `validation` in these files and `val` in the orchestrator's code.

## IEEE-CIS (Kaggle)

| File | Contents |
|---|---|
| `ieee_cis_time_split.csv` | The split by `TransactionDT`: the rule for each part, its time range, its rows and its fraud rows. Counts only |

Made by `eda/fraud/ieee_cis_split.py --write-splits splits`, on each person's own copy of the data. The reasoning is in `docs/fraud_dataset_recommendation.md`.

- There is **no ID file**. The Kaggle rules (section 7B) do not allow redistributing the data, and a list of its transaction IDs is derived from it. The split is a rule with two cut points, kept in `src/agentic_gan/utils/datasets.json` (`ieee_cis`, `split.cut_points`), so everyone gets the same split from their own copy.
- Rows are ordered by `TransactionDT`. The first 60% of rows is train, the next 20% validation and the last 20% test, counted after dropping the 3 exact duplicate rows. A row is in train if `TransactionDT` is below the first cut point, in validation if it is below the second, and in test otherwise.
- `train_sample` is a stratified draw of 100,000 rows from the train part (seed 42). The generator and the arms train on it. Validation and test stay whole.
