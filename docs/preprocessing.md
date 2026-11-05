# Shared preprocessing

Settings live in `src/agentic_gan/utils/datasets.json`: input file(s), target, categorical columns, imputation method, and split rules. Bank, Default and Polish use stratified splits; IEEE-CIS uses issue #50's fixed `TransactionDT` cutoffs and 100,000-row training sample.

```bash
uv run python -m agentic_gan.tools.preprocessing --dataset bank_marketing
uv run python -m agentic_gan.tools.preprocessing --dataset default
uv run python -m agentic_gan.tools.preprocessing --dataset polish
uv run python -m agentic_gan.tools.preprocessing --dataset ieee_cis
```

Bank, Default and Polish write `train_cleaned.csv`, `validation_cleaned.csv`, and `test_cleaned.csv` to `data/interim/<dataset>/`. The tables are cleaned and imputed, without encoding or scaling. Exclude `source_row` from model/GAN features.

The raw IEEE-CIS files are not stored in Git. Each Kaggle account must accept the [competition rules](https://www.kaggle.com/competitions/ieee-fraud-detection/rules) before downloading. TD confirmed this sponsored capstone qualifies as non-commercial academic research, and the project lead confirmed the rules are accepted on their Kaggle account. The rules prohibit redistribution, so keep the files in the ignored local data folder. After authenticating the [Kaggle CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md), download and extract the files with:

```bash
kaggle competitions download ieee-fraud-detection \
  -p data/external --unzip
```

This creates `data/external/train_transaction.csv` and `train_identity.csv` (or pass their paths through `--input` and `--identity-input`). The preprocessing script itself reads local inputs and does not access Kaggle. It left-joins the tables one-to-one on `TransactionID`; target labels come only from transactions. Exact copies, comparing every joined field except `TransactionID` and keeping the first, are removed before splitting. The split follows the configured `TransactionDT` cutoffs from issue #50; the 100,000 training rows are sampled from the training period only, stratified on `isFraud`, seed 42. Validation and test retain all rows. If `splits/ieee_cis_time_split.csv` is present, generated split counts are checked against it.

The model and GAN feature tables exclude `TransactionID`, `TransactionDT`, and `source_row`. The `has_identity` join-presence flag is retained. A training-sample missingness rate above 20% drops a feature's values; a varying missingness pattern still gets a binary `__missing` flag. Numeric medians, categorical vocabularies, rare-category buckets, missingness decisions, and scaling are fitted on the training sample only. Categories keep the 50 most common training values plus explicit missing and other buckets. Validation/test rows reuse those settings.

IEEE-CIS writes these local artifacts to `data/interim/ieee_cis/`:

- `train_cleaned.csv`, `validation_cleaned.csv`, `test_cleaned.csv`: imputed but unencoded/unscaled tables for GAN use, with `source_row` as an index column.
- `train_encoded.npz`, `validation_encoded.npz`, `test_encoded.npz`: sparse CSR float32 one-hot encoded and scaled feature matrices. Matching `*_encoded_rows.csv` files provide row order, `source_row`, and `isFraud`; the target and identifiers are not model features.
- `encoder_settings.json` and `preprocessor.pkl`: train-sample feature, vocabulary, imputation, and scaling settings and the reusable fitted encoder.
- `treatment_log.csv`: one row for each input or generated feature, documenting its treatment and reason.
- `manifest.json`: split counts, source hashes, and output hashes.

Use `load_preprocessed(output_dir)` to load the cleaned tables and fitted processor, or `load_encoded(output_dir, part)` to load one sparse matrix with its labels and source-row sequence. These outputs are derived from IEEE-CIS and remain local under the existing `.gitignore`; do not commit or redistribute them. Rerunning replaces the outputs for the selected dataset.

Bank, Default and Polish splits are stratified 60/20/20 with seed 42. Keep input rows/order, configuration, and software versions fixed to reproduce them. Their missing numeric values use the training median (configurable to mean); entirely missing numeric columns use zero. Categorical gaps use the training mode, with an all-missing fallback category. Validation and test use the training fill values. IEEE-CIS follows the time and training-sample rules above, with its own missingness and category handling.

- **Bank:** rename the day column to `day_of_month`, preserve/fill `unknown` categories, retain `pdays=-1`, add `previously_contacted`, and exclude `duration` by default. Map the target to 0/1.
- **Default:** drop ID, standardize target and repayment column names, and preserve undocumented codes as separate categories. The current data has no missing values.
- **Polish:** combine all five ARFF cohorts in configured order, retain the cohort label, replace infinite values with missing values, and impute financial ratios. Finite extreme values remain unchanged.

Both baseline arms for the earlier datasets read raw data and fit imputation, encoding, and scaling in memory within each CV training fold, including stacking inner folds. Final fitting learns preprocessing from training rows only. The IEEE-CIS training pipeline consumes its shared cleaned train/validation/test tables, and reuses the real-training-fitted transformations for synthetic rows without refitting on them.

Overrides: `--config /path/to/datasets.json`, `--input /path/to/source`, `--identity-input /path/to/train_identity.csv`, `--imputation mean`, and `--include-duration`. Input paths in configuration are relative to the repository root; `--root` overrides that root.

```python
from agentic_gan.tools.preprocessing import run_preprocessing

output = run_preprocessing("bank_marketing")
```

The `Preprocessor`, `profile_dataframe`, and `write_treatment_log` functions remain available to agent code for explicit use. IEEE-CIS outputs are only produced when the two local source CSVs are present.

## LLM agent (#74)

`agents.preprocessing.preprocess(request, run, client=...)` consumes a
`PreprocessInput` and returns `PreprocessOutput`. `PreprocessingAgent(client=...)`
provides the callable interface for the orchestrator's agent mapping. Existing CLI
preprocessing remains deterministic.

The agent profiles training rows only (types, missing counts, placeholder counts,
duplicates and numeric skew). It requests a typed treatment and one-line reason for
each affected column through the shared runtime, which uses temperature zero,
retries invalid output once and logs `LLM_FALLBACK` on failure. The table is enforced
in code: under 5% missing uses median/mode; at least 5% uses a seeded random forest
plus a missing indicator. Forest predictors exclude labels, provenance and excluded
features. If a forest cannot be fitted, training median/mode is used and logged.
Dataset semantic rules restrict choices further: Bank's unknown categories and
pdays sentinel remain informative; duration follows the config; IEEE keeps its
20% value-drop rule, missing flags and fixed category buckets. Outliers are profiled
and retained. A rule may consequently allow only one treatment.

Whole-row duplicates are removed only from training, ignoring the source-row ID.
Holdouts reuse the serialized training-fitted processor. Outputs are raw/cleaned,
unencoded CSVs, matching the current file-based agent convention, plus
`profile/profile.json`, `profile/treatment_log.csv`, and `profile/preprocessor.pkl`.
Source-row IDs are preserved when supplied. IEEE inputs must already be the joined,
time-split training sample and holdouts; the agent does not create a new split.

Run offline coverage with `uv run pytest tests/test_preprocessing_agent.py`.
No live LLM test or credentials are required.

### Pipeline integration

The default graph now runs `split_for_training` and `preprocess` before model
selection, quality checks and training. The splitter uses dataset loaders (including
Polish archives and the IEEE identity join), the frozen split recipe, and stable
`source_row` IDs. The older generic `split_data` API remains available.

Preprocessing also writes `profile/treatment_plan.json`, a typed, versioned recipe
with chosen actions, exclusions, source column names, and the dataset configuration.
`PreprocessOutput` carries its path and SHA-256 digest; the graph passes both to
`TrainingRequest`, and the manifest registers the plan. Training verifies the digest,
dataset settings, source rows and cleaned artifacts, then rebuilds the transformations
from the plan. It does not execute the preprocessing pickle to load a recipe.

For final fitting, each repeated-CV training fold, and stacking's inner folds,
repair models and statistics are fitted on that fit's real raw rows only. Synthetic
rows are already cleaned; they are encoded by the real-row-fitted transformer and
never train repair models or scaling. Missing flags are discrete inputs to SMOTENC.
Training-only deduplication is repeated within outer folds, and class counts reflect
retained rows. The plan's chosen actions stay fixed; no extra LLM calls occur in CV.

Legacy training requests without a plan retain deterministic preprocessing. Frozen
reference protocols and their existing drift checks are still required. Full-split
accepted GAN rows are not reused in repeated CV: a fold-local, quality-checked
`cv_gan_factory` is required, as before. The final evaluation agent now consumes
training results and writes a number-verified report; see `docs/evaluation.md`.

Offline integration coverage (all four dataset loaders, all training arms, repeated
CV, stacking, plan tampering, and artifact manifests):

```bash
uv run pytest tests/test_preprocessing_pipeline.py tests/test_ieee_preprocessing.py
```
