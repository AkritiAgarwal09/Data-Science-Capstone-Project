# Data-Science-Capstone-Project

DS Capstone — Agentic GAN Framework for Synthetic Financial Data Generation, Data Repair, and Bank Marketing Response Modeling

Columbia University — M.S. in Data Science
Course: Data Science Capstone
Industry Partner: TD Bank
Domain: Finance

Focus Areas: Generative AI · Machine Learning · Predictive Modeling · Reinforcement Learning

Overview

This project was developed as part of the Data Science Capstone for the M.S. in Data Science at Columbia University in collaboration with TD Bank.

The goal is to build an agentic generative modeling framework for financial tabular data that can:

generate high-fidelity synthetic financial/customer data,

identify and repair incomplete or low-quality records,

evaluate whether synthetic data preserves the statistical structure of real data, and

improve downstream bank marketing response prediction under realistic data-quality and class-imbalance constraints.

The project combines Generative Adversarial Networks (GANs), statistical data-quality evaluation, supervised machine learning, and agent-based orchestration into a reproducible end-to-end workflow.

Problem Statement

Financial institutions often work with datasets that contain missing values, imbalanced outcomes, limited samples, and privacy-sensitive customer attributes. These constraints can reduce the effectiveness of predictive models and make data sharing or experimentation difficult.

This capstone investigates whether synthetic-data generation and automated data-repair workflows can improve model development while preserving important properties of the original dataset.

## Setup

1. Install [uv](https://docs.astral.sh/uv/).
   - **Mac users:** also run `brew install libomp` (needed by XGBoost and LightGBM).
2. From the repo root, run `uv sync`. This creates `.venv` with the exact package versions in `uv.lock`.
3. Run code with `uv run ...`. For notebooks, run `uv run jupyter lab`, or select `.venv` as the kernel in VS Code.

To add a package, run `uv add <package>`, then commit both `pyproject.toml` and `uv.lock`. Don't use pip, conda or a requirements.txt.

Run the placeholder orchestrator with `uv run python -m agentic_gan.orchestrator --dataset bank_marketing`. It creates a folder under `runs/` with typed stub outputs, a manifest, `decisions.jsonl`, and a report. The stubs do not train models or evaluate the dataset.
Print its LangGraph Mermaid diagram with `uv run python -m agentic_gan.orchestrator --diagram`.

## Repository layout

- `src/agentic_gan/`: the system (orchestrator, agents, tools, schemas)
- `docs/`: architecture and design docs
- `eda/`: exploratory notebooks and baseline models
- `tests/`: automated tests
- `runs/`: pipeline run outputs, created at runtime and not tracked

Academic Context

This repository contains work completed for the Data Science Capstone within the M.S. in Data Science program at Columbia University.

The project is an academic collaboration with TD Bank and is intended for research, experimentation, and educational purposes. Any proprietary, confidential, or restricted data should not be committed to this public repository.

Contributors

Columbia University — M.S. in Data Science
Capstone Team — TD Bank Industry Project

Add team member names, faculty advisor, and industry mentor here.

License / Usage

This project was created for academic purposes. Usage of project code, data, or derived artifacts should comply with Columbia University requirements, applicable licenses, and any confidentiality or data-use agreements associated with the industry-sponsored capstone.

## Shared preprocessing

One script prepares Bank Marketing, Default, Polish Bankruptcy, and IEEE-CIS data:
`src/agentic_gan/tools/preprocessing.py`.

### Configuration

Edit [datasets.json](src/agentic_gan/utils/datasets.json) to set each dataset's input path, target column, categorical columns, imputation method, and split settings. Bank, Default, and Polish use stratified **60% training / 20% validation / 20% test**, **seed 42**, and **median imputation**. IEEE-CIS uses the fixed time split, deduplication, and training-only sample from issue #50. Bank's post-call `duration` feature is excluded by default.

Place the raw inputs at the configured paths (relative to the repo root):

| Dataset | Default input |
|---|---|
| Bank Marketing | `data/raw/bank_marketing.csv` |
| Default | `data/raw/default.csv` |
| Polish Bankruptcy | `data/raw/polish_companies_bankruptcy_uci365.zip` |
| IEEE-CIS | `data/external/train_transaction.csv` and `train_identity.csv` |

The script reads local inputs; it does not download them. See the [input format details](docs/preprocessing.md) before preparing the files.

### Run

After `uv sync`, run from the repo root:

```bash
uv run python -m agentic_gan.tools.preprocessing --dataset bank_marketing
uv run python -m agentic_gan.tools.preprocessing --dataset default
uv run python -m agentic_gan.tools.preprocessing --dataset polish
uv run python -m agentic_gan.tools.preprocessing --dataset ieee_cis
```

The script generates splits directly from the config. The same input rows in the same order, settings, and software versions reproduce the same splits. Imputation, category encoding, and scaling are learned **only from training rows**.

Optional overrides include `--config /path/to/datasets.json`, `--input /path/to/source`, `--imputation mean`, and `--include-duration` for a post-call Bank diagnostic.

### Outputs

Results are written to `data/interim/<dataset>/`:

Bank, Default and Polish each contain:

- `train_cleaned.csv`
- `validation_cleaned.csv`
- `test_cleaned.csv`

These tables contain corrected and imputed features before encoding/scaling. `source_row` identifies source rows and must be excluded from model/GAN features. The baseline runner fits encoding and scaling in memory using training rows only. IEEE-CIS uses the time split and training sample from issue #50, and also writes sparse scaled/encoded matrices, encoder settings, a treatment log, and a manifest under `data/interim/ieee_cis/`. Its identifier and time columns are excluded from features; all fitted cleaning and encoding rules come from the training sample. Raw inputs and generated outputs are ignored by Git.

See [preprocessing setup and tool API](docs/preprocessing.md) for dataset treatments and preprocessing tool reuse.

## Baseline models

Compare original and cleaned data with four fixed models: gradient boosting, LightGBM,
CatBoost, and stacking. Use Python 3.12 and `uv sync --frozen` first:

```bash
uv run python -m agentic_gan.tools.baselines --dataset default --arm both --cv-folds 0
uv run python -m agentic_gan.tools.baselines --dataset polish --arm both --cv-folds 0
uv run python -m agentic_gan.tools.baselines --dataset bank_marketing --arm both --cv-folds 0
uv run python -m agentic_gan.tools.baselines --dataset ieee_cis --arm both --cv-folds 0
```

Run these commands sequentially because they share `results/baselines.csv`. On the first
original-arm run, each model's threshold maximizes **validation F1**. Those numerical
thresholds are saved in `results/baseline_protocol_<dataset>_42_cv<folds>.json` and reused for
cleaned and augmented arms. IEEE-CIS uses `--cv-folds 0` because full-data CV is optional and
prohibitively slow at its 100k-row training size; the Training Agent reads this setting from
the protocol. An existing protocol is never silently overwritten or retuned.
Run `--arm both` or `--arm original` before running cleaned alone. Changed input data,
splits, model settings, or dependency versions require a separate experiment directory.

Both arms read raw inputs. Preprocessing fits inside training CV folds and stacking's
inner folds; the final fit uses training rows only. Models are ranked by validation
trapezoidal PR-AUC. Only the validation winner gets test metrics. CSV metrics prioritize
PR-AUC and F1, with accuracy last. Non-winner test cells are intentionally empty. The
per-run JSON files retain diagnostics, model settings, source/split hashes, and versions.
The refreshed protocols use seed 42 with optional training CV disabled for speed. CV, when
enabled, reports threshold-independent PR-AUC and ROC-AUC. Matching dataset/arm/seed/CV rows
are replaced on rerun.

The file-based Training Agent is `agents.training.run_training(request, run)`. It runs the
frozen suite on original, cleaned, SMOTENC, and quality-accepted CTGAN arms, saves models and
predictions, and returns `TrainingOutput`. See [docs/training.md](docs/training.md) for its
input contract, validation-only summaries, and optional repeated CV.

The lower-level in-memory helper also consumes the frozen protocol:

```python
from agentic_gan.agents.training import run_training_arms

results = run_training_arms(
    training_sets, X_val, y_val, X_test, y_test,
    protocol_path="results/baseline_protocol_bank_marketing_42_cv0.json",
    seed=42, cv_folds=0,
)
```

Supply the same real holdouts and feature schema for every arm. Synthetic rows belong
only in the training inputs and use the real-training-fitted encoder. For raw data,
supply an unfitted `preprocessor=`; for encoded data, omit it. The lower-level
`run_models(..., thresholds=protocol["thresholds"])` API is also available. Omitting
thresholds is reserved for original-arm calibration, not for comparison arms.

The augmented runner also requires frozen protocols and the full four-model suite.
It now defaults to seed 42, the seed frozen here. Additional seeds require reference
protocols calibrated on the same real holdouts; changed holdout labels/order are rejected.
It writes `results/augmented_bank_marketing_frozen.csv`, keeping the historical 0.5
results separate. This task does not rerun the historical augmentation experiments.

See [baseline results and rationale](docs/notes/baselines.md) for the model choices,
thresholds, measured results, and dataset-specific explanations.
