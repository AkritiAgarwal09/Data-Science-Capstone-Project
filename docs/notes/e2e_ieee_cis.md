# IEEE-CIS inside the pipeline: what blocks an end-to-end run (issue #85)

Measured 2026-11-05 on one MacBook (CPU only), from `main` at `16ef0f8`. The IEEE-CIS files stay
local (Kaggle rules, section 7B); this note holds timings and counts only, no rows.

**Result: the pipeline cannot finish on IEEE-CIS as it is built.** The split works. Preprocessing and
CTGAN each take hours to days on 100,000 training rows, and the training agent would repeat the
preprocessing cost inside every model fit. No run folder was completed, so there is no results table
for IEEE-CIS yet. The fixes below are small and each has an owner.

## What was run

`config_for_dataset("ieee_cis")` through the real `split_for_training` and `preprocess`, with the LLM
steps on their deterministic fallback (no Vertex settings on this machine). The files were linked
from `data/external/ieee-fraud-detection/` to the paths `datasets.json` expects.

| Stage | Result |
|---|---|
| Split (time cuts, 100,000-row training sample) | 76 s, 3.5 GB peak memory. Training sample has 3,383 positives and 96,617 negatives |
| Preprocessing, LLM loop | 364 columns, one LLM step each. Treatments: 255 `drop` (over 20% missing), 67 `model_missing`, 32 `median`, 10 `keep_category` |
| Preprocessing, fit | Not finished after 13 minutes, then stopped. It fits 67 RandomForest imputers (see below) |
| CTGAN input | 181 columns after the drops. One-hot width 8,648: `card1` alone has 7,663 categories |

## Blockers, in the order they hit

1. **Model-based imputation is too slow.** `TreatmentPreprocessor.fit` (`tools/preprocessing.py`)
   fits one RandomForest (32 trees, depth 8, one core) per `model_missing` column on all numeric
   predictors. One such model on 100,000 rows and 170 predictors took 258 s on this machine, so the 67
   columns need about 4.8 hours for one fit. The training agent refits the same preprocessor in every
   model fit and cross-validation fold (4 models, 5 seeds, every arm and fill level), so training
   would repeat it hundreds of times. *Fix:* for IEEE-CIS use `median` plus the missing-indicator flag
   that IEEE already gets, or cap the imputer's rows and predictors. Owner: Noor (preprocessing).
2. **One LLM call per column, with the whole profile in every prompt.** Each of the 364 steps sends
   about 47,000 characters, and the decision log reached 19.5 MB before the fit started. Every column
   has exactly one allowed treatment, so the model cannot choose anything. A live run would make 364
   Vertex calls of roughly 12,000 tokens each. *Fix:* skip the LLM when `allowed` has one option, and
   send only that column's profile. Owner: Noor (`agents/preprocessing.py`).
3. **CTGAN is about 35 hours at 300 epochs.** The time is about 0.0042 s per row per epoch, almost
   all of it the 8,648-wide one-hot layer.

   | Training rows | Per epoch | 300 epochs |
   |---|---|---|
   | 100,000 | about 420 s | about 35 h |
   | 20,000 | about 84 s | about 7 h |
   | 5,000 | about 21 s | about 1.7 h |

   *Fix:* bucket `card1`, `card2`, `addr1` and the other high-cardinality columns (top categories plus
   `other`), train on a stratified subsample, and run longer jobs on a GCP GPU. This is the decision in
   issue #90 (Kyle), and these numbers are the CPU test it asks for.
4. **The quality gate and the four-model training suite at 100,000 rows** were not reached, so their
   time is unknown.

## What to do next

1. Noor: fixes 1 and 2 (one rule change in the IEEE preprocessing tier and one `if` in the agent).
2. Kyle (#90): bucket the high-cardinality columns, pick the CTGAN sample and epoch budget, and record
   them in `utils/datasets.json`.
3. Rerun `uv run python -m agentic_gan.orchestrator --dataset ieee_cis` with live Vertex settings,
   then build the table with `uv run python -m agentic_gan.tools.cross_dataset runs/<id> --out
   results/cross_dataset.csv`.

## Reproduce the measurements

```bash
ln -s ieee-fraud-detection/train_transaction.csv data/external/train_transaction.csv
ln -s ieee-fraud-detection/train_identity.csv data/external/train_identity.csv
DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/sklearn/.dylibs" \
  .venv/bin/python -u -m agentic_gan.orchestrator --dataset ieee_cis
```

`DYLD_LIBRARY_PATH` is the macOS workaround for LightGBM's missing OpenMP library, as in
`docs/notes/e2e_default.md`. `data/external/` is git-ignored, so the links are not committed.
