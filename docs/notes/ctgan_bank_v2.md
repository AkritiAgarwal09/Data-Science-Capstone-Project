# CTGAN v2 on Bank Marketing

v1 failed the quality checks (#49). v2 fixes most of it but does not pass every check, and its
rows still do not beat real-only training. Files: `data/interim/bank_marketing/synthetic/bank_ctgan_v2_fill{10,25,50}.csv`
(same format as v1; fill = share of the 20,780-row class gap; the smaller files are the first rows
of the 50% file). Code: `eda/ctgan_bank/` (`train_variant.py`, `make_v2.py`, `ctgan_scoring.py`).
Every run is in `eda/ctgan_bank/artifacts/ctgan_runs.csv`.

## Result

| | v1 | v2 (50% fill) | Limit |
|---|---|---|---|
| Mean column distance | 0.106 | 0.042 | 0.05 |
| Worst column distance | 0.274 (`job`) | 0.117 (`balance`) | 0.10 |
| Spearman difference | 0.066 | 0.0451 | 0.045 |
| Technician share of subscribers | 42% | 14.5% | 15.3% real |
| Real-vs-synthetic AUC (seed 42) | 0.864 | 0.735 | about 0.52 (real vs real) |

Limits are 2x the real-vs-real noise floor (`suggest_thresholds`). v2 passes the mean check,
fails the worst-column check on `balance`, and is borderline on Spearman (0.039 and 0.043 at the
10% and 25% fills, 0.0451 at 50%, just over the limit). A classifier can still tell the rows
from real ones (AUC 0.735, down from 0.864).

Gate check (the 15 hard checks of the quality gate, suggested limits, seed 42): v1 fails 6 of 15
at the 25% fill, v2 fails 3 of 15 at 25% (worst column, and the PR-AUC and best-F1 gain checks,
-0.011 and -0.003) and 4 of 15 at 50% (the same three plus Spearman).

## What helped

Each change was tried alone first (scores below are the 50% fill, 10,390 rows).

| Change | Mean | AUC | Effect |
|---|---|---|---|
| v1 as saved | 0.106 | 0.864 | baseline |
| Add the missing `pdays`/`poutcome` rule filter | 0.104 | 0.859 | removes the 1.8% contradictory rows, little else |
| Sample without the target condition, keep subscriber rows | 0.094 | 0.832 | conditioning caused the `job` skew (technician 42% to 12%) |
| Rake the pool to real job, month, poutcome, marital, day-of-month shares | 0.043 | 0.803 | biggest fidelity gain |
| `pac` 5 (from 10) | 0.035 | 0.785 | smaller packs: AUC improves 20 < 10 < 5 |
| Discriminator steps 3 (from 1) | 0.040 | 0.781 | a stricter discriminator |
| `pac` 5 + steps 3 | 0.041 | 0.737 | best; saved as v2 |

The rake rows use the unconditional sampling and the rule filter too.

## What did not help

- **More epochs (600):** mean 0.051 and Spearman 0.115 after raking, worse than 300 epochs.
- **`pac` 20:** worse on every check (technician 74.5% when conditioned, AUC 0.92).

## Caveat

Raking matches the exact category shares the scorecard measures, so part of the fidelity gain is by
design. The AUC and the downstream result below are the independent checks.

## Downstream (#35 comparison, Bank, 5 seeds)

Test PR-AUC with real training rows plus synthetic subscribers:

| Fill | Real only | SMOTENC | CTGAN v1 | CTGAN v2 |
|---|---|---|---|---|
| 0% | 0.488 | | | |
| 10% | | 0.462 | 0.468 | 0.481 |
| 25% | | 0.455 | 0.438 | 0.471 |
| 50% | | 0.441 | 0.405 | 0.452 |

v2 removes most of v1's loss (at 50%, -0.083 becomes -0.036) and beats SMOTENC at every fill level,
but no synthetic arm beats real-only training. F1 at threshold 0.5 rises with synthetic rows
(0.359 to 0.448-0.480), mostly from the threshold shift, not better ranking. Models and threshold
are the provisional ones; rerun after #52 freezes them.

## Open

- `balance` (a skewed numeric column) is the remaining failing check.
- Not tried: `pac` below 5, more discriminator steps, other network sizes.
- The same recipe (unconditional sampling, rule filter, raking, small `pac`) should carry over to
  Default and Polish.
