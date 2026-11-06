"""Verify the 3year-only Polish split produces 297/99/99 minority rows."""

from pathlib import Path

from agentic_gan.orchestrator.split import split_dataset
from agentic_gan.tools.preprocessing import load_dataset, normalize
from agentic_gan.utils.datasets import DATASETS

config = DATASETS["polish"]
raw = load_dataset(config, Path(config.file))
data = normalize(raw, config, config.include_duration)
ids = split_dataset(raw, config)

parts = {name: data.loc[ids[name]] for name in ("train", "validation", "test")}
total = sum(len(p) for p in parts.values())
target = config.target

print(f"Total rows : {total}")
for name, part in parts.items():
    pos = int((part[target] == 1).sum())
    print(f"  {name:12}: {len(part)} rows, {pos} positives")

assert total == 10_503, f"Expected 10503 rows, got {total}"
assert int((parts['train'][target] == 1).sum()) == 297, \
    f"Expected 297 train positives, got {int((parts['train'][target] == 1).sum())}"
assert int((parts['validation'][target] == 1).sum()) == 99, \
    f"Expected 99 val positives, got {int((parts['validation'][target] == 1).sum())}"
assert int((parts['test'][target] == 1).sum()) == 99, \
    f"Expected 99 test positives, got {int((parts['test'][target] == 1).sum())}"
assert parts['train']['cohort'].nunique() == 1, "Expected single cohort value in train"
print("\nAll assertions passed.")

# Missing-indicator analysis
train_raw = data.loc[ids["train"]]
miss_rates = train_raw.isna().mean()
flagged = miss_rates[miss_rates >= 0.05]
print(f"\nColumns with >=5% missing in train ({len(flagged)} total):")
for col, rate in flagged.sort_values(ascending=False).items():
    print(f"  {col}: {rate:.1%}")
if flagged.empty:
    print("  WARNING: NO missing columns found — CTGAN will be blocked at type_routing!")
