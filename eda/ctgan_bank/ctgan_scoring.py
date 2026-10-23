"""Score Bank CTGAN synthetic tables with the quality-gate checks and log them as runs.

Each run is one row in eda/ctgan_bank/artifacts/ctgan_runs.csv: the settings that made the table,
the fidelity scorecard (tools/quality_fidelity, thresholds = 2x the real-vs-real noise floor) and
the real-vs-synthetic classifier AUC (tools/quality_privacy). Real subscribers in the training
split are the reference, as in the scorecard.

Score existing files from the command line (label=path pairs):

    uv run python eda/ctgan_bank/ctgan_scoring.py v1_fill50=<path to a synthetic CSV>
"""

import sys
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from agentic_gan.tools.quality_fidelity import (
    BANK_RULES,
    fidelity_checks,
    real_vs_real,
    suggest_thresholds,
)
from agentic_gan.tools.quality_privacy import distinguishability_check
from agentic_gan.utils.datasets import DATASETS

ROOT = Path(__file__).resolve().parents[2]
CONFIG = DATASETS["bank_marketing"]
FOLDER = ROOT / "data" / "interim" / CONFIG.name
RUNS_FILE = ROOT / "eda" / "ctgan_bank" / "artifacts" / "ctgan_runs.csv"
WATCH_COLUMNS = ("job", "month", "poutcome", "marital", "day_of_month", "pdays")
METADATA = {"source_row", "is_synthetic"}


class Scorer:
    """Real training and validation tables, the noise-floor thresholds and a ``score`` method."""

    def __init__(self) -> None:
        self.real_train = pd.read_csv(FOLDER / "train_cleaned.csv")
        self.real_val = pd.read_csv(FOLDER / "validation_cleaned.csv")
        self.target = CONFIG.target
        self.common = {
            "target": self.target,
            "positive": 1,
            "discrete": CONFIG.discrete,
            "rules": BANK_RULES,
        }
        self.reference = self.real_train[self.real_train[self.target] == 1]
        baseline = real_vs_real(self.real_train, self.real_val, **self.common)
        self.thresholds = suggest_thresholds(baseline.values)
        self.baseline = baseline.values
        self.features = [
            c for c in self.real_train if c not in {self.target, *METADATA}
        ]

    def score(self, synth: pd.DataFrame, label: str, settings: Mapping | None = None) -> dict:
        """One run row: settings, fidelity checks, per-column distances and classifier AUC."""
        report = fidelity_checks(
            self.reference, synth, thresholds=self.thresholds, **self.common
        )
        auc_check, details = distinguishability_check(
            self.real_train, self.real_val, synth, self.features, self.target, 1
        )
        row = {"label": label, **(settings or {}), "n_rows": len(synth)}
        row.update(report.values)
        row["worst_column"] = report.worst_column
        distances = report.columns["distance"]
        row.update({f"dist_{c}": float(distances[c]) for c in WATCH_COLUMNS if c in distances})
        row["technician_share"] = float((synth["job"] == "technician").mean())
        row["real_vs_synth_auc"] = auc_check.value
        row["real_vs_real_auc"] = float(details["real_vs_real_auc"])
        row["passed"] = report.passed
        row["failing"] = ";".join(c.name for c in report.failing() if c.hard)
        return row


def save_runs(rows: list[dict], path: Path = RUNS_FILE) -> pd.DataFrame:
    """Add rows to the table of runs, replacing any earlier run with the same label."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        table = pd.concat([old[~old["label"].isin(table["label"])], table], ignore_index=True)
    table.to_csv(path, index=False)
    return table


def main(argv: list[str]) -> None:
    if not argv or any("=" not in a for a in argv):
        raise SystemExit("usage: ctgan_scoring.py label=path.csv [label=path.csv ...]")
    scorer = Scorer()
    print("thresholds (2x real-vs-real):", {k: v for k, v in scorer.thresholds.items()
                                           if not k.startswith("rule:")})
    rows = []
    for arg in argv:
        label, path = arg.split("=", 1)
        rows.append(scorer.score(pd.read_csv(path), label))
    table = save_runs(rows)
    shown = ["label", "mean_column_distance", "worst_column_distance", "worst_column",
             "spearman_diff", "rule:pdays_poutcome", "technician_share", "real_vs_synth_auc",
             "passed"]
    print(table[shown].round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
