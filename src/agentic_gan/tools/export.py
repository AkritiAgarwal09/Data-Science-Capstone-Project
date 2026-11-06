"""Write the GAN-augmented training tables of a finished run.

The pipeline keeps the baseline training split and each synthetic set as separate files. This joins
them, one table per accepted fill level, with `is_synthetic` marking the synthetic rows. A level is
accepted when it passed the newest quality report, and, if the training agent saved a plan, was
also in its plan.

    uv run python -m agentic_gan.tools.export runs/<run_id>
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from agentic_gan.orchestrator.decisions import log_decision
from agentic_gan.orchestrator.manifest import RunDir
from agentic_gan.schemas import QualityReport, Verdict
from agentic_gan.utils.synthetic import IS_SYNTHETIC


def _token(level: float) -> str:
    return f"{level:.2f}".replace(".", "p")


def export_augmented(run: RunDir) -> list[Path]:
    """Write report/augmented_train_fill_<level>.csv for each accepted level and return the paths.

    No file is written, and the reason is logged, when the newest quality report is missing or
    rejected every level: that is a valid result, not an error.
    """
    reports = sorted(
        run.subdir("quality").glob("quality_report_attempt_*.json"),
        key=lambda path: int(re.search(r"(\d+)\.json$", path.name).group(1)),
    )
    if not reports:
        _log(run, [], "NO_QUALITY_REPORT", "No quality report in this run, so nothing was accepted")
        return []
    newest = reports[-1]
    attempt = int(re.search(r"(\d+)\.json$", newest.name).group(1))
    report = QualityReport.model_validate_json(newest.read_text())
    if report.verdict is not Verdict.PASS:
        _log(
            run,
            [],
            "NO_SYNTHETIC_DATA",
            f"The quality gate rejected every level at attempt {attempt}",
        )
        return []

    levels = report.passing_levels
    plan = run.subdir("models") / "training_plan.json"
    if plan.is_file():
        planned = json.loads(plan.read_text())["gan_levels"]
        levels = [level for level in levels if level in planned]
    real = pd.read_csv(run.subdir("data") / "baseline_train.csv").drop(columns="source_row")
    real[IS_SYNTHETIC] = 0
    paths = []
    for level in levels:
        synthetic = pd.read_csv(
            run.subdir("synthetic") / f"fill_{_token(level)}_attempt_{attempt}.csv"
        )
        path = run.subdir("report") / f"augmented_train_fill_{_token(level)}.csv"
        pd.concat([real, synthetic], ignore_index=True).to_csv(path, index=False)
        paths.append(path)
    _log(
        run,
        paths,
        "EXPORTED",
        f"Baseline training rows plus accepted synthetic rows, attempt {attempt}",
    )
    return paths


def _log(run: RunDir, paths: list[Path], outcome: str, rationale: str) -> None:
    log_decision(
        "export",
        "augmented_dataset",
        {"files": [p.name for p in paths]},
        outcome,
        rationale,
        run=run,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "run_dir", type=Path, help="a run folder, for example runs/bank_marketing_..."
    )
    run = RunDir(parser.parse_args().run_dir.resolve())
    paths = export_augmented(run)
    if not paths:
        print("No synthetic data was accepted in this run; see decisions.jsonl.")
        return 0
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
