"""Read-only audit of a run's targeted-retry decisions (issue #123).

Not part of the package. Empty violation list means the run passes the checks.

    uv run --no-sync python scripts/audit_retry.py runs/<run_id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agentic_gan.agents.model_selection import SEVEN_KNOBS, settings_key
from agentic_gan.orchestrator.decisions import read_decisions
from agentic_gan.orchestrator.manifest import RunDir


def _attempt(inputs) -> int | None:
    if not isinstance(inputs, dict) or "attempt" not in inputs:
        return None
    try:
        return int(inputs["attempt"])
    except (TypeError, ValueError):
        return None


def _seven(settings: dict | None) -> dict | None:
    if not isinstance(settings, dict):
        return None
    missing = [name for name in SEVEN_KNOBS if name not in settings]
    if missing:
        return None
    return {
        "epochs": int(settings["epochs"]),
        "batch_size": int(settings["batch_size"]),
        "pac": int(settings["pac"]),
        "discriminator_steps": int(settings["discriminator_steps"]),
        "sampling_mode": settings["sampling_mode"],
        "rule_filter": bool(settings["rule_filter"]),
        "rake_columns": list(settings["rake_columns"]),
    }


def _changed_knobs(previous: dict, applied: dict) -> list[str]:
    changed = []
    for name in SEVEN_KNOBS:
        if name == "rake_columns":
            if sorted(previous[name]) != sorted(applied[name]):
                changed.append(name)
        elif previous[name] != applied[name]:
            changed.append(name)
    return changed


def _hyperparameter_rows(decisions):
    rows = []
    for _, row in decisions.query("rule == 'hyperparameters'").iterrows():
        attempt = _attempt(row.inputs)
        settings = _seven(row.inputs)
        if attempt is None or settings is None:
            continue
        rows.append((attempt, settings, row))
    return rows


def _ladder_exhausted(inputs, rationale: str) -> bool:
    if isinstance(inputs, dict) and inputs.get("exhausted") is True:
        return True
    text = str(rationale or "").lower()
    return "every rung is used up" in text or "every rung" in text and "used up" in text


def audit(run_dir: Path | RunDir | str) -> list[str]:
    """Return violation messages for a run whose attempt 1 failed the quality gate."""
    run = run_dir if isinstance(run_dir, RunDir) else RunDir(Path(run_dir).resolve())
    decisions = read_decisions(run)
    violations: list[str] = []

    # 1. Attempt-2 hyperparameter_retry: source llm/ladder; applied differs from previous.
    retries = decisions.query("rule == 'hyperparameter_retry'")
    attempt2 = [
        row
        for _, row in retries.iterrows()
        if _attempt(row.inputs) == 2
    ]
    if not attempt2:
        violations.append(
            "missing hyperparameter_retry row for attempt 2 "
            "(expected after attempt 1 failed the gate)"
        )
    else:
        row = attempt2[-1]
        inputs = row.inputs if isinstance(row.inputs, dict) else {}
        source = inputs.get("source")
        if source not in {"llm", "ladder"}:
            violations.append(
                f"attempt 2 hyperparameter_retry source={source!r}; expected 'llm' or 'ladder'"
            )
        previous = _seven(inputs.get("previous_settings"))
        applied = _seven(inputs.get("applied"))
        if previous is None or applied is None:
            violations.append(
                "attempt 2 hyperparameter_retry missing previous_settings or applied seven knobs"
            )
        elif settings_key(previous) == settings_key(applied):
            violations.append(
                "attempt 2 hyperparameter_retry applied settings equal previous_settings"
            )

    # 2. retry_unchanged only when the ladder is exhausted.
    for _, row in decisions.query("rule == 'retry_unchanged'").iterrows():
        inputs = row.inputs if isinstance(row.inputs, dict) else {}
        if not _ladder_exhausted(inputs, row.rationale):
            attempt = _attempt(inputs)
            violations.append(
                f"retry_unchanged on attempt {attempt} without evidence the ladder was exhausted"
            )

    # 3. No unsupported_adjustment rows.
    unsupported = decisions.query("rule == 'unsupported_adjustment'")
    if not unsupported.empty:
        violations.append(
            f"unsupported_adjustment logged {len(unsupported)} time(s); must never appear"
        )

    # 4. Attempt 3 (if reached): hyperparameters differ from 1 and 2; no SKIPPED fallback.
    by_attempt = {attempt: settings for attempt, settings, _row in _hyperparameter_rows(decisions)}
    reached_three = 3 in by_attempt or any(
        _attempt(row.inputs) == 3
        for _, row in decisions.iterrows()
        if isinstance(row.inputs, dict)
        and row.rule in {"hyperparameter_retry", "retry_unchanged", "fallback", "quality_gate"}
    )
    if reached_three:
        if 3 not in by_attempt:
            violations.append("attempt 3 was reached but no hyperparameters row for attempt 3")
        else:
            key3 = settings_key(by_attempt[3])
            for earlier in (1, 2):
                if earlier in by_attempt and settings_key(by_attempt[earlier]) == key3:
                    violations.append(
                        f"attempt 3 hyperparameters match attempt {earlier}; must differ"
                    )
        skipped = [
            row
            for _, row in decisions.query("rule == 'fallback'").iterrows()
            if _attempt(row.inputs) == 3 and row.outcome == "SKIPPED"
        ]
        if skipped:
            violations.append("fallback SKIPPED logged for attempt 3")

    # 5. No stage_error FAILED.
    failed_stages = decisions.query("rule == 'stage_error' and outcome == 'FAILED'")
    if not failed_stages.empty:
        stages = [
            (row.inputs or {}).get("stage", "?")
            for _, row in failed_stages.iterrows()
        ]
        violations.append(f"stage_error FAILED present for stage(s): {stages}")

    return violations


def format_table(run_dir: Path | RunDir | str) -> str:
    """Short table: attempt, source, changed knobs, trigger check."""
    run = run_dir if isinstance(run_dir, RunDir) else RunDir(Path(run_dir).resolve())
    decisions = read_decisions(run)
    lines = ["attempt  source  changed_knobs  trigger"]
    lines.append("-" * 72)
    rows = decisions.query("rule in ['hyperparameter_retry', 'retry_unchanged']")
    if rows.empty:
        lines.append("(no hyperparameter_retry / retry_unchanged rows)")
        return "\n".join(lines)
    for _, row in rows.iterrows():
        inputs = row.inputs if isinstance(row.inputs, dict) else {}
        attempt = _attempt(inputs)
        source = inputs.get("source", "-")
        previous = _seven(inputs.get("previous_settings"))
        applied = _seven(inputs.get("applied"))
        if previous is not None and applied is not None:
            changed = ",".join(_changed_knobs(previous, applied)) or "(none)"
        else:
            changed = "?"
        trigger = inputs.get("trigger", "-")
        mark = "unchanged" if row.rule == "retry_unchanged" else source
        lines.append(f"{attempt}  {mark}  {changed}  {trigger}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "run_dir",
        type=Path,
        help="a run folder, for example runs/bank_marketing_...",
    )
    args = parser.parse_args(argv)
    run = RunDir(args.run_dir.resolve())
    if not run.root.exists():
        print(f"run folder not found: {run.root}", file=sys.stderr)
        return 1
    print(format_table(run))
    print()
    violations = audit(run)
    if not violations:
        print("audit: PASS (0 violations)")
        return 0
    print(f"audit: FAIL ({len(violations)} violation(s))")
    for message in violations:
        print(f"  - {message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
