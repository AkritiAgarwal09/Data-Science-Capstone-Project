"""Append-only decisions for one run."""

import json
import os
import sys
from datetime import datetime

import pandas as pd

from .manifest import RunDir


def log_progress(agent, action, *, tool=None, next_action=None) -> None:
    """Print concise live progress when the CLI script enables it."""
    if os.environ.get("AGENTIC_GAN_PROGRESS") != "1":
        return
    fields = [f"agent={agent}", action]
    if tool:
        fields.append(f"tool={tool}")
    if next_action:
        fields.append(f"next={next_action}")
    print(
        f"[{datetime.now().astimezone():%H:%M:%S}] " + " | ".join(fields),
        file=sys.stderr,
        flush=True,
    )


def log_decision(agent, rule, inputs, outcome, rationale, *, run: RunDir) -> None:
    entry = dict(agent=agent, rule=rule, inputs=inputs, outcome=outcome, rationale=rationale)
    with run.decisions_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, default=str) + "\n")
    label = outcome if isinstance(outcome, str) else json.dumps(outcome, default=str)[:160]
    log_progress(agent, f"decision={rule} outcome={label}")


def read_decisions(run: RunDir) -> pd.DataFrame:
    columns = ["agent", "rule", "inputs", "outcome", "rationale"]
    if not run.decisions_path.exists():
        return pd.DataFrame(columns=columns)
    with run.decisions_path.open(encoding="utf-8") as file:
        return pd.DataFrame((json.loads(line) for line in file), columns=columns)
