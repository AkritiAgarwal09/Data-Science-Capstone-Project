"""Audit a run folder and print a checklist table."""

import json
import sys
from pathlib import Path

import pandas as pd


def audit(run_path: Path) -> None:
    print(f"\n{'='*70}")
    print(f"RUN: {run_path}")
    print(f"{'='*70}")

    state = json.loads((run_path / "state.json").read_text())
    print("\n[1] STAGE STATUSES")
    for k, v in state.items():
        if k not in ("run_id", "run", "config"):
            print(f"  {k}: {v}")

    decisions_path = run_path / "decisions.jsonl"
    if not decisions_path.exists():
        print("  WARNING: decisions.jsonl not found")
        return
    decisions = pd.read_json(decisions_path, lines=True)

    # decisions.jsonl uses "agent" (not "stage") as the grouping field
    print("\n[2] SELECTION STEPS (attempt 1)")
    sel = decisions[decisions["agent"] == "model_selection"]
    gates = [
        "dcgan_excluded", "size_gate", "horizon_gate", "missing_gate",
        "leakage_gate", "minority_gate", "temporal_test", "type_routing", "fallback",
    ]
    for gate in gates:
        rows = sel[sel["rule"] == gate]
        if not rows.empty:
            r = rows.iloc[0]
            print(f"  {gate}: {r['outcome']}")
            if "rationale" in rows.columns and pd.notna(r.get("rationale")):
                print(f"    {str(r['rationale'])[:120]}")

    print("\n[3] HYPERPARAMETERS PER ATTEMPT")
    hyp_rules = [
        "hyperparameters", "hyperparameter_retry",
        "retry_unchanged", "unsupported_adjustment",
    ]
    for rule in hyp_rules:
        rows = decisions[decisions["rule"] == rule]
        for _, r in rows.iterrows():
            attempt = r.get("attempt", "?")
            inputs = str(r.get("inputs", ""))[:100]
            print(f"  attempt={attempt} {rule}: {r['outcome']} | {inputs}")

    print("\n[4] QUALITY RESULTS PER ATTEMPT")
    quality = decisions[decisions["agent"] == "quality_check"]
    for rule in ["quality_gate", "hard_check"]:
        rows = quality[quality["rule"] == rule]
        for _, r in rows.iterrows():
            attempt = r.get("attempt", "?")
            print(f"  attempt={attempt} {rule}: {r['outcome']} | {str(r.get('inputs',''))[:120]}")

    print("\n[5] LLM STEP COUNTS")
    if "llm_step" in decisions["rule"].values:
        llm = decisions[decisions["rule"] == "llm_step"]
        print(llm.groupby(["agent", "outcome"]).size().to_string())
    else:
        print("  No llm_step lines found.")

    print("\n[6] MANIFEST")
    manifest_path = run_path / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        print(f"  Artifacts registered: {len(manifest.get('artifacts', []))}")
        print(f"  config_sha256: {manifest.get('config_sha256','N/A')[:16]}...")
    else:
        print("  manifest.json not found")

    print("\n[7] TRAINING ARMS")
    metrics_path = run_path / "models" / "metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        if "arm" in metrics.columns and "fill_level" in metrics.columns:
            combos = metrics[["arm", "fill_level"]].drop_duplicates()
            print(f"  Arm/fill combos ({len(combos)}):")
            print(combos.to_string(index=False))
            print(f"  Total fits: {len(metrics)}")
        else:
            print(f"  columns: {list(metrics.columns)}")
    else:
        print("  models/metrics.csv not found")

    print("\n[8] SANITY")
    for split_name in ["train", "val", "validation", "test"]:
        for arm in ["baseline", "raw"]:
            p = run_path / "preprocessing" / f"{arm}_{split_name}.csv"
            if p.exists():
                full = pd.read_csv(p)
                if "is_synthetic" in full.columns:
                    n = int(full["is_synthetic"].sum())
                    flag = "WARNING: has is_synthetic rows!" if n > 0 else "✓ no is_synthetic rows"
                    print(f"  {p.name}: {flag}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python audit_run.py <run_folder> [<run_folder2> ...]")
    for path in sys.argv[1:]:
        audit(Path(path))
