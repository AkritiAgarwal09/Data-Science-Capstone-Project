"""One-process dashboard: serves the built frontend and a read-only JSON view of runs/.

Run `npm --prefix frontend run build`, then `uv run python -m agentic_gan.web`.
"""

import csv
import json
import glob
import mimetypes
import os
import re
import subprocess
import sys
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

from agentic_gan.schemas import STAGE_ORDER
from agentic_gan.utils.datasets import DATASETS

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
DIST = ROOT / "frontend" / "dist"
# Which agent logs its decisions under which stage. The split is deterministic code.
AGENT_OF = {"preprocessing": "preprocessing", "model_selection": "model_selection",
            "quality_check": "quality", "training": "training", "evaluation": "evaluation"}


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


_proc: subprocess.Popen | None = None


def datasets() -> dict:
    return {
        "datasets": [{"name": n, "target": c.target, "ready": (ROOT / c.file).exists()}
                     for n, c in DATASETS.items()],
        "running": _proc is not None and _proc.poll() is None,
    }


def start_run(name: str) -> int:
    """Launch the orchestrator for one known dataset. Returns an HTTP status."""
    global _proc
    if name not in DATASETS or not (ROOT / DATASETS[name].file).exists():
        return 400
    if datasets()["running"]:
        return 409
    env = dict(os.environ, LANGSMITH_TRACING="false", AGENTIC_GAN_PROGRESS="1")
    envfile = ROOT / ".env"  # same Vertex settings the e2e script loads
    for line in envfile.read_text(encoding="utf-8").splitlines() if envfile.exists() else []:
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env.setdefault(k.strip(), v.strip().strip("\"'"))
    libs = glob.glob(str(ROOT / ".venv/lib/python*/site-packages/sklearn/.dylibs"))
    if libs:  # OpenMP workaround for macOS, as in scripts/run_e2e_bank.sh
        env["DYLD_LIBRARY_PATH"] = libs[0] + os.pathsep + env.get("DYLD_LIBRARY_PATH", "")
    # No redirect: progress lines appear in the terminal running this server, like the e2e script.
    _proc = subprocess.Popen([sys.executable, "-u", "-m", "agentic_gan.orchestrator", "--dataset", name],
                             cwd=ROOT, env=env, start_new_session=True)
    return 202


def list_runs() -> list[str]:
    if not RUNS.is_dir():
        return []
    dirs = (d for d in RUNS.iterdir() if (d / "state.json").exists())
    return [d.name for d in sorted(dirs, key=lambda d: d.stat().st_mtime, reverse=True)]


def _rows(path: Path) -> int:
    with path.open("rb") as f:
        return max(sum(1 for _ in f) - 1, 0)


_LONG_NUM = re.compile(r"(?<![\w.])-?\d+\.\d{4,}")


def r3(text: str) -> str:
    """Round every long decimal in a text to three places."""
    return _LONG_NUM.sub(lambda m: f"{float(m.group()):.3f}", text)


METRICS = ("f1", "pr_auc", "precision", "recall")


GATES = {
    "dcgan_excluded": "DCGAN generates images, so it is never an option for a tabular table.",
    "size_gate": "Is there enough training data? A generator needs a minimum number of rows to learn the table.",
    "horizon_gate": "Datasets with several prediction horizons must be pooled consistently. Passes when there is one horizon, or none.",
    "missing_gate": "A generator cannot learn from gaps. Passes when no unresolved missing values remain after preprocessing.",
    "leakage_gate": "Columns that give away the target or identify rows must not be generated. Passes when none are in the table.",
    "minority_gate": "The rare class needs enough examples to be learned. Passes when its count reaches the minimum.",
    "temporal_test": "Does the data behave like a time series (entity key, time index, enough steps, no repeated observations)? Only then is TimeGAN considered. FAIL means a tabular generator is used instead.",
}


CHECKS = {
    "mean_column_distance": "Fidelity. Average, over all columns, of how far the synthetic column's distribution is from the real one (0 = identical, 1 = completely different).",
    "worst_column_distance": "Fidelity. The single worst column's distance. One badly modelled column fails this even when the average is fine.",
    "column_distance": "Fidelity, per column. Each column is scored on its own; a column over the limit is listed so you can see what the generator got wrong.",
    "spearman_diff": "Relationships between numeric columns. How much the pairwise correlations in the synthetic data differ from the real data.",
    "cramers_v_diff": "Relationships between categorical columns. How much the pairwise associations (Cramér's V) differ from the real data.",
    "rule": "Business sanity rule. The share of synthetic rows that break a rule the real data respects, such as impossible value combinations.",
    "label_balance": "The share of positive-class rows in the synthetic file against the share expected. Keeps the minority class from drifting.",
    "schema_match": "Structure. Counts real columns missing from the synthetic file; it must be zero.",
    "exact_train_copies": "Privacy. Synthetic rows identical to a real training row. Copies mean the generator memorised data.",
    "synthetic_duplicates": "Privacy and diversity. Rows repeated inside the synthetic file, a sign the generator collapsed onto a few patterns.",
    "dcr_ratio": "Privacy. Typical distance from a synthetic row to its nearest training row, divided by the same distance for real held-out rows. Below the limit means synthetic rows sit suspiciously close to training rows.",
    "real_vs_synthetic_auc": "Detectability. A classifier tries to tell real rows from synthetic ones; 0.5 means it cannot. Near 1.0 means the synthetic data is easy to spot.",
    "augmentation_pr_auc_gain": "Usefulness. Change in PR-AUC on real validation rows when a model also trains on the synthetic rows. Must be zero or higher.",
    "augmentation_best_f1_gain": "Usefulness. Change in best F1 on real validation rows when a model also trains on the synthetic rows. Must be zero or higher.",
}


def _check_rows(q: dict) -> list[dict]:
    """One row per check across fill levels: result, plain meaning, and the closest or worst value."""
    by: dict[str, list[dict]] = {}
    for level in q["levels"]:
        for c in level["checks"]:
            key = "column_distance" if c["name"].startswith("column_distance:") else c["name"]
            by.setdefault(key, []).append(c)
    rows = []
    for name, cs in by.items():
        failed = [c for c in cs if not c["passed"]]
        gap = lambda c: abs(c["value"] - c["threshold"])
        pick = max(failed, key=gap) if failed else min(cs, key=gap)
        outcome = "FAIL" if len(failed) == len(cs) else "MIXED" if failed else "PASS"
        cols = sorted({c["name"].split(":", 1)[1] for c in failed}) if name == "column_distance" else []
        detail = (f"{'Worst' if failed else 'Closest'} {pick['value']:.3f} against limit {pick['threshold']:.3f}; "
                  f"{'failed' if failed else 'passed'} {len(failed) or len(cs)} of {len(cs)}"
                  f"{' (' + ', '.join(cols) + ')' if cols else ''}. {'Hard check.' if pick['hard'] else 'Soft check.'}")
        rows.append({"name": name, "outcome": outcome,
                     "meaning": CHECKS.get(name.split(":")[0], ""), "detail": detail})
    return rows


def _evaluation(run: Path) -> dict | None:
    """Arm-vs-original test lifts as a table, plain findings, and the written report by section."""
    ev, comp = run / "report/evidence.json", run / "report/comparisons.csv"
    if not (ev.exists() and comp.exists()):
        return None
    verdicts = [v for v in json.loads(ev.read_text(encoding="utf-8"))["verdicts"] if v["reference"] == "original"]
    with comp.open(encoding="utf-8") as f:
        means = {(c["arm"], float(c["fill_level"]), c["metric"]): float(c["mean"])
                 for c in csv.DictReader(f) if c["split"] == "test" and c["reference"] == "original"}
    keys = sorted({(v["arm"], v["fill_level"]) for v in verdicts}, key=lambda k: (k[0] != "cleaned", k[1]))
    by = {(v["arm"], v["fill_level"], v["metric"]): v for v in verdicts}
    # The classifier the training agent picked (on validation PR-AUC) for each seed behind a mean.
    picked: dict[tuple[str, float], Counter] = {}
    metrics = run / "report/metrics.csv"
    if metrics.exists():
        with metrics.open(encoding="utf-8") as f:
            for c in csv.DictReader(f):
                if c["split"] == "test" and c["selected"] == "True":
                    picked.setdefault((c["arm"], float(c["fill_level"])), Counter())[c["model"]] += 1

    def classifier(arm: str, fill: float) -> str:
        return ", ".join(f"{m} ×{n}" if len(picked[(arm, fill)]) > 1 else m
                         for m, n in picked.get((arm, fill), Counter()).most_common()) or "unknown"

    rows = [{"arm": a, "fill": fl, "model": classifier(a, fl), "cells": [
        {"mean": means.get((a, fl, m)), "lift": by[(a, fl, m)]["lift"]["test"], "verdict": by[(a, fl, m)]["verdict"]}
        for m in METRICS]} for a, fl in keys]
    reference = [means.get(("original", 0.0, m)) for m in METRICS]

    findings = []
    for arm in ("cleaned", "smotenc"):
        for m in METRICS:
            vs = sorted((v for v in verdicts if v["arm"] == arm and v["metric"] == m), key=lambda v: v["fill_level"])
            if not vs:
                continue
            kinds = {v["verdict"] for v in vs}
            lifts = [v["lift"]["test"] for v in vs]
            span = f"{lifts[0]:+.3f}" if len(vs) == 1 else f"{min(lifts):+.3f} to {max(lifts):+.3f}"
            if len(kinds) == 1:
                findings.append({"arm": arm, "metric": m, "verdict": kinds.pop(), "text": span, "n": len(vs)})
            else:
                findings.append({"arm": arm, "metric": m, "verdict": "mixed", "text": span, "n": len(vs)})

    best = max(keys, key=lambda k: means.get((k[0], k[1], "f1"), 0))
    sections = []
    report = run / "report/final_report.md"
    if report.exists():
        for block in report.read_text(encoding="utf-8").split("\n## ")[1:]:
            title, _, body = block.partition("\n")
            paras = [r3(p.strip()) for p in body.split("\n\n") if p.strip() and not p.lstrip().startswith("|")]
            if paras:
                sections.append({"title": title.strip(), "paragraphs": paras})
    return {"metrics": list(METRICS), "reference": reference, "rows": rows, "findings": findings,
            "reference_model": classifier("original", 0.0),
            "best": {"arm": best[0], "fill": best[1], "model": classifier(*best), "f1": means.get((best[0], best[1], "f1"))},
            "sections": sections}


def _summaries(run: Path, decisions: list[dict]) -> dict[str, dict]:
    """A headline and bullet facts per stage, read from what the stage wrote. Missing files skip."""
    out: dict[str, dict] = {}

    def put(stage, headline, facts=()):
        out[stage] = {"headline": headline, "facts": list(facts)}

    def json_of(rel):
        f = run / rel
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8")) if (run / "config.yaml").exists() else {}
    r = cfg.get("split_ratios")
    if r and (run / "data/train.csv").exists():
        pct = "/".join(f"{round(r[k] * 100)}" for k in ("train", "val", "test"))
        n = {k: _rows(run / f"data/{k}.csv") for k in ("train", "val", "test")}
        put("split", f"Chose a {pct} train/val/test split (seed {cfg.get('split_seed')})",
            [f"{k}: {v:,} rows" for k, v in n.items()] + [f"target: {cfg.get('target')} (positive = {cfg.get('positive_class')})"])

    plan, prof = json_of("profile/treatment_plan.json"), json_of("profile/profile.json")
    if plan and prof:
        t = plan["treatments"]
        put("preprocessing", f"Profiled {prof['rows']:,} training rows and chose treatments for {len(t)} columns",
            [f"{c}: {a}" for c, a in t.items()]
            + [f"{c}: {i['missing']} missing" for c, i in prof["columns"].items() if i["missing"]])

    sel = {d["rule"]: d for d in decisions if d["agent"] == "model_selection"}
    attempts = sum(d["rule"] == "type_routing" for d in decisions if d["agent"] == "model_selection")
    if "type_routing" in sel:
        put("model_selection", sel["type_routing"]["rationale"] + (f" ({attempts} attempts)" if attempts > 1 else ""),
            [sel["hyperparameters"]["rationale"]] if "hyperparameters" in sel else [])
        out["model_selection"]["gates"] = [
            {"name": g, "outcome": sel[g]["outcome"], "meaning": text, "detail": sel[g]["rationale"]}
            for g, text in GATES.items() if g in sel]

    reports = sorted((run / "quality").glob("quality_report_attempt_*.json"))
    if reports:
        q = json.loads(reports[-1].read_text(encoding="utf-8"))
        ok = [str(l["fill_level"]) for l in q["levels"] if l["passed"]]
        dg = q.get("diagnosis") or {}
        put("quality_check", f"Verdict {q['verdict']}: {len(ok)} of {len(q['levels'])} fill levels passed (attempt {len(reports)})",
            ([f"accepted levels: {', '.join(ok)}"] if ok else [])
            + ([f"failed checks: {', '.join(dg['failed_checks'])}"] if dg.get("failed_checks") else []))
        out["quality_check"]["gates"] = _check_rows(q)

    tr = [d for d in decisions if d["agent"] == "training" and isinstance(d["outcome"], dict)]
    summary = next((d["outcome"]["summary"] for d in reversed(tr) if "summary" in d["outcome"]), None)
    plan_d = next((d["outcome"] for d in tr if "smote_levels" in d["outcome"]), None)
    if summary:
        put("training", summary, [f"SMOTENC levels: {plan_d['smote_levels']}, CTGAN levels: {plan_d['gan_levels'] or 'none'}, seeds: {plan_d['seeds']}"] if plan_d else [])

    evaluation = _evaluation(run)
    if evaluation:
        b = evaluation["best"]
        put("evaluation", f"Best test F1: {b['arm']} at fill {b['fill']:.3f} ({b['f1']:.3f}) with {b['model']}, vs original {evaluation['reference'][0]:.3f} with {evaluation['reference_model']}")
    # One- or two-sentence agent explanation (model selection gates, quality checks), when logged.
    for stage in ("model_selection", "quality_check"):
        text = next((d["outcome"].get("explanation") or d["outcome"].get("likely_cause")
                     for d in reversed(decisions)
                     if d["agent"] == stage and d["rule"] == "llm_step" and isinstance(d["outcome"], dict)), None)
        if text and stage in out:
            out[stage]["explain"] = text
    # Every number shown anywhere is rounded to three decimals.
    for v in out.values():
        if "explain" in v:
            v["explain"] = r3(v["explain"])
        v["headline"] = r3(v["headline"])
        v["facts"] = [r3(f) for f in v["facts"]]
        for g in v.get("gates", []):
            g["detail"] = r3(g["detail"])
    return out


def run_view(run_id: str) -> dict | None:
    run = RUNS / run_id
    # Reject path tricks: the id must be a direct child of runs/.
    if run.parent != RUNS or not (run / "state.json").exists():
        return None
    state = json.loads((run / "state.json").read_text(encoding="utf-8"))
    decisions = _jsonl(run / "decisions.jsonl")
    models = {}
    for d in decisions:
        model = (d.get("inputs") or {}).get("model")
        if d["rule"] == "llm_step" and model:
            models[d["agent"]] = model
    # Drop prompts/context: large, and the UI only needs the outcome and rationale.
    slim = [{k: d.get(k) for k in ("agent", "rule", "outcome", "rationale")} for d in decisions]
    return {"run_id": run_id, "state": state, "stages": [str(s) for s in STAGE_ORDER],
            "agent_of": AGENT_OF, "models": models, "decisions": slim,
            "summaries": _summaries(run, decisions),
            "evaluation": _evaluation(run)}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, code: int = 200) -> None:
        self._send(code, json.dumps(payload, default=str).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/api/datasets":
            return self._json(datasets())
        if path == "/api/runs":
            return self._json(list_runs())
        if path.startswith("/api/runs/"):
            view = run_view(path.removeprefix("/api/runs/"))
            return self._json(view, 200) if view else self._json({"error": "no such run"}, 404)
        # Static files; unknown paths fall back to index.html (single page app).
        target = (DIST / path.lstrip("/")).resolve()
        if not target.is_file() or DIST.resolve() not in target.parents:
            target = DIST / "index.html"
        if not target.is_file():
            return self._send(503, b"Frontend not built: npm --prefix frontend run build",
                              "text/plain")
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._send(200, target.read_bytes(), ctype)

    def do_POST(self) -> None:  # noqa: N802
        # A JSON content type forces a CORS preflight, so other web pages cannot start runs.
        if self.path != "/api/start" or self.headers.get("Content-Type") != "application/json":
            return self._json({"error": "not found"}, 404)
        try:
            name = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))["dataset"]
        except (ValueError, KeyError, TypeError):
            return self._json({"error": "send {\"dataset\": name}"}, 400)
        code = start_run(str(name))
        self._json({"ok": code == 202}, code)

    def log_message(self, *args) -> None:
        pass


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
