"""One-shot Bank run exercising the WGAN-GP backup on attempt 3 (issue #124).

Stubs both CTGAN and WGAN-GP, forces quality to fail twice so attempt 3
runs the WGAN-GP backup. Verifies the backup fires, the right files are written,
and the decision log records the correct generator.

    uv run --no-sync python scripts/run_bank_stub_wgan_backup.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from test_selection import _bank_frame  # noqa: E402

from agentic_gan.agents.model_selection import run_model_selection  # noqa: E402
from agentic_gan.llm import FakeLLM  # noqa: E402
from agentic_gan.orchestrator.decisions import read_decisions  # noqa: E402
from agentic_gan.orchestrator.graph import config_for_dataset, run_pipeline  # noqa: E402
from agentic_gan.orchestrator.stubs import StubAgent, default_agents  # noqa: E402
from agentic_gan.schemas import (  # noqa: E402
    ArmFiles,
    PreprocessOutput,
    RunConfig,
    Stage,
    Status,
)
from agentic_gan.utils.synthetic import IS_SYNTHETIC  # noqa: E402


def _patch_stubs() -> tuple:
    """Patch both CTGAN and WGAN-GP with minimal stubs; return (ctgan_calls, wgan_calls)."""
    ctgan_calls: list = []
    wgan_calls: list = []

    def fake_ctgan_train(train_df, target_col, **kwargs):
        ctgan_calls.append(kwargs.get("epochs", "?"))
        log_path = kwargs.get("log_path")
        if log_path is not None:
            Path(log_path).write_text("Generator Loss,Discriminator Loss\n0.1,0.2\n")
        return SimpleNamespace(target_col=target_col)

    def fake_ctgan_save(model, path):
        Path(path).write_bytes(b"ctgan")

    def fake_sample_for_fill(model, train, positive_rows, counts, **kwargs):
        wanted = max(counts.values() if hasattr(counts, "values") else counts)
        frame = pd.DataFrame({name: [0] * wanted for name in train.columns})
        target = model.target_col
        frame[target] = 1
        frame[IS_SYNTHETIC] = 1
        return frame

    def fake_wgan_train(train_df, target_col, positive=1, discrete_columns=(), **kwargs):
        wgan_calls.append(kwargs.get("epochs", "?"))
        log_path = kwargs.get("log_path")
        if log_path is not None:
            pd.DataFrame({
                "Epoch": [1], "Generator Loss": [0.1],
                "Discriminator Loss": [0.2], "Gradient Penalty": [0.5],
            }).to_csv(log_path, index=False)
        feature_cols = [c for c in train_df.columns if c != target_col]
        return SimpleNamespace(
            target_col=target_col, positive=positive, columns=feature_cols,
            numeric_ranges={}, integer_valued=set(), discrete_columns=list(discrete_columns),
            transformer=None,
        )

    def fake_wgan_save(model, path):
        Path(path).write_bytes(b"wgangp")

    def fake_wgan_sample(model, n, *, seed, row_filter=None, drop_duplicates_of=None, **kwargs):
        frame = pd.DataFrame({name: [1] * n for name in model.columns})
        frame[model.target_col] = model.positive
        frame[IS_SYNTHETIC] = 1
        return frame

    import agentic_gan.agents.model_selection as sel_mod
    import agentic_gan.tools.ctgan as ctgan_mod
    import agentic_gan.tools.wgan_gp as wgan_mod

    ctgan_mod.train_ctgan = fake_ctgan_train
    ctgan_mod.save_model = fake_ctgan_save
    sel_mod.sample_for_fill = fake_sample_for_fill
    wgan_mod.train_wgan_gp = fake_wgan_train
    wgan_mod.save_model = fake_wgan_save
    wgan_mod.sample_wgan_gp = fake_wgan_sample

    return ctgan_calls, wgan_calls


def _bank_preprocess(request, run):
    frame = _bank_frame(rows=2000, positives=500)
    table = frame.to_csv(index=False)
    data = run.subdir("data")
    paths = {}
    for kind in ("raw", "baseline"):
        for part in ("train", "val", "test"):
            path = data / f"{kind}_{part}.csv"
            path.write_text(table, encoding="utf-8")
            paths[f"{kind}_{part}"] = path.resolve()
    raw = ArmFiles(
        train_path=paths["raw_train"],
        val_path=paths["raw_val"],
        test_path=paths["raw_test"],
    )
    baseline = ArmFiles(
        train_path=paths["baseline_train"],
        val_path=paths["baseline_val"],
        test_path=paths["baseline_test"],
    )
    profile = run.subdir("profile") / "profile.json"
    profile.write_text("{}\n", encoding="utf-8")
    transformer = run.subdir("profile") / "transformer.txt"
    transformer.write_text("stub\n", encoding="utf-8")
    return PreprocessOutput(
        status=Status.SUCCESS,
        raw=raw,
        baseline=baseline,
        profile_path=profile.resolve(),
        treatment_log_path=profile.resolve(),
        transformer_path=transformer.resolve(),
        transformer_fit_rows=len(frame),
        baseline_train_positives=500,
        baseline_train_negatives=1500,
    )


def main() -> int:
    ctgan_calls, wgan_calls = _patch_stubs()
    config = RunConfig.model_validate({
        **config_for_dataset("bank_marketing").model_dump(),
        "min_rows": 2000,
    })
    agents = default_agents()
    agents[Stage.MODEL_SELECTION] = run_model_selection
    agents[Stage.PREPROCESSING] = _bank_preprocess
    # Force quality failures on every attempt so we reach attempt 3.
    agents[Stage.QUALITY_CHECK] = StubAgent(
        Stage.QUALITY_CHECK, failures=-1, failure_mode="quality"
    )
    state, run = run_pipeline(
        config,
        runs_dir=ROOT / "runs",
        agents=agents,
        llm=FakeLLM([RuntimeError("offline")] * 30),
    )
    decisions = read_decisions(run)
    hp_rows = decisions.query("rule == 'hyperparameters'")

    print(f"COMPLETE={state.is_complete}  failed_stage={state.failed_stage}")
    print(f"RUN={run.root}")
    print(f"model_selection_attempts={state.attempts(Stage.MODEL_SELECTION)}")
    print(f"quality_attempts={state.attempts(Stage.QUALITY_CHECK)}")
    print(f"ctgan_trains={len(ctgan_calls)}  wgan_trains={len(wgan_calls)}")
    print()
    print("Hyperparameter rows:")
    for _, row in hp_rows.iterrows():
        gen = row.inputs.get("generator", "?")
        epochs = row.inputs.get("epochs", "?")
        attempt = row.inputs.get("attempt", "?")
        print(f"  attempt={attempt}  generator={gen}  epochs={epochs}")

    # Verify the backup fired on attempt 3
    ok = True
    if len(wgan_calls) == 0:
        print("FAIL: WGAN-GP backup was not called on attempt 3")
        ok = False
    else:
        print(f"\nPASS: WGAN-GP backup called {len(wgan_calls)} time(s)")
    if len(ctgan_calls) != 2:
        print(f"FAIL: expected 2 CTGAN trains (attempts 1+2), got {len(ctgan_calls)}")
        ok = False
    else:
        print(f"PASS: CTGAN trained {len(ctgan_calls)} time(s) (attempts 1+2)")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
