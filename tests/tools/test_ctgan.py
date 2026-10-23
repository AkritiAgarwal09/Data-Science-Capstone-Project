import numpy as np
import pandas as pd
import pytest

from agentic_gan.tools.ctgan import (
    fill_counts,
    load_model,
    sample_synthetic,
    save_model,
    train_ctgan,
)


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(0)
    n = 300
    y = (rng.random(n) < 0.2).astype(int)
    return pd.DataFrame({
        "age": rng.integers(18, 80, n),
        "income": rng.normal(50, 10, n) + 20 * y,
        "job": rng.choice(["a", "b", "c"], n),
        "y": y,
    })


@pytest.fixture(scope="module")
def trained(df, tmp_path_factory):
    log = tmp_path_factory.mktemp("logs") / "loss.csv"
    model = train_ctgan(df, "y", epochs=3, batch_size=50, pac=5, log_path=log)
    assert log.exists()
    return model


def test_train_logs_losses(trained):
    assert len(trained.loss_history) == 3
    assert trained.train_seconds > 0
    assert set(trained.discrete_columns) == {"job", "y"}


def test_sample_minority_flagged(trained):
    out = sample_synthetic(trained, 40, target_value=1, seed=1)
    assert len(out) == 40
    assert (out["y"] == 1).all() and (out["is_synthetic"] == 1).all()
    assert out["age"].dtype.kind == "i"


def test_unconditional_sample_still_keeps_target(trained):
    out = sample_synthetic(trained, 20, target_value=1, seed=1, conditional=False)
    assert len(out) == 20
    assert (out["y"] == 1).all()


def test_dedupe_against_real(trained, df):
    out = sample_synthetic(trained, 30, drop_duplicates_of=df)
    assert len(out) == 30
    merged = pd.concat([df, out[df.columns]]).duplicated().sum()
    assert merged <= df.duplicated().sum()


def test_save_load_roundtrip(trained, tmp_path):
    save_model(trained, tmp_path / "m.pkl")
    reloaded = load_model(tmp_path / "m.pkl")
    assert len(sample_synthetic(reloaded, 10)) == 10


def test_nan_raises_and_fill_works(df):
    bad = df.copy()
    bad.loc[:5, "income"] = np.nan
    with pytest.raises(ValueError, match="income"):
        train_ctgan(bad, "y", epochs=1, batch_size=50, pac=5)
    train_ctgan(bad, "y", epochs=1, batch_size=50, pac=5, fill_missing={"income": -1})


def test_fill_counts():
    y = pd.Series([0] * 23953 + [1] * 3173)
    assert fill_counts(y) == {0.10: 2078, 0.25: 5195, 0.50: 10390}


def test_sample_clipped_rounded_and_filtered(trained, df):
    lo, hi = df["income"].min(), df["income"].max()
    out = sample_synthetic(trained, 30, row_filter=lambda b: b["age"] >= 40)
    assert len(out) == 30
    assert (out["age"] >= 40).all()
    assert out["income"].between(lo, hi).all()
    assert "age" in trained.integer_valued and "income" not in trained.integer_valued
