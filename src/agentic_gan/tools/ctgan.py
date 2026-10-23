"""Dataset-agnostic CTGAN tool: train on a cleaned table, sample minority-class rows.

The Model Selection Agent calls ``train_ctgan`` and ``sample_synthetic`` as its generate
tool. The input is the cleaned (not encoded or scaled) training split. CTGAN cannot handle
missing values in numeric columns, so callers fill them first (or pass ``fill_missing``).
"""

from __future__ import annotations

import logging
import pickle
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from ctgan import CTGAN

from agentic_gan.utils.synthetic import IS_SYNTHETIC, fill_counts

__all__ = [
    "IS_SYNTHETIC", "TrainedCTGAN", "fill_counts", "load_model", "sample_synthetic",
    "save_model", "train_ctgan",
]

logger = logging.getLogger(__name__)

@dataclass
class TrainedCTGAN:
    """A fitted CTGAN plus what is needed to sample from it and to audit the run."""

    model: CTGAN
    target_col: str
    discrete_columns: list[str]
    columns: list[str]
    dtypes: dict[str, str]
    params: dict[str, Any]
    loss_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    train_seconds: float = 0.0
    n_train_rows: int = 0
    numeric_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    integer_valued: list[str] = field(default_factory=list)


def _infer_discrete(df: pd.DataFrame, target_col: str) -> list[str]:
    # Explicit inference: CTGAN does not detect pandas 3 "str" columns as discrete.
    cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if target_col not in cols:
        cols.append(target_col)
    return cols


def _check_no_missing(df: pd.DataFrame) -> None:
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        raise ValueError(
            "CTGAN cannot handle missing values (its transformer adds a null-indicator "
            "column that breaks inverse_transform). Fill these columns first or pass "
            f"fill_missing: {missing.to_dict()}"
        )


def train_ctgan(
    train_df: pd.DataFrame,
    target_col: str,
    discrete_columns: Sequence[str] | None = None,
    epochs: int = 300,
    batch_size: int = 500,
    pac: int = 10,
    discriminator_steps: int = 1,
    generator_dim: Sequence[int] = (256, 256),
    discriminator_dim: Sequence[int] = (256, 256),
    device: str = "cpu",
    seed: int = 42,
    fill_missing: Mapping[str, Any] | None = None,
    log_path: str | Path | None = None,
) -> TrainedCTGAN:
    """Fit CTGAN on the cleaned training split (features plus target, no NaN).

    ``discrete_columns=None`` treats every non-numeric column and ``target_col`` as
    discrete. ``fill_missing`` maps column -> fill value and is applied before the NaN
    check; without it any NaN raises. ``device`` is "cpu", "cuda" or "auto". Per-epoch
    generator and discriminator losses are returned and, if ``log_path`` is given,
    written to CSV.
    """
    if target_col not in train_df.columns:
        raise ValueError(f"target_col {target_col!r} not in train_df")
    if device not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be 'cpu', 'cuda' or 'auto'")
    if batch_size % pac != 0:
        raise ValueError("batch_size must be divisible by pac")

    df = train_df.copy()
    if IS_SYNTHETIC in df.columns:
        raise ValueError(f"train_df must not already contain {IS_SYNTHETIC!r}")
    if fill_missing:
        df = df.fillna(dict(fill_missing))
    _check_no_missing(df)

    if discrete_columns is None:
        discrete = _infer_discrete(df, target_col)
    else:
        discrete = list(discrete_columns)
    if target_col not in discrete:
        discrete.append(target_col)

    params = {
        "epochs": epochs, "batch_size": batch_size, "pac": pac,
        "discriminator_steps": discriminator_steps,
        "generator_dim": tuple(generator_dim), "discriminator_dim": tuple(discriminator_dim),
        "device": device, "seed": seed,
    }
    model = CTGAN(
        epochs=epochs, batch_size=batch_size, pac=pac, discriminator_steps=discriminator_steps,
        generator_dim=tuple(generator_dim), discriminator_dim=tuple(discriminator_dim),
        enable_gpu=device != "cpu", verbose=False,
    )
    model.set_random_state(seed)

    logger.info("Training CTGAN on %d rows x %d columns (%s)", len(df), df.shape[1], device)
    start = time.perf_counter()
    model.fit(df, discrete_columns=discrete)
    seconds = time.perf_counter() - start

    losses = model.loss_values.copy()
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        losses.to_csv(log_path, index=False)
    logger.info("CTGAN trained in %.1fs; final losses G=%.3f D=%.3f", seconds,
                losses["Generator Loss"].iloc[-1], losses["Discriminator Loss"].iloc[-1])

    numeric = [c for c in df.columns if c not in discrete and pd.api.types.is_numeric_dtype(df[c])]
    return TrainedCTGAN(
        model=model, target_col=target_col, discrete_columns=discrete,
        columns=df.columns.tolist(), dtypes={c: str(t) for c, t in df.dtypes.items()},
        params=params, loss_history=losses, train_seconds=seconds, n_train_rows=len(df),
        numeric_ranges={c: (float(df[c].min()), float(df[c].max())) for c in numeric},
        integer_valued=[c for c in numeric if bool((df[c] == df[c].round()).all())],
    )


def _restore_dtypes(sample: pd.DataFrame, model: TrainedCTGAN) -> pd.DataFrame:
    """Clip numeric columns to the training range and round whole-number columns."""
    for col, (low, high) in model.numeric_ranges.items():
        sample[col] = sample[col].clip(low, high)
    for col in model.integer_valued:
        sample[col] = sample[col].round()
    for col, dtype in model.dtypes.items():
        if dtype.startswith(("int", "uint")):
            sample[col] = sample[col].round().astype(dtype)
    return sample


def sample_synthetic(
    model: TrainedCTGAN,
    n: int,
    target_value: Any = 1,
    seed: int | None = None,
    drop_duplicates_of: pd.DataFrame | None = None,
    row_filter: Callable[[pd.DataFrame], pd.Series] | None = None,
    max_rounds: int = 20,
    conditional: bool = True,
) -> pd.DataFrame:
    """Return exactly ``n`` synthetic rows of ``target_col == target_value`` with is_synthetic=1.

    Sampling is conditional on the target unless ``conditional`` is false. Either way, rows
    with the wrong target are dropped, so the returned table is still the requested class.
    Numeric columns are clipped to their training range and whole-number columns are
    rounded. ``row_filter`` takes a batch and returns a boolean mask of rows to keep, for
    dataset rules such as "these two columns must agree" (rejected rows are resampled).
    If ``drop_duplicates_of``
    is given (e.g. the real training split, same columns), rows that exactly copy a real
    row or an earlier synthetic row are dropped and resampled, up to ``max_rounds``.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if seed is not None:
        model.model.set_random_state(seed)

    seen: set[tuple] = set()
    if drop_duplicates_of is not None:
        seen.update(map(tuple, drop_duplicates_of[model.columns].itertuples(index=False)))

    kept: list[pd.DataFrame] = []
    total = drawn = on_target = usable = 0
    for _ in range(max_rounds):
        need = n - total
        # Size each draw from the share of rows that survive so far (1.0 until measured).
        match_rate = max(usable / drawn, 0.02) if drawn else 1.0
        request = min(int(need / match_rate * 1.2) + 10, 100_000)
        if conditional:
            batch = model.model.sample(
                request, condition_column=model.target_col, condition_value=target_value,
            )
        else:
            batch = model.model.sample(request)
        drawn += len(batch)
        batch = _restore_dtypes(batch[model.columns].reset_index(drop=True), model)
        batch = batch[batch[model.target_col] == target_value].reset_index(drop=True)
        on_target += len(batch)
        if row_filter is not None:
            batch = batch[row_filter(batch).to_numpy()].reset_index(drop=True)
        usable += len(batch)
        if drop_duplicates_of is not None:
            keep = []
            for i, row in enumerate(batch.itertuples(index=False)):
                key = tuple(row)
                if key not in seen:
                    seen.add(key)
                    keep.append(i)
            batch = batch.iloc[keep]
        kept.append(batch.head(need))
        total += len(kept[-1])
        if total >= n:
            break
    else:
        logger.warning("Only %d of %d valid synthetic rows after %d rounds", total, n, max_rounds)

    logger.info("Sampled %d rows to keep %d; target-match rate %.1f%%, passed row_filter %.1f%%",
                drawn, total, 100 * on_target / max(drawn, 1), 100 * usable / max(on_target, 1))
    out = pd.concat(kept, ignore_index=True)
    out[IS_SYNTHETIC] = 1
    return out


def save_model(model: TrainedCTGAN, path: str | Path) -> None:
    """Pickle the trained generator and its metadata (moved to CPU so it reloads anywhere)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.model.set_device("cpu")
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str | Path) -> TrainedCTGAN:
    """Reload a model written by ``save_model``. Only load files you created yourself."""
    with open(path, "rb") as f:
        return pickle.load(f)
