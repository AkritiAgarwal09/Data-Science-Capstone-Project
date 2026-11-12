"""Dataset-agnostic WGAN-GP tool: train on positive-class rows, sample minority-class rows.

The Model Selection Agent calls ``train_wgan_gp`` and ``sample_wgan_gp`` as the
attempt-3 backup generator when CTGAN fails two quality-gate attempts on a table
whose selection-rule fallback is wgan_gp.  Training data is the positive class
only; the target column is never modelled and is added back as a constant after
sampling.
"""

from __future__ import annotations

import logging
import pickle
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from agentic_gan.utils.synthetic import IS_SYNTHETIC

__all__ = [
    "IS_SYNTHETIC",
    "DEFAULTS",
    "TrainedWGANGP",
    "Critic",
    "Generator",
    "apply_activate",
    "gradient_penalty",
    "load_model",
    "prepare_positive_rows",
    "sample_wgan_gp",
    "save_model",
    "train_wgan_gp",
]

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "epochs": 300,
    "batch_size": 256,
    "embedding_dim": 128,
    "generator_dim": (256, 256),
    "critic_dim": (256, 256),
    "n_critic": 5,
    "gp_lambda": 10.0,
    "lr": 1e-4,
    "betas": (0.5, 0.9),
}


@dataclass
class TrainedWGANGP:
    """A fitted WGAN-GP plus everything needed to sample from it and to audit the run."""

    generator: nn.Module
    critic: nn.Module
    transformer: Any  # ctgan.data_transformer.DataTransformer
    target_col: str
    positive: Any
    columns: list[str]          # full original column order INCLUDING the target
    feature_columns: list[str]  # original columns without the target
    discrete_columns: list[str] # discrete feature names (no target)
    dtypes: dict[str, str]
    params: dict[str, Any]
    loss_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    train_seconds: float = 0.0
    n_train_rows: int = 0
    numeric_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    integer_valued: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_positive_rows(
    train_df: pd.DataFrame,
    target_col: str,
    positive: Any,
    discrete_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Return (features_df, discrete_features) containing only positive-class rows.

    ``features_df`` has the target column removed and its index reset.
    ``discrete_features`` is ``discrete_columns`` minus the target, extended with
    any column whose dtype is not numeric (the same inference ctgan uses for pandas 3
    str columns).  Raises ``ValueError`` for any input that would corrupt training.
    """
    if target_col not in train_df.columns:
        raise ValueError(f"target_col {target_col!r} not in train_df")
    if IS_SYNTHETIC in train_df.columns:
        raise ValueError(
            f"train_df must not already contain {IS_SYNTHETIC!r}"
        )
    missing = train_df.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        raise ValueError(
            "WGAN-GP cannot handle missing values (the transformer adds a "
            "null-indicator column that breaks inverse_transform). "
            f"Fill these columns first: {missing.to_dict()}"
        )
    pos_rows = train_df[train_df[target_col] == positive]
    if len(pos_rows) == 0:
        raise ValueError(
            f"No rows with {target_col!r} == {positive!r} in train_df"
        )
    if len(pos_rows) < 2:
        raise ValueError(
            f"At least 2 positive rows are required, found {len(pos_rows)}"
        )
    features = pos_rows.drop(columns=[target_col]).reset_index(drop=True)
    # Preserve order: provided discrete (minus target), then non-numeric additions.
    seen: set[str] = set()
    discrete_features: list[str] = []
    for c in discrete_columns:
        if c != target_col and c in features.columns and c not in seen:
            discrete_features.append(c)
            seen.add(c)
    for c in features.columns:
        if not pd.api.types.is_numeric_dtype(features[c]) and c not in seen:
            discrete_features.append(c)
            seen.add(c)
    return features, discrete_features


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------

def apply_activate(data: torch.Tensor, output_info_list: list) -> torch.Tensor:
    """Apply tanh to numeric (alpha) spans and Gumbel-softmax (tau=0.2) to one-hot spans.

    Mirrors CTGAN's ``_apply_activate``.  ``output_info_list`` is the list of
    lists of SpanInfo objects from a fitted ``DataTransformer``.  Each SpanInfo
    has ``dim`` (int) and ``activation_fn`` (str, 'tanh' or 'softmax').
    Returns a tensor of the same shape as ``data``.
    """
    parts: list[torch.Tensor] = []
    st = 0
    for column_info in output_info_list:
        for span_info in column_info:
            ed = st + span_info.dim
            if span_info.activation_fn == "tanh":
                parts.append(torch.tanh(data[:, st:ed]))
            elif span_info.activation_fn == "softmax":
                parts.append(
                    F.gumbel_softmax(data[:, st:ed], tau=0.2, hard=False)
                )
            else:
                raise ValueError(
                    f"Unexpected activation function {span_info.activation_fn!r}"
                )
            st = ed
    return torch.cat(parts, dim=1)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """Plain MLP generator: embedding_dim -> hidden layers (ReLU) -> data_dim.

    Uses no residual connections or BatchNorm (unlike ctgan's Generator), to
    match the WGAN-GP spec of a simple feed-forward network.
    """

    def __init__(
        self,
        embedding_dim: int,
        generator_dim: tuple[int, ...],
        data_dim: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = embedding_dim
        for h_dim in generator_dim:
            layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, data_dim))
        self.seq = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class Critic(nn.Module):
    """MLP critic: data_dim -> hidden layers -> scalar.  No sigmoid.

    Uses LeakyReLU(0.2) and Dropout(0.25) in each hidden block.  The final
    Linear outputs a raw real-valued score per row (Wasserstein loss requires
    no output activation).
    """

    def __init__(
        self,
        data_dim: int,
        critic_dim: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = data_dim
        for h_dim in critic_dim:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.25),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.seq = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


# ---------------------------------------------------------------------------
# Gradient penalty
# ---------------------------------------------------------------------------

def gradient_penalty(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    gp_lambda: float,
    device: str,
) -> torch.Tensor:
    """WGAN-GP gradient penalty: gp_lambda * E[(||∇_x̂ D(x̂)||_2 - 1)^2].

    ``real`` and ``fake`` must already be on ``device`` and have the same shape.
    ``create_graph=True`` keeps the penalty in the computational graph so the
    critic optimiser can differentiate through it.
    """
    batch = real.size(0)
    alpha = torch.rand(batch, 1, device=device)
    interpolates = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    crit_out = critic(interpolates)
    grads = torch.autograd.grad(
        outputs=crit_out.sum(),
        inputs=interpolates,
        create_graph=True,
    )[0]
    return gp_lambda * ((grads.norm(2, dim=1) - 1.0) ** 2).mean()


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

def _validate_args(
    epochs: int,
    batch_size: int,
    embedding_dim: int,
    generator_dim: tuple,
    critic_dim: tuple,
    n_critic: int,
    gp_lambda: float,
    lr: float,
    device: str,
) -> None:
    for name, val in [("epochs", epochs), ("embedding_dim", embedding_dim),
                      ("n_critic", n_critic)]:
        if not isinstance(val, int) or isinstance(val, bool) or val < 1:
            raise ValueError(f"{name} must be a positive integer, got {val!r}")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 2:
        raise ValueError(f"batch_size must be an integer >= 2, got {batch_size!r}")
    if not generator_dim:
        raise ValueError("generator_dim must be a non-empty tuple")
    if not critic_dim:
        raise ValueError("critic_dim must be a non-empty tuple")
    if gp_lambda < 0:
        raise ValueError(f"gp_lambda must be >= 0, got {gp_lambda!r}")
    if lr <= 0:
        raise ValueError(f"lr must be > 0, got {lr!r}")
    if device not in {"cpu", "cuda", "auto"}:
        raise ValueError(f"device must be 'cpu', 'cuda' or 'auto', got {device!r}")


# ---------------------------------------------------------------------------
# Save / load  (implemented here; training in train_wgan_gp fills the model)
# ---------------------------------------------------------------------------

def save_model(model: TrainedWGANGP, path: str | Path) -> None:
    """Pickle the trained generator and its metadata (moved to CPU for portability).

    Only load files you created yourself.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.generator.cpu()
    model.critic.cpu()
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str | Path) -> TrainedWGANGP:
    """Reload a model written by ``save_model``.  Only load files you created."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_wgan_gp(
    train_df: pd.DataFrame,
    target_col: str,
    positive: Any,
    discrete_columns: list[str],
    epochs: int = 300,
    batch_size: int = 256,
    embedding_dim: int = 128,
    generator_dim: tuple[int, ...] = (256, 256),
    critic_dim: tuple[int, ...] = (256, 256),
    n_critic: int = 5,
    gp_lambda: float = 10.0,
    lr: float = 1e-4,
    device: str = "cpu",
    seed: int = 42,
    log_path: str | Path | None = None,
) -> TrainedWGANGP:
    """Fit WGAN-GP on positive-class rows of the cleaned training split.

    Per-epoch losses are epoch means (averaged over all iterations in the epoch).
    The loss history DataFrame has columns: Epoch, Generator Loss,
    Discriminator Loss, Gradient Penalty.
    """
    _validate_args(epochs, batch_size, embedding_dim, generator_dim, critic_dim,
                   n_critic, gp_lambda, lr, device)

    _device = (
        "cuda" if (device == "cuda" or (device == "auto" and torch.cuda.is_available()))
        else "cpu"
    )

    features, discrete_features = prepare_positive_rows(
        train_df, target_col, positive, discrete_columns
    )

    # Seed before transformer fit: DataTransformer.fit uses numpy for GMM fitting,
    # so numpy must be seeded to get deterministic transformer parameters.
    torch.manual_seed(seed)
    np.random.seed(seed)

    from ctgan.data_transformer import DataTransformer  # lazy: ctgan imports are heavy
    transformer = DataTransformer()
    transformer.fit(features, discrete_columns=discrete_features)

    real_data = transformer.transform(features)
    real_tensor = torch.tensor(real_data, dtype=torch.float32, device=_device)
    n_rows, data_dim = real_tensor.shape

    # Record ranges and integer-valued flags from positive rows only
    numeric_cols = [
        c for c in features.columns
        if pd.api.types.is_numeric_dtype(features[c]) and c not in discrete_features
    ]
    numeric_ranges = {c: (float(features[c].min()), float(features[c].max())) for c in numeric_cols}
    integer_valued = [
        c for c in numeric_cols
        if bool((features[c] == features[c].round()).all())
    ]

    # Networks (generator first so n_critic test can distinguish by creation order)
    generator = Generator(embedding_dim, generator_dim, data_dim).to(_device)
    critic_net = Critic(data_dim, critic_dim).to(_device)

    gen_opt = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    crit_opt = torch.optim.Adam(critic_net.parameters(), lr=lr, betas=(0.5, 0.9))

    # Dedicated PRNG for noise and batch indices; does not consume the global state
    # (global state is used by gradient_penalty's torch.rand and gumbel_softmax).
    rng = torch.Generator(device=_device)
    rng.manual_seed(seed)

    steps_per_epoch = max(n_rows // batch_size, 1)
    loss_rows: list[dict] = []
    start = time.perf_counter()

    for epoch in range(epochs):
        generator.train()
        critic_net.train()

        ep_gen: list[float] = []
        ep_crit: list[float] = []
        ep_gp: list[float] = []

        for _ in range(steps_per_epoch):
            # --- n_critic critic updates ---
            for _ in range(n_critic):
                # Real batch: sample with replacement so size is always batch_size
                idx = torch.randint(0, n_rows, (batch_size,), generator=rng, device=_device)
                real_batch = real_tensor[idx]

                # Fake batch: detach so critic backward doesn't flow into generator
                noise = torch.randn(batch_size, embedding_dim, generator=rng, device=_device)
                with torch.no_grad():
                    fake_batch = apply_activate(generator(noise), transformer.output_info_list)

                gp = gradient_penalty(critic_net, real_batch, fake_batch, gp_lambda, _device)
                crit_loss = critic_net(fake_batch).mean() - critic_net(real_batch).mean() + gp

                crit_opt.zero_grad()
                crit_loss.backward()
                crit_opt.step()

                ep_crit.append(crit_loss.item())
                ep_gp.append(gp.item())

            # --- ONE generator update (critic params must not change) ---
            noise = torch.randn(batch_size, embedding_dim, generator=rng, device=_device)
            fake_activated = apply_activate(generator(noise), transformer.output_info_list)
            gen_loss = -critic_net(fake_activated).mean()

            gen_opt.zero_grad()
            gen_loss.backward()
            gen_opt.step()

            ep_gen.append(gen_loss.item())

        mean_gen = float(np.mean(ep_gen))
        mean_crit = float(np.mean(ep_crit))
        mean_gp = float(np.mean(ep_gp))

        if not all(np.isfinite([mean_gen, mean_crit, mean_gp])):
            raise RuntimeError(f"WGAN-GP diverged at epoch {epoch + 1}")

        loss_rows.append({
            "Epoch": epoch + 1,
            "Generator Loss": mean_gen,
            "Discriminator Loss": mean_crit,
            "Gradient Penalty": mean_gp,
        })

    train_seconds = time.perf_counter() - start
    loss_history = pd.DataFrame(loss_rows)

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        loss_history.to_csv(log_path, index=False)

    generator.eval()
    critic_net.eval()

    logger.info(
        "WGAN-GP trained on %d rows, %d features, device=%s in %.1fs; "
        "final G=%.3f D=%.3f",
        n_rows, len(features.columns), _device, train_seconds,
        loss_history["Generator Loss"].iloc[-1],
        loss_history["Discriminator Loss"].iloc[-1],
    )

    return TrainedWGANGP(
        generator=generator,
        critic=critic_net,
        transformer=transformer,
        target_col=target_col,
        positive=positive,
        columns=train_df.columns.tolist(),
        feature_columns=features.columns.tolist(),
        discrete_columns=discrete_features,
        dtypes={c: str(t) for c, t in train_df.dtypes.items()},
        params={
            "epochs": epochs, "batch_size": batch_size, "embedding_dim": embedding_dim,
            "generator_dim": generator_dim, "critic_dim": critic_dim,
            "n_critic": n_critic, "gp_lambda": gp_lambda, "lr": lr,
            "device": device, "seed": seed,
        },
        loss_history=loss_history,
        train_seconds=train_seconds,
        n_train_rows=n_rows,
        numeric_ranges=numeric_ranges,
        integer_valued=integer_valued,
    )


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

def sample_wgan_gp(
    model: TrainedWGANGP,
    n: int,
    *,
    seed: int,
    row_filter: Callable[[pd.DataFrame], pd.Series] | None = None,
    drop_duplicates_of: pd.DataFrame | None = None,
    max_rounds: int = 100,
) -> pd.DataFrame:
    """Sample ``n`` synthetic positive-class rows from a fitted WGAN-GP.

    A fresh ``torch.Generator`` seeded from ``seed`` is created so the output is
    reproducible and independent of earlier calls. Each round overshoots the need
    by 20 % plus 10 rows, capped at 100,000, guided by keep_rate (fraction of
    generated rows that survive filtering).  If max_rounds are exhausted with
    fewer than ``n`` rows, logs a warning and returns what is available; the
    caller decides how to handle the shortfall.

    The model is not mutated.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(seed, int):
        raise ValueError("seed must be an int")

    # Seed both the dedicated Generator and the global state.
    # apply_activate calls F.gumbel_softmax which uses the global random state,
    # so it must also be reset to make the output reproducible.
    rng = torch.Generator()
    rng.manual_seed(seed)
    torch.manual_seed(seed)

    embedding_dim = model.params["embedding_dim"]

    seen: set[tuple] = set()
    if drop_duplicates_of is not None:
        seen.update(map(tuple, drop_duplicates_of[model.columns].itertuples(index=False)))

    kept: list[pd.DataFrame] = []
    total = 0
    drawn = 0
    usable = 0
    keep_rate = 1.0

    for _ in range(max_rounds):
        need = n - total
        request = min(int(need / max(keep_rate, 0.02) * 1.2) + 10, 100_000)

        noise = torch.randn(request, embedding_dim, generator=rng)
        with torch.no_grad():
            raw = model.generator(noise)
            activated = apply_activate(raw, model.transformer.output_info_list)

        samples_np = activated.cpu().numpy()
        batch = model.transformer.inverse_transform(samples_np)
        if not isinstance(batch, pd.DataFrame):
            batch = pd.DataFrame(batch, columns=model.feature_columns)

        drawn += len(batch)

        # Restore dtypes: clip numeric to training range, round integer columns
        for col, (low, high) in model.numeric_ranges.items():
            if col in batch.columns:
                batch[col] = batch[col].clip(low, high)
        for col in model.integer_valued:
            if col in batch.columns:
                batch[col] = batch[col].round()
        for col, dtype in model.dtypes.items():
            if col in batch.columns and dtype.startswith(("int", "uint")):
                batch[col] = batch[col].round().astype(dtype)

        # Restore target column and original column order
        batch[model.target_col] = model.positive
        batch = batch[model.columns].reset_index(drop=True)

        if row_filter is not None:
            mask = row_filter(batch)
            batch = batch[mask.to_numpy()].reset_index(drop=True)

        usable += len(batch)

        if drop_duplicates_of is not None:
            keep_rows: list[bool] = []
            for row in batch.itertuples(index=False):
                key = tuple(row)
                if key not in seen:
                    seen.add(key)
                    keep_rows.append(True)
                else:
                    keep_rows.append(False)
            batch = batch[keep_rows].reset_index(drop=True)

        kept.append(batch.head(need))
        total += len(kept[-1])
        keep_rate = usable / max(drawn, 1)

        if total >= n:
            break
    else:
        logger.warning(
            "Only %d of %d valid synthetic rows after %d rounds",
            total, n, max_rounds,
        )

    if not kept:
        return pd.DataFrame(columns=model.columns + [IS_SYNTHETIC])

    out = pd.concat(kept, ignore_index=True).head(n)
    out[IS_SYNTHETIC] = 1
    return out[model.columns + [IS_SYNTHETIC]].reset_index(drop=True)
