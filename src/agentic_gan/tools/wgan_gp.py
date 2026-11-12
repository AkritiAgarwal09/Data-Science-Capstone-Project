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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
# Training loop  (Step 2)
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

    Implemented in Step 2.
    """
    raise NotImplementedError("train_wgan_gp is implemented in Step 2")


# ---------------------------------------------------------------------------
# Sampler  (Step 3)
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

    Implemented in Step 3.
    """
    raise NotImplementedError("sample_wgan_gp is implemented in Step 3")
