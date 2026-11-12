"""Tests for tools/wgan_gp.py building blocks.

Step 1: prepare_positive_rows, apply_activate, Critic, Generator, gradient_penalty.
Step 2 (training) and Step 3 (sampler) tests are appended below as they are built.
Run this file alone to avoid the known torch/OpenMP segfault:
    uv run --no-sync pytest tests/tools/test_wgan_gp.py -q
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn  # noqa: F401

# ---------------------------------------------------------------------------
# Module accessor (lazy import keeps torch out of the test-collection phase)
# ---------------------------------------------------------------------------

def _wgan():
    from agentic_gan.tools import wgan_gp
    return wgan_gp


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def toy_df():
    rng = np.random.default_rng(1)
    n = 120
    return pd.DataFrame({
        "num1": rng.normal(0, 1, n).astype(float),
        "num2": rng.normal(5, 2, n).astype(float),
        "cat1": rng.choice(["a", "b", "c"], n),
        "y": [1] * 72 + [0] * 48,
    })


@pytest.fixture(scope="module")
def transformer(toy_df):
    """DataTransformer fitted on positive-row features (no target)."""
    from ctgan.data_transformer import DataTransformer
    torch.manual_seed(42)
    np.random.seed(42)
    dt = DataTransformer()
    features = toy_df[toy_df["y"] == 1].drop(columns=["y"]).reset_index(drop=True)
    dt.fit(features, discrete_columns=["cat1"])
    return dt


# ---------------------------------------------------------------------------
# SpanInfo stub for apply_activate tests (avoids a real DataTransformer fit)
# ---------------------------------------------------------------------------

_SpanInfo = namedtuple("_SpanInfo", ["dim", "activation_fn"])


def _info(specs):
    """specs: list[list[tuple[dim, activation_fn]]]"""
    return [[_SpanInfo(d, a) for d, a in col] for col in specs]


# ===========================================================================
# Step 1 tests: building blocks
# ===========================================================================

# ---------------------------------------------------------------------------
# prepare_positive_rows
# ---------------------------------------------------------------------------

class TestPreparePositiveRows:
    def test_keeps_only_positive_rows(self, toy_df):
        features, _ = _wgan().prepare_positive_rows(toy_df, "y", 1, ["cat1", "y"])
        assert len(features) == 72

    def test_drops_target_column(self, toy_df):
        features, _ = _wgan().prepare_positive_rows(toy_df, "y", 1, ["cat1", "y"])
        assert "y" not in features.columns

    def test_discrete_excludes_target(self, toy_df):
        _, discrete = _wgan().prepare_positive_rows(toy_df, "y", 1, ["cat1", "y"])
        assert "y" not in discrete

    def test_discrete_includes_named_col(self, toy_df):
        _, discrete = _wgan().prepare_positive_rows(toy_df, "y", 1, ["cat1", "y"])
        assert "cat1" in discrete

    def test_non_numeric_col_inferred_as_discrete(self, toy_df):
        # cat1 is str — should be inferred even when not explicitly listed
        _, discrete = _wgan().prepare_positive_rows(toy_df, "y", 1, [])
        assert "cat1" in discrete

    def test_raises_on_missing_target(self, toy_df):
        with pytest.raises(ValueError, match="target_col"):
            _wgan().prepare_positive_rows(toy_df, "no_such_col", 1, [])

    def test_raises_on_is_synthetic(self, toy_df):
        df = toy_df.copy()
        df["is_synthetic"] = 0
        with pytest.raises(ValueError, match="is_synthetic"):
            _wgan().prepare_positive_rows(df, "y", 1, [])

    def test_raises_on_nan(self, toy_df):
        df = toy_df.copy()
        df.loc[0, "num1"] = None
        with pytest.raises(ValueError):
            _wgan().prepare_positive_rows(df, "y", 1, [])

    def test_raises_on_no_positives(self, toy_df):
        df = toy_df.copy()
        df["y"] = 0
        with pytest.raises(ValueError):
            _wgan().prepare_positive_rows(df, "y", 1, [])

    def test_raises_when_fewer_than_two_positives(self, toy_df):
        df = toy_df.copy()
        df["y"] = 0
        df.loc[0, "y"] = 1
        with pytest.raises(ValueError):
            _wgan().prepare_positive_rows(df, "y", 1, [])

    def test_index_is_reset(self, toy_df):
        features, _ = _wgan().prepare_positive_rows(toy_df, "y", 1, [])
        assert features.index.tolist() == list(range(len(features)))


# ---------------------------------------------------------------------------
# apply_activate
# ---------------------------------------------------------------------------

class TestApplyActivate:
    def test_output_shape_matches_input(self):
        output_info = _info([[(1, "tanh"), (3, "softmax")], [(4, "softmax")]])
        data = torch.randn(32, 8)
        out = _wgan().apply_activate(data, output_info)
        assert out.shape == data.shape

    def test_tanh_spans_within_minus1_plus1(self):
        output_info = _info([[(2, "tanh")]])
        data = torch.randn(32, 2) * 10  # large logits to stress tanh
        out = _wgan().apply_activate(data, output_info)
        assert (out >= -1.0).all() and (out <= 1.0).all()

    def test_softmax_spans_sum_to_one(self):
        output_info = _info([[(4, "softmax")]])
        data = torch.randn(32, 4)
        out = _wgan().apply_activate(data, output_info)
        assert (out >= 0).all()
        assert torch.allclose(out.sum(dim=1), torch.ones(32), atol=1e-5)

    def test_softmax_spans_non_negative(self):
        output_info = _info([[(3, "softmax")], [(5, "softmax")]])
        data = torch.randn(32, 8)
        out = _wgan().apply_activate(data, output_info)
        assert (out >= 0).all()

    def test_mixed_info_all_conditions(self):
        # 1 tanh + 3 softmax + 4 softmax
        output_info = _info([[(1, "tanh"), (3, "softmax")], [(4, "softmax")]])
        data = torch.randn(16, 8)
        out = _wgan().apply_activate(data, output_info)
        # tanh span: col 0
        assert (out[:, 0:1] >= -1).all() and (out[:, 0:1] <= 1).all()
        # first softmax block: cols 1:4
        assert torch.allclose(out[:, 1:4].sum(dim=1), torch.ones(16), atol=1e-5)
        # second softmax block: cols 4:8
        assert torch.allclose(out[:, 4:8].sum(dim=1), torch.ones(16), atol=1e-5)

    def test_unknown_activation_raises(self):
        output_info = _info([[(2, "relu")]])
        data = torch.randn(4, 2)
        with pytest.raises(ValueError, match="relu"):
            _wgan().apply_activate(data, output_info)


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

class TestCritic:
    def test_last_module_is_linear(self):
        critic = _wgan().Critic(8)
        last = list(critic.seq.children())[-1]
        assert isinstance(last, nn.Linear)

    def test_no_sigmoid_anywhere(self):
        critic = _wgan().Critic(8)
        for mod in critic.modules():
            assert not isinstance(mod, nn.Sigmoid)

    def test_dropout_p_is_025(self):
        critic = _wgan().Critic(8)
        dropouts = [m for m in critic.modules() if isinstance(m, nn.Dropout)]
        assert len(dropouts) > 0
        for d in dropouts:
            assert abs(d.p - 0.25) < 1e-6

    def test_leaky_relu_slope_is_02(self):
        critic = _wgan().Critic(8)
        relus = [m for m in critic.modules() if isinstance(m, nn.LeakyReLU)]
        assert len(relus) > 0
        for r in relus:
            assert abs(r.negative_slope - 0.2) < 1e-6

    def test_output_shape(self):
        critic = _wgan().Critic(8)
        x = torch.randn(16, 8)
        out = critic(x)
        assert out.shape == (16, 1)

    def test_custom_critic_dim(self):
        critic = _wgan().Critic(8, critic_dim=(64,))
        x = torch.randn(4, 8)
        assert critic(x).shape == (4, 1)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class TestGenerator:
    def test_output_width_matches_transformer(self, transformer):
        wgan = _wgan()
        data_dim = transformer.output_dimensions
        gen = wgan.Generator(128, (256, 256), data_dim)
        noise = torch.randn(16, 128)
        out = gen(noise)
        assert out.shape == (16, data_dim)

    def test_output_shape_general(self):
        gen = _wgan().Generator(64, (128, 128), 20)
        noise = torch.randn(8, 64)
        assert gen(noise).shape == (8, 20)

    def test_no_batchnorm(self):
        gen = _wgan().Generator(128, (256, 256), 16)
        for mod in gen.modules():
            assert not isinstance(mod, nn.BatchNorm1d)

    def test_no_residual_connections(self):
        # Plain MLP: the seq children should only be Linear and ReLU
        gen = _wgan().Generator(128, (256, 256), 16)
        for mod in gen.seq.children():
            assert isinstance(mod, (nn.Linear, nn.ReLU))


# ---------------------------------------------------------------------------
# gradient_penalty
# ---------------------------------------------------------------------------

class TestGradientPenalty:
    def test_analytic_value_linear_critic(self):
        """For f(x)=w·x, GP = gp_lambda * (||w||_2 - 1)^2 (tolerance 1e-5)."""
        wgan = _wgan()
        dim = 8
        critic = nn.Linear(dim, 1, bias=False)
        with torch.no_grad():
            w_norm = critic.weight.norm().item()
        gp_lambda = 10.0
        torch.manual_seed(0)
        real = torch.randn(16, dim)
        fake = torch.randn(16, dim)
        gp = wgan.gradient_penalty(critic, real, fake, gp_lambda, "cpu")
        expected = gp_lambda * (w_norm - 1.0) ** 2
        assert abs(gp.item() - expected) < 1e-5

    def test_zero_when_unit_norm(self):
        """GP is 0 when the critic weight is already unit-norm."""
        wgan = _wgan()
        dim = 8
        critic = nn.Linear(dim, 1, bias=False)
        with torch.no_grad():
            critic.weight.div_(critic.weight.norm())
        torch.manual_seed(1)
        real = torch.randn(16, dim)
        fake = torch.randn(16, dim)
        gp = wgan.gradient_penalty(critic, real, fake, 10.0, "cpu")
        assert abs(gp.item()) < 1e-5

    def test_gp_lambda_zero_gives_zero(self):
        wgan = _wgan()
        critic = nn.Linear(4, 1, bias=False)
        real = torch.randn(8, 4)
        fake = torch.randn(8, 4)
        gp = wgan.gradient_penalty(critic, real, fake, 0.0, "cpu")
        assert abs(gp.item()) < 1e-9

    def test_gp_scales_with_lambda(self):
        """GP with lambda=20 is exactly twice GP with lambda=10."""
        wgan = _wgan()
        torch.manual_seed(2)
        dim = 8
        critic = nn.Linear(dim, 1, bias=False)
        real = torch.randn(16, dim)
        fake = torch.randn(16, dim)
        gp10 = wgan.gradient_penalty(critic, real, fake, 10.0, "cpu")
        gp20 = wgan.gradient_penalty(critic, real, fake, 20.0, "cpu")
        assert abs(gp20.item() - 2.0 * gp10.item()) < 1e-4

    def test_gp_has_grad(self):
        """The GP must stay in the computational graph (create_graph=True)."""
        wgan = _wgan()
        dim = 8
        critic = nn.Linear(dim, 1, bias=False)
        real = torch.randn(16, dim)
        fake = torch.randn(16, dim)
        gp = wgan.gradient_penalty(critic, real, fake, 1.0, "cpu")
        assert gp.requires_grad
