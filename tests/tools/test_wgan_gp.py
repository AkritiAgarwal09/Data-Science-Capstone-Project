"""Tests for tools/wgan_gp.py building blocks, training loop, and sampler.

Steps 1–3: prepare_positive_rows, apply_activate, Critic, Generator,
gradient_penalty, train_wgan_gp, sample_wgan_gp.
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


# ===========================================================================
# Step 2 tests: training loop
# ===========================================================================

def _small_train(toy_df, **kwargs):
    """Train a small WGAN-GP quickly for testing."""
    defaults = dict(epochs=2, batch_size=16, n_critic=3, seed=42)
    defaults.update(kwargs)
    return _wgan().train_wgan_gp(toy_df, "y", 1, ["cat1", "y"], **defaults)


class TestTrainWGANGP:
    def test_returns_trained_wgan_gp(self, toy_df):
        wgan = _wgan()
        result = _small_train(toy_df)
        assert isinstance(result, wgan.TrainedWGANGP)

    def test_loss_history_shape(self, toy_df):
        epochs = 3
        result = _small_train(toy_df, epochs=epochs)
        assert len(result.loss_history) == epochs
        assert set(result.loss_history.columns) >= {
            "Epoch", "Generator Loss", "Discriminator Loss", "Gradient Penalty"
        }

    def test_loss_history_epoch_column(self, toy_df):
        result = _small_train(toy_df, epochs=2)
        assert result.loss_history["Epoch"].tolist() == [1, 2]

    def test_log_file_written_and_parent_created(self, toy_df, tmp_path):
        log = tmp_path / "nested" / "dir" / "loss.csv"
        _small_train(toy_df, log_path=log)
        assert log.exists()
        df = pd.read_csv(log)
        assert len(df) == 2

    def test_n_critic_ratio(self, toy_df, monkeypatch):
        """critic steps == n_critic * generator steps."""
        step_counts: dict[int, int] = {}
        creation_order: list[int] = []

        _orig_init = torch.optim.Adam.__init__
        _orig_step = torch.optim.Adam.step

        def _tracked_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            creation_order.append(id(self))
            step_counts[id(self)] = 0

        def _tracked_step(self, closure=None):
            step_counts[id(self)] = step_counts.get(id(self), 0) + 1
            return _orig_step(self, closure)

        monkeypatch.setattr(torch.optim.Adam, "__init__", _tracked_init)
        monkeypatch.setattr(torch.optim.Adam, "step", _tracked_step)

        n_c = 3
        _small_train(toy_df, n_critic=n_c, epochs=2, batch_size=8)

        # generator opt is created first, critic opt second
        gen_id, crit_id = creation_order[0], creation_order[1]
        assert step_counts[crit_id] == n_c * step_counts[gen_id]

    def test_critic_params_unchanged_during_generator_step(self):
        """gen_opt.step() must not update critic parameters."""
        wgan = _wgan()
        # Minimal generator step manually
        gen = wgan.Generator(8, (16,), 4)
        critic = wgan.Critic(4, (16,))
        gen_opt = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0.5, 0.9))

        # Snapshot critic weights
        critic_before = {k: v.clone() for k, v in critic.state_dict().items()}

        # Simulate one generator step (no apply_activate needed; raw output is fine)
        noise = torch.randn(8, 8)
        fake = gen(noise)
        gen_loss = -critic(fake).mean()
        gen_opt.zero_grad()
        gen_loss.backward()
        gen_opt.step()

        for k, v in critic.state_dict().items():
            assert torch.allclose(v, critic_before[k]), f"Critic param {k} changed"

    def test_gp_lambda_affects_critic_loss(self, toy_df):
        """Different gp_lambda values must yield different critic losses."""
        r0 = _small_train(toy_df, gp_lambda=0.0, epochs=1, seed=7)
        r10 = _small_train(toy_df, gp_lambda=10.0, epochs=1, seed=7)
        c0 = r0.loss_history["Discriminator Loss"].iloc[0]
        c10 = r10.loss_history["Discriminator Loss"].iloc[0]
        assert abs(c0 - c10) > 1e-6

    def test_determinism_same_seed(self, toy_df):
        r1 = _small_train(toy_df, seed=99)
        r2 = _small_train(toy_df, seed=99)
        pd.testing.assert_frame_equal(r1.loss_history, r2.loss_history)
        for k in r1.generator.state_dict():
            assert torch.allclose(r1.generator.state_dict()[k], r2.generator.state_dict()[k])

    def test_determinism_different_seed(self, toy_df):
        r1 = _small_train(toy_df, seed=1)
        r2 = _small_train(toy_df, seed=2)
        # At least one parameter should differ
        any_diff = any(
            not torch.allclose(r1.generator.state_dict()[k], r2.generator.state_dict()[k])
            for k in r1.generator.state_dict()
        )
        assert any_diff

    def test_nan_guard(self, toy_df, monkeypatch):
        """RuntimeError raised when a loss becomes NaN."""
        import torch as _torch
        _orig_mean = _torch.Tensor.mean

        call_count = [0]

        def _nan_mean(self, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] > 10:
                return _torch.tensor(float("nan"), requires_grad=True)
            return _orig_mean(self, *args, **kwargs)

        monkeypatch.setattr(_torch.Tensor, "mean", _nan_mean)

        with pytest.raises(RuntimeError, match="diverged"):
            _wgan().train_wgan_gp(toy_df, "y", 1, ["cat1", "y"], epochs=5, batch_size=8, seed=0)

    def test_save_load_round_trip(self, toy_df, tmp_path):
        wgan = _wgan()
        model = _small_train(toy_df)
        path = tmp_path / "model.pkl"
        wgan.save_model(model, path)
        loaded = wgan.load_model(path)
        for k in model.generator.state_dict():
            assert torch.allclose(
                model.generator.state_dict()[k].cpu(),
                loaded.generator.state_dict()[k].cpu(),
            )
        assert loaded.params == model.params

    def test_uses_only_positive_rows(self, toy_df):
        """Negative rows with out-of-range values must not appear in numeric_ranges."""
        df = toy_df.copy()
        # Inject a very large value in a negative row
        df.loc[df["y"] == 0, "num1"] = 1e9
        result = _small_train(df)
        low, high = result.numeric_ranges["num1"]
        assert high < 1e8, "numeric_ranges must be computed from positive rows only"
        assert result.n_train_rows == (df["y"] == 1).sum()

    def test_train_seconds_positive(self, toy_df):
        result = _small_train(toy_df)
        assert result.train_seconds > 0

    def test_generator_critic_in_eval_mode(self, toy_df):
        result = _small_train(toy_df)
        assert not result.generator.training
        assert not result.critic.training


# ===========================================================================
# Step 3 tests: sampler
# ===========================================================================

@pytest.fixture(scope="module")
def trained_model(toy_df):
    """Train once per module; 2 epochs is enough for sampler correctness tests."""
    return _wgan().train_wgan_gp(
        toy_df, "y", 1, ["cat1", "y"], epochs=2, batch_size=16, seed=42
    )


@pytest.fixture(scope="module")
def toy_df_with_int():
    rng = np.random.default_rng(2)
    n = 120
    return pd.DataFrame({
        "num1": rng.normal(0, 1, n).astype(float),
        "age": rng.integers(20, 60, n),  # integer column
        "cat1": rng.choice(["a", "b", "c"], n),
        "y": [1] * 72 + [0] * 48,
    })


@pytest.fixture(scope="module")
def trained_model_int(toy_df_with_int):
    return _wgan().train_wgan_gp(
        toy_df_with_int, "y", 1, ["cat1", "y"], epochs=2, batch_size=16, seed=42
    )


class TestSampleWGANGP:
    def test_output_columns_and_length(self, trained_model, toy_df):
        wgan = _wgan()
        n = 20
        out = wgan.sample_wgan_gp(trained_model, n, seed=0)
        assert len(out) == n
        assert list(out.columns) == trained_model.columns + [wgan.IS_SYNTHETIC]

    def test_target_equals_positive(self, trained_model):
        out = _wgan().sample_wgan_gp(trained_model, 20, seed=0)
        assert (out[trained_model.target_col] == trained_model.positive).all()

    def test_is_synthetic_equals_one(self, trained_model):
        wgan = _wgan()
        out = wgan.sample_wgan_gp(trained_model, 20, seed=0)
        assert (out[wgan.IS_SYNTHETIC] == 1).all()

    def test_no_nan(self, trained_model):
        out = _wgan().sample_wgan_gp(trained_model, 20, seed=0)
        assert not out.isna().any().any()

    def test_categorical_values_in_training_categories(self, trained_model, toy_df):
        pos_cats = set(toy_df[toy_df["y"] == 1]["cat1"].unique())
        all_cats = set(toy_df["cat1"].unique())
        out = _wgan().sample_wgan_gp(trained_model, 50, seed=1)
        synth_cats = set(out["cat1"].unique())
        # Prefer the stronger check; fall back to all-training categories
        if synth_cats.issubset(pos_cats):
            assert synth_cats.issubset(pos_cats)
        else:
            assert synth_cats.issubset(all_cats), (
                f"Synthetic categories {synth_cats} not in training categories {all_cats}"
            )

    def test_numeric_values_within_training_range(self, trained_model):
        out = _wgan().sample_wgan_gp(trained_model, 50, seed=2)
        for col, (low, high) in trained_model.numeric_ranges.items():
            assert (out[col] >= low - 1e-6).all(), f"{col} below training min"
            assert (out[col] <= high + 1e-6).all(), f"{col} above training max"

    def test_integer_columns_whole_numbers_and_int_dtype(self, trained_model_int):
        out = _wgan().sample_wgan_gp(trained_model_int, 20, seed=3)
        for col in trained_model_int.integer_valued:
            assert (out[col] == out[col].round()).all(), f"{col} has non-integer values"
            assert pd.api.types.is_integer_dtype(out[col].dtype), f"{col} not integer dtype"

    def test_drop_duplicates_no_train_copy(self, trained_model, toy_df):
        out = _wgan().sample_wgan_gp(
            trained_model, 30, seed=4, drop_duplicates_of=toy_df
        )
        # No synthetic row should equal any training row (merge on model.columns)
        merged = out[trained_model.columns].merge(
            toy_df[trained_model.columns], on=trained_model.columns, how="inner"
        )
        assert len(merged) == 0, "Synthetic output contains a copy of a training row"

    def test_drop_duplicates_no_intra_duplicates(self, trained_model, toy_df):
        out = _wgan().sample_wgan_gp(
            trained_model, 30, seed=4, drop_duplicates_of=toy_df
        )
        assert not out[trained_model.columns].duplicated().any(), "Duplicate synthetic rows found"

    def test_row_filter_applied(self, trained_model):
        # Keep only rows where num1 > its training median
        col = "num1"
        low, high = trained_model.numeric_ranges[col]
        threshold = (low + high) / 2

        def keep_above(df: pd.DataFrame) -> pd.Series:
            return df[col] > threshold

        out = _wgan().sample_wgan_gp(trained_model, 20, seed=5, row_filter=keep_above)
        assert (out[col] > threshold).all()

    def test_same_seed_gives_identical_frames(self, trained_model):
        wgan = _wgan()
        out1 = wgan.sample_wgan_gp(trained_model, 20, seed=10)
        out2 = wgan.sample_wgan_gp(trained_model, 20, seed=10)
        pd.testing.assert_frame_equal(out1, out2)

    def test_different_seed_gives_different_frames(self, trained_model):
        wgan = _wgan()
        out1 = wgan.sample_wgan_gp(trained_model, 20, seed=10)
        out2 = wgan.sample_wgan_gp(trained_model, 20, seed=11)
        # At least one row should differ
        eq = (out1.values == out2.values).all(axis=1)
        assert not eq.all()

    def test_calling_twice_does_not_mutate_model(self, trained_model):
        wgan = _wgan()
        sd_before = {k: v.clone() for k, v in trained_model.generator.state_dict().items()}
        wgan.sample_wgan_gp(trained_model, 20, seed=20)
        wgan.sample_wgan_gp(trained_model, 20, seed=21)
        for k, v in trained_model.generator.state_dict().items():
            assert torch.allclose(v, sd_before[k]), f"Generator param {k} changed after sampling"

    def test_max_rounds_impossible_filter_returns_partial(self, trained_model, caplog):
        import logging

        def impossible(df: pd.DataFrame) -> pd.Series:
            return pd.Series(False, index=df.index)

        with caplog.at_level(logging.WARNING, logger="agentic_gan.tools.wgan_gp"):
            out = _wgan().sample_wgan_gp(
                trained_model, 50, seed=99, row_filter=impossible, max_rounds=1
            )
        assert len(out) < 50
        assert any("valid synthetic rows" in m for m in caplog.messages)
