"""Default gan_bounds from config_for_dataset (issue #123 step 1)."""

from __future__ import annotations

from dataclasses import asdict

from agentic_gan.agents.model_selection import CTGAN_DEFAULTS, _rule_hash
from agentic_gan.orchestrator.graph import DEFAULT_GAN_BOUNDS, config_for_dataset
from agentic_gan.orchestrator.manifest import create_run
from agentic_gan.schemas import RunConfig
from agentic_gan.tools.ctgan_sampling import recipe_for
from agentic_gan.utils.datasets import DATASETS

EXPECTED_BOUNDS = {
    "epochs": (300, 600),
    "pac": (5, 10),
    "discriminator_steps": (1, 5),
    "batch_size": (250, 1000),
}


def test_default_gan_bounds_constant_matches_expected():
    assert DEFAULT_GAN_BOUNDS == EXPECTED_BOUNDS


def test_every_dataset_config_has_exactly_these_four_bounds():
    assert set(DATASETS) == {"bank_marketing", "default", "polish", "ieee_cis"}
    for name in DATASETS:
        bounds = config_for_dataset(name).gan_bounds
        assert set(bounds) == set(EXPECTED_BOUNDS)
        assert bounds == EXPECTED_BOUNDS


def test_two_configs_do_not_share_the_same_dict_object():
    first = config_for_dataset("bank_marketing")
    second = config_for_dataset("polish")
    assert first.gan_bounds == second.gan_bounds
    assert first.gan_bounds is not second.gan_bounds
    assert first.gan_bounds is not DEFAULT_GAN_BOUNDS
    first.gan_bounds["epochs"] = (1, 2)
    assert second.gan_bounds["epochs"] == EXPECTED_BOUNDS["epochs"]
    assert DEFAULT_GAN_BOUNDS["epochs"] == EXPECTED_BOUNDS["epochs"]


def test_yaml_round_trip_keeps_gan_bounds_and_rule_hash(tmp_path):
    # to_yaml / from_yaml: tuples become lists in JSON/YAML; RunConfig and _rule_hash must agree.
    config = config_for_dataset("bank_marketing")
    restored = RunConfig.from_yaml(config.to_yaml())
    assert restored.gan_bounds == config.gan_bounds == EXPECTED_BOUNDS
    assert _rule_hash(restored) == _rule_hash(config)
    # Same pattern as test_rule_hash_survives_the_config_yaml_round_trip via create_run.
    run = create_run(config, run_id="bounds-hash", runs_dir=tmp_path / "runs")
    assert _rule_hash(run.load_config()) == _rule_hash(config)


def test_attempt_1_settings_lie_inside_the_bounds():
    # CTGAN_DEFAULTS: epochs 300, batch 500, pac 10, steps 1.
    for name, value in CTGAN_DEFAULTS.items():
        low, high = EXPECTED_BOUNDS[name]
        assert low <= value <= high, name
    # Bank recipe: pac 5, steps 3, batch 500, epochs 300 — and every other dataset recipe.
    for dataset in DATASETS:
        recipe = asdict(recipe_for(dataset))
        for name in EXPECTED_BOUNDS:
            low, high = EXPECTED_BOUNDS[name]
            assert low <= recipe[name] <= high, f"{dataset}.{name}"
    assert CTGAN_DEFAULTS["epochs"] == 300
    assert CTGAN_DEFAULTS["batch_size"] == 500
    assert CTGAN_DEFAULTS["pac"] == 10
    assert CTGAN_DEFAULTS["discriminator_steps"] == 1
    bank = asdict(recipe_for("bank_marketing"))
    assert bank["epochs"] == 300
    assert bank["batch_size"] == 500
    assert bank["pac"] == 5
    assert bank["discriminator_steps"] == 3


def test_changing_one_bound_changes_the_rule_hash():
    config = config_for_dataset("bank_marketing")
    base_hash = _rule_hash(config)
    moved = config.model_copy(
        update={"gan_bounds": {**config.gan_bounds, "epochs": (300, 700)}}
    )
    assert _rule_hash(moved) != base_hash


def test_bounds_are_legal_for_runconfig_validator():
    # RunConfig._bounds_ordered requires low <= high (schemas.py).
    RunConfig(
        dataset_name="bank_marketing",
        dataset_path=config_for_dataset("bank_marketing").dataset_path,
        target="y",
        positive_class="yes",
        gan_bounds=dict(EXPECTED_BOUNDS),
    )
