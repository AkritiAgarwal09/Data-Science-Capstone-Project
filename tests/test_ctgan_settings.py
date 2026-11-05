import json
from pathlib import Path

import pytest

from agentic_gan.utils.ctgan_settings import load_ctgan_settings


def test_v1_is_the_bank_run():
    settings = load_ctgan_settings("v1")
    assert settings.version == "v1"
    assert settings.epochs == 300
    assert settings.batch_size == 500
    assert settings.pac == 10
    assert settings.discriminator_steps == 1
    assert settings.generator_dim == (256, 256)
    assert settings.conditional is True
    assert settings.rake is False
    assert settings.pool_rows is None


def test_command_line_overrides_epochs_device_and_seed():
    settings = load_ctgan_settings("v1", epochs=20, device="cpu", seed=7)
    assert settings.epochs == 20
    assert settings.seed == 7
    assert settings.pac == 10


def test_v2_is_the_bank_recipe():
    settings = load_ctgan_settings("v2")
    assert settings.version == "v2"
    assert settings.source == "bank v2"
    assert settings.epochs == 300
    assert settings.batch_size == 500
    assert settings.pac == 5
    assert settings.discriminator_steps == 3
    assert settings.generator_dim == (256, 256)
    assert settings.discriminator_dim == (256, 256)
    assert settings.conditional is False
    assert settings.rake is True
    assert settings.pool_rows == 60_000
    assert settings.seed == 42


def test_missing_settings_file_is_refused(tmp_path: Path):
    missing = tmp_path / "missing.json"
    with pytest.raises(SystemExit, match="does not exist"):
        load_ctgan_settings("v2", path=missing)


def test_v2_file_is_loaded_when_present(tmp_path: Path):
    path = tmp_path / "ctgan_v2_settings.json"
    path.write_text(json.dumps({
        "epochs": 400,
        "batch_size": 500,
        "pac": 5,
        "discriminator_steps": 3,
        "generator_dim": [128, 128],
        "discriminator_dim": [128, 128],
        "conditional": False,
        "rake": True,
        "pool_rows": 8000,
    }))
    settings = load_ctgan_settings("v2", path=path)
    assert settings.version == "v2"
    assert settings.epochs == 400
    assert settings.pac == 5
    assert settings.discriminator_steps == 3
    assert settings.conditional is False
    assert settings.rake is True
    assert settings.pool_rows == 8000
    assert settings.source == str(path)
