"""CTGAN run settings for the Default and Polish synthetic-data scripts.

v1 is the Bank run: 300 epochs, batch size 500, pac 10, one discriminator step,
conditional sampling, no raking. v2 is not chosen yet. When it is, save
``eda/ctgan_v2_settings.json`` and rerun with ``--settings v2``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

V1 = {
    "epochs": 300,
    "batch_size": 500,
    "pac": 10,
    "discriminator_steps": 1,
    "generator_dim": (256, 256),
    "discriminator_dim": (256, 256),
    "device": "cpu",
    "seed": 42,
    "conditional": True,
    "rake": False,
    "pool_rows": None,
}

_REQUIRED = (
    "epochs",
    "batch_size",
    "pac",
    "discriminator_steps",
    "generator_dim",
    "discriminator_dim",
    "conditional",
    "rake",
)
_INT_KEYS = ("epochs", "batch_size", "pac", "discriminator_steps")


@dataclass(frozen=True)
class CTGANRunSettings:
    version: str
    epochs: int
    batch_size: int
    pac: int
    discriminator_steps: int
    generator_dim: tuple[int, ...]
    discriminator_dim: tuple[int, ...]
    device: str
    seed: int
    conditional: bool
    rake: bool
    pool_rows: int | None
    source: str


def v2_settings_path() -> Path:
    return Path(__file__).resolve().parents[3] / "eda" / "ctgan_v2_settings.json"


def load_ctgan_settings(
    version: str,
    *,
    path: Path | None = None,
    epochs: int | None = None,
    device: str | None = None,
    seed: int | None = None,
) -> CTGANRunSettings:
    """Return v1, or v2 once ``eda/ctgan_v2_settings.json`` exists.

    ``epochs``, ``device`` and ``seed`` override the bundle when passed on the command line.
    """
    if version == "v1":
        raw = dict(V1)
        source = "bank v1"
    elif version == "v2":
        settings_path = path if path is not None else v2_settings_path()
        raw = _read_v2(settings_path)
        source = str(settings_path)
    else:
        raise SystemExit("settings must be v1 or v2")
    if epochs is not None:
        raw["epochs"] = epochs
    if device is not None:
        raw["device"] = device
    if seed is not None:
        raw["seed"] = seed
    return _build(version, raw, source)


def _read_v2(path: Path) -> dict:
    if not path.is_file():
        keys = ", ".join((*_REQUIRED, "pool_rows"))
        raise SystemExit(
            "CTGAN v2 settings are not available yet. "
            f"Save the chosen settings as {path} and rerun with --settings v2. "
            f"Required keys: {keys}. "
            "pool_rows is required when rake is true. "
            "device and seed are optional and default to cpu and 42."
        )
    try:
        loaded = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    missing = [key for key in _REQUIRED if key not in loaded]
    if missing:
        raise SystemExit(f"{path} is missing keys: {', '.join(missing)}")
    return loaded


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SystemExit(f"{name} must be a positive integer")
    return value


def _dims(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise SystemExit(f"{name} must be a non-empty list of positive integers")
    return tuple(_positive_int(item, name) for item in value)


def _build(version: str, raw: dict, source: str) -> CTGANRunSettings:
    epochs = _positive_int(raw["epochs"], "epochs")
    batch_size = _positive_int(raw["batch_size"], "batch_size")
    pac = _positive_int(raw["pac"], "pac")
    steps = _positive_int(raw["discriminator_steps"], "discriminator_steps")
    if batch_size % pac != 0:
        raise SystemExit("batch_size must be divisible by pac")
    conditional = raw["conditional"]
    rake = raw["rake"]
    if not isinstance(conditional, bool) or not isinstance(rake, bool):
        raise SystemExit("conditional and rake must be true or false")
    device = raw.get("device", "cpu")
    if device not in {"cpu", "cuda", "auto"}:
        raise SystemExit("device must be cpu, cuda, or auto")
    seed = raw.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SystemExit("seed must be an integer")
    pool_rows = raw.get("pool_rows")
    if rake:
        pool_rows = _positive_int(pool_rows, "pool_rows")
    elif pool_rows is not None:
        pool_rows = _positive_int(pool_rows, "pool_rows")
    return CTGANRunSettings(
        version=version,
        epochs=epochs,
        batch_size=batch_size,
        pac=pac,
        discriminator_steps=steps,
        generator_dim=_dims(list(raw["generator_dim"]), "generator_dim"),
        discriminator_dim=_dims(list(raw["discriminator_dim"]), "discriminator_dim"),
        device=device,
        seed=seed,
        conditional=conditional,
        rake=rake,
        pool_rows=pool_rows,
        source=source,
    )
