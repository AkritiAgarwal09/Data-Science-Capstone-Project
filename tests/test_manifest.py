"""Tests for the run folder and manifest."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_gan.orchestrator import manifest as manifest_module
from agentic_gan.orchestrator.manifest import (
    SUBDIRS,
    ManifestEntry,
    ManifestError,
    Problem,
    RunDir,
    create_run,
    package_versions,
)
from agentic_gan.orchestrator.state import RunState
from agentic_gan.schemas import RunConfig, SplitOutput, Stage, Status

NOW = datetime(2026, 10, 14, 10, 15, 0, tzinfo=UTC)
RUN_ID = "bank_marketing_20261014-101500"


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "bank-full.csv"
    path.write_text("age,y\n30,no\n41,yes\n")
    return path


@pytest.fixture
def config(dataset):
    return RunConfig(
        dataset_name="bank_marketing", dataset_path=dataset, target="y", positive_class="yes"
    )


@pytest.fixture
def run(config, tmp_path):
    return create_run(config, runs_dir=tmp_path / "runs", now=lambda: NOW)


@pytest.fixture
def manifest(run):
    return run.load_manifest()


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write(run: RunDir, rel: str, text: str = "data") -> Path:
    path = run.root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def split_output(
    train="data/train.csv", val="data/val.csv", test="data/test.csv", **overrides
) -> SplitOutput:
    fields = {
        "status": Status.SUCCESS,
        "train_path": Path(train),
        "val_path": Path(val),
        "test_path": Path(test),
        "target": "y",
        "seed": 42,
        "train_positives": 1,
        "train_negatives": 9,
    }
    return SplitOutput(**{**fields, **overrides})


def write_split_files(run: RunDir) -> None:
    for name in ("train", "val", "test"):
        write(run, f"data/{name}.csv", name)


# --- create_run -------------------------------------------------------------------------------


def test_create_run_makes_the_folder_tree(run, tmp_path):
    assert run.root == tmp_path / "runs" / RUN_ID
    assert all(run.subdir(name).is_dir() for name in SUBDIRS)
    assert run.config_path.is_file()
    assert run.state_path.is_file()
    assert run.manifest_path.is_file()


def test_run_id_defaults_to_dataset_name_and_start_time(run):
    assert run.run_id == RUN_ID


def test_an_explicit_run_id_is_used(config, tmp_path):
    run = create_run(config, run_id="my_run", runs_dir=tmp_path / "runs", now=lambda: NOW)
    assert run.run_id == "my_run"


def test_the_default_run_id_is_made_safe_for_odd_dataset_names(config, tmp_path):
    odd = config.updated(dataset_name="Bank Marketing (UCI)")
    run = create_run(odd, runs_dir=tmp_path / "runs", now=lambda: NOW)
    assert run.run_id == "Bank-Marketing-UCI_20261014-101500"


def test_the_saved_config_rebuilds_the_run_config(run, config):
    assert run.load_config() == config


def test_the_initial_state_has_nothing_started(run):
    assert run.load_state() == RunState.new(RUN_ID)


def test_the_manifest_header_pins_the_inputs(run, dataset):
    manifest = run.load_manifest()
    assert manifest.run_id == RUN_ID
    assert manifest.created_at == NOW
    assert manifest.dataset_sha256 == hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert manifest.config_sha256 == hashlib.sha256(run.config_path.read_bytes()).hexdigest()
    assert manifest.packages["python"]
    assert manifest.entries == []


def test_an_existing_run_is_never_overwritten(run, config, tmp_path):
    before = run.config_path.read_text()
    with pytest.raises(FileExistsError, match="already exists"):
        create_run(config.updated(target="other"), runs_dir=tmp_path / "runs", now=lambda: NOW)
    assert run.config_path.read_text() == before


def test_a_missing_dataset_creates_nothing(config, tmp_path):
    missing = config.updated(dataset_path=tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError):
        create_run(missing, runs_dir=tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("run_id", ["../escape", "a/b", "with space", ".hidden", ""])
def test_an_unsafe_run_id_creates_nothing(config, tmp_path, run_id):
    with pytest.raises(ValidationError):
        create_run(config, run_id=run_id, runs_dir=tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_a_failed_setup_removes_the_half_made_folder(config, tmp_path, monkeypatch):
    def broken_save(self, run):
        raise OSError("disk full")

    monkeypatch.setattr(manifest_module.Manifest, "save", broken_save)
    with pytest.raises(OSError, match="disk full"):
        create_run(config, runs_dir=tmp_path / "runs", now=lambda: NOW)
    assert not (tmp_path / "runs" / RUN_ID).exists()
    monkeypatch.undo()
    assert create_run(config, runs_dir=tmp_path / "runs", now=lambda: NOW).root.is_dir()


def test_the_default_clock_is_timezone_aware(config, tmp_path):
    run = create_run(config, runs_dir=tmp_path / "runs")
    assert run.load_manifest().created_at.tzinfo is not None


def test_package_versions_cover_python_and_the_dependencies():
    versions = package_versions()
    assert {"python", "pydantic"} <= set(versions)
    assert all(isinstance(v, str) and v for v in versions.values())


# --- RunDir -----------------------------------------------------------------------------------


def test_run_dir_paths(run):
    assert run.state_path.name == "state.json"
    assert run.manifest_path.name == "manifest.json"
    assert run.decisions_path.name == "decisions.jsonl"
    assert run.subdir("models") == run.root / "models"


def test_run_dir_rejects_an_unknown_subfolder(run):
    with pytest.raises(ValueError, match="unknown run subfolder"):
        run.subdir("scratch")


def test_relative_accepts_paths_inside_the_run_folder(run):
    assert run.relative(Path("data/train.csv")) == Path("data/train.csv")
    assert run.relative(run.root / "data" / "train.csv") == Path("data/train.csv")
    assert run.relative(run.root / "data" / ".." / "models" / "m.pkl") == Path("models/m.pkl")


@pytest.mark.parametrize("path", ["../elsewhere.csv", "data/../../elsewhere.csv", "/etc/hosts"])
def test_relative_refuses_paths_outside_the_run_folder(run, path):
    with pytest.raises(ManifestError, match="outside the run folder"):
        run.relative(Path(path))


# --- register ---------------------------------------------------------------------------------


def test_register_hashes_each_file_and_records_the_agent(run, manifest):
    write_split_files(run)
    registered = manifest.register(run, Stage.SPLIT, split_output())
    assert {e.path: (e.sha256, e.agent) for e in registered.entries} == {
        Path("data/train.csv"): (sha("train"), Stage.SPLIT),
        Path("data/val.csv"): (sha("val"), Stage.SPLIT),
        Path("data/test.csv"): (sha("test"), Stage.SPLIT),
    }


def test_register_returns_a_new_manifest(run, manifest):
    write_split_files(run)
    manifest.register(run, Stage.SPLIT, split_output())
    assert manifest.entries == []


def test_register_stores_absolute_paths_inside_the_run_as_relative(run, manifest):
    write_split_files(run)
    absolute = split_output(
        train=str(run.root / "data/train.csv"),
        val=str(run.root / "data/val.csv"),
        test=str(run.root / "data/test.csv"),
    )
    registered = manifest.register(run, Stage.SPLIT, absolute)
    assert {e.path for e in registered.entries} == {
        Path("data/train.csv"),
        Path("data/val.csv"),
        Path("data/test.csv"),
    }


def test_register_resolves_relative_paths_against_the_run_folder(
    run, manifest, tmp_path, monkeypatch
):
    write_split_files(run)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert len(manifest.register(run, Stage.SPLIT, split_output()).entries) == 3


@pytest.mark.parametrize("bad", ["../outside.csv", "/etc/hosts"])
def test_register_refuses_files_outside_the_run_folder(run, manifest, tmp_path, bad):
    write_split_files(run)
    (tmp_path / "outside.csv").write_text("x")
    with pytest.raises(ManifestError, match="outside the run folder"):
        manifest.register(run, Stage.SPLIT, split_output(train=bad))


def test_register_refuses_a_missing_file(run, manifest):
    write(run, "data/train.csv")
    with pytest.raises(ManifestError, match="data/val.csv: not a file"):
        manifest.register(run, Stage.SPLIT, split_output())


def test_register_refuses_a_folder(run, manifest):
    write_split_files(run)
    (run.root / "data" / "folder").mkdir()
    with pytest.raises(ManifestError, match="not a file"):
        manifest.register(run, Stage.SPLIT, split_output(train="data/folder"))


def test_registering_a_file_again_by_the_same_agent_replaces_its_entry(run, manifest):
    write_split_files(run)
    first = manifest.register(run, Stage.SPLIT, split_output())
    write(run, "data/train.csv", "train v2")
    second = first.register(run, Stage.SPLIT, split_output())
    assert len(second.entries) == 3
    assert {e.path: e.sha256 for e in second.entries}[Path("data/train.csv")] == sha("train v2")


def test_a_file_cannot_be_claimed_by_two_agents(run, manifest):
    write_split_files(run)
    registered = manifest.register(run, Stage.SPLIT, split_output())
    with pytest.raises(ManifestError, match="already produced by split"):
        registered.register(run, Stage.PREPROCESSING, split_output())


def test_a_failed_output_registers_nothing(run, manifest):
    failed = SplitOutput(status=Status.FAILED, reason="schema check failed")
    assert manifest.register(run, Stage.SPLIT, failed) == manifest


def test_produced_by_lists_one_agents_files(run, manifest):
    write_split_files(run)
    write(run, "profile/profile.json")
    registered = manifest.register(run, Stage.SPLIT, split_output())
    registered = registered.updated(
        entries=[
            *registered.entries,
            ManifestEntry(
                path=Path("profile/profile.json"), sha256=sha("data"), agent=Stage.PREPROCESSING
            ),
        ]
    )
    assert [e.path for e in registered.produced_by(Stage.PREPROCESSING)] == [
        Path("profile/profile.json")
    ]
    assert len(registered.produced_by(Stage.SPLIT)) == 3
    assert registered.produced_by(Stage.TRAINING) == []


# --- manifest validation ----------------------------------------------------------------------


def test_a_file_can_appear_only_once(manifest):
    entry = ManifestEntry(path=Path("data/a.csv"), sha256=sha("a"), agent=Stage.SPLIT)
    with pytest.raises(ValidationError, match="only once"):
        manifest.updated(entries=[entry, entry])


@pytest.mark.parametrize(
    "fields",
    [
        {"path": Path("/abs/file.csv")},
        {"path": Path("../file.csv")},
        {"path": Path("data/../../file.csv")},
        {"path": Path(".")},
        {"sha256": "abc"},
        {"sha256": "G" * 64},
        {"agent": "not_a_stage"},
    ],
    ids=["absolute", "parent", "nested parent", "empty", "short hash", "non-hex hash", "bad agent"],
)
def test_manifest_entries_reject_bad_values(fields):
    values = {"path": Path("data/a.csv"), "sha256": sha("a"), "agent": Stage.SPLIT}
    with pytest.raises(ValidationError):
        ManifestEntry(**{**values, **fields})


def test_the_start_time_must_carry_a_timezone(manifest):
    with pytest.raises(ValidationError):
        manifest.updated(created_at=datetime(2026, 10, 14, 10, 15))


# --- verify -----------------------------------------------------------------------------------


@pytest.fixture
def registered(run, manifest):
    write_split_files(run)
    return manifest.register(run, Stage.SPLIT, split_output())


def test_a_clean_run_verifies(run, registered):
    assert registered.verify(run) == []


def test_verify_reports_a_missing_file(run, registered):
    (run.root / "data" / "val.csv").unlink()
    assert registered.verify(run) == [Problem(path=Path("data/val.csv"), kind="missing")]


def test_verify_reports_a_changed_file(run, registered):
    write(run, "data/test.csv", "tampered")
    assert registered.verify(run) == [Problem(path=Path("data/test.csv"), kind="changed")]


def test_verify_reports_every_problem(run, registered):
    (run.root / "data" / "train.csv").unlink()
    write(run, "data/val.csv", "tampered")
    problems = registered.verify(run)
    assert {(p.path, p.kind) for p in problems} == {
        (Path("data/train.csv"), "missing"),
        (Path("data/val.csv"), "changed"),
    }


def test_verify_checks_the_config_too(run, registered):
    run.config_path.write_text(run.config_path.read_text() + "# edited\n")
    assert registered.verify(run) == [Problem(path=Path("config.yaml"), kind="changed")]
    run.config_path.unlink()
    assert registered.verify(run) == [Problem(path=Path("config.yaml"), kind="missing")]


def test_a_file_nobody_registered_is_not_a_problem(run, registered):
    write(run, "data/scratch.csv")
    assert registered.verify(run) == []


# --- save and load ----------------------------------------------------------------------------


def test_save_and_load_round_trip(run, registered):
    registered.save(run)
    assert run.load_manifest() == registered
    assert run.load_manifest().verify(run) == []


def test_the_saved_manifest_lists_path_hash_and_agent(run, registered):
    registered.save(run)
    first = json.loads(run.manifest_path.read_text())["entries"][0]
    assert first == {"path": "data/train.csv", "sha256": sha("train"), "agent": "split"}


def test_a_corrupt_manifest_is_an_error(run):
    run.manifest_path.write_text('{"run_id": ')
    with pytest.raises(ValidationError):
        run.load_manifest()


# --- a whole run ------------------------------------------------------------------------------


def test_a_run_can_be_audited_after_it_finishes(run):
    state = run.load_state().start(Stage.SPLIT)
    write_split_files(run)
    manifest = run.load_manifest().register(run, Stage.SPLIT, split_output())
    state = state.finish(Stage.SPLIT, Status.SUCCESS)
    state.save(run.state_path)
    manifest.save(run)

    reloaded_state, reloaded_manifest = run.load_state(), run.load_manifest()
    assert reloaded_state.is_done(Stage.SPLIT)
    assert reloaded_manifest.verify(run) == []

    write(run, "data/train.csv", "changed after the run")
    assert [p.kind for p in reloaded_manifest.verify(run)] == ["changed"]



def test_relative_runs_directory_produces_unambiguous_artifact_paths(config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = create_run(config, runs_dir=Path("runs"))
    assert run.root.is_absolute()
    artifact = run.subdir("data") / "train.csv"
    assert run.relative(artifact) == Path("data/train.csv")
    assert run.relative(Path("data/train.csv")) == Path("data/train.csv")
