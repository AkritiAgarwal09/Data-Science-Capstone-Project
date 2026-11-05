"""Run folder and manifest: where a run's files live, and how to show they are unchanged.

`create_run` makes runs/<run_id>/ with its config, an initial state and an empty manifest. As each
agent finishes, `Manifest.register` hashes the files its output lists. `Manifest.verify` later
confirms every file is still there and unchanged (architecture.md section 4, success criterion).

Paths in an agent output are relative to the run folder, or absolute inside it. Anything outside
the folder is refused, so an agent cannot write results somewhere the manifest does not cover.
"""

import platform
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from agentic_gan.orchestrator.state import STATE_FILE, RunState
from agentic_gan.schemas import Outcome, RunConfig, RunId, Schema, Stage
from agentic_gan.utils.files import sha256_file, write_text_atomic

SUBDIRS = ("data", "profile", "synthetic", "quality", "models", "explain", "report")
CONFIG_FILE = "config.yaml"
MANIFEST_FILE = "manifest.json"
DECISIONS_FILE = "decisions.jsonl"

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ManifestError(ValueError):
    """A file cannot be registered: it is missing, outside the run folder, or claimed twice."""


@dataclass(frozen=True)
class RunDir:
    """The runs/<run_id>/ folder and the fixed files inside it."""

    root: Path

    @property
    def run_id(self) -> str:
        return self.root.name

    @property
    def config_path(self) -> Path:
        return self.root / CONFIG_FILE

    @property
    def state_path(self) -> Path:
        return self.root / STATE_FILE

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_FILE

    @property
    def decisions_path(self) -> Path:
        return self.root / DECISIONS_FILE

    def subdir(self, name: str) -> Path:
        if name not in SUBDIRS:
            raise ValueError(f"unknown run subfolder {name!r}; expected one of {SUBDIRS}")
        return self.root / name

    def relative(self, path: Path) -> Path:
        """Path of a file relative to the run folder. Raises ManifestError if it is outside."""
        try:
            return (self.root / path).resolve().relative_to(self.root.resolve())
        except ValueError:
            raise ManifestError(f"{path}: outside the run folder {self.root}") from None

    def load_config(self) -> RunConfig:
        return RunConfig.from_file(self.config_path)

    def load_state(self) -> RunState:
        return RunState.load(self.state_path)

    def load_manifest(self) -> "Manifest":
        return Manifest.load(self)


class ManifestEntry(Schema):
    path: Path
    sha256: Sha256
    agent: Stage

    @field_validator("path")
    @classmethod
    def _relative(cls, value: Path) -> Path:
        if value.is_absolute() or ".." in value.parts or not value.parts:
            raise ValueError("path must be relative to the run folder, without '..'")
        return value


class Problem(Schema):
    """A file the manifest vouches for that is no longer as recorded."""

    path: Path
    kind: Literal["missing", "changed"]


def package_versions() -> dict[str, str]:
    """Python and every declared dependency of this project, as installed."""
    names = ["agentic-gan"]
    try:
        names += [
            re.split(r"[ ;<>=!~\[(]", req, maxsplit=1)[0]
            for req in metadata.requires(names[0]) or []
        ]
    except metadata.PackageNotFoundError:
        pass
    versions = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


class Manifest(Schema):
    """Index of every artifact in a run, plus what is needed to rebuild the run.

    dataset_sha256 and config_sha256 pin the inputs: a run is reproducible from config.yaml and
    the raw file. packages records the environment, which the config does not.
    """

    run_id: RunId
    created_at: AwareDatetime
    dataset_sha256: Sha256
    config_sha256: Sha256
    packages: dict[str, str] = Field(default_factory=dict)
    entries: list[ManifestEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _paths_unique(self) -> Self:
        paths = [e.path for e in self.entries]
        if len(set(paths)) != len(paths):
            raise ValueError("a file can appear in the manifest only once")
        return self

    def register(self, run: RunDir, agent: Stage, output: Outcome) -> Self:
        """Add every file the output lists, hashed now. Nothing is added if any file is refused.

        Registering the same file again by the same agent replaces its entry, as on a retry. A
        file already registered by a different agent is an error.
        """
        entries = {e.path: e for e in self.entries}
        for path in output.artifact_paths():
            rel = run.relative(path)
            if not (run.root / rel).is_file():
                raise ManifestError(f"{rel}: not a file in the run folder")
            claimed = entries.get(rel)
            if claimed is not None and claimed.agent is not agent:
                raise ManifestError(f"{rel}: already produced by {claimed.agent}")
            digest = sha256_file(run.root / rel)
            entries[rel] = ManifestEntry(path=rel, sha256=digest, agent=agent)
        return self.updated(entries=list(entries.values()))

    def verify(self, run: RunDir) -> list[Problem]:
        """Files that are missing or changed since they were registered. Empty means all good.

        config.yaml is checked too, since the run is rebuilt from it.
        """
        expected = [(Path(CONFIG_FILE), self.config_sha256)]
        expected += [(e.path, e.sha256) for e in self.entries]
        problems = []
        for rel, digest in expected:
            file = run.root / rel
            if not file.is_file():
                problems.append(Problem(path=rel, kind="missing"))
            elif sha256_file(file) != digest:
                problems.append(Problem(path=rel, kind="changed"))
        return problems

    def produced_by(self, agent: Stage) -> list[ManifestEntry]:
        return [e for e in self.entries if e.agent is agent]

    def save(self, run: RunDir) -> None:
        write_text_atomic(run.manifest_path, self.model_dump_json(indent=2))

    @classmethod
    def load(cls, run: RunDir) -> Self:
        return cls.model_validate_json(run.manifest_path.read_text(encoding="utf-8"))


def _default_run_id(config: RunConfig, created_at: datetime) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", config.dataset_name).strip("-.") or "run"
    return f"{name}_{created_at:%Y%m%d-%H%M%S}"


def create_run(
    config: RunConfig,
    *,
    run_id: str | None = None,
    runs_dir: Path = Path("runs"),
    now: Callable[[], datetime] | None = None,
) -> RunDir:
    """Create runs/<run_id>/ with its subfolders, config.yaml, state.json and manifest.json.

    The default run id is the dataset name plus the UTC start time. An existing run folder is
    never reused or overwritten. If setup fails part-way, the folder made here is removed.
    """
    created_at = (now or (lambda: datetime.now(UTC)))()
    run_id = _default_run_id(config, created_at) if run_id is None else run_id
    state = RunState.new(run_id)  # rejects an unsafe run id before anything is created
    dataset_sha256 = sha256_file(config.dataset_path)  # a run needs its raw file to be rebuilt

    # Stage outputs use run.subdir(); absolute roots prevent prefixing runs/ twice.
    root = (Path(runs_dir) / run_id).resolve()
    try:
        root.mkdir(parents=True)
    except FileExistsError:
        raise FileExistsError(f"run folder already exists: {root}") from None
    run = RunDir(root)
    try:
        for name in SUBDIRS:
            run.subdir(name).mkdir()
        write_text_atomic(run.config_path, config.to_yaml())
        state.save(run.state_path)
        Manifest(
            run_id=run_id,
            created_at=created_at,
            dataset_sha256=dataset_sha256,
            config_sha256=sha256_file(run.config_path),
            packages=package_versions(),
        ).save(run)
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return run
