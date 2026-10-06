from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd
import pyarrow  # noqa: F401  (imported for the writer version string)

from ml_workbench.state import StepEntry

VERSION_RE: Final = re.compile(r"^v\d{5}$")
RUN_ID_RE: Final = re.compile(r"^run_[A-Za-z0-9]{8}$")
PROJECT_ID_RE: Final = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
STEPS_LOG_VERSION: Final = 1
PARQUET_COMPRESSION: Final = "snappy"


class WorkspaceError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class VersionInfo:
    version: str
    path: Path
    data_hash: str
    row_count: int
    loaded_at: str
    schema: tuple[dict[str, Any], ...] = ()
    writer: str = ""


@dataclass(frozen=True)
class Workspace:
    project_id: str
    root: Path

    def __post_init__(self) -> None:
        if not PROJECT_ID_RE.match(self.project_id):
            raise WorkspaceError(
                f"invalid project id '{self.project_id}' (allowed: letters, digits, '_', '-')"
            )

    @property
    def project_dir(self) -> Path:
        return self.root / "projects" / self.project_id

    @property
    def datasets_dir(self) -> Path:
        return self.project_dir / "datasets"

    @property
    def models_dir(self) -> Path:
        return self.project_dir / "models"

    @property
    def preprocessing_dir(self) -> Path:
        return self.project_dir / "preprocessing"

    @property
    def cache_dir(self) -> Path:
        return self.project_dir / "cache"

    @property
    def reports_dir(self) -> Path:
        return self.project_dir / "reports"

    @property
    def steps_path(self) -> Path:
        return self.project_dir / "steps.json"

    def ensure(self) -> None:
        self.project_dir.mkdir(parents=True, exist_ok=True)
        for directory in (
            self.datasets_dir,
            self.models_dir,
            self.preprocessing_dir,
            self.cache_dir,
            self.reports_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def version_path(self, version: str) -> Path:
        if not VERSION_RE.match(version):
            raise WorkspaceError(f"invalid version '{version}'")
        return self.datasets_dir / f"{version}.parquet"

    def schema_path(self, version: str) -> Path:
        if not VERSION_RE.match(version):
            raise WorkspaceError(f"invalid version '{version}'")
        return self.datasets_dir / f"{version}.schema.json"

    def list_versions(self) -> list[str]:
        self.ensure()
        return sorted(
            path.stem
            for path in self.datasets_dir.glob("v*.parquet")
            if VERSION_RE.match(path.stem)
        )

    def write_version(
        self, frame: pd.DataFrame, *, schema: Sequence[dict[str, Any]] | None = None
    ) -> VersionInfo:
        self.ensure()
        versions = self.list_versions()
        number = int(versions[-1][1:]) + 1 if versions else 1
        version = f"v{number:05d}"
        path = self.version_path(version)
        frame.to_parquet(path, index=False, compression=PARQUET_COMPRESSION)
        info = VersionInfo(
            version=version,
            path=path,
            data_hash=hash_file(path),
            row_count=len(frame),
            loaded_at=utc_now(),
            schema=tuple(schema or ()),
            writer=f"pyarrow/{pyarrow.__version__}",
        )
        payload = {
            "version": info.version,
            "data_hash": info.data_hash,
            "row_count": info.row_count,
            "loaded_at": info.loaded_at,
            "writer": info.writer,
            "schema": list(info.schema),
        }
        self.schema_path(version).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return info

    def read_version(self, version: str) -> pd.DataFrame:
        path = self.version_path(version)
        if not path.is_file():
            raise WorkspaceError(f"version '{version}' not found")
        return pd.read_parquet(path)

    def read_version_info(self, version: str) -> VersionInfo:
        sidecar = self.schema_path(version)
        if not sidecar.is_file():
            raise WorkspaceError(f"schema sidecar for '{version}' not found")
        payload = json.loads(sidecar.read_text())
        return VersionInfo(
            version=version,
            path=self.version_path(version),
            data_hash=str(payload["data_hash"]),
            row_count=int(payload["row_count"]),
            loaded_at=str(payload["loaded_at"]),
            schema=tuple(payload.get("schema", ())),
            writer=str(payload.get("writer", "")),
        )

    def read_steps(self) -> list[StepEntry]:
        if not self.steps_path.is_file():
            return []
        payload = json.loads(self.steps_path.read_text())
        return [
            StepEntry(
                id=int(entry["id"]),
                tab=str(entry["tab"]),
                op=str(entry["op"]),
                params=dict(entry.get("params", {})),
                dataset_hash_before=str(entry.get("dataset_hash_before", "")),
                dataset_hash_after=str(entry.get("dataset_hash_after", "")),
                created_at=str(entry.get("created_at", "")),
            )
            for entry in payload.get("entries", [])
        ]

    def write_steps(self, entries: Sequence[StepEntry]) -> None:
        self.ensure()
        payload = {
            "version": STEPS_LOG_VERSION,
            "project_id": self.project_id,
            "entries": [asdict(entry) for entry in entries],
        }
        self.steps_path.write_text(json.dumps(payload, indent=2) + "\n")

    def append_step(self, entry: StepEntry) -> list[StepEntry]:
        entries = self.read_steps()
        entries.append(entry)
        self.write_steps(entries)
        return entries

    def pop_step(self, tab: str | None = None) -> StepEntry | None:
        entries = self.read_steps()
        index: int | None = None
        for position in range(len(entries) - 1, -1, -1):
            if tab is None or entries[position].tab == tab:
                index = position
                break
        if index is None:
            return None
        popped = entries.pop(index)
        self.write_steps(entries)
        return popped

    def run_dir(self, run_id: str) -> Path:
        if not RUN_ID_RE.match(run_id):
            raise WorkspaceError(f"invalid run id '{run_id}'")
        return self.models_dir / run_id

    def write_run_json(self, run_id: str, name: str, payload: dict[str, object]) -> Path:
        path = self.run_dir(run_id) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return path

    def read_run_json(self, run_id: str, name: str) -> dict[str, Any]:
        path = self.run_dir(run_id) / name
        if not path.is_file():
            raise WorkspaceError(f"'{name}' for run '{run_id}' not found")
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise WorkspaceError(f"'{name}' for run '{run_id}' is not a JSON object")
        return payload
