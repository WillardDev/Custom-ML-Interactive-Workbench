from __future__ import annotations

import sqlite3
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from openpyxl import Workbook

from ml_workbench.services import data_service
from ml_workbench.services.data_service import (
    DataError,
    insert_dataset,
    list_samples,
    load_bytes,
    load_path,
    load_sqlite,
    read_sample,
    schema_report,
    sqlite_tables,
    validate_size,
)
from ml_workbench.services.workspace import Workspace, WorkspaceError
from ml_workbench.state import ProjectState

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"


def test_sample_list_and_read() -> None:
    samples = list_samples()
    assert "binary_classification.csv" in samples
    frame = read_sample("binary_classification.csv")
    assert frame.shape == (400, 6)
    with pytest.raises(DataError):
        read_sample("../../secrets.csv")


def test_load_path_dispatch() -> None:
    csv_path = SAMPLES_DIR / "binary_classification.csv"
    assert load_path(csv_path).shape == (400, 6)
    with pytest.raises(DataError):
        load_path(SAMPLES_DIR / "missing.csv")
    with pytest.raises(DataError):
        load_path(Path("/tmp/archive.zip"))


def test_validate_size() -> None:
    validate_size(0)
    with pytest.raises(DataError):
        validate_size(data_service.MAX_UPLOAD_BYTES + 1)


def test_load_bytes_csv_tsv() -> None:
    base = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    for suffix, sep in ((".csv", ","), (".tsv", "\t")):
        text = base.to_csv(sep=sep, index=False).encode()
        loaded = load_bytes(text, f"upload{suffix}")
        assert loaded.shape == (2, 2)


def test_load_bytes_excel() -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["a", "b"])
    sheet.append([1, "x"])
    sheet.append([2, "y"])
    buffer = BytesIO()
    workbook.save(buffer)
    loaded = load_bytes(buffer.getvalue(), "upload.xlsx")
    assert loaded.shape == (2, 2)


def test_load_bytes_parquet() -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    buffer = BytesIO()
    frame.to_parquet(buffer, index=False)
    loaded = load_bytes(buffer.getvalue(), "upload.parquet")
    assert loaded.shape == (2, 2)


def test_load_bytes_unsupported() -> None:
    with pytest.raises(DataError):
        load_bytes(b"payload", "file.xyz")


def test_sqlite_helpers(tmp_path: Path) -> None:
    db = tmp_path / "sample.sqlite"
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE people (id INT, name TEXT)")
    connection.executemany("INSERT INTO people (id, name) VALUES (?, ?)", [(1, "a"), (2, "b")])
    connection.commit()
    connection.close()

    assert sqlite_tables(db) == ["people"]
    assert load_sqlite(db, "SELECT * FROM people").shape == (2, 2)
    assert load_bytes(db.read_bytes(), "sample.sqlite").shape == (2, 2)


def test_schema_report() -> None:
    frame = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0] + [None] * 22,
            "cat": ["a", "b", "a"] + [None] * 22,
            "dt": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] + [None] * 22),
            "txt": [f"unique value number {index}" for index in range(25)],
        }
    )
    report = {entry["column"]: entry for entry in schema_report(frame)}
    assert report["num"]["kind"] == "numeric"
    assert report["num"]["missing"] == 22
    assert report["num"]["missing_pct"] == pytest.approx(22 / 25)
    assert report["num"]["unique"] == 3
    assert report["cat"]["kind"] == "categorical"
    assert report["dt"]["kind"] == "datetime"
    assert report["txt"]["kind"] == "text"
    assert report["txt"]["unique"] == 25


def test_workspaces_reject_unsafe_project_ids(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceError):
        Workspace(project_id="../../etc/passwd", root=tmp_path)


def test_insert_dataset_versions_and_reset(tmp_path: Path) -> None:
    workspace = Workspace(project_id="p_data", root=tmp_path)
    state = ProjectState(project_id="p_data")
    frame = pd.DataFrame({"a": [1.0, None], "b": ["x", "y"]})

    version = insert_dataset(state, workspace, frame, source="test.csv")
    assert version.version == "v00001"
    assert state.dataset is not None and state.dataset.version == "v00001"
    assert state.task is None

    parquet = workspace.datasets_dir / "v00001.parquet"
    sidecar = workspace.datasets_dir / "v00001.schema.json"
    assert parquet.is_file() and sidecar.is_file()

    info = workspace.read_version_info("v00001")
    assert info.data_hash.startswith("sha256:")
    assert info.data_hash == state.dataset.data_hash
    assert info.schema

    reloaded = workspace.read_version("v00001")
    assert reloaded.equals(frame)
    assert len(state.steps) == 1 and state.steps[0].op == "load"

    insert_dataset(state, workspace, frame, source="test2.csv")
    assert state.dataset is not None and state.dataset.version == "v00002"
    assert len(state.steps) == 1 and state.steps[0].params["source"] == "test2.csv"
    assert workspace.list_versions() == ["v00001", "v00002"]

    with pytest.raises(DataError):
        insert_dataset(state, workspace, pd.DataFrame({"a": []}), source="empty.csv")
