from __future__ import annotations

import io
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Final

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

from ml_workbench.rules.staleness import mark_downstream_stale
from ml_workbench.services.workspace import VersionInfo, Workspace, utc_now
from ml_workbench.state import DatasetInfo, ProjectState, StepEntry

MAX_UPLOAD_BYTES: Final = 200 * 1024 * 1024
REPO_ROOT: Final = Path(__file__).resolve().parents[3]
SAMPLES_DIR: Final = REPO_ROOT / "data" / "samples"
CATEGORICAL_CARDINALITY: Final = 20  # ⚠ presentation heuristic, not a rule

TABLE_SUFFIXES: Final = frozenset({".csv", ".tsv", ".txt"})
EXCEL_SUFFIXES: Final = frozenset({".xlsx", ".xls"})
SQLITE_SUFFIXES: Final = frozenset({".sqlite", ".sqlite3", ".db"})
SUPPORTED_SUFFIXES: Final = TABLE_SUFFIXES | EXCEL_SUFFIXES | {".parquet"} | SQLITE_SUFFIXES


class DataError(ValueError):
    pass


def validate_size(size_bytes: int) -> None:
    if size_bytes > MAX_UPLOAD_BYTES:
        raise DataError(
            f"file is {size_bytes:,} bytes; the upload limit is {MAX_UPLOAD_BYTES:,} bytes"
        )


def list_samples() -> list[str]:
    return sorted(path.name for path in SAMPLES_DIR.glob("*.csv"))


def read_sample(name: str) -> pd.DataFrame:
    if name not in list_samples():
        raise DataError(f"unknown sample dataset '{name}'")
    return pd.read_csv(SAMPLES_DIR / name)


def load_path(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise DataError(f"file not found: {path}")
    validate_size(path.stat().st_size)
    suffix = path.suffix.lower()
    if suffix in TABLE_SUFFIXES:
        return pd.read_csv(path, sep="\t" if suffix in {".tsv", ".txt"} else ",")
    if suffix in EXCEL_SUFFIXES:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in SQLITE_SUFFIXES:
        raise DataError("sqlite sources need a query: use the SQLite source option")
    raise DataError(
        f"unsupported file type '{suffix}' (allowed: {', '.join(sorted(SUPPORTED_SUFFIXES))})"
    )


def load_bytes(data: bytes, filename: str) -> pd.DataFrame:
    validate_size(len(data))
    suffix = Path(filename).suffix.lower()
    if suffix in TABLE_SUFFIXES:
        return pd.read_csv(io.BytesIO(data), sep="\t" if suffix in {".tsv", ".txt"} else ",")
    if suffix in EXCEL_SUFFIXES:
        return pd.read_excel(io.BytesIO(data))
    if suffix == ".parquet":
        return pd.read_parquet(io.BytesIO(data))
    if suffix in SQLITE_SUFFIXES:
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as handle:
            handle.write(data)
            temp_path = Path(handle.name)
        try:
            return _read_sqlite_path(temp_path, sqlite_tables(temp_path)[0])
        finally:
            temp_path.unlink(missing_ok=True)
    raise DataError(
        f"unsupported file type '{suffix}' (allowed: {', '.join(sorted(SUPPORTED_SUFFIXES))})"
    )


def sqlite_tables(path: Path) -> list[str]:
    if not path.is_file():
        raise DataError(f"file not found: {path}")
    try:
        connection = sqlite3.connect(path)
    except sqlite3.Error as exc:
        raise DataError(f"not a valid SQLite database: {exc}") from exc
    try:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ]
        if not tables:
            raise DataError("the SQLite database contains no tables")
        return tables
    except sqlite3.Error as exc:
        raise DataError(f"could not read SQLite tables: {exc}") from exc
    finally:
        connection.close()


def load_sqlite(path: Path, query: str) -> pd.DataFrame:
    validate_size(path.stat().st_size)
    try:
        connection = sqlite3.connect(path)
        try:
            return pd.read_sql_query(query, connection)
        except sqlite3.Error as exc:
            raise DataError(f"SQL error: {exc}") from exc
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise DataError(f"could not open SQLite database: {exc}") from exc


def _read_sqlite_path(path: Path, table: str) -> pd.DataFrame:
    connection = sqlite3.connect(path)
    try:
        return pd.read_sql_query(f'SELECT * FROM "{table}"', connection)
    except sqlite3.Error as exc:
        raise DataError(f"SQL error: {exc}") from exc
    finally:
        connection.close()


def write_temp_bytes(data: bytes, suffix: str) -> Path:
    """Write upload bytes to a temp file; the caller must unlink the result."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(data)
        return Path(handle.name)


def infer_kind(series: pd.Series) -> str:
    if is_bool_dtype(series):
        return "boolean"
    if is_datetime64_any_dtype(series):
        return "datetime"
    if is_numeric_dtype(series):
        return "numeric"
    if is_string_dtype(series) or is_object_dtype(series):
        unique = int(series.nunique(dropna=True))
        return "categorical" if unique <= CATEGORICAL_CARDINALITY else "text"
    return "other"


def schema_report(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in frame.columns:
        series = frame[column]
        missing = int(series.isna().sum())
        rows.append(
            {
                "column": str(column),
                "dtype": str(series.dtype),
                "kind": infer_kind(series),
                "missing": missing,
                "missing_pct": round(missing / len(frame), 4) if len(frame) else 0.0,
                "unique": int(series.nunique(dropna=True)),
            }
        )
    return rows


def insert_dataset(
    state: ProjectState, workspace: Workspace, frame: pd.DataFrame, *, source: str
) -> VersionInfo:
    if frame.empty:
        raise DataError("the dataset is empty (0 rows); load a file with at least one row")
    version = workspace.write_version(frame, schema=schema_report(frame))
    entry = StepEntry(
        id=1,
        tab="data",
        op="load",
        params={"source": source, "version": version.version, "rows": version.row_count},
        dataset_hash_before="",
        dataset_hash_after=version.data_hash,
        created_at=utc_now(),
    )
    workspace.write_steps([entry])
    state.dataset = DatasetInfo(
        version=version.version,
        path=str(version.path),
        data_hash=version.data_hash,
        row_count=version.row_count,
        loaded_at=version.loaded_at,
    )
    state.frame = frame
    state.task = None
    state.split = None
    state.pipeline = None
    state.models = []
    state.active_model_id = None
    state.steps = workspace.read_steps()
    mark_downstream_stale(state, "data")
    return version
