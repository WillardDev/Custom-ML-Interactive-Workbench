from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = REPO_ROOT / "data" / "samples"

EXPECTED = [
    "binary_classification.csv",
    "regression.csv",
    "clustering.csv",
    "messy_classification.csv",
]


def _rows(name: str) -> list[list[str]]:
    with (SAMPLES_DIR / name).open(newline="") as handle:
        return list(csv.reader(handle))


def test_all_sample_files_exist() -> None:
    for name in EXPECTED:
        assert (SAMPLES_DIR / name).is_file(), name


def test_rows_are_consistent_and_nonempty() -> None:
    for name in EXPECTED:
        rows = _rows(name)
        width = len(rows[0])
        assert width > 1, name
        assert len(rows) > 100, name
        assert all(len(row) == width for row in rows), name


def test_messy_data_has_missing_values_and_duplicates() -> None:
    rows = _rows("messy_classification.csv")
    header, body = rows[0], rows[1:]
    assert any(cell == "" or cell == "N/A" for row in body for cell in row)
    assert len(body) != len({tuple(row) for row in body}), "expected duplicate rows"
    assert header[0] == "customer_id"


def test_clustering_labels_are_scoped() -> None:
    rows = _rows("clustering.csv")
    labels = {row[2] for row in rows[1:]}
    assert labels == {"0", "1", "2"}


def test_targets_look_binary_in_classification_sample() -> None:
    rows = _rows("binary_classification.csv")
    targets = {row[5] for row in rows[1:]}
    assert targets == {"0", "1"}
