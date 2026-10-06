from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_workbench.rules.eda import EDA_MAX_ROWS
from ml_workbench.rules.task import build_task
from ml_workbench.services.eda_service import eda_report
from ml_workbench.state import TaskDefinition


def _cls_task() -> TaskDefinition:
    frame = _cls_frame()
    return build_task(frame, "supervised", target="churned", task_type="binary")


def _cls_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "income": [50.0, 60.0, 80.0, 40.0, np.nan, 90.0, 55.0, 70.0],
            "region": ["north", "south", "north", "west", "south", "south", "west", "north"],
            "churned": [0, 1, 0, 1, 0, 1, 1, 0],
        }
    )


def test_eda_base_report_shape_dtypes_missing() -> None:
    frame = _cls_frame()
    task = _cls_task()
    report = eda_report(frame, task)
    assert report.base.shape == (8, 3)
    assert report.base.rows == 8
    assert report.base.columns == 3
    assert report.base.missing == {"income": 1}
    assert report.base.numeric_describe is not None
    assert report.base.numeric_pearson is not None
    assert report.base.numeric_spearman is not None


def test_eda_cramers_v_for_categorical_pairs() -> None:
    frame = pd.DataFrame(
        {
            "occupation": ["a", "b", "a", "b", "a", "b", "a", "b"],
            "region": ["north", "north", "west", "west", "south", "south", "north", "south"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    task = build_task(frame, "supervised", target="target", task_type="binary")
    report = eda_report(frame, task)
    assert report.base.cramers_v is not None
    matrix = report.base.cramers_v
    assert set(matrix.columns) == {"occupation", "region"}
    assert float(matrix.loc["occupation", "region"]) >= 0.0


def test_eda_classification_views_and_chi_square() -> None:
    frame = _cls_frame()
    report = eda_report(frame, _cls_task())
    assert report.classification is not None
    counts = report.classification.class_counts
    assert "count" in counts.columns
    assert "fraction" in counts.columns
    assert float(counts["fraction"].sum()) == pytest.approx(1.0)
    assert report.classification.chi_square is not None
    assert report.classification.feature_by_class  # at least the region crosstab


def test_eda_regression_views_skew_hint() -> None:
    frame = pd.DataFrame(
        {
            "x": range(40),
            "price": [1.5 ** (index + 1) + 1.0 for index in range(40)],
        }
    )
    task = build_task(frame, "supervised", target="price", task_type="regression")
    report = eda_report(frame, task)
    assert report.regression is not None
    assert report.regression.target_skew > 1.0
    assert report.regression.skew_hint is not None
    assert "log transform is suggested" in report.regression.skew_hint


def test_eda_regression_symmetric_target_no_hint() -> None:
    frame = pd.DataFrame({"x": range(40), "price": [float(5 + index**0.5) for index in range(40)]})
    task = build_task(frame, "supervised", target="price", task_type="regression")
    report = eda_report(frame, task)
    assert report.regression.skew_hint is None


def test_eda_sampling_large_frames() -> None:
    rows = EDA_MAX_ROWS + 5000
    frame = pd.DataFrame({"x": range(rows), "y": [float(v % 7) for v in range(rows)]})
    task = build_task(frame, "supervised", target="y", task_type="multiclass")
    report = eda_report(frame, task, seed=3)
    assert report.base.rows == rows
    assert report.base.sampled_rows == EDA_MAX_ROWS
