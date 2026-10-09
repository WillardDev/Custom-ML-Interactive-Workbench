from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_workbench.services.error_service import ErrorAnalysisError, error_views
from ml_workbench.services.training_service import train_model


def _train(state, ws, model_id: str, params: dict) -> str:
    result = train_model(state, ws, model_id, params)
    return result.run_id


def test_error_views_binary_confusion_roc_and_worst_n(
    prepared_workspace, binary_frame: pd.DataFrame
) -> None:
    state, ws = prepared_workspace(binary_frame, target="target", task_type="binary")
    run_id = _train(
        state, ws, "logistic_regression", {"C": 1.0, "penalty": "l2", "solver": "lbfgs"}
    )
    report = error_views(state, ws, run_id)
    assert report.task_type == "binary"
    view_names = [str(view["view"]) for view in report.views]
    assert "confusion_matrix" in view_names and "roc" in view_names
    assert "pr_curve" in view_names and "calibration" in view_names

    confusion = next(v for v in report.views if str(v["view"]) == "confusion_matrix")["confusion"]
    assert confusion.shape == (2, 2)

    roc = next(v for v in report.views if str(v["view"]) == "roc")
    assert 0 <= roc["auc"] <= 1
    assert len(roc["fpr"]) == len(roc["tpr"])

    threshold = next(v for v in report.views if str(v["view"]) == "threshold_analysis")
    assert len(threshold["rows"]) == 9

    calibration = next(v for v in report.views if str(v["view"]) == "calibration")
    assert len(calibration["centers"]) == 10

    assert not report.worst_n.empty and len(report.worst_n) <= 10
    assert "true" in report.worst_n.columns and "prediction" in report.worst_n.columns
    assert "error" in report.worst_n.columns
    assert list(report.worst_n["error"]) == sorted(report.worst_n["error"], reverse=True)

    assert report.segments is not None and not report.segments.empty
    assert "mean_abs_error" in report.segments.columns


def test_error_views_binary_string_labels(
    prepared_workspace, string_binary_frame: pd.DataFrame
) -> None:
    state, ws = prepared_workspace(string_binary_frame, target="target", task_type="binary")
    run_id = _train(
        state, ws, "logistic_regression", {"C": 1.0, "penalty": "l2", "solver": "lbfgs"}
    )
    report = error_views(state, ws, run_id)
    assert report.task_type == "binary"

    confusion = next(v for v in report.views if str(v["view"]) == "confusion_matrix")["confusion"]
    assert list(confusion.index) == ["true_No", "true_Yes"]
    assert list(confusion.columns) == ["pred_No", "pred_Yes"]

    roc = next(v for v in report.views if str(v["view"]) == "roc")
    assert 0 <= roc["auc"] <= 1

    threshold = next(v for v in report.views if str(v["view"]) == "threshold_analysis")
    assert len(threshold["rows"]) == 9

    assert not report.worst_n.empty
    assert list(report.worst_n["error"]) == sorted(report.worst_n["error"], reverse=True)

    assert report.segments is not None and not report.segments.empty


def test_error_views_regression_residuals(prepared_workspace) -> None:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "x": rng.uniform(1, 10, size=60),
            "label": rng.choice(["a", "b"], size=60),
            "target": 2.5 * rng.uniform(1, 10, size=60) + 1,
        }
    )
    state, ws = prepared_workspace(frame, target="target", task_type="regression")
    run_id = _train(state, ws, "linear_regression", {"fit_intercept": True})
    report = error_views(state, ws, run_id)
    names = [str(view["view"]) for view in report.views]
    assert "residuals_vs_predicted" in names
    assert "residual_histogram" in names
    assert "qq_plot" in names
    assert "heteroscedasticity" in names
    assert "error_by_target_quantile" in names

    residuals = next(v for v in report.views if str(v["view"]) == "residuals_vs_predicted")[
        "points"
    ]
    assert len(residuals) == len(report.predicted)
    hetero = next(v for v in report.views if str(v["view"]) == "heteroscedasticity")
    assert -1 <= hetero["spearman"] <= 1


def test_error_views_requires_split(prepared_workspace, binary_frame: pd.DataFrame) -> None:
    state, ws = prepared_workspace(binary_frame, target="target", task_type="binary")
    state.split = None
    from ml_workbench.services.error_service import error_views

    with pytest.raises(ErrorAnalysisError):
        error_views(state, ws, "run_00000000")
