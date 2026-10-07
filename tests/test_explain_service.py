from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ml_workbench.services.explain_service import ExplainError, explain_model, shap_available
from ml_workbench.services.jobs import JobQueue
from ml_workbench.services.training_service import train_model


def _train(state, ws, model_id: str, params: dict) -> str:
    return train_model(state, ws, model_id, params).run_id


def test_explain_linear_coefficients_cached(prepared_workspace, binary_frame: pd.DataFrame) -> None:
    state, ws = prepared_workspace(binary_frame, target="target", task_type="binary")
    run_id = _train(
        state, ws, "logistic_regression", {"C": 1.0, "penalty": "l2", "solver": "lbfgs"}
    )
    first = explain_model(state, ws, run_id)
    assert first.cached is False
    assert first.method == "linear"
    panels = {panel["panel"]: panel for panel in first.panels}
    assert panels["coefficients"]["top_features"]
    assert panels["local_attribution"]["attribution"]

    cache_path = Path(first.cache_file)
    assert cache_path.is_file()

    second = explain_model(state, ws, run_id)
    assert second.cached is True
    assert second.method == "linear"
    payload = json.loads(cache_path.read_text())
    assert payload["model_id"] == "logistic_regression"
    assert payload["background_rows"] == first.background_rows


def test_explain_tree_uses_sklearn_importances_without_shap(
    prepared_workspace, binary_frame: pd.DataFrame
) -> None:
    state, ws = prepared_workspace(binary_frame, target="target", task_type="binary")
    run_id = _train(state, ws, "decision_tree", {"max_depth": 3, "criterion": "gini"})
    report = explain_model(state, ws, run_id)
    if shap_available():
        assert report.method == "tree_shap"
    else:
        assert report.method == "tree_importances"
        panels = {panel["panel"]: panel for panel in report.panels}
        assert "table" in panels["importances"]
        assert panels["importances"]["bias_note"]


def test_explain_runs_through_job_queue(prepared_workspace, binary_frame: pd.DataFrame) -> None:
    state, ws = prepared_workspace(binary_frame, target="target", task_type="binary")
    run_id = _train(
        state, ws, "logistic_regression", {"C": 1.0, "penalty": "l2", "solver": "lbfgs"}
    )
    queue = JobQueue(max_workers=1)
    try:
        handle = queue.submit(
            "shap", lambda update: explain_model(state, ws, run_id, on_progress=update)
        )
        done = queue.wait(handle.job_id, timeout=30)
        assert done.state == "done"
        assert done.kind == "shap"
        assert done.result.cached is False
        assert done.progress == 1.0
    finally:
        queue.shutdown()


def test_explain_requires_trained_run(prepared_workspace, binary_frame: pd.DataFrame) -> None:
    state, ws = prepared_workspace(binary_frame, target="target", task_type="binary")
    with pytest.raises(ExplainError):
        explain_model(state, ws, "run_00000000")
