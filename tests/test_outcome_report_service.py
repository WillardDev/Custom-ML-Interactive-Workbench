from __future__ import annotations

import json

import joblib
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from ml_workbench.services.outcome_service import OutcomeError, build_supervised_outcome
from ml_workbench.services.report_service import build_report, render_script
from ml_workbench.services.training_service import train_model


def _train(state, ws, model_id: str, params: dict) -> str:
    return train_model(state, ws, model_id, params).run_id


def test_outcome_packages_deliverables(prepared_workspace, binary_frame: pd.DataFrame) -> None:
    state, ws = prepared_workspace(binary_frame, target="target", task_type="binary")
    run_id = _train(
        state, ws, "logistic_regression", {"C": 1.0, "penalty": "l2", "solver": "lbfgs"}
    )
    result = build_supervised_outcome(state, ws, run_id)
    assert result.files

    outcome = ws.outcome_dir
    assert (outcome / "predictions.csv").is_file()
    assert (outcome / "threshold.json").is_file()
    assert (outcome / "model_card.json").is_file()

    predictions = pd.read_csv(outcome / "predictions.csv")
    assert "prediction" in predictions.columns
    assert "prediction@threshold" in predictions.columns
    assert len(predictions) == len(binary_frame)

    card = json.loads((outcome / "model_card.json").read_text())
    assert card["model_id"] == "logistic_regression"
    assert card["target"] == "target"
    assert card["deliverables"]

    threshold = json.loads((outcome / "threshold.json").read_text())
    assert threshold["threshold"] == 0.5 and threshold["applied"] is True

    loaded = joblib.load(outcome / "pipeline.joblib")
    assert hasattr(loaded, "predict")


def test_outcome_refit_on_all_uses_full_data(
    prepared_workspace, binary_frame: pd.DataFrame
) -> None:
    state, ws = prepared_workspace(binary_frame, target="target", task_type="binary")
    run_id = _train(
        state, ws, "logistic_regression", {"C": 1.0, "penalty": "l2", "solver": "lbfgs"}
    )
    result = build_supervised_outcome(state, ws, run_id, refit_on_all=True)
    assert any("pipeline.joblib" in name for name in result.files)
    pipeline = joblib.load(ws.outcome_dir / "pipeline.joblib")
    assert isinstance(pipeline, Pipeline)


def test_outcome_requires_known_run(prepared_workspace, binary_frame: pd.DataFrame) -> None:
    state, ws = prepared_workspace(binary_frame, target="target", task_type="binary")
    with pytest.raises(OutcomeError):
        build_supervised_outcome(state, ws, "run_00000000")


def test_report_manifest_and_script(prepared_workspace, binary_frame: pd.DataFrame) -> None:
    state, ws = prepared_workspace(binary_frame, target="target", task_type="binary")
    run_id = _train(
        state, ws, "logistic_regression", {"C": 1.0, "penalty": "l2", "solver": "lbfgs"}
    )
    outcome = build_supervised_outcome(state, ws, run_id)

    report = build_report(state, ws, run_id, outcome.files)
    assert report.report
    assert report.script
    assert report.manifest

    reports = ws.reports_dir
    assert (reports / "report.html").is_file()
    assert (reports / "reproduce.py").is_file()
    assert (reports / "manifest.json").is_file()

    script = (reports / "reproduce.py").read_text()
    assert "reproduce-script" in script
    assert "train_model" in script
    assert state.dataset.data_hash in script
    assert f"PROJECT_ID = {state.project_id!r}" in script
    assert "DATA_HASH =" in script

    manifest = json.loads((reports / "manifest.json").read_text())
    assert manifest["data_hash"] == state.dataset.data_hash
    assert manifest["seed"] == state.seed
    assert manifest["model"]["id"] == "logistic_regression"

    html = (reports / "report.html").read_text()
    assert "<h1>Model report</h1>" in html
    assert state.project_id in html


def test_render_script_replays_recorded_cleaning_order(
    prepared_workspace, binary_frame: pd.DataFrame
) -> None:
    state, ws = prepared_workspace(binary_frame, target="target", task_type="binary")
    _train(state, ws, "logistic_regression", {"C": 1.0, "penalty": "l2", "solver": "lbfgs"})
    script = render_script(state, ws)
    dedupe_pos = script.find("remove_duplicates(frame)")
    assert dedupe_pos >= 0
    assert script.find("op='replay'") >= 0
    assert script.find("run_preprocessing(state, ws, PreprocessingOptions(") >= 0
    assert dedupe_pos < script.find("write_version(state.frame)")
