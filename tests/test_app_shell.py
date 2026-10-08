from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ml_workbench.services.workspace import Workspace
from ml_workbench.state import ProjectState
from ml_workbench.ui.explainability_tab import EXPLAIN_JOBS

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "ml_workbench" / "app.py"


@pytest.fixture
def app(tmp_path: Path) -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=10)
    app.session_state["workspace"] = Workspace(project_id="p_apptest", root=tmp_path)
    return app


def _load_sample(app: AppTest) -> AppTest:
    app.selectbox("data_sample").set_value("binary_classification.csv")
    app.button("data_load_sample").click().run()
    assert not app.exception
    return app


def _set_task(app: AppTest) -> AppTest:
    app.selectbox("task_target").set_value("churned")
    app.run()
    app.button("task_set").click().run()
    assert not app.exception
    return app


def _clean(app: AppTest) -> AppTest:
    app.sidebar.radio("workflow_nav").set_value("cleaning").run()
    assert not app.exception
    app.button("clean_run").click().run()
    assert not app.exception
    return app


def test_app_boots_on_data_tab(app: AppTest) -> None:
    app.run()
    assert not app.exception
    assert app.header[0].value == "1. Data Insertion"
    assert app.sidebar.radio("workflow_nav").value == "data"


def test_locked_tab_shows_gate_rule(app: AppTest) -> None:
    app.run()
    app.sidebar.radio("workflow_nav").set_value("training").run()
    assert not app.exception
    assert app.header[0].value == "6. Training"
    assert "Locked:" in app.info[0].value
    assert "GATE-01" in app.info[0].value


def test_real_flow_unlocks_tabs(app: AppTest) -> None:
    app.run()
    _load_sample(app)

    state: ProjectState = app.session_state["state"]
    assert state.dataset is not None
    assert state.dataset.version == "v00001"
    assert state.task is None

    _set_task(app)
    state = app.session_state["state"]
    assert state.task is not None and state.task.target == "churned"

    _clean(app)
    state = app.session_state["state"]
    cleaning_steps = [step for step in state.steps if step.tab == "cleaning"]
    assert cleaning_steps
    assert state.cleaned
    assert state.stale["cleaning"] is False
    assert sum(1 for value in state.stale.values() if value) == 8

    app.sidebar.radio("workflow_nav").set_value("preprocessing").run()
    assert not any("Locked:" in item.value for item in app.info)

    app.sidebar.radio("workflow_nav").set_value("training").run()
    assert "Locked:" in app.info[0].value
    assert "GATE-01" in app.info[0].value


def test_edit_data_marks_downstream_stale(app: AppTest) -> None:
    app.run()
    _load_sample(app)
    _set_task(app)
    _clean(app)

    app.sidebar.radio("workflow_nav").set_value("data").run()
    _set_task(app)
    state: ProjectState = app.session_state["state"]
    assert state.stale["cleaning"] is True

    app.sidebar.radio("workflow_nav").set_value("cleaning").run()
    assert not app.exception
    assert any("stale" in item.value.lower() for item in app.warning)
    assert any("Data Insertion changed" in item.value for item in app.warning)


def test_undo_restores_previous_version(app: AppTest) -> None:
    app.run()
    _load_sample(app)
    _set_task(app)
    _clean(app)

    state: ProjectState = app.session_state["state"]
    cleaning_before = len([step for step in state.steps if step.tab == "cleaning"])
    version_before = state.dataset.version if state.dataset else ""
    assert cleaning_before >= 2

    app.button("clean_undo").click().run()
    assert not app.exception
    state = app.session_state["state"]
    cleaning_after = [step for step in state.steps if step.tab == "cleaning"]
    assert len(cleaning_after) == cleaning_before - 1
    assert state.dataset is not None and state.dataset.version != version_before


def test_preprocessing_build_unlocks_training_and_eda(app: AppTest) -> None:
    app.run()
    _load_sample(app)
    _set_task(app)
    _clean(app)

    app.sidebar.radio("workflow_nav").set_value("preprocessing").run()
    assert not app.exception
    assert app.header[0].value == "3. Data Preprocessing"

    state: ProjectState = app.session_state["state"]
    assert state.split is None and state.pipeline is None

    app.button("prep_build").click().run()
    assert not app.exception
    state = app.session_state["state"]
    assert state.split is not None and state.split.fitted
    assert state.pipeline is not None and state.pipeline.fitted
    preprocessing_ops = [step.op for step in state.steps if step.tab == "preprocessing"]
    assert "split" in preprocessing_ops
    assert "encode" in preprocessing_ops

    app.sidebar.radio("workflow_nav").set_value("eda").run()
    assert not app.exception
    assert app.header[0].value == "4. Exploratory Data Analysis"
    assert any("Classification views" in item.value for item in app.subheader)

    app.sidebar.radio("workflow_nav").set_value("training").run()
    assert not app.exception
    assert app.header[0].value == "6. Training"
    assert not any("Locked:" in item.value for item in app.info)


def test_phase4_train_and_predict_end_to_end(app: AppTest) -> None:
    app.run()
    _load_sample(app)
    _set_task(app)
    _clean(app)

    app.sidebar.radio("workflow_nav").set_value("preprocessing").run()
    app.button("prep_build").click().run()
    assert not app.exception

    app.sidebar.radio("workflow_nav").set_value("modelling").run()
    assert not app.exception
    assert app.header[0].value == "5. Modelling"
    assert any("Model registry" in item.value for item in app.subheader)

    app.button("mod_queue_all").click().run()
    assert not app.exception
    state: ProjectState = app.session_state["state"]
    queued = [model for model in state.models if model.status == "queued"]
    assert queued, "modelling tab queues at least one job"
    assert app.session_state["training_queue"]

    app.sidebar.radio("workflow_nav").set_value("training").run()
    assert app.header[0].value == "6. Training"
    app.button("train_run").click().run()
    assert not app.exception
    state = app.session_state["state"]
    done = state.trained_models
    assert done and not app.session_state["training_queue"]
    assert state.active_model_id == done[-1].run_id
    assert any(step.tab == "training" for step in state.steps)

    app.sidebar.radio("workflow_nav").set_value("prediction").run()
    assert not app.exception
    assert app.header[0].value == "7. Prediction"
    assert app.selectbox("pred_model").value == done[-1].run_id
    assert app.button("pred_single")
    app.button("pred_single").click().run()
    assert not app.exception
    assert app.json, "single-row prediction renders a JSON payload"


def test_phase5_error_explain_outcome_end_to_end(app: AppTest) -> None:
    app.run()
    _load_sample(app)
    _set_task(app)
    _clean(app)

    app.sidebar.radio("workflow_nav").set_value("preprocessing").run()
    app.button("prep_build").click().run()
    assert not app.exception

    app.sidebar.radio("workflow_nav").set_value("modelling").run()
    assert not app.exception
    app.button("mod_queue_all").click().run()
    assert not app.exception

    app.sidebar.radio("workflow_nav").set_value("training").run()
    app.button("train_run").click().run()
    assert not app.exception
    state: ProjectState = app.session_state["state"]
    assert state.trained_models
    run_id = state.active_model_id
    assert run_id is not None

    app.sidebar.radio("workflow_nav").set_value("error_analysis").run()
    assert not app.exception
    assert app.header[0].value == "8. Error Analysis"
    assert any("Error views" in item.value for item in app.subheader)
    assert app.dataframe, "binary error views render dataframes"

    app.sidebar.radio("workflow_nav").set_value("explainability").run()
    assert not app.exception
    assert app.header[0].value == "9. Model Explainability"
    assert app.button("explain_run")
    app.button("explain_run").click().run()
    assert not app.exception
    handle = app.session_state.get("explain_handle")
    if handle is not None:
        done = EXPLAIN_JOBS.wait(handle.job_id, timeout=30)
        assert done.state == "done"
    app.run()
    assert not app.exception
    assert any("explanation" in item.value.lower() for item in app.subheader)

    app.sidebar.radio("workflow_nav").set_value("outcome").run()
    assert not app.exception
    assert app.header[0].value == "10. Outcome and Final Prediction"
    app.button("outcome_build").click().run()
    assert not app.exception
    state = app.session_state["state"]
    assert any(step.tab == "outcome" for step in state.steps)
    payload = app.session_state["outcome_report"]
    assert payload["files"]

    workspace: Workspace = app.session_state["workspace"]
    assert (workspace.reports_dir / "report.html").is_file()
    assert (workspace.reports_dir / "reproduce.py").is_file()
    assert (workspace.reports_dir / "manifest.json").is_file()
    assert (workspace.outcome_dir / "predictions.csv").is_file()
