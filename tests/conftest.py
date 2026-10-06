from __future__ import annotations

import pytest

from ml_workbench.state import (
    DatasetInfo,
    ModelRun,
    PipelineInfo,
    ProjectState,
    SplitInfo,
    StepEntry,
    TaskDefinition,
)


@pytest.fixture
def empty_state() -> ProjectState:
    return ProjectState(project_id="p_test")


@pytest.fixture
def ready_state() -> ProjectState:
    state = ProjectState(project_id="p_test")
    state.dataset = DatasetInfo(
        version="v1",
        path="data/samples/binary_classification.csv",
        data_hash="sha256:test",
        row_count=400,
        loaded_at="2026-10-06T00:00:00Z",
    )
    state.task = TaskDefinition(learning_type="supervised", task_type="binary", target="churned")
    state.steps = [StepEntry(id=1, tab="cleaning", op="dedupe", created_at="test")]
    state.split = SplitInfo(strategy="stratified_kfold", params={"n_splits": 5}, fitted=True)
    state.pipeline = PipelineInfo(fitted=True, path="models/test/pipeline")
    state.models = [ModelRun(run_id="r_001", model_id="logistic_regression", status="done")]
    state.active_model_id = "logistic_regression"
    return state
