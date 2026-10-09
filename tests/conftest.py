from __future__ import annotations

import numpy as np
import pandas as pd
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


@pytest.fixture
def binary_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "units": rng.integers(1, 9, size=60).astype(float),
            "band": rng.choice(["a", "b", "c"], size=60),
            "score": rng.normal(0, 1, size=60),
            "target": rng.integers(0, 2, size=60).astype(int),
        }
    )


@pytest.fixture
def string_binary_frame() -> pd.DataFrame:
    """Binary target with string labels ('No'/'Yes') — exercises label encoding paths."""
    rng = np.random.default_rng(11)
    signal = rng.normal(0, 1, size=80)
    return pd.DataFrame(
        {
            "units": rng.integers(1, 9, size=80).astype(float),
            "band": rng.choice(["a", "b", "c"], size=80),
            "score": signal,
            "target": np.where(signal > 0, "Yes", "No"),
        }
    )


@pytest.fixture
def prepared_workspace(tmp_path):
    """Factory: a cleaned + preprocessed (train/test split) project ready for training."""

    def make(
        frame: pd.DataFrame,
        *,
        target: str = "target",
        task_type: str = "binary",
        learning_type: str = "supervised",
        eval_labels: str | None = None,
        time_column: str | None = None,
        confirm_forecasting: bool = False,
        options=None,
    ):
        from ml_workbench.rules.task import build_task
        from ml_workbench.services.data_service import insert_dataset
        from ml_workbench.services.preprocessing_service import (
            PreprocessingOptions,
            run_preprocessing,
        )
        from ml_workbench.services.workspace import Workspace, utc_now

        ws = Workspace(project_id="p_svc", root=tmp_path)
        state = ProjectState(project_id="p_svc")
        state.frame = frame
        state.dataset = insert_dataset(state, ws, frame, source="test")
        ws.write_steps(
            [
                StepEntry(
                    id=1,
                    tab="cleaning",
                    op="dedupe",
                    dataset_hash_before=state.dataset.data_hash,
                    dataset_hash_after=state.dataset.data_hash,
                    created_at=utc_now(),
                )
            ]
        )
        state.steps = ws.read_steps()
        if learning_type == "unsupervised":
            state.task = build_task(
                frame,
                "unsupervised",
                task_type=task_type,
                eval_labels=eval_labels,
            )
        else:
            state.task = build_task(
                frame,
                "supervised",
                target=target,
                task_type=task_type,
                time_column=time_column,
                confirm_forecasting=confirm_forecasting,
            )
        run_preprocessing(state, ws, options or PreprocessingOptions())
        return state, ws

    return make
