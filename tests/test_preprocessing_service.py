from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ml_workbench.rules.task import build_task
from ml_workbench.services.data_service import insert_dataset
from ml_workbench.services.preprocessing_service import (
    PreprocessingError,
    PreprocessingOptions,
    apply_target_transform,
    build_pipeline,
    materialize_split,
    run_preprocessing,
)
from ml_workbench.services.workspace import Workspace, utc_now
from ml_workbench.state import ProjectState, StepEntry


def _ready_state(
    tmp_path: Path,
    frame: pd.DataFrame,
    *,
    target: str = "target",
    task_type: str = "binary",
) -> tuple[ProjectState, Workspace]:
    ws = Workspace(project_id="p_prep", root=tmp_path)
    state = ProjectState(project_id="p_prep")
    state.dataset = insert_dataset(state, ws, frame, source="test")
    state.frame = frame
    ws.write_steps(
        [
            StepEntry(
                id=2,
                tab="cleaning",
                op="dedupe",
                params={},
                dataset_hash_before=state.dataset.data_hash,
                dataset_hash_after=state.dataset.data_hash,
                created_at=utc_now(),
            )
        ]
    )
    state.steps = ws.read_steps()
    learning_type = "unsupervised" if target is None else "supervised"
    state.task = build_task(frame, learning_type, target=target, task_type=task_type)
    return state, ws


def _binary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "units": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "region": ["north", "south", "north", "west", "south", "west", "north", "south"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


def test_run_preprocessing_builds_leak_safe_pipeline(tmp_path: Path) -> None:
    frame = _binary_frame()
    state, ws = _ready_state(tmp_path, frame, target="target", task_type="binary")

    result = run_preprocessing(
        state, ws, PreprocessingOptions(encoder="one_hot", scaler="standard")
    )
    assert result.strategy == "stratified"
    assert result.rule_id == "SPLIT-03"
    assert result.train_rows + result.test_rows == len(frame)
    assert result.feature_count == 2

    split = json.loads((ws.preprocessing_dir / "split.json").read_text())
    assert len(split["train"]) == result.train_rows
    assert len(split["test"]) == result.test_rows
    assert state.split is not None and state.split.fitted
    assert state.pipeline is not None and state.pipeline.fitted

    pipeline = joblib.load(ws.preprocessing_dir / "pipeline.joblib")
    named = pipeline.named_steps["columns"].named_transformers_["num"]
    scaler_mean = float(np.ravel(named.named_steps["scaler"].mean_)[0])
    train_mean = float(frame["units"].iloc[split["train"]].mean())
    assert scaler_mean == pytest.approx(train_mean, abs=1e-9)

    ops = [entry.op for entry in state.steps if entry.tab == "preprocessing"]
    assert ops == ["split", "encode", "scale"]
    assert state.stale["training"] is True
    assert state.stale["preprocessing"] is False


def test_split_seed_deterministic(tmp_path: Path) -> None:
    frame = _binary_frame()
    state, ws = _ready_state(tmp_path, frame, target="target", task_type="binary")
    first = run_preprocessing(state, ws, PreprocessingOptions(scaler=None))
    split_first = json.loads((ws.preprocessing_dir / "split.json").read_text())

    state2, _ = _ready_state(tmp_path, frame, target="target", task_type="binary")
    ws2 = ws
    second = run_preprocessing(state2, ws2, PreprocessingOptions(scaler=None))
    split_second = json.loads((ws2.preprocessing_dir / "split.json").read_text())
    assert first.train_rows == second.train_rows
    assert split_first["train"] == split_second["train"]


def test_regression_split_is_random(tmp_path: Path) -> None:
    frame = pd.DataFrame({"x": range(10), "target": [float(v * 1.5) for v in range(10)]})
    state, ws = _ready_state(tmp_path, frame, target="target", task_type="regression")
    result = run_preprocessing(state, ws, PreprocessingOptions())
    assert result.strategy == "random"
    assert result.rule_id == "SPLIT-04"


def test_chronological_split_never_shuffled() -> None:
    from ml_workbench.rules.split import choose_split

    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="MS"),
            "sales": [float(v) for v in range(10)],
        }
    )
    task = build_task(
        frame, "supervised", target="sales", time_column="date", confirm_forecasting=True
    )
    rule = choose_split(task)
    assert rule.strategy == "chronological"
    result = materialize_split(frame, task, rule, test_size=0.2, seed=0)
    assert list(result.train) == list(range(8))
    assert list(result.test) == [8, 9]


def test_pipeline_rejects_unknown_columns(tmp_path: Path) -> None:
    frame = _binary_frame()
    state, ws = _ready_state(tmp_path, frame, target="target", task_type="binary")
    task = state.task
    report = build_pipeline(frame, np.arange(6), task, encoder="one_hot", scaler="standard")
    transformed = report.pipeline.transform(frame)
    assert transformed.shape[1] == 4  # 1 numeric + 3 one-hot regions


def test_one_hot_ignores_test_only_category(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0],
            "cat": ["seed", "known", "known", "rare"],
            "target": [0, 1, 0, 1],
        }
    )
    state, _ = _ready_state(tmp_path, frame, target="target", task_type="binary")
    train = np.array([0, 1, 2])
    report = build_pipeline(frame, train, state.task, encoder="one_hot", scaler=None)
    out = report.pipeline.transform(frame)
    assert out.shape[0] == 4
    rare_hot = np.abs(out[3, 1:])
    assert float(np.sum(rare_hot)) < 1e-12


def test_cleaning_required_before_preprocessing(tmp_path: Path) -> None:
    ws = Workspace(project_id="p_prep", root=tmp_path)
    state = ProjectState(project_id="p_prep")
    frame = _binary_frame()
    state.dataset = insert_dataset(state, ws, frame, source="test")
    state.frame = frame
    state.task = build_task(frame, "supervised", target="target", task_type="binary")
    with pytest.raises(PreprocessingError, match="Data Cleaning"):
        run_preprocessing(state, ws, PreprocessingOptions())


def test_task_required_before_preprocessing(tmp_path: Path) -> None:
    ws = Workspace(project_id="p_prep", root=tmp_path)
    state = ProjectState(project_id="p_prep")
    with pytest.raises(PreprocessingError, match="load a dataset"):
        run_preprocessing(state, ws, PreprocessingOptions())


def test_target_transform_log_then_inverse() -> None:
    series = pd.Series([1.0, 2.0, 4.0, 8.0, 16.0])
    transformed, _ = apply_target_transform(series, "log")
    assert np.allclose(transformed, np.log1p(series))
    assert np.allclose(np.expm1(transformed), series)
