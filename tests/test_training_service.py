from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ml_workbench.rules.training import artifact_files
from ml_workbench.services.preprocessing_service import PreprocessingOptions
from ml_workbench.services.training_service import (
    train_model,
    tune_hyperparameters,
)
from ml_workbench.state import ModelRun


def _regression_frame(rows: int = 60, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(1, 10, size=rows)
    return pd.DataFrame(
        {"x": x, "label": rng.choice(["a", "b"], size=rows), "target": x**2 / 4 + 1}
    )


def test_train_creates_pipeline_meta_and_metrics(
    prepared_workspace, binary_frame: pd.DataFrame
) -> None:
    state, ws = prepared_workspace(binary_frame, target="target", task_type="binary")
    result = train_model(
        state, ws, "logistic_regression", {"C": 1.0, "penalty": "l2", "solver": "lbfgs"}
    )
    assert result.model_status == "done"
    assert result.n_splits == 5

    run_dir = ws.run_dir(result.run_id)
    for filename in artifact_files():
        assert (run_dir / filename).is_file(), f"{filename} missing (TRAIN-06)"

    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["data_hash"] == state.dataset.data_hash
    assert meta["seed"] == state.seed
    assert meta["primary"] == result.primary

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["model_id"] == "logistic_regression"
    assert metrics["primary"] == result.primary
    assert metrics["rule_id"] == "METRIC-01"
    assert len(metrics["folds"]) == result.n_splits

    trained = state.models[-1]
    assert trained.status == "done" and trained.run_id == result.run_id
    assert any(step.tab == "training" and step.op == "train" for step in state.steps)


def test_train_with_run_id_replaces_queued_entry(
    prepared_workspace, binary_frame: pd.DataFrame
) -> None:
    state, ws = prepared_workspace(binary_frame, target="target", task_type="binary")
    state.models = [
        ModelRun(run_id="run_queued12", model_id="logistic_regression", status="queued")
    ]
    result = train_model(
        state,
        ws,
        "logistic_regression",
        {"C": 1.0, "penalty": "l2", "solver": "lbfgs"},
        run_id="run_queued12",
    )
    assert result.run_id == "run_queued12"
    runs = state.models
    assert all(model.status != "queued" for model in runs)
    assert runs[-1].run_id == "run_queued12" and runs[-1].status == "done"


def test_leaderboard_reads_sidecars_only(prepared_workspace, binary_frame: pd.DataFrame) -> None:
    from ml_workbench.services.training_service import leaderboard

    state, ws = prepared_workspace(binary_frame, target="target", task_type="binary")
    train_model(state, ws, "logistic_regression", {"C": 1.0, "penalty": "l2", "solver": "lbfgs"})
    train_model(state, ws, "decision_tree", {"max_depth": 4})
    table = leaderboard(ws, state.models)
    assert len(table) == 2
    assert {"run_id", "model", "mean", "std", "gap"}.issubset(table.columns)
    assert table["gap"].notna().all()
    assert "folds" not in table.columns


def test_tuning_returns_best_params_and_prunes(
    prepared_workspace, binary_frame: pd.DataFrame
) -> None:
    state, ws = prepared_workspace(binary_frame, target="target", task_type="binary")
    result = tune_hyperparameters(state, ws, "random_forest", num_trials=4, seed=0)
    assert result.model_id == "random_forest"
    assert result.best_params
    assert result.trials <= 4
    assert result.pruned >= 0
    assert result.best_score > 0


def test_regression_saves_target_transform_and_predicts_in_domain(
    prepared_workspace,
) -> None:
    frame = _regression_frame()
    state, ws = prepared_workspace(
        frame,
        target="target",
        task_type="regression",
        options=PreprocessingOptions(target_transform="log"),
    )
    result = train_model(state, ws, "linear_regression", {})
    assert (ws.run_dir(result.run_id) / "target_transform.joblib").is_file()
    assert result.primary in {"rmse", "mae"}

    with pytest.raises(TypeError):
        train_model(state, ws, "linear_regression", {"nonsense_param": 1})
