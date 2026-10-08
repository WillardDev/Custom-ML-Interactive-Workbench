from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml_workbench.registry import load_registry
from ml_workbench.rules.metrics import (
    CLASSIFICATION_TASK_TYPES,
    anomaly_labeled_scores,
    labeled_cluster_scores,
    metric_plan,
)
from ml_workbench.rules.prediction import (
    anomaly_scoring_plan,
    calibration_curve,
    threshold_metrics,
)
from ml_workbench.services.model_cache import ModelCache
from ml_workbench.services.training_service import (
    _anomaly_decision,
    _binary_outlier_labels,
    classification_scores,
    inverse_target,
    regression_scores,
    unsupervised_scores,
)
from ml_workbench.services.workspace import Workspace
from ml_workbench.state import ModelRun, ProjectState, feature_columns


class PredictionError(ValueError):
    pass


@dataclass(frozen=True)
class PredictionResult:
    df: pd.DataFrame
    has_proba: bool


@dataclass(frozen=True)
class TestEvaluation:
    y_true: np.ndarray | None
    y_pred: np.ndarray | None
    y_proba: np.ndarray | None
    scores: dict[str, float]


def load_pipeline(workspace: Workspace, run_id: str) -> Any:
    artifact = workspace.run_dir(run_id) / "pipeline.joblib"
    if not artifact.is_file():
        raise PredictionError(f"no pipeline artifact for run '{run_id}' — train the model first")
    return joblib.load(artifact)


def get_pipeline(workspace: Workspace, run_id: str, cache: ModelCache | None = None) -> Any:
    """PERF-02: load through the bounded LRU cache (lazy, evicts least recently used)."""
    if cache is None:
        return load_pipeline(workspace, run_id)
    cached = cache.get(run_id)
    if cached is not None:
        return cached
    return cache.put(run_id, load_pipeline(workspace, run_id))


def _load_target_transform(workspace: Workspace, run_id: str) -> Any:
    path = workspace.run_dir(run_id) / "target_transform.joblib"
    if not path.is_file():
        return None
    return joblib.load(path)


def _run_spec(state: ProjectState, run_id: str) -> tuple[ModelRun, Any]:
    run = next((model for model in state.models if model.run_id == run_id), None)
    if run is None:
        raise PredictionError(f"unknown run '{run_id}' — train a model first")
    spec = load_registry().get(run.model_id)
    if spec is None:
        raise PredictionError(f"unknown model '{run.model_id}'")
    return run, spec


def predict_frame(
    state: ProjectState,
    workspace: Workspace,
    run_id: str,
    frame: pd.DataFrame,
    *,
    cache: ModelCache | None = None,
) -> PredictionResult:
    run, spec = _run_spec(state, run_id)
    pipeline = get_pipeline(workspace, run_id, cache)
    task = state.task
    features = feature_columns(task, frame) if task is not None else list(frame.columns)
    result = frame.copy()
    has_predict = bool(spec.flags.get("has_predict"))
    has_transform = bool(spec.flags.get("has_transform"))
    if has_transform:
        transformed = np.asarray(pipeline.transform(frame[features]))
        for k in range(transformed.shape[1]):
            result[f"pc{k + 1}"] = transformed[:, k]
    if not has_predict:
        if not has_transform:
            raise PredictionError(
                f"'{spec.id}' can neither assign nor project new rows (PRED-04/PRED-05): "
                f"train a surrogate nearest-centroid model instead"
            )
        return PredictionResult(df=result, has_proba=False)
    if task is not None and task.task_type == "anomaly_detection":
        # PRED-06: score every row (higher = more anomalous) and flag at the plan threshold.
        plan = anomaly_scoring_plan(spec.id)
        score = _anomaly_decision(
            pipeline.named_steps["model"], _transform_matrix(pipeline, frame, features)
        )
        result["score"] = score
        result["prediction"] = (score > float(plan["threshold"])).astype(int)
        return PredictionResult(df=result, has_proba=False)
    y_pred = np.asarray(pipeline.predict(frame[features]))
    result["prediction"] = y_pred
    # has_proba is a model capability; only classifiers actually expose probabilities.
    has_proba = (
        bool(spec.flags.get("has_proba"))
        and task is not None
        and (task.task_type in CLASSIFICATION_TASK_TYPES)
    )
    if has_proba:
        proba = np.asarray(pipeline.predict_proba(frame[features]))
        classes = getattr(pipeline.named_steps["model"], "classes_", [])
        for index, label in enumerate(classes):
            result[f"p_{label}"] = proba[:, index]
    else:
        inverse = _load_target_transform(workspace, run_id)
        if inverse is not None:
            result["prediction"] = inverse_target(inverse, y_pred)
    return PredictionResult(df=result, has_proba=has_proba)


def predict_single(
    state: ProjectState,
    workspace: Workspace,
    run_id: str,
    row: dict[str, object],
    *,
    cache: ModelCache | None = None,
) -> dict[str, object]:
    if state.frame is None:
        raise PredictionError("load a dataset first")
    aligned = {column: row.get(column) for column in state.frame.columns}
    one_row = pd.DataFrame([aligned])
    result = predict_frame(state, workspace, run_id, one_row, cache=cache)
    first = result.df.iloc[0]
    output: dict[str, object] = {"prediction": first["prediction"]}
    if "score" in result.df.columns:
        output["score"] = first["score"]
    if result.has_proba:
        for column in result.df.columns:
            if str(column).startswith("p_"):
                output[str(column)] = first[column]
    return output


def _test_indices(state: ProjectState) -> np.ndarray:
    if state.split is None or state.split.indices_path is None:
        raise PredictionError("no test split saved — run Preprocessing")
    indices = json.loads(Path(state.split.indices_path).read_text())
    test_idx = np.asarray(indices["test"], dtype=int)
    if len(test_idx) == 0:
        raise PredictionError("the split strategy produced no test rows")
    return test_idx


def _transform_matrix(pipeline: Any, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    preprocess = pipeline.named_steps["preprocess"]
    return np.asarray(preprocess.transform(frame[features]))


def _evaluate_unsupervised(
    state: ProjectState, workspace: Workspace, run_id: str, *, cache: ModelCache | None = None
) -> TestEvaluation:
    assert state.frame is not None and state.task is not None
    task = state.task
    test_idx = _test_indices(state)
    run, spec = _run_spec(state, run_id)
    pipeline = get_pipeline(workspace, run_id, cache)
    test_frame = state.frame.iloc[test_idx]
    features = feature_columns(task, state.frame)
    x_test = _transform_matrix(pipeline, test_frame, features)
    model = pipeline.named_steps["model"]
    scores = unsupervised_scores(task.task_type, model, x_test)
    y_true: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    if task.eval_labels is not None:
        y_true = np.asarray(state.frame[task.eval_labels].iloc[test_idx])
    plan = metric_plan(task.task_type, labeled=task.eval_labels is not None)
    if task.task_type == "clustering":
        y_pred = np.asarray(model.predict(x_test))
        if y_true is not None:
            scores.update(labeled_cluster_scores(y_true, y_pred))
    elif task.task_type == "anomaly_detection":
        decision = _anomaly_decision(model, x_test)
        threshold = float(anomaly_scoring_plan(spec.id)["threshold"])
        y_pred = (decision > threshold).astype(int)
        if y_true is not None:
            scores.update(anomaly_labeled_scores(_binary_outlier_labels(y_true), decision))
    filtered = {
        name: value
        for name, value in sorted(scores.items())
        if name in plan.metrics or name == plan.primary
    }
    return TestEvaluation(y_true=y_true, y_pred=y_pred, y_proba=None, scores=filtered)


def evaluate_test(
    state: ProjectState,
    workspace: Workspace,
    run_id: str,
    *,
    cache: ModelCache | None = None,
) -> TestEvaluation:
    if state.frame is None:
        raise PredictionError("no dataset loaded")
    if state.task is None:
        raise PredictionError("no task set — set a task in Data Insertion")
    if state.task.learning_type == "unsupervised":
        return _evaluate_unsupervised(state, workspace, run_id, cache=cache)
    assert state.task.target is not None
    test_idx = _test_indices(state)
    run, spec = _run_spec(state, run_id)
    pipeline = get_pipeline(workspace, run_id, cache)
    test_frame = state.frame.iloc[test_idx]
    features = feature_columns(state.task, test_frame)
    y_true = np.asarray(state.frame[state.task.target].iloc[test_idx])
    y_pred = np.asarray(pipeline.predict(test_frame[features]))
    proba: np.ndarray | None = None
    has_proba = bool(spec.flags.get("has_proba")) and (
        state.task.task_type in CLASSIFICATION_TASK_TYPES
    )
    if has_proba:
        proba = np.asarray(pipeline.predict_proba(test_frame[features]))
    else:
        inverse = _load_target_transform(workspace, run_id)
        if inverse is not None:
            y_pred = inverse_target(inverse, y_pred)
    if has_proba:
        scores = classification_scores(y_true, y_pred, proba)
    else:
        scores = regression_scores(y_true, y_pred)
    return TestEvaluation(y_true=y_true, y_pred=y_pred, y_proba=proba, scores=scores)


def threshold_eval(evaluation: TestEvaluation, threshold: float = 0.5) -> dict[str, float]:
    """PRED-01: adjustable-threshold decision rule for binary classifiers."""
    if evaluation.y_proba is None:
        raise PredictionError("model has no probabilities — thresholding is not available")
    positive = (
        evaluation.y_proba[:, 1] if evaluation.y_proba.shape[1] > 1 else evaluation.y_proba[:, 0]
    )
    return threshold_metrics(evaluation.y_true, positive, threshold)


def calibration(evaluation: TestEvaluation, bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """PRED-01: calibration (expected-vs-observed) table for binary classifiers."""
    if evaluation.y_proba is None:
        raise PredictionError("model has no probabilities")
    positive = (
        evaluation.y_proba[:, 1] if evaluation.y_proba.shape[1] > 1 else evaluation.y_proba[:, 0]
    )
    return calibration_curve(positive, evaluation.y_true, bins)
