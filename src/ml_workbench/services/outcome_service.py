from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from ml_workbench.registry import load_registry
from ml_workbench.rules.outcome import supervised_deliverables
from ml_workbench.rules.prediction import apply_threshold
from ml_workbench.rules.staleness import mark_downstream_stale
from ml_workbench.services.prediction_service import predict_frame
from ml_workbench.services.preprocessing_service import apply_target_transform, build_pipeline
from ml_workbench.services.training_service import build_estimator
from ml_workbench.services.workspace import Workspace, utc_now
from ml_workbench.state import ProjectState, StepEntry


class OutcomeError(ValueError):
    pass


@dataclass(frozen=True)
class OutcomeResult:
    run_id: str
    model_id: str
    files: tuple[str, ...]


def build_supervised_outcome(
    state: ProjectState,
    workspace: Workspace,
    run_id: str,
    *,
    threshold: float = 0.5,
    refit_on_all: bool = False,
) -> OutcomeResult:
    """OUT-01: package the supervised deliverables for the active run."""
    if state.frame is None or state.task is None or state.task.target is None:
        raise OutcomeError("load a dataset and set a task first")
    run = next((model for model in state.models if model.run_id == run_id), None)
    if run is None:
        raise OutcomeError(f"unknown run '{run_id}' — train a model first")
    spec = load_registry().get(run.model_id)
    if spec is None:
        raise OutcomeError(f"unknown model '{run.model_id}'")
    meta = workspace.read_run_json(run_id, "meta.json")
    metrics = workspace.read_run_json(run_id, "metrics.json")
    data_hash = str(meta.get("data_hash", state.dataset.data_hash if state.dataset else ""))

    workspace.ensure()
    outcome = workspace.outcome_dir

    predictions = predict_frame(state, workspace, run_id, state.frame)
    predictions_df = predictions.df
    if predictions.has_proba and state.task.task_type == "binary":
        positive = [column for column in predictions_df.columns if str(column).startswith("p_")]
        if positive:
            hard = apply_threshold(np.asarray(predictions_df[positive[0]], dtype=float), threshold)
            predictions_df["prediction@threshold"] = hard
    predictions_path = outcome / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    if refit_on_all:
        pipeline_path = _refit_on_all(
            state,
            workspace,
            spec,
            dict(meta.get("params", {}) or {}),
            int(meta.get("seed", state.seed)),
        )
    else:
        source = workspace.run_dir(run_id) / "pipeline.joblib"
        if not source.is_file():
            raise OutcomeError(f"missing pipeline artifact for '{run_id}'")
        pipeline_path = outcome / "pipeline.joblib"
        shutil.copy2(source, pipeline_path)

    threshold_path = outcome / "threshold.json"
    threshold_path.write_text(_json(threshold_payload(threshold, state.task.task_type)))
    card_path = outcome / "model_card.json"
    card_path.write_text(_json(_model_card(spec, meta, metrics, data_hash)))

    files = tuple(
        path.relative_to(workspace.project_dir).as_posix()
        for path in (pipeline_path, predictions_path, threshold_path, card_path)
        if path.is_file()
    )
    _log_outcome_step(state, workspace, run_id, spec.id, files)
    return OutcomeResult(run_id=run_id, model_id=spec.id, files=files)


def _log_outcome_step(
    state: ProjectState,
    workspace: Workspace,
    run_id: str,
    model_id: str,
    files: tuple[str, ...],
) -> None:
    """OUT-01: record the packaging step so the report manifest traces the run."""
    created_at = utc_now()
    steps = workspace.read_steps()
    workspace.write_steps(
        steps
        + [
            StepEntry(
                id=max((entry.id for entry in steps), default=0) + 1,
                tab="outcome",
                op="package",
                params={"run_id": run_id, "model": model_id, "files": list(files)},
                dataset_hash_before=state.dataset.data_hash if state.dataset else "",
                dataset_hash_after=state.dataset.data_hash if state.dataset else "",
                created_at=created_at,
            )
        ]
    )
    state.steps = workspace.read_steps()
    mark_downstream_stale(state, "outcome")


def threshold_payload(threshold: float, task_type: str) -> dict[str, Any]:
    """OUT-01: records the chosen threshold/interval method for the outcome."""
    return {"threshold": threshold, "applied": bool(task_type == "binary")}


def _refit_on_all(
    state: ProjectState,
    workspace: Workspace,
    spec: Any,
    params: dict[str, Any],
    seed: int,
) -> Path:
    frame, task = state.frame, state.task
    assert frame is not None and task is not None and task.target is not None
    choices = _preprocessing_choices(state.steps)
    reporter = build_pipeline(
        frame,
        np.arange(len(frame)),
        task,
        encoder=str(choices.get("encoder") or "ordinal"),
        scaler=choices.get("scaler"),
    )
    target_transform: object = None
    if task.task_type == "regression" and choices.get("target_transform") is not None:
        y_to_fit, target_transform = apply_target_transform(
            frame[task.target], str(choices["target_transform"])
        )
    else:
        y_to_fit = frame[task.target]
    estimator = build_estimator(spec, params, task.task_type, seed)
    combined = Pipeline([("preprocess", reporter.pipeline), ("model", estimator)])
    combined.fit(frame, np.asarray(y_to_fit))
    path = workspace.outcome_dir / "pipeline.joblib"
    joblib.dump(combined, path)
    if target_transform is not None:
        joblib.dump(target_transform, workspace.outcome_dir / "target_transform.joblib")
    return path


def _preprocessing_choices(steps: list[StepEntry]) -> dict[str, Any]:
    encoder: str = "ordinal"
    scaler: str | None = "standard"
    target_transform: str | None | object = None
    for entry in reversed(steps):
        if entry.tab != "preprocessing":
            continue
        if entry.op == "encode":
            encoder = str(entry.params.get("encoder") or "ordinal")
        elif entry.op == "scale":
            value = entry.params.get("scaler")
            scaler = str(value) if value is not None else None
        elif entry.op == "target_transform":
            value = entry.params.get("transform")
            target_transform = None if value is None else str(value)
    return {"encoder": encoder, "scaler": scaler, "target_transform": target_transform}


def _model_card(
    spec: Any, meta: dict[str, Any], metrics: dict[str, Any], data_hash: str
) -> dict[str, Any]:
    versions = dict(meta.get("versions", {}) or {})
    versions.update(meta.get("library_versions", {}) or {})
    return {
        "model_id": spec.id,
        "model_name": spec.name,
        "library": spec.library,
        "family": spec.family,
        "task_type": str(metrics.get("task_type", "")),
        "target": str(meta.get("task", {}).get("target", "")),
        "primary": str(metrics.get("primary", "")),
        "mean_score": metrics.get("mean"),
        "std_score": metrics.get("std"),
        "params": dict(meta.get("params", {}) or {}),
        "data_hash": data_hash,
        "seed": meta.get("seed"),
        "split": meta.get("split"),
        "trained_at": meta.get("trained_at"),
        "library_versions": versions,
        "explain_method": spec.flags.get("explain_method"),
        "deliverables": list(supervised_deliverables()),
    }


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
