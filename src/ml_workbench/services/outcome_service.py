from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ml_workbench.registry import load_registry
from ml_workbench.rules.outcome import outcome_deliverables, supervised_deliverables
from ml_workbench.rules.prediction import apply_threshold
from ml_workbench.rules.staleness import mark_downstream_stale
from ml_workbench.services.prediction_service import predict_frame
from ml_workbench.services.preprocessing_service import apply_target_transform, build_pipeline
from ml_workbench.services.training_service import _anomaly_decision, build_estimator
from ml_workbench.services.workspace import Workspace, utc_now
from ml_workbench.state import ProjectState, StepEntry, feature_columns


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
    """OUT-01/04: records the chosen threshold/interval method for the outcome."""
    return {
        "threshold": threshold,
        "applied": bool(task_type in {"binary", "anomaly_detection"}),
    }


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
    spec: Any,
    meta: dict[str, Any],
    metrics: dict[str, Any],
    data_hash: str,
    deliverables: list[dict[str, str]] | None = None,
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
        "deliverables": list(deliverables or supervised_deliverables()),
    }


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"


def build_unsupervised_outcome(
    state: ProjectState,
    workspace: Workspace,
    run_id: str,
) -> OutcomeResult:
    """OUT-02/03/04: package the unsupervised deliverables for the active run."""
    if state.frame is None or state.task is None:
        raise OutcomeError("load a dataset and set a task first")
    if state.task.learning_type != "unsupervised":
        raise OutcomeError("build_supervised_outcome is the supervised packaging path")
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
    task_type = state.task.task_type

    source = workspace.run_dir(run_id) / "pipeline.joblib"
    if not source.is_file():
        raise OutcomeError(f"missing pipeline artifact for '{run_id}'")
    pipeline_path = outcome / "pipeline.joblib"
    shutil.copy2(source, pipeline_path)

    predictions = predict_frame(state, workspace, run_id, state.frame).df
    declared = outcome_deliverables(task_type)
    files = [pipeline_path]

    if task_type == "clustering":
        clusters_df = state.frame.copy()
        clusters_df["cluster_id"] = predictions["prediction"]
        clusters_path = outcome / "clusters.csv"
        clusters_df.to_csv(clusters_path, index=False)
        files.append(clusters_path)
        profiles, personas = _cluster_profiles(state, clusters_df)
        profiles_path = outcome / "profiles.csv"
        profiles.to_csv(profiles_path, index=False)
        files.append(profiles_path)
        personas_path = outcome / "personas.json"
        personas_path.write_text(_json(personas))
        files.append(personas_path)
    elif task_type == "dimensionality_reduction":
        embeddings = state.frame.copy()
        pc_columns = [column for column in predictions.columns if str(column).startswith("pc")]
        for column in pc_columns:
            embeddings[column] = predictions[column]
        embeddings_path = outcome / "embeddings.csv"
        embeddings.to_csv(embeddings_path, index=False)
        files.append(embeddings_path)
        summary = _dimred_summary(pipeline_path, state, metrics, data_hash)
        summary_path = outcome / "summary.json"
        summary_path.write_text(_json(summary))
        files.append(summary_path)
    elif task_type == "anomaly_detection":
        features = list(feature_columns(state.task, state.frame))
        loaded = joblib.load(source)
        x = np.asarray(loaded.named_steps["preprocess"].transform(state.frame[features]))
        decision = _anomaly_decision(loaded.named_steps["model"], x)
        flagged = state.frame.copy()
        flagged["score"] = decision
        flagged["flagged"] = (decision > 0.0).astype(int)
        flagged_path = outcome / "flagged.csv"
        flagged.to_csv(flagged_path, index=False)
        files.append(flagged_path)
        distribution = _anomaly_distribution(pd.Series(decision))
        distribution_path = outcome / "score_distribution.json"
        distribution_path.write_text(_json(distribution))
        files.append(distribution_path)
        threshold_path = outcome / "threshold.json"
        threshold_path.write_text(_json(threshold_payload(0.0, "anomaly_detection")))
        files.append(threshold_path)
    elif task_type == "association":
        # OUT-05: the mined rules table is the association deliverable.
        model = joblib.load(source).named_steps["model"]
        rules = pd.DataFrame(getattr(model, "rules_", []))
        if not rules.empty:
            rules["antecedent"] = rules["antecedent"].map(lambda ante: " + ".join(ante))
            rules["consequent"] = rules["consequent"].map(lambda conc: " + ".join(conc))
        rules_path = outcome / "rules.csv"
        rules.to_csv(rules_path, index=False)
        files.append(rules_path)

    card_path = outcome / "model_card.json"
    card_path.write_text(_json(_model_card(spec, meta, metrics, data_hash, list(declared))))
    files.append(card_path)

    relative = tuple(
        path.relative_to(workspace.project_dir).as_posix() for path in files if path.is_file()
    )
    _log_outcome_step(state, workspace, run_id, spec.id, relative)
    return OutcomeResult(run_id=run_id, model_id=spec.id, files=relative)


def _cluster_profiles(
    state: ProjectState, clusters_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    assert state.frame is not None and state.task is not None
    features = list(feature_columns(state.task, state.frame))
    profiles: list[dict[str, Any]] = []
    personas: list[dict[str, Any]] = []
    overall = clusters_df[features].mean(numeric_only=True)
    for cluster_id in sorted(int(value) for value in clusters_df["cluster_id"].unique()):
        members = clusters_df.loc[clusters_df["cluster_id"] == cluster_id, features]
        row: dict[str, Any] = {"cluster_id": cluster_id, "size": int(len(members))}
        for feature in features:
            row[f"{feature}_mean"] = round(float(members[feature].mean()), 4)
            row[f"{feature}_std"] = round(float(members[feature].std(ddof=0)), 4)
        profiles.append(row)
        deviations = (members.mean() - overall).abs().sort_values(ascending=False)
        personas.append(
            {
                "cluster": int(cluster_id),
                "size": int(len(members)),
                "top_features": [
                    {
                        "feature": str(feature),
                        "mean": round(float(members[feature].mean()), 4),
                        "deviation": round(float(members[feature].mean() - overall[feature]), 4),
                    }
                    for feature in deviations.index[:3]
                ],
            }
        )
    return pd.DataFrame(profiles), {"personas": personas}


def _dimred_summary(
    pipeline_path: Path, state: ProjectState, metrics: dict[str, Any], data_hash: str
) -> dict[str, Any]:
    assert state.frame is not None and state.task is not None
    features = list(feature_columns(state.task, state.frame))
    pipeline = joblib.load(pipeline_path)
    x = np.asarray(pipeline.named_steps["preprocess"].transform(state.frame[features]))
    model = pipeline.named_steps["model"]
    embedding = np.asarray(model.transform(x))
    explained = np.asarray(getattr(model, "explained_variance_ratio_", []))
    summary: dict[str, Any] = {
        "n_components": embedding.shape[1],
        "explained_variance_ratio": [round(float(value), 4) for value in explained],
        "cumulative_variance": [
            round(float(np.sum(explained[: i + 1])), 4) for i in range(len(explained))
        ],
        "data_hash": data_hash,
        "metrics": metrics,
    }
    if hasattr(model, "inverse_transform"):
        reconstructed = np.asarray(model.inverse_transform(embedding))
        summary["mean_reconstruction_error"] = round(float(((x - reconstructed) ** 2).mean()), 4)
    return summary


def _anomaly_distribution(score: pd.Series) -> dict[str, Any]:
    described = score.describe()
    return {
        "count": int(described["count"]),
        "flagged": int((score.to_numpy() > 0.0).sum()),
        "mean": round(float(described["mean"]), 4),
        "std": round(float(described["std"]), 4),
        "min": round(float(described["min"]), 4),
        "p25": round(float(described["25%"]), 4),
        "median": round(float(described["50%"]), 4),
        "p75": round(float(described["75%"]), 4),
        "max": round(float(described["max"]), 4),
    }
