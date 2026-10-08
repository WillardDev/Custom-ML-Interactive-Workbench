from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    silhouette_samples,
)

from ml_workbench.rules.error_analysis import error_view_plan, neural_diagnostics
from ml_workbench.rules.prediction import calibration_curve, threshold_metrics
from ml_workbench.services.model_cache import ModelCache
from ml_workbench.services.prediction_service import get_pipeline
from ml_workbench.services.training_service import _anomaly_decision
from ml_workbench.services.workspace import Workspace, WorkspaceError
from ml_workbench.state import ProjectState, feature_columns


class ErrorAnalysisError(ValueError):
    pass


WORST_N: int = 10
SEGMENT_MAX_CATEGORIES: int = 8


@dataclass(frozen=True)
class ErrorReport:
    task_type: str
    views: tuple[dict[str, Any], ...]
    worst_n: pd.DataFrame
    segments: pd.DataFrame | None
    predicted: pd.DataFrame


def error_views(
    state: ProjectState,
    workspace: Workspace,
    run_id: str,
    *,
    cache: ModelCache | None = None,
) -> ErrorReport:
    """ERR-01..08: task-specific error views on the held-out test split."""
    if state.frame is None or state.task is None:
        raise ErrorAnalysisError("load a dataset and set a task first")
    if state.task.learning_type == "unsupervised":
        return _unsupervised_error_views(state, workspace, run_id, cache=cache)
    if state.task.target is None:
        raise ErrorAnalysisError("a supervised task needs a target column")
    if state.split is None or state.split.indices_path is None:
        raise ErrorAnalysisError("no test split saved — run Preprocessing first")

    indices = json.loads(Path(state.split.indices_path).read_text())
    test_idx = np.asarray(indices["test"], dtype=int)
    if len(test_idx) == 0:
        raise ErrorAnalysisError("the split produced no test rows")

    pipeline = get_pipeline(workspace, run_id, cache)
    test_frame = state.frame.iloc[test_idx].reset_index(drop=True)
    features = [column for column in test_frame.columns if column != state.task.target]
    y_true = np.asarray(state.frame[state.task.target].iloc[test_idx])
    y_pred = np.asarray(pipeline.predict(test_frame[features]))
    proba = _positive_proba(pipeline, test_frame[features], state.task.task_type, y_pred)

    predictions = test_frame[[*features, state.task.target]].copy()
    predictions["true"] = y_true
    predictions["prediction"] = y_pred

    views = [
        {"view": name, **payload}
        for name, payload in zip(
            _view_names(state.task.task_type),
            _build_views(state.task.task_type, y_true, y_pred, proba),
            strict=False,
        )
    ]
    worst_n = _worst_n_rows(predictions, state.task.target)
    segments = _segment_slices(predictions, y_true, y_pred)
    views.extend(_neural_error_views(workspace, run_id))
    return ErrorReport(
        task_type=state.task.task_type,
        views=tuple(views),
        worst_n=worst_n,
        segments=segments,
        predicted=predictions,
    )


def _view_names(task_type: str) -> list[str]:
    return [str(plan["view"]) for plan in error_view_plan(task_type)]


def _neural_error_views(workspace: Workspace, run_id: str) -> list[dict[str, Any]]:
    """ERR-08: learning curves, overfitting diagnostics and per-epoch metrics for neural runs."""
    try:
        payload = workspace.read_run_json(run_id, "metrics.json")
    except WorkspaceError:
        return []
    neural = payload.get("neural")
    if not isinstance(neural, dict):
        return []
    train_curve = [float(value) for value in neural.get("loss_curve", [])]
    val_curve = [float(value) for value in neural.get("val_loss_curve", [])]
    if not train_curve:
        return []
    diagnostics = neural_diagnostics(
        train_curve,
        val_curve,
        best_epoch=int(neural.get("best_epoch", -1)),
        epochs_run=int(neural.get("epochs_run", len(train_curve))),
        epochs=int(neural.get("epochs", len(train_curve))),
        patience=int(neural.get("patience", 10)),
    )
    rows = [
        {"epoch": index + 1, "train_loss": train_loss, "val_loss": val_loss}
        for index, (train_loss, val_loss) in enumerate(zip(train_curve, val_curve, strict=False))
    ]
    return [
        {
            "view": "learning_curve",
            "train_loss": train_curve,
            "val_loss": val_curve,
            "best_epoch": diagnostics["best_epoch"],
        },
        {"view": "overfitting_diagnostics", **diagnostics},
        {"view": "per_epoch_metrics", "rows": rows},
    ]


def _positive_proba(
    pipeline: Any, test_frame: pd.DataFrame, task_type: str, y_pred: np.ndarray
) -> np.ndarray | None:
    if task_type != "binary":
        return None
    if not hasattr(pipeline, "predict_proba"):
        return None
    proba = np.asarray(pipeline.predict_proba(test_frame))
    if proba.ndim != 2:
        return None
    return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]


def _build_views(
    task_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None,
) -> tuple[dict[str, Any], ...]:
    if task_type == "binary":
        return _binary_views(y_true, y_pred, proba)
    if task_type in {"multiclass", "multilabel"}:
        return _multiclass_views(y_true, y_pred)
    if task_type in {"regression", "forecasting"}:
        return _regression_views(y_true, y_pred)
    return ({"view": "prediction_table"},)


def _binary_views(
    y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None
) -> tuple[dict[str, Any], ...]:
    classes = np.unique(np.concatenate([y_true, y_pred])).tolist()
    score = float(np.mean(y_true == y_pred))
    matrix = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=classes),
        index=[f"true_{c}" for c in classes],
        columns=[f"pred_{c}" for c in classes],
    )
    if proba is None:
        fpr = [0.0, 1.0]
        tpr = [0.0, 1.0]
        precision = [float(score), 1.0]
        recall = [1.0, float(score)]
        average_precision = score
    else:
        fpr, tpr, _ = roc_curve(y_true, proba)
        precision, recall, _ = precision_recall_curve(y_true, proba)
        average_precision = average_precision_score(y_true, proba)
        centers, observed = calibration_curve(proba, y_true)
    rows = []
    for threshold in np.linspace(0.0, 1.0, 9):
        rows.append(
            threshold_metrics(
                y_true, proba if proba is not None else y_pred, threshold=float(threshold)
            )
        )
    return (
        {"confusion": matrix.round(4)},
        {
            "fpr": [float(v) for v in fpr],
            "tpr": [float(v) for v in tpr],
            "auc": score if proba is None else roc_auc_score(y_true, proba),
        },
        {
            "precision": [float(v) for v in precision],
            "recall": [float(v) for v in recall],
            "average_precision": float(average_precision),
        },
        {"rows": rows},
        (
            {"centers": [float(v) for v in centers], "observed": [float(v) for v in observed]}
            if proba is not None
            else {}
        ),
    )


def _multiclass_views(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict[str, Any], ...]:
    classes = np.unique(np.concatenate([y_true, y_pred])).tolist()
    matrix = confusion_matrix(y_true, y_pred, labels=classes)
    row_sum = matrix.sum(axis=1, keepdims=True)
    normalized = matrix / row_sum if row_sum.any() else matrix
    precision, recall, _, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    per_class = [
        {
            "class": str(label),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(classes)
    ]
    pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j:
                pairs.append((int(matrix[i, j]), str(classes[i]), str(classes[j])))
    most_confused = [
        {"count": count, "true": true, "predicted": predicted}
        for count, true, predicted in sorted(pairs, reverse=True)[:5]
    ]
    return (
        {"matrix": pd.DataFrame(normalized, index=classes, columns=classes).round(4)},
        {"per_class": per_class},
        {"most_confused": most_confused},
    )


def _regression_views(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict[str, Any], ...]:
    residuals = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    counts, edges = np.histogram(residuals, bins=20)
    quantiles = (np.arange(len(residuals)) + 0.5) / max(len(residuals), 1)
    theoretical = _norm.ppf(quantiles)
    spearman = float(
        spearmanr(np.abs(residuals), np.abs(np.asarray(y_pred, dtype=float))).statistic
    )
    try:
        error_by_quantile = (
            pd.DataFrame(
                {"bin": pd.qcut(y_true, q=4, duplicates="drop"), "abs_error": np.abs(residuals)}
            )
            .groupby("bin", observed=True)["abs_error"]
            .mean()
        )
        by_quantile = [
            {"bin": str(level), "mean_abs_error": float(value)}
            for level, value in error_by_quantile.items()
        ]
    except (ValueError, TypeError):
        by_quantile = []
    return (
        {"points": [(float(p), float(r)) for p, r in zip(y_pred, residuals, strict=False)]},
        {"counts": [int(c) for c in counts], "edges": [float(e) for e in edges]},
        {"x": [float(q) for q in theoretical], "y": [float(r) for r in np.sort(residuals)]},
        {"spearman": spearman},
        {"by_quantile": by_quantile},
    )


def _worst_n_rows(predictions: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = predictions.drop(columns=[target]).copy()
    rows["error"] = np.abs(rows["prediction"].astype(float) - rows["true"].astype(float))
    rows = rows.sort_values("error", ascending=False).head(WORST_N).reset_index(drop=True)
    return rows


def _segment_slices(
    predictions: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray
) -> pd.DataFrame | None:
    categorical = [
        column
        for column in predictions.columns
        if column not in {"true", "prediction", "error"}
        and not pd.api.types.is_numeric_dtype(predictions[column])
    ]
    for column in categorical:
        cardinality = predictions[column].nunique(dropna=True)
        if cardinality <= SEGMENT_MAX_CATEGORIES:
            errors = np.abs(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float))
            segment = pd.DataFrame(
                {"column": column, "value": predictions[column], "error": errors}
            )
            grouped = (
                segment.groupby(["column", "value"], dropna=False)["error"]
                .agg(["mean", "count"])
                .reset_index()
            )
            grouped = grouped.sort_values("mean", ascending=False).reset_index(drop=True)
            return grouped.rename(columns={"mean": "mean_abs_error", "count": "rows"})
    return None


def _resolve_unsupervised_views(
    report_payloads: list[dict[str, Any]], task_type: str
) -> tuple[dict[str, Any], ...]:
    """Order accessor payloads to match the ERR-04/05/06 view plan."""
    positions = {plan["view"]: index for index, plan in enumerate(error_view_plan(task_type))}
    ordered: list[dict[str, Any]] = [{} for _ in positions]
    for payload in report_payloads:
        name = str(payload["view"])
        ordered[positions[name]] = payload
    return tuple(ordered)


def _clustering_error_payloads(x: np.ndarray, labels: np.ndarray) -> list[dict[str, Any]]:
    counts = np.bincount(labels.astype(int), minlength=int(labels.max()) + 1)
    sizes = [int(round(float(v))) for v in counts.tolist()]
    ratio = round(float(max(sizes) / min(sizes)), 4) if len(sizes) > 1 and min(sizes) > 0 else None
    if len(np.unique(labels)) > 1 and len(x) >= 4:
        per_sample = silhouette_samples(x, labels)
        table = pd.DataFrame({"row": np.arange(len(per_sample)), "silhouette": per_sample})
        low = table[table["silhouette"] < 0.0].sort_values("silhouette")
        low_points = low if len(low) else table.sort_values("silhouette")
        payloads = []
        payloads.append(
            {
                "view": "silhouette_per_sample",
                "mean": round(float(np.mean(per_sample)), 4),
                "values": [round(float(v), 4) for v in per_sample[:200].tolist()],
            }
        )
        payloads.append(
            {
                "view": "low_silhouette_points",
                "count": int((per_sample < 0.0).sum()),
                "table": low_points.head(WORST_N).reset_index(drop=True),
            }
        )
    else:
        payloads = [
            {"view": "silhouette_per_sample", "mean": float("nan"), "values": []},
            {"view": "low_silhouette_points", "count": 0, "table": pd.DataFrame()},
        ]
    payloads.append(
        {
            "view": "cluster_size_imbalance",
            "labels": list(range(len(sizes))),
            "counts": sizes,
            "max_min_ratio": ratio,
        }
    )
    payloads.append(
        {
            "view": "stability_warning",
            "warning": (
                f"largest cluster is {ratio}x the smallest — consider re-centering or a different k"
                if ratio is not None and ratio >= 2.0
                else "cluster sizes are reasonably balanced"
            ),
        }
    )
    return payloads


def _dimred_error_payloads(model: Any, x: np.ndarray) -> list[dict[str, Any]]:
    if not hasattr(model, "inverse_transform"):
        empty = pd.DataFrame(columns=["row", "reconstruction_error"])
        return [
            {"view": "reconstruction_error_per_row", "mean": float("nan"), "table": empty},
            {"view": "poorly_embedded_points", "count": 0, "threshold": None},
        ]
    embedding = np.asarray(model.transform(x))
    reconstructed = np.asarray(model.inverse_transform(embedding))
    per_row = np.sqrt(((x - reconstructed) ** 2).sum(axis=1))
    threshold = float(np.percentile(per_row, 80))
    table = pd.DataFrame(
        {"row": np.arange(len(per_row)), "reconstruction_error": per_row}
    ).sort_values("reconstruction_error", ascending=False)
    return [
        {
            "view": "reconstruction_error_per_row",
            "mean": round(float(np.mean(per_row)), 4),
            "table": table.head(WORST_N).reset_index(drop=True),
        },
        {
            "view": "poorly_embedded_points",
            "count": int((per_row > threshold).sum()),
            "threshold": round(threshold, 4),
        },
    ]


def _anomaly_error_payloads(decision: np.ndarray, score: pd.Series) -> list[dict[str, Any]]:
    top = score.sort_values(ascending=False).head(WORST_N)
    top_table = pd.DataFrame({"row": top.index.to_numpy(), "score": top.to_numpy()})
    return [
        {
            "view": "score_distribution",
            "mean": round(float(decision.mean()), 4),
            "std": round(float(decision.std()), 4),
            "flagged": int((decision > 0.0).sum()),
        },
        {
            "view": "top_flagged_rows",
            "count": int((decision > 0.0).sum()),
            "quantiles": {
                str(q): round(float(score.quantile(q)), 4) for q in (0.5, 0.9, 0.95, 0.99)
            },
            "table": top_table,
        },
    ]


def _unsupervised_error_views(
    state: ProjectState,
    workspace: Workspace,
    run_id: str,
    *,
    cache: ModelCache | None,
) -> ErrorReport:
    assert state.frame is not None and state.task is not None
    task = state.task
    if state.split is None or state.split.indices_path is None:
        raise ErrorAnalysisError("no test split saved — run Preprocessing first")
    indices = json.loads(Path(state.split.indices_path).read_text())
    test_idx = np.asarray(indices["test"], dtype=int)
    if len(test_idx) == 0:
        raise ErrorAnalysisError(
            "the split produced no test rows (add evaluation labels or split the data)"
        )
    pipeline = get_pipeline(workspace, run_id, cache)
    test_frame = state.frame.iloc[test_idx].reset_index(drop=True)
    features = list(feature_columns(task, state.frame))
    x_test = np.asarray(pipeline.named_steps["preprocess"].transform(test_frame[features]))
    model = pipeline.named_steps["model"]
    predictions = test_frame.copy()
    payloads: list[dict[str, Any]] = []
    task_type = task.task_type
    if task.eval_labels is not None:
        predictions["true"] = np.asarray(state.frame[task.eval_labels].iloc[test_idx])
    if task_type == "clustering":
        labels = np.asarray(model.predict(x_test))
        predictions["prediction"] = labels
        payloads = _clustering_error_payloads(x_test, labels)
    elif task_type == "dimensionality_reduction":
        embedding = np.asarray(model.transform(x_test))
        for k in range(embedding.shape[1]):
            predictions[f"pc{k + 1}"] = embedding[:, k]
        payloads = _dimred_error_payloads(model, x_test)
    elif task_type == "anomaly_detection":
        decision = _anomaly_decision(model, x_test)
        predictions["score"] = decision
        predictions["prediction"] = (decision > 0.0).astype(int)
        payloads = _anomaly_error_payloads(decision, predictions["score"])
        if task.eval_labels is not None:
            binary = (predictions["true"].to_numpy() != 0).astype(int)
            flag = (decision > 0.0).astype(int)
            payloads.append(
                {
                    "view": "false_positives_negatives",
                    "tp": int(((flag == 1) & (binary == 1)).sum()),
                    "fp": int(((flag == 1) & (binary == 0)).sum()),
                    "fn": int(((flag == 0) & (binary == 1)).sum()),
                    "tn": int(((flag == 0) & (binary == 0)).sum()),
                }
            )
    views = _resolve_unsupervised_views(payloads, task_type)
    views = (*views, *_neural_error_views(workspace, run_id))
    worst_n = (
        predictions.sort_values(["score", "prediction"], ascending=False).head(WORST_N)
        if "score" in predictions.columns
        else predictions.head(WORST_N)
    )
    worst_n = worst_n.reset_index(drop=True)
    return ErrorReport(
        task_type=task_type,
        views=views,
        worst_n=worst_n,
        segments=None,
        predicted=predictions,
    )
