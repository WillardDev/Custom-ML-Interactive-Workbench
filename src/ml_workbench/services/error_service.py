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
)

from ml_workbench.rules.error_analysis import error_view_plan
from ml_workbench.rules.prediction import calibration_curve, threshold_metrics
from ml_workbench.services.model_cache import ModelCache
from ml_workbench.services.prediction_service import get_pipeline
from ml_workbench.services.workspace import Workspace
from ml_workbench.state import ProjectState


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
    """ERR-01/02/03/07: task-specific error views on the held-out test split."""
    if state.frame is None or state.task is None or state.task.target is None:
        raise ErrorAnalysisError("load a dataset and set a task first")
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
    return ErrorReport(
        task_type=state.task.task_type,
        views=tuple(views),
        worst_n=worst_n,
        segments=segments,
        predicted=predictions,
    )


def _view_names(task_type: str) -> list[str]:
    return [str(plan["view"]) for plan in error_view_plan(task_type)]


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
