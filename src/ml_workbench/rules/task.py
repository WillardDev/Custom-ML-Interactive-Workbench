from __future__ import annotations

from typing import Final

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_integer_dtype, is_numeric_dtype

from ml_workbench.state import TaskDefinition, TaskError

MULTICLASS_UNIQUE_THRESHOLD: Final = 20

SUPERVISED_TASK_TYPES: Final[frozenset[str]] = frozenset(
    {"binary", "multiclass", "multilabel", "regression", "forecasting"}
)
UNSUPERVISED_TASK_TYPES: Final[frozenset[str]] = frozenset(
    {"clustering", "dimensionality_reduction", "anomaly_detection", "association"}
)

TASK_FAMILY: Final[dict[str, str]] = {
    "binary": "classification",
    "multiclass": "classification",
    "multilabel": "classification",
}


def registry_task_type(task_type: str) -> str:
    """Map a TaskDefinition.task_type to the registry's coarse task family."""
    return TASK_FAMILY.get(task_type, task_type)


def suggest_task_type(series: pd.Series) -> str:
    """TASK-01: infer a supervised task type from the target column."""
    if is_datetime64_any_dtype(series):
        return "forecasting"
    nunique = series.nunique(dropna=True)
    if nunique < 2:
        raise TaskError("target must contain at least 2 distinct values")
    if nunique == 2:
        return "binary"
    if is_numeric_dtype(series):
        if is_integer_dtype(series) and nunique <= MULTICLASS_UNIQUE_THRESHOLD:
            return "multiclass"
        return "regression"
    return "multiclass"


def build_task(
    frame: pd.DataFrame,
    learning_type: str,
    *,
    target: str | None = None,
    group: str | None = None,
    time_column: str | None = None,
    eval_labels: str | None = None,
    task_type: str | None = None,
    confirm_forecasting: bool = False,
) -> TaskDefinition:
    """Build a validated task definition (TASK-01, TASK-02, TASK-03)."""
    if learning_type not in {"supervised", "unsupervised"}:
        raise TaskError(f"unknown learning type '{learning_type}'")

    chosen = [name for name in (target, group, time_column, eval_labels) if name is not None]
    missing = [name for name in chosen if name not in frame.columns]
    if missing:
        raise TaskError(f"unknown column(s): {', '.join(missing)}")

    if learning_type == "unsupervised":
        if target is not None:
            raise TaskError("unsupervised tasks must not set a target column (TASK-03)")
        if task_type is None:
            task_type = "clustering"
        if task_type not in UNSUPERVISED_TASK_TYPES:
            raise TaskError(
                f"unknown unsupervised task type '{task_type}' "
                f"(expected one of {', '.join(sorted(UNSUPERVISED_TASK_TYPES))})"
            )
        return TaskDefinition(
            learning_type=learning_type,
            task_type=task_type,
            target=None,
            group=group,
            time_column=time_column,
            eval_labels=eval_labels,
        )

    if target is None:
        raise TaskError("supervised tasks require a target column")

    if time_column is not None:
        if not confirm_forecasting:
            raise TaskError(
                "a time column selects a forecasting task only with explicit confirmation (TASK-02)"
            )
        resolved = "forecasting"
    elif task_type is not None:
        resolved = task_type
    else:
        resolved = suggest_task_type(frame[target])
        if resolved == "forecasting" and not confirm_forecasting:
            raise TaskError(
                "target is a datetime column; confirm forecasting to use it (TASK-02), "
                "otherwise choose a numeric target"
            )

    if resolved not in SUPERVISED_TASK_TYPES:
        raise TaskError(
            f"unknown supervised task type '{resolved}' "
            f"(expected one of {', '.join(sorted(SUPERVISED_TASK_TYPES))})"
        )
    if resolved == "forecasting" and time_column is None:
        raise TaskError("forecasting requires a time column")

    return TaskDefinition(
        learning_type=learning_type,
        task_type=resolved,
        target=target,
        group=group,
        time_column=time_column,
        eval_labels=eval_labels,
    )
