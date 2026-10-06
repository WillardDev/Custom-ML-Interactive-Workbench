from __future__ import annotations

from typing import Final

import pandas as pd
from pandas.api.types import is_numeric_dtype

from ml_workbench.rules.preprocessing import SKEW_THRESHOLD
from ml_workbench.state import TaskDefinition

EDA_MAX_ROWS: Final = 100_000  # EDA-09: keep EDA fast above this threshold
CATEGORICAL_CARDINALITY_CAP: Final = 20  # Cramér's V pairs above this are skipped

BASE_VIEWS: Final[tuple[str, ...]] = (
    "shape",
    "dtypes",
    "missing_heatmap",
    "distributions",
    "correlation",
)

CLASSIFICATION_VIEWS: Final[tuple[str, ...]] = (
    "class_balance",
    "feature_by_class",
    "chi_square",
)

REGRESSION_VIEWS: Final[tuple[str, ...]] = (
    "target_histogram",
    "qq_plot",
    "feature_target_scatter",
)


def base_views() -> tuple[str, ...]:
    """EDA-01: views available for every task."""
    return BASE_VIEWS


def task_views(task: TaskDefinition) -> tuple[str, ...]:
    """EDA-02..08: task-specific view ids layered on top of the base views."""
    if task.task_type in {"binary", "multiclass", "multilabel"}:
        return CLASSIFICATION_VIEWS
    if task.task_type == "regression":
        return REGRESSION_VIEWS
    if task.task_type == "forecasting":
        return ("decomposition", "acf_pacf", "rolling_stats", "stationarity")
    if task.task_type == "clustering":
        return ("hopkins", "pca_umap_preview", "scree")
    if task.task_type == "anomaly_detection":
        return ("zscore_iqr_flags", "mahalanobis")
    if task.task_type == "dimensionality_reduction":
        return ("correlation_groups", "vif")
    if task.task_type == "association":
        return ("item_frequency", "basket_size")
    return ()


def eda_plan(task: TaskDefinition) -> tuple[str, ...]:
    """Full view plan for the task: base views + task-specific views."""
    return (*BASE_VIEWS, *task_views(task))


def skew(series: pd.Series) -> float:
    """Fisher–Pearson sample skew; NaN-safe (0.0 for empty/constant series)."""
    if len(series) == 0:
        return 0.0
    numeric = pd.to_numeric(series.dropna(), errors="coerce").astype(float)
    if len(numeric) < 3 or float(numeric.nunique()) < 2:
        return 0.0
    mean = float(numeric.mean())
    std = float(numeric.std())
    if std == 0:
        return 0.0
    size = float(len(numeric))
    centered = (numeric - mean) / std
    return (size / ((size - 1) * (size - 2))) * float((centered**3).sum())


def skew_hint(series: pd.Series) -> str | None:
    """HINT-02: |skew(target)| > 1 suggests a log transform."""
    value = skew(series)
    if abs(value) > SKEW_THRESHOLD:
        return f"|skew| = {abs(value):.2f} > 1 — a log transform is suggested (HINT-02/TT-01)."
    return None


def sample_frame_for_eda(
    frame: pd.DataFrame, *, limit: int = EDA_MAX_ROWS, seed: int = 0
) -> pd.DataFrame:
    """EDA-09: sample large frames so EDA stays fast; deterministic under a seed."""
    if len(frame) <= limit:
        return frame
    return frame.sample(limit, random_state=seed)


def categorical_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if not is_numeric_dtype(frame[column])]


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if is_numeric_dtype(frame[column])]


def categorical_cardinalities(frame: pd.DataFrame) -> dict[str, int]:
    """Nail down high-cardinality categoricals for Cramér's V; mirrors the EDA-01 cap."""
    return {
        column: int(frame[column].nunique(dropna=True))
        for column in categorical_columns(frame)
        if frame[column].nunique(dropna=True) <= CATEGORICAL_CARDINALITY_CAP
    }
