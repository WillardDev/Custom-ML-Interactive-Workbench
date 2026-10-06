from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from ml_workbench.rules.eda import (
    EDA_MAX_ROWS,
    eda_plan,
    numeric_columns,
    sample_frame_for_eda,
    skew,
    skew_hint,
)
from ml_workbench.state import TaskDefinition


@dataclass(frozen=True)
class EdaException(ValueError):
    pass


@dataclass(frozen=True)
class BaseReport:
    rows: int
    columns: int
    shape: tuple[int, int]
    sampled_rows: int
    dtypes: dict[str, str]
    missing: dict[str, int]
    numeric_describe: pd.DataFrame | None
    numeric_pearson: pd.DataFrame | None
    numeric_spearman: pd.DataFrame | None
    cramers_v: pd.DataFrame | None


@dataclass(frozen=True)
class ClassificationReport:
    class_counts: pd.DataFrame
    feature_by_class: dict[str, pd.DataFrame] = field(default_factory=dict)
    chi_square: pd.DataFrame | None = None


@dataclass(frozen=True)
class RegressionReport:
    target_skew: float
    skew_hint: str | None
    target_histogram: pd.Series
    feature_scatter: pd.DataFrame | None


@dataclass(frozen=True)
class EdaResult:
    plan: tuple[str, ...]
    base: BaseReport
    views: tuple[str, ...]
    classification: ClassificationReport | None = None
    regression: RegressionReport | None = None


def _categorical(target: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [column for column in target.columns if is_numeric_dtype(target[column])]
    categorical = [column for column in target.columns if column not in numeric]
    return numeric, categorical


def _cramers_v(observed: np.ndarray) -> float:
    table = np.asarray(observed, dtype=float)
    total = float(table.sum())
    if total == 0:
        return 0.0
    row_tot = table.sum(axis=1, keepdims=True)
    col_tot = table.sum(axis=0, keepdims=True)
    expected = row_tot * col_tot / total
    chi2 = float(np.nansum(np.where(expected > 0, (table - expected) ** 2 / expected, 0.0)))
    denominator = total * (min(table.shape) - 1)
    if denominator == 0:
        return 0.0
    return float(np.sqrt(max(0.0, chi2 / denominator)))


def cramers_v_matrix(frame: pd.DataFrame, columns: list[str], cap: int = 20) -> pd.DataFrame | None:
    """Cramér's V for categorical column pairs (EDA-01)."""
    low = [column for column in columns if frame[column].nunique(dropna=True) <= cap]
    if len(low) == 0:
        return None
    matrix = pd.DataFrame(np.eye(len(low)), index=low, columns=low)
    for i, left in enumerate(low):
        for right in low[i + 1 :]:
            table = pd.crosstab(frame[left], frame[right])
            value = _cramers_v(table.to_numpy())
            matrix.loc[left, right] = value
            matrix.loc[right, left] = value
    return matrix


def _chi_square(frame: pd.DataFrame, target: str, feature_cols: list[str]) -> pd.DataFrame | None:
    rows: list[dict[str, object]] = []
    for feature in feature_cols:
        table = pd.crosstab(frame[feature], frame[target])
        observed = table.to_numpy()
        total = float(observed.sum())
        if total == 0:
            continue
        row_tot = observed.sum(axis=1, keepdims=True)
        col_tot = observed.sum(axis=0, keepdims=True)
        expected = row_tot * col_tot / total
        chi2 = float(np.nansum((observed - expected) ** 2 / np.where(expected > 0, expected, 1)))
        rows.append(
            {
                "feature": feature,
                "chi2": round(chi2, 4),
                "dof": int((observed.shape[0] - 1) * (observed.shape[1] - 1)),
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows)


def base_report(
    frame: pd.DataFrame,
    *,
    seed: int = 0,
    max_rows: int = EDA_MAX_ROWS,
) -> BaseReport:
    sampled_rows = len(frame)
    sample = sample_frame_for_eda(frame, limit=max_rows, seed=seed)
    sampled_rows = len(sample)
    numeric = numeric_columns(sample)

    numeric_describe = sample[numeric].describe().transpose() if numeric else None
    pearson = sample[numeric].corr(method="pearson") if numeric else None
    spearman = sample[numeric].corr(method="spearman") if numeric else None

    _, categorical = _categorical(sample)
    cramers = cramers_v_matrix(sample, categorical) if categorical else None

    return BaseReport(
        rows=len(frame),
        columns=len(frame.columns),
        shape=sample.shape,
        sampled_rows=sampled_rows,
        dtypes={str(column): str(dtype) for column, dtype in sample.dtypes.items()},
        missing={
            column: int(sample[column].isna().sum())
            for column in sample.columns
            if int(sample[column].isna().sum()) > 0
        },
        numeric_describe=numeric_describe,
        numeric_pearson=pearson,
        numeric_spearman=spearman,
        cramers_v=cramers,
    )


def classification_report(frame: pd.DataFrame, task: TaskDefinition) -> ClassificationReport:
    assert task.target is not None
    target = frame[task.target]
    counts = target.value_counts(dropna=True).rename("count")
    fractions = (target.value_counts(dropna=True, normalize=True)).rename("fraction")
    table = pd.DataFrame({"count": counts, "fraction": fractions})

    features = [column for column in frame.columns if column != task.target]
    feature_by_class: dict[str, pd.DataFrame] = {}
    for feature in features:
        if frame[feature].nunique(dropna=True) <= 20:
            feature_by_class[feature] = pd.crosstab(frame[feature], target)
    categorical_features = [column for column in features if not is_numeric_dtype(frame[column])]
    chi = _chi_square(frame, task.target, categorical_features)
    return ClassificationReport(
        class_counts=table, feature_by_class=feature_by_class, chi_square=chi
    )


def regression_report(frame: pd.DataFrame, task: TaskDefinition) -> RegressionReport:
    assert task.target is not None
    values = pd.to_numeric(frame[task.target], errors="coerce").dropna()
    histogram = values.value_counts(bins=30, sort=False).sort_index()
    features = [column for column in frame.columns if column != task.target]
    sampled = sample_frame_for_eda(frame, limit=EDA_MAX_ROWS)
    scatter = sampled[features + [task.target]].dropna() if len(sampled) <= 5000 else None
    return RegressionReport(
        target_skew=skew(values),
        skew_hint=skew_hint(values),
        target_histogram=histogram,
        feature_scatter=scatter,
    )


def eda_report(frame: pd.DataFrame, task: TaskDefinition, *, seed: int = 0) -> EdaResult:
    plan = eda_plan(task)
    base = base_report(frame, seed=seed)
    views: list[str] = []
    classification = None
    regression = None
    if task.task_type in {"binary", "multiclass", "multilabel"}:
        classification = classification_report(frame, task)
        views = ["class_balance", "feature_by_class", "chi_square"]
    elif task.task_type == "regression":
        regression = regression_report(frame, task)
        views = ["target_histogram", "qq_plot", "feature_target_scatter"]
    return EdaResult(
        plan=plan,
        base=base,
        views=tuple(views),
        classification=classification,
        regression=regression,
    )
