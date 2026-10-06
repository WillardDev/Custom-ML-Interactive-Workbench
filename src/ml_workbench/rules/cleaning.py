from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer

from ml_workbench.state import TaskError

SENTINEL_STRINGS: Final[frozenset[str]] = frozenset(
    {"", "n/a", "na", "null", "none", "nan", "-", "?"}
)
IMPUTATION_STRATEGIES: Final[tuple[str, ...]] = (
    "median",
    "mean",
    "knn",
    "iterative",
    "mode",
    "constant",
)
NUMERIC_IMPUTATION_STRATEGIES: Final[frozenset[str]] = frozenset(
    {"median", "mean", "knn", "iterative"}
)
OUTLIER_OPS: Final[tuple[str, ...]] = ("keep", "flag", "clip", "winsorize")
CLASSIFICATION_TASK_TYPES: Final[frozenset[str]] = frozenset({"binary", "multiclass", "multilabel"})
MINORITY_FRACTION_THRESHOLD: Final = 0.20


@dataclass(frozen=True)
class CleanReport:
    rows_before: int
    rows_after: int
    duplicates_removed: int
    dtype_changes: dict[str, str]
    sentinels_replaced: int


@dataclass(frozen=True)
class NanDecision:
    handles_nan: bool
    allow_leave_nan: bool
    imputation_required: bool
    strategies: tuple[str, ...]


@dataclass(frozen=True)
class OutlierDecision:
    allowed: tuple[str, ...]
    default: str
    recommend: str
    rule_id: str


@dataclass(frozen=True)
class ClassBalance:
    counts: dict[str, int]
    fractions: dict[str, float]
    minority_fraction: float
    imbalanced: bool
    threshold: float = MINORITY_FRACTION_THRESHOLD


def remove_duplicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    rows_before = len(frame)
    out = frame.drop_duplicates().reset_index(drop=True)
    return out, rows_before - len(out)


def fix_dtypes(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str], int]:
    out = frame.copy()
    dtype_changes: dict[str, str] = {}
    sentinels_replaced = 0

    for column in out.columns:
        series = out[column]
        if not is_string_dtype(series):
            continue
        stripped = series.str.strip()
        sentinel_mask = stripped.str.lower().isin(SENTINEL_STRINGS)
        cleaned = stripped.mask(sentinel_mask)
        sentinels_replaced += int(sentinel_mask.sum())
        converted = pd.to_numeric(cleaned, errors="coerce")
        failed = cleaned.notna() & converted.isna()
        if not bool(failed.any()) and bool(cleaned.notna().any()):
            out[column] = converted
            if str(converted.dtype) != str(series.dtype):
                dtype_changes[column] = f"{series.dtype} -> {converted.dtype}"
        else:
            out[column] = cleaned

    return out, dtype_changes, sentinels_replaced


def dedupe_and_fix_dtypes(frame: pd.DataFrame) -> tuple[pd.DataFrame, CleanReport]:
    deduped, duplicates_removed = remove_duplicates(frame)
    fixed, dtype_changes, sentinels_replaced = fix_dtypes(deduped)
    return fixed, CleanReport(
        rows_before=len(frame),
        rows_after=len(fixed),
        duplicates_removed=duplicates_removed,
        dtype_changes=dtype_changes,
        sentinels_replaced=sentinels_replaced,
    )


def drop_missing_target(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in frame.columns:
        raise TaskError(f"unknown target column '{target}'")
    return frame.loc[frame[target].notna()].reset_index(drop=True)


def nan_decision(handles_nan: bool) -> NanDecision:
    return NanDecision(
        handles_nan=handles_nan,
        allow_leave_nan=handles_nan,
        imputation_required=not handles_nan,
        strategies=IMPUTATION_STRATEGIES,
    )


def columns_with_missing(frame: pd.DataFrame, exclude: Sequence[str] = ()) -> list[str]:
    excluded = set(exclude)
    return [
        str(column)
        for column in frame.columns
        if column not in excluded and bool(frame[column].isna().any())
    ]


def apply_imputation(
    frame: pd.DataFrame,
    columns: Sequence[str],
    strategy: str,
    *,
    target: str | None = None,
    fill_value: Any = None,
    seed: int = 0,
) -> pd.DataFrame:
    if strategy not in IMPUTATION_STRATEGIES:
        raise TaskError(f"unknown imputation strategy '{strategy}'")
    if not columns:
        raise TaskError("no columns selected for imputation")
    if target is not None and target in columns:
        raise TaskError("the target column is never imputed; drop those rows instead (CLEAN-02)")
    unknown = [column for column in columns if column not in frame.columns]
    if unknown:
        raise TaskError(f"unknown column(s): {', '.join(unknown)}")

    out = frame.copy()
    if strategy in NUMERIC_IMPUTATION_STRATEGIES:
        non_numeric = [column for column in columns if not is_numeric_dtype(out[column])]
        if non_numeric:
            raise TaskError(
                f"strategy '{strategy}' needs numeric columns: {', '.join(non_numeric)}"
            )

    if strategy in {"median", "mean"}:
        for column in columns:
            fill = out[column].median() if strategy == "median" else out[column].mean()
            out[column] = out[column].fillna(fill)
    elif strategy == "mode":
        for column in columns:
            modes = out[column].mode(dropna=True)
            if modes.empty:
                raise TaskError(f"column '{column}' has no mode value to fill with")
            out[column] = out[column].fillna(modes.iloc[0])
    elif strategy == "constant":
        if fill_value is None:
            raise TaskError("constant imputation requires a fill value")
        out = out.fillna({column: fill_value for column in columns})
    else:
        imputer = (
            KNNImputer() if strategy == "knn" else IterativeImputer(random_state=seed, max_iter=10)
        )
        values = imputer.fit_transform(out[list(columns)])
        for index, column in enumerate(columns):
            out[column] = values[:, index]
    return out


def outlier_decision(task_type: str, family: str | None) -> OutlierDecision:
    if task_type == "anomaly_detection":
        return OutlierDecision(
            allowed=("keep", "flag"),
            default="flag",
            recommend=(
                "keep every row and flag outliers — in anomaly detection the outliers "
                "are the signal (CLEAN-04a)"
            ),
            rule_id="CLEAN-04a",
        )
    if family == "tree":
        return OutlierDecision(
            allowed=("keep", "flag"),
            default="flag",
            recommend="keep and flag outliers — the model labels noise itself (CLEAN-04b)",
            rule_id="CLEAN-04b",
        )
    return OutlierDecision(
        allowed=("keep", "clip", "winsorize"),
        default="keep",
        recommend=(
            "treatment is optional: winsorize or clip extreme values, or use robust "
            "scaling downstream (CLEAN-04c)"
        ),
        rule_id="CLEAN-04c",
    )


def apply_outlier_treatment(
    frame: pd.DataFrame,
    columns: Sequence[str],
    op: str,
    *,
    factor: float = 1.5,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    if op not in OUTLIER_OPS:
        raise TaskError(f"unknown outlier treatment '{op}' (allowed: {', '.join(OUTLIER_OPS)})")
    if op == "keep" or not columns:
        return frame
    unknown = [column for column in columns if column not in frame.columns]
    if unknown:
        raise TaskError(f"unknown column(s): {', '.join(unknown)}")
    non_numeric = [column for column in columns if not is_numeric_dtype(frame[column])]
    if non_numeric:
        raise TaskError(f"outlier treatment needs numeric columns: {', '.join(non_numeric)}")

    out = frame.copy()
    for column in columns:
        series = out[column]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = float(q1 - factor * iqr)
        upper = float(q3 + factor * iqr)
        if op == "flag":
            out[f"{column}_outlier"] = (series < lower) | (series > upper)
        elif op == "clip":
            out[column] = series.clip(lower, upper)
        else:
            out[column] = series.clip(
                float(series.quantile(lower_quantile)), float(series.quantile(upper_quantile))
            )
    return out


def class_balance(frame: pd.DataFrame, target: str, task_type: str) -> ClassBalance | None:
    if task_type not in CLASSIFICATION_TASK_TYPES:
        return None
    if target not in frame.columns:
        raise TaskError(f"unknown target column '{target}'")
    value_counts = frame[target].value_counts(dropna=True)
    total = int(value_counts.sum())
    if total == 0:
        raise TaskError("target column has no values")
    counts = {str(label): int(count) for label, count in value_counts.items()}
    fractions = {label: count / total for label, count in counts.items()}
    minority = min(fractions.values())
    return ClassBalance(
        counts=counts,
        fractions=fractions,
        minority_fraction=minority,
        imbalanced=minority < MINORITY_FRACTION_THRESHOLD,
    )
