from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

from ml_workbench.rules.split import SplitRule, choose_split, default_holdout_fraction
from ml_workbench.rules.staleness import mark_downstream_stale
from ml_workbench.services.workspace import Workspace, utc_now
from ml_workbench.state import PipelineInfo, ProjectState, SplitInfo, StepEntry, TaskDefinition
from ml_workbench.state import feature_columns as frame_features


class PreprocessingError(ValueError):
    pass


@dataclass(frozen=True)
class PreprocessingOptions:
    scaler: str | None = "standard"
    encoder: str = "ordinal"
    target_transform: str | None = "log"
    test_size: float = default_holdout_fraction()
    seed: int | None = None


@dataclass(frozen=True)
class SplitResult:
    train: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class PipelineReport:
    pipeline: Any
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    encoder: str
    scaler: str | None


@dataclass(frozen=True)
class PreprocessingResult:
    strategy: str
    rule_id: str
    train_rows: int
    test_rows: int
    feature_count: int
    encoder: str
    scaler: str | None
    target_transform: str | None
    pipeline_path: str


SCALERS: dict[str, object] = {
    "standard": StandardScaler(),
    "robust": RobustScaler(),
    "minmax": MinMaxScaler(),
}


def materialize_split(
    frame: pd.DataFrame,
    task: TaskDefinition,
    rule: SplitRule,
    *,
    test_size: float,
    seed: int,
) -> SplitResult:
    """Turn a SplitRule into concrete train/test row indices (SPLIT-01..07)."""
    rows = np.arange(len(frame))

    if rule.strategy == "none":
        return SplitResult(train=rows, test=np.array([], dtype=int))

    if rule.strategy == "chronological":
        assert task.time_column is not None
        order = np.argsort(frame[task.time_column].to_numpy(), kind="stable")
        cut = int(len(frame) * (1.0 - test_size))
        ordered = order[:cut] if cut < len(frame) else order[:-1]
        return SplitResult(train=ordered, test=np.setdiff1d(rows, ordered))

    if rule.strategy == "grouped":
        assert task.group is not None
        groups = frame[task.group].to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train, test = next(splitter.split(rows, groups=groups))
        return SplitResult(train=train, test=test)

    if rule.strategy in {"stratified", "eval_labels"}:
        labels = (
            frame[task.target].to_numpy()
            if task.target is not None
            else frame[task.eval_labels].to_numpy()
        )
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train, test = next(splitter.split(rows, labels))
        return SplitResult(train=train, test=test)

    train, test = train_test_split(rows, test_size=test_size, random_state=seed, shuffle=True)
    return SplitResult(train=train, test=test)


def _column_groups(frame: pd.DataFrame, task: TaskDefinition) -> tuple[list[str], list[str]]:
    features = list(frame_features(task, frame))
    numeric = [column for column in features if is_numeric_dtype(frame[column])]
    categorical = [column for column in features if column not in numeric]
    return numeric, categorical


def build_pipeline(
    frame: pd.DataFrame,
    train_indices: np.ndarray,
    task: TaskDefinition,
    *,
    encoder: str = "ordinal",
    scaler: str | None = "standard",
) -> PipelineReport:
    """PIPE-01/ENC-05/SCALE-04: build and fit the pipeline on the training fold only."""
    numeric, categorical = _column_groups(frame, task)
    train_frame = frame.iloc[train_indices]

    if task.task_type == "association":
        # Association mines raw item strings — skip encoding/scaling entirely.
        passthrough = Pipeline([("columns", ColumnTransformer([], remainder="passthrough"))])
        passthrough.fit(train_frame)
        return PipelineReport(
            pipeline=passthrough,
            numeric_columns=(),
            categorical_columns=(),
            encoder=encoder,
            scaler=scaler,
        )

    transformers: list[tuple[str, object, list[str]]] = []
    if numeric:
        numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
        if scaler is not None:
            if scaler not in SCALERS:
                raise PreprocessingError(f"unknown scaler '{scaler}'")
            numeric_steps.append(("scaler", SCALERS[scaler]))
        transformers.append(("num", Pipeline(numeric_steps), numeric))
    if categorical:
        if encoder == "one_hot":
            transformer: object = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        elif encoder == "ordinal":
            transformer = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2
            )
        else:
            raise PreprocessingError(f"unknown encoder '{encoder}'")
        transformers.append(("cat", transformer, categorical))

    column_transformer = ColumnTransformer(transformers, remainder="drop")
    pipeline = Pipeline([("columns", column_transformer)])
    pipeline.fit(train_frame)
    return PipelineReport(
        pipeline=pipeline,
        numeric_columns=tuple(numeric),
        categorical_columns=tuple(categorical),
        encoder=encoder,
        scaler=scaler,
    )


def apply_target_transform(
    series: pd.Series, name: str | None, *, seed: int = 0
) -> tuple[pd.Series, object]:
    """TT-01: fit a target transform on the training target (leak-safe)."""
    if name is None or name == "log":
        transformer: Any = FunctionTransformer(np.log1p, np.expm1)
        return pd.Series(
            np.log1p(pd.to_numeric(series, errors="coerce")), index=series.index
        ), transformer
    transformer = PowerTransformer(method="box-cox" if name == "boxcox" else "yeojohnson")
    fitted = transformer.fit(series.to_numpy().reshape(-1, 1))
    return pd.Series(
        np.ravel(fitted.transform(series.to_numpy().reshape(-1, 1))), index=series.index
    ), fitted


def _next_step_id(entries: list[StepEntry]) -> int:
    return max((entry.id for entry in entries), default=0) + 1


def run_preprocessing(
    state: ProjectState, workspace: Workspace, options: PreprocessingOptions
) -> PreprocessingResult:
    if state.frame is None or state.dataset is None or state.task is None:
        raise PreprocessingError("load a dataset and set a task before preprocessing")
    if not state.cleaned:
        raise PreprocessingError("run Data Cleaning before preprocessing")

    frame, task = state.frame, state.task
    seed = options.seed if options.seed is not None else state.seed
    rule = choose_split(task)
    split = materialize_split(frame, task, rule, test_size=options.test_size, seed=seed)
    report = build_pipeline(
        frame, split.train, task, encoder=options.encoder, scaler=options.scaler
    )

    workspace.ensure()
    pipeline_path = workspace.preprocessing_dir / "pipeline.joblib"
    joblib.dump(report.pipeline, pipeline_path)

    split_path = workspace.preprocessing_dir / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "strategy": rule.strategy,
                "rule_id": rule.rule_id,
                "test_size": options.test_size,
                "train": [int(index) for index in split.train],
                "test": [int(index) for index in split.test],
            },
            indent=2,
        )
        + "\n"
    )

    dataset = state.dataset
    base = {
        "version_before": dataset.version,
        "version_after": dataset.version,
    }
    steps = workspace.read_steps()
    entries = [
        StepEntry(
            id=_next_step_id(steps),
            tab="preprocessing",
            op="split",
            params={
                **base,
                "strategy": rule.strategy,
                "rule_id": rule.rule_id,
                "test_size": options.test_size,
            },
            dataset_hash_before=dataset.data_hash,
            dataset_hash_after=dataset.data_hash,
            created_at=utc_now(),
        ),
        StepEntry(
            id=_next_step_id(steps) + 1,
            tab="preprocessing",
            op="encode",
            params={
                **base,
                "encoder": options.encoder,
                "columns": list(report.categorical_columns),
            },
            dataset_hash_before=dataset.data_hash,
            dataset_hash_after=dataset.data_hash,
            created_at=utc_now(),
        ),
    ]
    if options.scaler is not None:
        entries.append(
            StepEntry(
                id=_next_step_id(steps) + 2,
                tab="preprocessing",
                op="scale",
                params={**base, "scaler": options.scaler, "columns": list(report.numeric_columns)},
                dataset_hash_before=dataset.data_hash,
                dataset_hash_after=dataset.data_hash,
                created_at=utc_now(),
            )
        )
    if options.target_transform is not None and task.task_type == "regression":
        entries.append(
            StepEntry(
                id=_next_step_id(steps) + (3 if options.scaler is not None else 2),
                tab="preprocessing",
                op="target_transform",
                params={**base, "transform": options.target_transform},
                dataset_hash_before=dataset.data_hash,
                dataset_hash_after=dataset.data_hash,
                created_at=utc_now(),
            )
        )
    workspace.write_steps(steps + entries)

    state.split = SplitInfo(
        strategy=rule.strategy,
        params={"test_size": options.test_size, "rule_id": rule.rule_id},
        fitted=True,
        indices_path=str(split_path),
    )
    state.pipeline = PipelineInfo(fitted=True, path=str(pipeline_path))
    state.steps = workspace.read_steps()
    mark_downstream_stale(state, "preprocessing")

    return PreprocessingResult(
        strategy=rule.strategy,
        rule_id=rule.rule_id,
        train_rows=len(split.train),
        test_rows=len(split.test),
        feature_count=len(report.numeric_columns) + len(report.categorical_columns),
        encoder=options.encoder,
        scaler=options.scaler,
        target_transform=options.target_transform,
        pipeline_path=str(pipeline_path),
    )


def load_pipeline(path: Path | str) -> object:
    return joblib.load(str(path))
