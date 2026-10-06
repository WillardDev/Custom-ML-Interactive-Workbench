from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ml_workbench.registry import load_registry
from ml_workbench.rules.cleaning import (
    ClassBalance,
    apply_imputation,
    apply_outlier_treatment,
    class_balance,
    drop_missing_target,
    fix_dtypes,
    outlier_decision,
    remove_duplicates,
)
from ml_workbench.rules.staleness import mark_downstream_stale
from ml_workbench.services.data_service import schema_report
from ml_workbench.services.workspace import Workspace, utc_now
from ml_workbench.state import DatasetInfo, ProjectState, StepEntry, TaskError


class CleaningServiceError(ValueError):
    pass


@dataclass(frozen=True)
class CleaningOptions:
    drop_missing_target: bool = True
    imputations: tuple[tuple[str, str], ...] = ()
    constant_values: dict[str, Any] = field(default_factory=dict)
    outlier_columns: tuple[str, ...] = ()
    outlier_op: str = "keep"
    outlier_factor: float = 1.5


@dataclass(frozen=True)
class OpReport:
    op: str
    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    params: dict[str, Any]


@dataclass(frozen=True)
class CleaningResult:
    ops: tuple[OpReport, ...]
    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    balance: ClassBalance | None
    versions: tuple[str, ...]


def model_policy(state: ProjectState) -> tuple[bool, str | None]:
    """(handles_nan, model family) for the active model; (False, None) when none trained yet."""
    if state.active_model_id is None:
        return False, None
    spec = load_registry().get(state.active_model_id)
    if spec is None:
        return False, None
    return bool(spec.flags["handles_nan"]), spec.family


def _next_step_id(entries: list[StepEntry]) -> int:
    return max((entry.id for entry in entries), default=0) + 1


def run_cleaning(
    state: ProjectState, workspace: Workspace, options: CleaningOptions
) -> CleaningResult:
    if state.frame is None or state.dataset is None:
        raise CleaningServiceError("load a dataset before running Data Cleaning")
    if state.task is None:
        raise CleaningServiceError("set a task before running Data Cleaning")

    task = state.task
    frame = state.frame
    rows_before = len(frame)
    columns_before = len(frame.columns)
    previous_version = state.dataset.version
    current_hash = state.dataset.data_hash
    entries: list[StepEntry] = []
    reports: list[OpReport] = []

    def _apply(
        op: str, current: pd.DataFrame, updated: pd.DataFrame, params: dict[str, Any]
    ) -> None:
        nonlocal previous_version, current_hash
        version = workspace.write_version(updated, schema=schema_report(updated))
        entries.append(
            StepEntry(
                id=_next_step_id(workspace.read_steps() + entries),
                tab="cleaning",
                op=op,
                params={
                    **params,
                    "version_before": previous_version,
                    "version_after": version.version,
                },
                dataset_hash_before=current_hash,
                dataset_hash_after=version.data_hash,
                created_at=utc_now(),
            )
        )
        reports.append(
            OpReport(
                op=op,
                rows_before=len(current),
                rows_after=len(updated),
                columns_before=len(current.columns),
                columns_after=len(updated.columns),
                params=dict(params),
            )
        )
        previous_version = version.version
        current_hash = version.data_hash

    # CLEAN-01: duplicates and dtypes are always cleaned, reported before/after.
    deduped, duplicates_removed = remove_duplicates(frame)
    _apply("dedupe", frame, deduped, {"duplicates_removed": duplicates_removed})
    cleaned, dtype_changes, sentinels = fix_dtypes(deduped)
    _apply(
        "fix_dtypes",
        deduped,
        cleaned,
        {"dtype_changes": dtype_changes, "sentinels_replaced": sentinels},
    )
    frame = cleaned

    # CLEAN-02: missing targets are dropped, never imputed.
    target = task.target
    if (
        options.drop_missing_target
        and task.learning_type == "supervised"
        and target is not None
        and bool(frame[target].isna().any())
    ):
        updated = drop_missing_target(frame, target)
        _apply(
            "drop_rows",
            frame,
            updated,
            {
                "reason": "missing_target",
                "target": target,
                "rows_dropped": len(frame) - len(updated),
            },
        )
        frame = updated

    # CLEAN-03: imputation choice per strategy; constant imputation is per-column.
    by_strategy: dict[str, list[str]] = {}
    for column, strategy in options.imputations:
        by_strategy.setdefault(strategy, []).append(column)
    for strategy, columns in by_strategy.items():
        if strategy == "constant":
            for column in columns:
                if column not in options.constant_values:
                    raise TaskError(f"constant imputation needs a fill value for column '{column}'")
                updated = apply_imputation(
                    frame,
                    [column],
                    "constant",
                    target=target,
                    fill_value=options.constant_values[column],
                )
                _apply(
                    "impute",
                    frame,
                    updated,
                    {
                        "columns": [column],
                        "strategy": "constant",
                        "value": options.constant_values[column],
                    },
                )
                frame = updated
        else:
            updated = apply_imputation(frame, columns, strategy, target=target)
            _apply("impute", frame, updated, {"columns": list(columns), "strategy": strategy})
            frame = updated

    # CLEAN-04: outlier policy from task/family; op must be allowed by the decision.
    handles_nan, family = model_policy(state)
    decision = outlier_decision(task.task_type, family)
    if options.outlier_op != "keep":
        if options.outlier_op not in decision.allowed:
            raise CleaningServiceError(
                f"outlier op '{options.outlier_op}' is not allowed here ({decision.rule_id}): "
                f"allowed = {', '.join(decision.allowed)}"
            )
        if not options.outlier_columns:
            raise TaskError("no columns selected for outlier treatment")
        updated = apply_outlier_treatment(
            frame, options.outlier_columns, options.outlier_op, factor=options.outlier_factor
        )
        _apply(
            "outlier_treatment",
            frame,
            updated,
            {
                "columns": list(options.outlier_columns),
                "op": options.outlier_op,
                "factor": options.outlier_factor,
            },
        )
        frame = updated

    if entries:
        workspace.write_steps(workspace.read_steps() + entries)
    state.frame = frame
    info = workspace.read_version_info(previous_version)
    state.dataset = DatasetInfo(
        version=info.version,
        path=str(info.path),
        data_hash=info.data_hash,
        row_count=info.row_count,
        loaded_at=info.loaded_at,
    )
    state.steps = workspace.read_steps()
    mark_downstream_stale(state, "cleaning")

    balance = class_balance(frame, target, task.task_type) if target is not None else None
    return CleaningResult(
        ops=tuple(reports),
        rows_before=rows_before,
        rows_after=len(frame),
        columns_before=columns_before,
        columns_after=len(frame.columns),
        balance=balance,
        versions=tuple(
            str(entry.params["version_after"]) for entry in state.steps if entry.tab == "cleaning"
        ),
    )


def undo_last_cleaning_step(state: ProjectState, workspace: Workspace) -> StepEntry | None:
    entry = workspace.pop_step("cleaning")
    if entry is None:
        return None
    version_before = str(entry.params.get("version_before", ""))
    if not version_before:
        raise CleaningServiceError("the step has no 'version_before' to restore")
    info = workspace.read_version_info(version_before)
    state.frame = workspace.read_version(version_before)
    state.dataset = DatasetInfo(
        version=info.version,
        path=str(info.path),
        data_hash=info.data_hash,
        row_count=info.row_count,
        loaded_at=info.loaded_at,
    )
    state.steps = workspace.read_steps()
    mark_downstream_stale(state, "cleaning")
    return entry
