from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype

from ml_workbench.rules import (
    class_balance,
    columns_with_missing,
    nan_decision,
    outlier_decision,
)
from ml_workbench.services.cleaning_service import (
    CleaningOptions,
    CleaningServiceError,
    model_policy,
    run_cleaning,
    undo_last_cleaning_step,
)
from ml_workbench.services.workspace import Workspace, WorkspaceError
from ml_workbench.state import ProjectState, TaskDefinition, TaskError
from ml_workbench.ui.theme import card

LEAVE_AS_NAN = "(leave as NaN)"


def render_cleaning_tab(state: ProjectState, workspace: Workspace) -> None:
    frame = state.frame
    task = state.task
    if frame is None or task is None:
        st.caption("Load a dataset and set a task first (Data Insertion).")
        return
    _render_options(state, workspace, frame, task)
    _render_overview(state)
    _render_balance(state)
    _render_step_log(state, workspace)


def _render_options(
    state: ProjectState, workspace: Workspace, frame: pd.DataFrame, task: TaskDefinition
) -> None:
    with card("Cleaning options"):
        handles_nan, family = model_policy(state)
        policy = nan_decision(handles_nan)
        outliers = outlier_decision(task.task_type, family)

        missing = columns_with_missing(frame, exclude=[task.target] if task.target else [])
        target_missing = bool(task.target and bool(frame[task.target].isna().any()))

        drop_target = True
        imputations: tuple[tuple[str, str], ...] = ()
        constant_values: dict[str, object] = {}
        outlier_columns: tuple[str, ...] = ()
        outlier_factor = 1.5

        with st.expander(
            "Missing values (CLEAN-02, CLEAN-03)", expanded=bool(missing) or target_missing
        ):
            if target_missing and task.target:
                st.caption(
                    f"{int(frame[task.target].isna().sum())} rows have a missing target — "
                    "they are dropped, never imputed (CLEAN-02)."
                )
            if task.target is not None:
                drop_target = st.checkbox(
                    "Drop rows with missing target", value=True, key="clean_drop_target"
                )
            if not missing:
                st.caption("No missing values in feature columns.")
            else:
                if policy.allow_leave_nan:
                    st.caption(
                        "Selected model handles missing values — leaving them as NaN is allowed."
                    )
                else:
                    st.caption(
                        "Missing feature values must be imputed "
                        "(median/mean/KNN/iterative/mode/constant) — CLEAN-03."
                    )
                strategies = list(policy.strategies)
                if policy.allow_leave_nan:
                    strategies = [LEAVE_AS_NAN, *strategies]
                impute_columns = st.multiselect(
                    "Columns to impute", missing, key="clean_impute_columns"
                )
                strategy = st.selectbox(
                    "Imputation strategy", strategies, key="clean_impute_strategy"
                )
                if strategy == "constant":
                    raw = st.text_input("Constant fill value", key="clean_fill_value")
                    constant_values = {column: _coerce_scalar(raw) for column in impute_columns}
                imputations = (
                    ()
                    if strategy == LEAVE_AS_NAN
                    else tuple((column, strategy) for column in impute_columns)
                )

        with st.expander("Outliers (CLEAN-04)"):
            st.caption(outliers.recommend)
            numeric_columns = [
                column
                for column in frame.columns
                if is_numeric_dtype(frame[column]) and column != task.target
            ]
            allowed = list(outliers.allowed)
            outlier_op = st.selectbox(
                "Treatment",
                allowed,
                index=allowed.index(outliers.default),
                key="clean_outlier_op",
            )
            outlier_columns = tuple(
                st.multiselect("Columns", numeric_columns, key="clean_outlier_columns")
            )
            if outlier_op == "clip":
                outlier_factor = st.number_input(
                    "IQR factor", 1.0, 5.0, 1.5, 0.5, key="clean_outlier_factor"
                )

        if st.button("Run cleaning", key="clean_run", type="primary"):
            options = CleaningOptions(
                drop_missing_target=drop_target,
                imputations=imputations,
                constant_values=constant_values,
                outlier_columns=outlier_columns,
                outlier_op=outlier_op,
                outlier_factor=outlier_factor,
            )
            try:
                result = run_cleaning(state, workspace, options)
            except (CleaningServiceError, TaskError, WorkspaceError) as exc:
                st.error(str(exc))
            else:
                st.success(
                    f"Cleaning complete — {result.rows_before} → {result.rows_after} rows, "
                    f"{result.columns_before} → {result.columns_after} columns."
                )
                if result.ops:
                    st.dataframe(
                        pd.DataFrame([asdict(report) for report in result.ops]),
                        hide_index=True,
                    )
                st.caption(
                    "Every step is versioned and appended to the step log with before/after "
                    "hashes (CLEAN-01, CLEAN-06)."
                )


def _render_overview(state: ProjectState) -> None:
    with card("Dataset"):
        frame = state.frame
        if frame is None:
            return
        left, mid, right = st.columns(3)
        left.metric("Rows", len(frame))
        mid.metric("Columns", len(frame.columns))
        right.metric("Version", state.dataset.version if state.dataset else "-")
        missing_cells = int(frame.isna().sum().sum())
        st.caption(
            f"{missing_cells} missing cell(s) across the dataset"
            + (
                f"; columns with missing values: {', '.join(columns_with_missing(frame))}"
                if missing_cells
                else ""
            )
        )


def _render_balance(state: ProjectState) -> None:
    task = state.task
    frame = state.frame
    if task is None or task.target is None or frame is None:
        return
    balance = class_balance(frame, task.target, task.task_type)
    if balance is None:
        return
    with card("Class balance (CLEAN-05)"):
        table = [
            {
                "class": label,
                "count": balance.counts[label],
                "fraction": round(balance.fractions[label], 4),
            }
            for label in balance.counts
        ]
        st.dataframe(pd.DataFrame(table), hide_index=True)
        if balance.imbalanced:
            st.warning(
                f"Minority class is {balance.minority_fraction:.1%} of rows "
                f"(< {balance.threshold:.0%}) — consider class weights or resampling (CLEAN-05)."
            )


def _render_step_log(state: ProjectState, workspace: Workspace) -> None:
    with card("Step log (CLEAN-06)"):
        if not state.steps:
            st.caption("No steps recorded yet.")
            return
        rows = [
            {
                "id": entry.id,
                "tab": entry.tab,
                "op": entry.op,
                "params": json.dumps(entry.params, sort_keys=True),
                "hash_before": _short(entry.dataset_hash_before),
                "hash_after": _short(entry.dataset_hash_after),
                "created_at": entry.created_at,
            }
            for entry in state.steps
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        if any(entry.tab == "cleaning" for entry in state.steps):
            if st.button("Undo last cleaning step", key="clean_undo"):
                try:
                    undone = undo_last_cleaning_step(state, workspace)
                except (CleaningServiceError, WorkspaceError) as exc:
                    st.error(str(exc))
                else:
                    if undone is not None:
                        st.success(
                            f"Undid step {undone.id} ({undone.op}); dataset restored to "
                            f"{undone.params.get('version_before', '?')}."
                        )
                    else:
                        st.info("Nothing to undo.")


def _coerce_scalar(value: str) -> object:
    value = value.strip()
    if value == "":
        return value
    try:
        return float(value)
    except ValueError:
        return value


def _short(data_hash: str) -> str:
    return data_hash[:12] if data_hash else ""
