from __future__ import annotations

import pandas as pd
import streamlit as st

from ml_workbench.rules import build_task, mark_downstream_stale, suggest_task_type
from ml_workbench.services.data_service import (
    DataError,
    insert_dataset,
    list_samples,
    load_bytes,
    load_sqlite,
    read_sample,
    schema_report,
    sqlite_tables,
    write_temp_bytes,
)
from ml_workbench.services.workspace import Workspace
from ml_workbench.state import ProjectState, TaskError

NONE_OPTION = "(none)"

UNSUPERVISED_TASK_TYPES = (
    "clustering",
    "dimensionality_reduction",
    "anomaly_detection",
    "association",
)


def render_data_tab(state: ProjectState, workspace: Workspace) -> None:
    _render_insert(state, workspace)
    _render_dataset_summary(state)
    _render_task_form(state)


def _render_insert(state: ProjectState, workspace: Workspace) -> None:
    st.subheader("Insert data")
    source = st.radio(
        "Data source",
        ("Sample dataset", "Upload file", "SQLite database"),
        horizontal=True,
        key="data_source",
    )
    if source == "Sample dataset":
        name = st.selectbox("Sample dataset", list_samples(), key="data_sample")
        if st.button("Load sample", key="data_load_sample"):
            _load_data(state, workspace, read_sample(name), source=f"sample:{name}")
    elif source == "Upload file":
        upload = st.file_uploader(
            "CSV, TSV, Excel, or Parquet",
            type=["csv", "tsv", "txt", "xlsx", "xls", "parquet"],
            key="data_upload",
        )
        if st.button("Load file", key="data_load_file"):
            if upload is None:
                st.error("choose a file to upload first")
            else:
                _load_data(
                    state, workspace, load_bytes(upload.getvalue(), upload.name), source=upload.name
                )
    else:
        upload = st.file_uploader(
            "SQLite database file",
            type=["sqlite", "sqlite3", "db"],
            key="data_upload_sqlite",
        )
        if upload is not None:
            path = write_temp_bytes(upload.getvalue(), ".sqlite")
            try:
                tables = sqlite_tables(path)
                query = st.text_input(
                    "SQL query",
                    value=f'SELECT * FROM "{tables[0]}"',
                    key="data_sql_query",
                )
            finally:
                path.unlink(missing_ok=True)
        else:
            query = st.text_input("SQL query", value="", key="data_sql_query")
        if st.button("Run query", key="data_load_sql"):
            if upload is None:
                st.error("choose a SQLite database file first")
            elif not query.strip():
                st.error("enter a SQL query")
            else:
                path = write_temp_bytes(upload.getvalue(), ".sqlite")
                try:
                    _load_data(
                        state, workspace, load_sqlite(path, query), source=f"sqlite:{upload.name}"
                    )
                finally:
                    path.unlink(missing_ok=True)


def _load_data(
    state: ProjectState, workspace: Workspace, frame: pd.DataFrame, *, source: str
) -> None:
    try:
        insert_dataset(state, workspace, frame, source=source)
    except DataError as exc:
        st.error(str(exc))
    else:
        st.success(
            f"Loaded {frame.shape[0]} rows × {frame.shape[1]} columns "
            f"— dataset version {state.dataset.version if state.dataset else '?'}."
        )


def _render_dataset_summary(state: ProjectState) -> None:
    st.subheader("Dataset")
    if state.dataset is None or state.frame is None:
        st.caption("No dataset loaded yet.")
        return
    dataset = state.dataset
    left, right = st.columns(2)
    left.metric("Rows", dataset.row_count)
    right.metric("Version", dataset.version)
    st.dataframe(pd.DataFrame(schema_report(state.frame)), hide_index=True)
    st.caption(f"Data hash {dataset.data_hash} · loaded {dataset.loaded_at}")


def _optional_column(label: str, options: list[str], key: str) -> str | None:
    choice = st.selectbox(label, [NONE_OPTION, *options], key=key)
    return None if choice == NONE_OPTION else choice


def _render_task_form(state: ProjectState) -> None:
    st.subheader("Task")
    frame = state.frame
    if frame is None:
        st.caption("Load a dataset to define the task.")
        return
    if state.task is not None:
        task = state.task
        detail = f"target `{task.target}`" if task.target else f"type `{task.task_type}`"
        st.success(f"Current task: {task.learning_type} / {task.task_type} ({detail})")

    columns = [str(column) for column in frame.columns]
    learning_type = st.radio(
        "Learning type", ("supervised", "unsupervised"), horizontal=True, key="task_learning"
    )

    target = group = time_column = eval_labels = None
    task_type = None
    if learning_type == "supervised":
        target = _optional_column("Target column", columns, "task_target")
        if target is not None:
            try:
                suggestion = suggest_task_type(frame[target])
            except TaskError as exc:
                st.warning(str(exc))
            else:
                st.caption(
                    f"Suggested task type from `{target}`: **{suggestion}** (TASK-01). "
                    "Override below if needed."
                )
            task_type = st.selectbox(
                "Task type",
                ["binary", "multiclass", "multilabel", "regression", "forecasting"],
                key="task_type_override",
            )
        time_column = _optional_column("Time column (optional)", columns, "task_time")
        if time_column is not None:
            st.checkbox(
                "Confirm this is a forecasting problem (TASK-02)",
                key="task_confirm_forecast",
            )
        group = _optional_column("Group column (optional)", columns, "task_group")
    else:
        task_type = st.selectbox("Task type", list(UNSUPERVISED_TASK_TYPES), key="task_type_unsup")
        eval_labels = _optional_column("Evaluation labels (optional)", columns, "task_eval_labels")
        group = _optional_column("Group column (optional)", columns, "task_group")

    if st.button("Set task", key="task_set"):
        confirmed = bool(st.session_state.get("task_confirm_forecast", False))
        try:
            task = build_task(
                frame,
                str(learning_type),
                target=target,
                group=group,
                time_column=time_column,
                eval_labels=eval_labels,
                task_type=task_type,
                confirm_forecasting=confirmed,
            )
        except TaskError as exc:
            st.error(str(exc))
        else:
            state.task = task
            mark_downstream_stale(state, "data")
            st.success(
                f"Task set: {task.learning_type} / {task.task_type} "
                f"{f'on `{task.target}`' if task.target else ''}."
            )
