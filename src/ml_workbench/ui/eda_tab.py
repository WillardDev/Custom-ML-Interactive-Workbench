from __future__ import annotations

import pandas as pd
import streamlit as st

from ml_workbench.rules import eda_plan, skew_hint
from ml_workbench.services.eda_service import EdaResult, eda_report
from ml_workbench.services.workspace import Workspace
from ml_workbench.state import ProjectState, TaskDefinition


def render_eda_tab(state: ProjectState, workspace: Workspace) -> None:
    frame = state.frame
    task = state.task
    if frame is None or task is None:
        st.caption("Load a dataset and set a task first (Data Insertion).")
        return
    st.subheader("EDA plan (EDA-01, EDA-02, EDA-03)")
    st.caption(f"Task: {task.task_type} — " + ", ".join(eda_plan(task)))

    report = eda_report(frame, task)
    _render_base(state, report)
    _render_classification(state, report)
    _render_regression(state, task, report)
    st.caption(
        "EDA views are read-only; upstream edits mark this tab stale "
        "(run Cleaning again to refresh)."
    )


def _render_base(state: ProjectState, report: EdaResult) -> None:
    st.subheader("Dataset overview (EDA-01)")
    left, mid, right = st.columns(3)
    left.metric("Rows", report.base.rows)
    mid.metric("Columns", report.base.columns)
    right.metric("Sampled rows", report.base.sampled_rows)
    if report.base.dtypes:
        st.caption("Column dtypes (sampled frame):")
        dtype_rows = [
            {"column": column, "dtype": dtype} for column, dtype in report.base.dtypes.items()
        ]
        st.dataframe(pd.DataFrame(dtype_rows, columns=["column", "dtype"]), hide_index=True)
    if report.base.missing:
        st.markdown("**Missing values**")
        missing_rows = [
            {"column": column, "missing": count} for column, count in report.base.missing.items()
        ]
        st.dataframe(pd.DataFrame(missing_rows), hide_index=True)
    else:
        st.caption("No missing values.")
    if report.base.numeric_describe is not None:
        st.markdown("**Numeric summary**")
        st.dataframe(report.base.numeric_describe, hide_index=True)
    if report.base.numeric_pearson is not None:
        st.markdown("**Pearson correlation (numeric)**")
        st.dataframe(report.base.numeric_pearson)
    if report.base.numeric_spearman is not None:
        st.markdown("**Spearman correlation (numeric)**")
        st.dataframe(report.base.numeric_spearman)
    if report.base.cramers_v is not None:
        st.markdown("**Cramér's V (categorical pairs)**")
        st.dataframe(report.base.cramers_v)


def _render_classification(state: ProjectState, report: EdaResult) -> None:
    if report.classification is None:
        return
    st.subheader("Classification views (EDA-02)")
    st.markdown("**Class balance**")
    st.dataframe(report.classification.class_counts, hide_index=True)
    if report.classification.feature_by_class:
        st.markdown("**Class vs. feature crosstabs**")
        for feature, table in report.classification.feature_by_class.items():
            st.caption(feature)
            st.dataframe(table)
    if report.classification.chi_square is not None:
        st.markdown("**Chi-square tests (categorical features vs target)**")
        st.dataframe(report.classification.chi_square, hide_index=True)


def _render_regression(state: ProjectState, task: TaskDefinition, report: EdaResult) -> None:
    if report.regression is None or task.target is None:
        return
    st.subheader("Regression views (EDA-03, HINT-02)")
    hint = skew_hint(state.frame[task.target]) if state.frame is not None else None
    if hint:
        st.warning(hint)
    st.metric("Target skew", round(report.regression.target_skew, 3))
    st.markdown("**Target histogram (bins)**")
    st.bar_chart(report.regression.target_histogram)
    if report.regression.feature_scatter is not None:
        st.markdown("**Features vs target (sampled)**")
        st.dataframe(report.regression.feature_scatter, hide_index=True, height=240)
