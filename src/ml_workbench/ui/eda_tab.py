from __future__ import annotations

import pandas as pd
import streamlit as st

from ml_workbench.rules import eda_plan, skew_hint
from ml_workbench.services.eda_service import EdaResult, eda_report
from ml_workbench.services.workspace import Workspace
from ml_workbench.state import ProjectState, TaskDefinition
from ml_workbench.ui.theme import card


def render_eda_tab(state: ProjectState, workspace: Workspace) -> None:
    frame = state.frame
    task = state.task
    if frame is None or task is None:
        st.caption("Load a dataset and set a task first (Data Insertion).")
        return
    with card("EDA plan (EDA-01, EDA-02, EDA-03)"):
        st.caption(f"Task: {task.task_type} — " + ", ".join(eda_plan(task)))

    report = eda_report(frame, task)
    _render_base(state, report)
    _render_classification(state, report)
    _render_regression(state, task, report)
    _render_timeseries(state, report)
    _render_clustering(state, report)
    _render_anomaly(state, report)
    _render_dimred(state, report)
    _render_association(state, report)
    st.caption(
        "EDA views are read-only; upstream edits mark this tab stale "
        "(run Cleaning again to refresh)."
    )


def _render_base(state: ProjectState, report: EdaResult) -> None:
    with card("Dataset overview (EDA-01)"):
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
                {"column": column, "missing": count}
                for column, count in report.base.missing.items()
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
    with card("Classification views (EDA-02)"):
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
    with card("Regression views (EDA-03, HINT-02)"):
        hint = skew_hint(state.frame[task.target]) if state.frame is not None else None
        if hint:
            st.warning(hint)
        st.metric("Target skew", round(report.regression.target_skew, 3))
        st.markdown("**Target histogram (bins)**")
        st.bar_chart(report.regression.target_histogram)
        if report.regression.feature_scatter is not None:
            st.markdown("**Features vs target (sampled)**")
            st.dataframe(report.regression.feature_scatter, hide_index=True, height=240)


def _render_timeseries(state: ProjectState, report: EdaResult) -> None:
    if report.timeseries is None:
        return
    ts = report.timeseries
    with card("Time-series views (EDA-04)"):
        st.caption(f"seasonal period {ts.period}")
        st.markdown("**Decomposition (trend / seasonal / residual)**")
        st.line_chart(
            pd.DataFrame({"trend": ts.trend, "seasonal": ts.seasonal, "residual": ts.residual})
        )
        acf_column, pacf_column = st.columns(2)
        with acf_column:
            st.markdown("**ACF**")
            st.bar_chart(ts.acf.set_index("lag")["acf"])
        with pacf_column:
            st.markdown("**PACF**")
            st.bar_chart(ts.pacf.set_index("lag")["pacf"])
        st.markdown("**Rolling statistics**")
        st.line_chart(ts.rolling)
        verdict = (
            "stationary" if ts.stationary else "unit root likely (difference before modelling)"
        )
        st.caption(
            f"ADF statistic {ts.stationarity_stat} vs 5% critical "
            f"{ts.stationarity_critical} — {verdict}"
        )


def _render_clustering(state: ProjectState, report: EdaResult) -> None:
    if report.clustering is None:
        return
    with card("Clustering views (EDA-05)"):
        st.metric("Hopkins statistic", report.clustering.hopkins)
        st.caption(
            "Hopkins ≈ 0.5 suggests near-random data; "
            "values above 0.7 suggest clusterable structure."
        )
        if report.clustering.components_for_90 is not None:
            st.metric("Components for 90% variance", report.clustering.components_for_90)
        if report.clustering.scree is not None:
            st.markdown("**Scree plot (PCA preview)**")
            st.dataframe(report.clustering.scree, hide_index=True)
            st.bar_chart(
                report.clustering.scree.set_index("component")["explained_variance"],
                height=240,
            )


def _render_anomaly(state: ProjectState, report: EdaResult) -> None:
    if report.anomaly is None:
        return
    with card("Anomaly views (EDA-06)"):
        st.markdown("**Univariate z-score flags (|z| > 3)**")
        if report.anomaly.zscore_flags is not None and not report.anomaly.zscore_flags.empty:
            st.dataframe(report.anomaly.zscore_flags, hide_index=True)
        else:
            st.caption("No feature exceeds the |z| > 3 threshold.")
        st.markdown("**IQR flags (below/above 1.5·IQR)**")
        if report.anomaly.iqr_flags is not None and not report.anomaly.iqr_flags.empty:
            st.dataframe(report.anomaly.iqr_flags, hide_index=True)
        else:
            st.caption("No feature has IQR outliers.")
        st.markdown("**Mahalanobis distance (multivariate)**")
        if report.anomaly.mahalanobis is not None:
            st.dataframe(report.anomaly.mahalanobis, hide_index=True)
        else:
            st.caption("Not enough variation to fit a Mahalanobis model.")


def _render_dimred(state: ProjectState, report: EdaResult) -> None:
    if report.dimred is None:
        return
    with card("Dimensionality reduction views (EDA-07)"):
        st.markdown("**Correlation groups (|ρ| ≥ 0.7)**")
        if report.dimred.correlation_groups:
            for group in report.dimred.correlation_groups:
                st.write(", ".join(group))
        else:
            st.caption("No correlated feature groups found.")
        st.markdown("**Variance inflation factors (VIF)**")
        if report.dimred.vif is not None:
            st.dataframe(report.dimred.vif, hide_index=True)
            st.caption("VIF ≥ 10 indicates strong multicollinearity.")
        else:
            st.caption("VIF not computable with the current features.")


def _render_association(state: ProjectState, report: EdaResult) -> None:
    if report.association is None:
        return
    assoc = report.association
    with card("Association views (EDA-08)"):
        st.caption(f"{assoc.baskets} baskets, {assoc.unique_items} unique items")
        st.markdown("**Item frequency (share of baskets)**")
        if not assoc.item_frequency.empty:
            st.bar_chart(assoc.item_frequency.set_index("item")["frequency"])
        st.markdown("**Basket size distribution**")
        if not assoc.basket_size.empty:
            st.bar_chart(assoc.basket_size)
