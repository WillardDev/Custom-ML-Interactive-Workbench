from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ml_workbench.services.error_service import ErrorAnalysisError, error_views
from ml_workbench.services.model_cache import ModelCache
from ml_workbench.services.workspace import Workspace
from ml_workbench.state import ProjectState

MODEL_CACHE = ModelCache(capacity=2)


def render_error_analysis_tab(state: ProjectState, workspace: Workspace) -> None:
    trained = state.trained_models
    if not trained:
        st.info("No trained models yet. Train one in the Training tab.")
        return
    if state.frame is None or state.task is None:
        st.caption("Reload the dataset first (Data Insertion).")
        return

    run_id = st.selectbox(
        "Model",
        [run.run_id for run in trained],
        format_func=lambda run_id: _format_model(trained, run_id),
        key="err_model",
    )
    try:
        report = error_views(state, workspace, run_id, cache=MODEL_CACHE)
    except ErrorAnalysisError as exc:
        st.error(str(exc))
        return

    views = {str(view["view"]): view for view in report.views}
    st.subheader("Error views")
    if state.task.task_type == "binary":
        _render_binary(views)
    elif state.task.task_type in {"multiclass", "multilabel"}:
        _render_multiclass(views)
    elif state.task.task_type in {"regression", "forecasting"}:
        _render_regression(views)

    st.markdown("##### Worst-N rows (ERR-07)")
    if not report.worst_n.empty:
        st.dataframe(report.worst_n, use_container_width=True)
    if report.segments is not None and not report.segments.empty:
        st.markdown("##### Segment slices (ERR-07)")
        st.dataframe(report.segments, use_container_width=True)


def _render_binary(views: dict[str, Any]) -> None:
    confusion = views.get("confusion_matrix", {}).get("confusion")
    if confusion is not None:
        st.markdown("##### Confusion matrix")
        st.dataframe(confusion, use_container_width=True)
    roc = views.get("roc", {})
    if roc:
        if roc.get("auc") is not None:
            st.markdown(f"##### ROC curve (AUC = {roc['auc']:.4f})")
        fig = go.Figure(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name="ROC"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dot"}))
        fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(fig, use_container_width=True)
    pr = views.get("pr_curve", {})
    if pr:
        st.markdown(f"#### Precision-recall (AP = {pr['average_precision']:.4f})")
        fig = go.Figure(go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines", name="PR"))
        fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig, use_container_width=True)
    threshold = views.get("threshold_analysis", {}).get("rows")
    if threshold:
        st.markdown("##### Threshold analysis")
        st.dataframe(pd.DataFrame(threshold).round(4), use_container_width=True)
    calibration_data = views.get("calibration", {})
    if calibration_data and calibration_data.get("centers"):
        st.markdown("##### Calibration (expected-vs-observed)")
        st.table(
            pd.DataFrame(
                {"expected": calibration_data["centers"], "observed": calibration_data["observed"]}
            ).round(4)
        )


def _render_multiclass(views: dict[str, Any]) -> None:
    matrix = views.get("confusion_matrix_normalized", {}).get("matrix")
    if matrix is not None:
        st.markdown("##### Normalized confusion matrix")
        st.dataframe(matrix, use_container_width=True)
    per_class = views.get("per_class_precision_recall", {}).get("per_class")
    if per_class:
        st.markdown("##### Per-class precision / recall / support")
        st.dataframe(pd.DataFrame(per_class), use_container_width=True)
    most_confused = views.get("most_confused_pairs", {}).get("most_confused")
    if most_confused:
        st.markdown("##### Most-confused pairs")
        st.dataframe(pd.DataFrame(most_confused), use_container_width=True)


def _render_regression(views: dict[str, Any]) -> None:
    residuals = views.get("residuals_vs_predicted", {}).get("points")
    if residuals:
        fig = go.Figure(
            go.Scatter(x=[p for p, _ in residuals], y=[r for _, r in residuals], mode="markers")
        )
        fig.add_hline(y=0.0, line_dash="dot")
        fig.update_layout(xaxis_title="Predicted", yaxis_title="Residual")
        st.markdown("##### Residuals vs predicted")
        st.plotly_chart(fig, use_container_width=True)
    histogram = views.get("residual_histogram", {})
    if histogram:
        st.markdown("##### Residual histogram")
        fig = go.Figure(go.Bar(x=histogram["edges"][:-1], y=histogram["counts"]))
        st.plotly_chart(fig, use_container_width=True)
    qq = views.get("qq_plot", {})
    if qq:
        st.markdown("##### Q-Q plot")
        fig = go.Figure(go.Scatter(x=qq["x"], y=qq["y"], mode="markers"))
        fig.add_trace(go.Scatter(x=qq["x"], y=qq["x"], mode="lines", line={"dash": "dot"}))
        st.plotly_chart(fig, use_container_width=True)
    hetero = views.get("heteroscedasticity", {})
    if hetero:
        st.markdown(
            f"##### Heteroscedasticity (Spearman |residual|,|predicted|) = {hetero['spearman']:.4f}"
        )
    by_quantile = views.get("error_by_target_quantile", {}).get("by_quantile")
    if by_quantile:
        st.markdown("##### Mean abs error by target quantile")
        st.dataframe(pd.DataFrame(by_quantile), use_container_width=True)


def _format_model(trained: list[Any], run_id: str) -> str:
    run = next((model for model in trained if model.run_id == run_id), None)
    return f"{run.model_id} ({run.run_id})" if run else run_id
