from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ml_workbench.services.explain_service import explain_model
from ml_workbench.services.jobs import JobHandle, JobQueue
from ml_workbench.services.model_cache import ModelCache
from ml_workbench.services.workspace import Workspace
from ml_workbench.state import ProjectState

MODEL_CACHE = ModelCache(capacity=2)
EXPLAIN_JOBS = JobQueue(max_workers=2)


def render_explainability_tab(state: ProjectState, workspace: Workspace) -> None:
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
        key="explain_model",
    )
    report = st.session_state.get("explain_report")
    handle: JobHandle | None = st.session_state.get("explain_handle")
    if handle is not None and handle.state in {"queued", "running"}:
        handle = EXPLAIN_JOBS.poll(handle.job_id)
        st.progress(handle.progress, text=handle.message or "explaining…")
        st.caption(f"job {handle.job_id} ({handle.kind}) is running")
        return

    if handle is not None and handle.state == "done":
        st.session_state["explain_report"] = handle.result
        st.session_state.pop("explain_handle", None)
        report = handle.result
        st.success(
            f"Explanation ready (background job `{handle.job_id}`, cached={handle.result.cached})"
        )

    if st.button("Explain model", key="explain_run", type="primary"):
        st.session_state.pop("explain_report", None)
        run_id_to_explain = run_id
        handle = EXPLAIN_JOBS.submit(
            "shap",
            lambda update: explain_model(
                state, workspace, run_id_to_explain, cache=MODEL_CACHE, on_progress=update
            ),
        )
        st.session_state["explain_handle"] = handle
        st.rerun()

    if report is None:
        st.info("Explainability dispatches panels by `explain_method` (EXPL-01/02/03/04/05).")
        return
    _render_report(report)


def _render_report(report: Any) -> None:
    st.subheader(f"{report.method} explanation")
    for panel in report.panels:
        name = str(panel["panel"])
        if name == "coefficients":
            table = panel.get("table")
            st.markdown("##### Standardized coefficients (EXPL-01)")
            if table:
                st.dataframe(pd.DataFrame(table).round(4), use_container_width=True)
        elif name == "local_attribution":
            attribution = panel.get("attribution")
            st.markdown("##### Local attribution (first row, EXPL-05)")
            if attribution:
                rows = sorted(attribution, key=lambda row: abs(row["contribution"]), reverse=True)
                st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True)
        elif name == "importances":
            st.markdown("##### Feature importances (EXPL-02)")
            st.dataframe(pd.DataFrame(panel["table"]).round(4), use_container_width=True)
            if panel.get("bias_note"):
                st.caption(panel["bias_note"])
        elif name == "permutation_importance":
            st.markdown("##### Permutation importance (EXPL-03)")
            st.dataframe(pd.DataFrame(panel["rows"]).round(4), use_container_width=True)
            if panel.get("note"):
                st.caption(panel["note"])
        elif name == "gradient_attributions":
            st.markdown("##### Integrated Gradients — global attributions (EXPL-04)")
            st.dataframe(pd.DataFrame(panel["table"]).round(4), use_container_width=True)
            if panel.get("note"):
                st.caption(panel["note"])
        elif name == "gradient_local":
            st.markdown("##### Integrated Gradients — first row (EXPL-04)")
            attribution = panel.get("attribution") or []
            if attribution:
                st.dataframe(pd.DataFrame(attribution).round(4), use_container_width=True)
        elif name == "pdp_ice":
            _render_pdp(panel)
        elif name == "centroid_heatmap":
            st.markdown("##### Cluster centroids (EXPL-06)")
            st.dataframe(pd.DataFrame(panel["table"]), use_container_width=True)
            if panel.get("note"):
                st.caption(panel["note"])
        elif name == "anova":
            st.markdown("##### Per-feature F-ratio across clusters (EXPL-06)")
            st.dataframe(pd.DataFrame(panel["table"]), use_container_width=True)
        elif name == "surrogate_tree":
            st.markdown("##### Surrogate decision tree importances (EXPL-06)")
            st.dataframe(pd.DataFrame(panel["table"]).round(4), use_container_width=True)
            if panel.get("bias_note"):
                st.caption(panel["bias_note"])
        elif name == "personas":
            st.markdown("##### Cluster personas (EXPL-06)")
            for persona in panel["personas"]:
                with st.expander(f"Cluster {persona['cluster']} · {persona['size']} rows"):
                    st.dataframe(pd.DataFrame(persona["top_features"]), use_container_width=True)
        elif name == "loadings_table":
            st.markdown("##### Component loadings (EXPL-07)")
            st.dataframe(pd.DataFrame(panel["table"]), use_container_width=True)
            if panel.get("note"):
                st.caption(panel["note"])
        elif name == "variance_explained":
            explained = panel.get("explained") or []
            st.markdown("##### Explained variance ratio (EXPL-07)")
            fig = go.Figure(go.Bar(x=list(range(1, len(explained) + 1)), y=explained))
            fig.update_layout(xaxis_title="Component", yaxis_title="Explained variance ratio")
            st.plotly_chart(fig, use_container_width=True)
        elif name == "biplot_coordinates":
            st.markdown("##### Feature loading biplot — PC1 vs PC2 (EXPL-07)")
            rows = panel["table"]
            fig = go.Figure(
                go.Scatter(
                    x=[row["x"] for row in rows],
                    y=[row["y"] for row in rows],
                    mode="markers+text",
                    text=[row["feature"] for row in rows],
                    textposition="top center",
                )
            )
            fig.add_hline(y=0.0, line_dash="dot")
            fig.add_vline(x=0.0, line_dash="dot")
            fig.update_layout(xaxis_title="PC1 loading", yaxis_title="PC2 loading")
            st.plotly_chart(fig, use_container_width=True)
        elif name == "reconstruction_error_per_feature":
            st.markdown("##### Reconstruction error per feature (EXPL-07)")
            st.dataframe(pd.DataFrame(panel["table"]).round(4), use_container_width=True)
        elif name == "per_feature_deviation":
            st.markdown("##### Flagged-row feature deviation (EXPL-08)")
            st.dataframe(pd.DataFrame(panel["table"]).round(4), use_container_width=True)
            if panel.get("note"):
                st.caption(panel["note"])

    if report.warnings:
        for warning in report.warnings:
            st.warning(warning)
    st.caption(f"background rows: {report.background_rows} · cached: {report.cached}")


def _render_pdp(panel: dict[str, Any]) -> None:
    average = panel.get("average") or []
    ice = panel.get("ice") or []
    st.markdown("##### PDP / ICE (EXPL-05)")
    for curve in average:
        fig = go.Figure(
            go.Scatter(
                x=curve.get("grid", list(range(len(curve["average"])))),
                y=curve["average"],
                mode="lines+markers",
                name="PDP",
            )
        )
        for icurve in ice:
            if icurve["feature"] == curve["feature"]:
                for row in icurve.get("curves", [])[:5]:
                    fig.add_trace(go.Scatter(x=curve["grid"], y=row, mode="lines", opacity=0.35))
                break
        fig.update_layout(xaxis_title=str(curve["feature"]), yaxis_title="prediction")
        st.plotly_chart(fig, use_container_width=True)


def _format_model(trained: list[Any], run_id: str) -> str:
    run = next((model for model in trained if model.run_id == run_id), None)
    return f"{run.model_id} ({run.run_id})" if run else run_id
