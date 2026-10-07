from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from ml_workbench.services.outcome_service import OutcomeError, build_supervised_outcome
from ml_workbench.services.report_service import build_report
from ml_workbench.services.workspace import Workspace
from ml_workbench.state import ProjectState


def render_outcome_tab(state: ProjectState, workspace: Workspace) -> None:
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
        key="outcome_model",
    )
    threshold = st.slider(
        "Decision threshold",
        0.0,
        1.0,
        0.5,
        key="outcome_threshold",
        help="reuses the Prediction tab threshold when set",
    )
    refit = st.checkbox("Refit the pipeline on all data (OUT-01)", key="outcome_refit")

    if st.button("Build outcome, report and script", key="outcome_build", type="primary"):
        try:
            result = build_supervised_outcome(
                state, workspace, run_id, threshold=threshold, refit_on_all=refit
            )
            report = build_report(state, workspace, run_id, result.files, threshold=threshold)
        except OutcomeError as exc:
            st.error(str(exc))
        else:
            st.session_state["outcome_report"] = {
                "files": result.files,
                "report": report.report,
                "script": report.script,
                "manifest": report.manifest,
            }
            st.rerun()

    payload = st.session_state.get("outcome_report")
    if payload is None:
        st.info(
            "Package the outcome: deliverables, HTML report and reproducible script "
            "(OUT-01, EXPORT-01/02)."
        )
        return

    st.subheader("Deliverables (OUT-01)")
    st.table(pd.DataFrame({"file": payload["files"]}))
    st.subheader("Report bundle (EXPORT-01)")
    st.table(pd.DataFrame({"file": [payload["report"], payload["script"], payload["manifest"]]}))

    model_card = workspace.outcome_dir / "model_card.json"
    if model_card.is_file():
        st.markdown("##### Model card")
        st.json(model_card.read_text(), expanded=False)

    predictions = workspace.outcome_dir / "predictions.csv"
    if predictions.is_file():
        st.markdown("##### Predictions preview")
        st.dataframe(pd.read_csv(predictions).head(20), use_container_width=True)


def _format_model(trained: list[Any], run_id: str) -> str:
    run = next((model for model in trained if model.run_id == run_id), None)
    return f"{run.model_id} ({run.run_id})" if run else run_id
