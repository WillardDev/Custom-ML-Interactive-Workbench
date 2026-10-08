from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import streamlit as st

if __package__ in {None, ""}:  # allow `streamlit run src/ml_workbench/app.py` from any Python
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_workbench.rules import evaluate_gate, gates, stale_banner
from ml_workbench.rules.gating import ALL_REQS, DESCRIPTIONS, REQUIREMENTS, GateDecision
from ml_workbench.services.workspace import Workspace
from ml_workbench.state import ProjectState
from ml_workbench.tabs import TAB_BY_ID, TAB_ORDER, TITLES
from ml_workbench.ui import (
    render_cleaning_tab,
    render_data_tab,
    render_eda_tab,
    render_error_analysis_tab,
    render_explainability_tab,
    render_modelling_tab,
    render_outcome_tab,
    render_prediction_tab,
    render_preprocessing_tab,
    render_training_tab,
)
from ml_workbench.ui.theme import begin_run, inject_theme, page_header

DEFAULT_WORKSPACE_ROOT = Path(__file__).resolve().parents[2] / "workspace"

RENDERERS: dict[str, Callable[[ProjectState, Workspace], None]] = {
    "data": render_data_tab,
    "cleaning": render_cleaning_tab,
    "preprocessing": render_preprocessing_tab,
    "eda": render_eda_tab,
    "modelling": render_modelling_tab,
    "training": render_training_tab,
    "prediction": render_prediction_tab,
    "error_analysis": render_error_analysis_tab,
    "explainability": render_explainability_tab,
    "outcome": render_outcome_tab,
}


def _state() -> ProjectState:
    if "state" not in st.session_state:
        st.session_state["state"] = ProjectState(project_id="p_local")
    state: ProjectState = st.session_state["state"]
    return state


def _workspace(state: ProjectState) -> Workspace:
    if "workspace" not in st.session_state:
        st.session_state["workspace"] = Workspace(
            project_id=state.project_id, root=DEFAULT_WORKSPACE_ROOT
        )
    workspace: Workspace = st.session_state["workspace"]
    return workspace


def _navigation(state: ProjectState) -> str:
    decisions = gates(state)
    current_id = st.session_state.get("workflow_nav") or TAB_ORDER[0]
    current_no = TAB_BY_ID[current_id].number

    def label(tab_id: str) -> str:
        info = TAB_BY_ID[tab_id]
        if decisions[tab_id].locked:
            glyph = "🔒 "
        elif info.number < current_no:
            glyph = "✓ "
        else:
            glyph = ""
        return f"{glyph}{info.number}. {info.title}"

    return st.sidebar.radio("Workflow", TAB_ORDER, format_func=label, key="workflow_nav")


def _sidebar_status(state: ProjectState) -> None:
    decisions = gates(state)
    lines = []
    for tab_id in TAB_ORDER:
        parts = []
        if decisions[tab_id].locked:
            parts.append("locked")
        if state.stale.get(tab_id, False):
            parts.append("stale")
        if parts:
            info = TAB_BY_ID[tab_id]
            lines.append(f"- {info.number}. {info.title} — {', '.join(parts)}")
    if lines:
        st.sidebar.markdown("**Tab status**\n" + "\n".join(lines))


def _requirements_panel(tab_id: str, decision: GateDecision, state: ProjectState) -> None:
    with st.expander("Requirements for this tab"):
        rows = [
            {
                "requirement": DESCRIPTIONS[req],
                "status": "missing" if req in decision.missing else "met",
            }
            for req in ALL_REQS
            if req in REQUIREMENTS[tab_id]
        ]
        st.table(rows)


def _render_tab(state: ProjectState, workspace: Workspace, tab_id: str) -> None:
    info = TAB_BY_ID[tab_id]
    decision = evaluate_gate(tab_id, state)
    page_header(info)

    if decision.locked:
        st.info(decision.message)
        _requirements_panel(tab_id, decision, state)
        return

    banner = stale_banner(state, tab_id)
    if banner:
        st.warning(banner)

    renderer = RENDERERS.get(tab_id)
    if renderer is not None:
        renderer(state, workspace)
    else:
        st.write(f"Design specification: docs/tabs.md section {info.design_section}.")
        st.info(f"{TITLES[tab_id]} arrives in Phase {info.implemented_phase}.")


def main() -> None:
    st.set_page_config(page_title="ML Workbench", layout="wide")
    begin_run()
    inject_theme()
    st.sidebar.title("ML Workbench")
    st.sidebar.caption("Phase 8: Time Series and Association are live")

    state = _state()
    workspace = _workspace(state)
    tab_id = _navigation(state)
    _render_tab(state, workspace, tab_id)
    _sidebar_status(state)


if __name__ == "__main__":
    main()
