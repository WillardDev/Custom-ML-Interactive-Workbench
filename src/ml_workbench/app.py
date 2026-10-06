from __future__ import annotations

import streamlit as st

from ml_workbench.rules import (
    complete_rerun,
    evaluate_gate,
    gates,
    mark_downstream_stale,
    stale_banner,
)
from ml_workbench.rules.gating import ALL_REQS, DESCRIPTIONS, REQUIREMENTS, GateDecision
from ml_workbench.state import (
    DatasetInfo,
    ModelRun,
    PipelineInfo,
    ProjectState,
    SplitInfo,
    StepEntry,
    TaskDefinition,
)
from ml_workbench.tabs import TAB_BY_ID, TAB_ORDER, TITLES

SCENARIOS = {
    "Nothing loaded": 0,
    "Dataset loaded": 1,
    "Task selected": 2,
    "Cleaning done": 3,
    "Split fitted": 4,
    "Model trained": 5,
}


def _state() -> ProjectState:
    if "state" not in st.session_state:
        st.session_state["state"] = ProjectState(project_id="scaffold-demo")
    state: ProjectState = st.session_state["state"]
    return state


def _apply_scenario(state: ProjectState, level: int) -> None:
    state.dataset = None
    state.task = None
    state.steps = []
    state.split = None
    state.pipeline = None
    state.models = []
    state.active_model_id = None

    if level >= 1:
        state.dataset = DatasetInfo(
            version="v1",
            path="data/samples/binary_classification.csv",
            data_hash="sha256:scaffold",
            row_count=400,
            loaded_at="scaffold",
        )
    if level >= 2:
        state.task = TaskDefinition(
            learning_type="supervised", task_type="binary", target="churned"
        )
    if level >= 3:
        state.steps.append(StepEntry(id=1, tab="cleaning", op="dedupe", created_at="scaffold"))
    if level >= 4:
        state.split = SplitInfo(strategy="stratified_kfold", params={"n_splits": 5}, fitted=True)
        state.pipeline = PipelineInfo(fitted=True, path="workspace/scaffold/pipeline")
    if level >= 5:
        state.models.append(
            ModelRun(run_id="r_scaffold", model_id="hist_gradient_boosting", status="done")
        )
        state.active_model_id = "hist_gradient_boosting"


def _sync_simulators(state: ProjectState) -> None:
    st.sidebar.subheader("Scaffold: simulate state")
    choice = st.sidebar.radio("Simulated state", list(SCENARIOS), key="sim_scenario")
    _apply_scenario(state, SCENARIOS[choice])


def _navigation(state: ProjectState) -> str:
    st.sidebar.header("Workflow")

    def label(tab_id: str) -> str:
        info = TAB_BY_ID[tab_id]
        return f"{info.number}. {info.title}"

    return st.sidebar.radio("Workflow tabs", TAB_ORDER, format_func=label, key="workflow_nav")


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


def _render_scaffold_actions(state: ProjectState, tab_id: str) -> None:
    st.divider()
    st.caption("Scaffold actions (removed in Phase 2)")
    left, right = st.columns(2)
    with left:
        if st.button("Simulate edit of this tab", key="sim_edit"):
            marked = mark_downstream_stale(state, tab_id)
            if marked:
                st.write("Marked stale:", ", ".join(TITLES[item] for item in marked))
            else:
                st.write("No downstream tabs to mark stale.")
    with right:
        if st.button("Simulate re-run of this tab", key="sim_rerun"):
            complete_rerun(state, tab_id)
            st.write(f"{TITLES[tab_id]} refreshed; downstream staleness unchanged (STALE-03).")


def _render_tab(state: ProjectState, tab_id: str) -> None:
    info = TAB_BY_ID[tab_id]
    decision = evaluate_gate(tab_id, state)
    st.header(f"{info.number}. {info.title}")

    if decision.locked:
        st.info(decision.message)
        _requirements_panel(tab_id, decision, state)
        return

    banner = stale_banner(state, tab_id)
    if banner:
        st.warning(banner)

    st.write(f"Design specification: docs/tabs.md section {info.design_section}.")
    st.write(f"This tab is implemented in Phase {info.implemented_phase}.")
    _render_scaffold_actions(state, tab_id)


def main() -> None:
    st.set_page_config(page_title="ML Workbench", layout="wide")
    st.sidebar.title("ML Workbench")
    st.sidebar.caption("Phase 1 scaffold: navigation, gating, and staleness only")

    state = _state()
    _sync_simulators(state)
    tab_id = _navigation(state)
    _render_tab(state, tab_id)
    _sidebar_status(state)


if __name__ == "__main__":
    main()
