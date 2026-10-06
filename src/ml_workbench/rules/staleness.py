from __future__ import annotations

from ml_workbench.state import ProjectState
from ml_workbench.tabs import TAB_ORDER, TITLES


def mark_downstream_stale(state: ProjectState, tab_id: str) -> list[str]:
    index = TAB_ORDER.index(tab_id)
    state.stale[tab_id] = False
    newly_marked = [later for later in TAB_ORDER[index + 1 :] if not state.stale[later]]
    for later in newly_marked:
        state.stale[later] = True
    return newly_marked


def complete_rerun(state: ProjectState, tab_id: str) -> None:
    state.stale[tab_id] = False


def stale_tabs(state: ProjectState) -> list[str]:
    return [tab_id for tab_id in TAB_ORDER if state.stale[tab_id]]


def _cause_tab(state: ProjectState) -> str | None:
    stale_ids = stale_tabs(state)
    if not stale_ids:
        return None
    first_stale_index = TAB_ORDER.index(stale_ids[0])
    if first_stale_index == 0:
        return None
    return TAB_ORDER[first_stale_index - 1]


def stale_banner(state: ProjectState, tab_id: str) -> str | None:
    if not state.stale.get(tab_id, False):
        return None
    cause = _cause_tab(state)
    if cause is None:
        return (
            f"{TITLES[tab_id]} is stale: upstream results changed after these "
            "results were computed. Re-run to refresh."
        )
    return (
        f"{TITLES[tab_id]} is stale: {TITLES[cause]} changed after these "
        "results were computed. Re-run to refresh."
    )
