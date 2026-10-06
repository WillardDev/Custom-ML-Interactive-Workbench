from __future__ import annotations

from ml_workbench.rules import (
    complete_rerun,
    evaluate_gate,
    gates,
    mark_downstream_stale,
    stale_banner,
    stale_tabs,
)
from ml_workbench.state import ProjectState
from ml_workbench.tabs import TAB_ORDER


def test_edit_upstream_marks_downstream_stale(ready_state: ProjectState) -> None:
    marked = mark_downstream_stale(ready_state, "cleaning")
    index = TAB_ORDER.index("cleaning")
    expected = list(TAB_ORDER[index + 1 :])
    assert marked == expected
    assert stale_tabs(ready_state) == expected
    assert not ready_state.stale["data"]
    assert not ready_state.stale["cleaning"]
    assert mark_downstream_stale(ready_state, "cleaning") == []


def test_stale_banner_and_rerun_prompt(ready_state: ProjectState) -> None:
    mark_downstream_stale(ready_state, "cleaning")
    banner = stale_banner(ready_state, "preprocessing")
    assert banner is not None
    assert "Data Cleaning" in banner
    assert "Re-run" in banner
    assert stale_banner(ready_state, "cleaning") is None
    assert stale_banner(ready_state, "data") is None


def test_rerun_clears_only_rerun_tab(ready_state: ProjectState) -> None:
    mark_downstream_stale(ready_state, "cleaning")
    complete_rerun(ready_state, "preprocessing")
    assert not ready_state.stale["preprocessing"]
    assert ready_state.stale["eda"]
    assert ready_state.stale["outcome"]


def test_stale_and_gated_are_orthogonal(ready_state: ProjectState) -> None:
    comparison = gates(ready_state)
    mark_downstream_stale(ready_state, "cleaning")
    after_edit = gates(ready_state)
    for tab_id in TAB_ORDER:
        assert after_edit[tab_id] == comparison[tab_id], tab_id

    ready_state.split = None
    ready_state.pipeline = None
    before = evaluate_gate("training", ready_state)
    mark_downstream_stale(ready_state, "cleaning")
    after = evaluate_gate("training", ready_state)
    assert after == before
    assert before.locked and before.rule_id == "GATE-01"
