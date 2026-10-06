from __future__ import annotations

from pathlib import Path

from ml_workbench.rules import evaluate_gate, gates
from ml_workbench.rules.gating import REQUIREMENTS
from ml_workbench.state import ModelRun, ProjectState, SplitInfo
from ml_workbench.tabs import TAB_ORDER, TABS, TITLES

REPO_ROOT = Path(__file__).resolve().parents[2]
TABS_DOC = REPO_ROOT / "docs" / "tabs.md"

REQUIREMENT_COLUMNS = ["project", "task", "cleaned", "split_or_pipeline", "model"]


def _matrix_from_docs() -> dict[str, frozenset[str]]:
    title_to_id = {tab.title: tab.id for tab in TABS}
    matrix: dict[str, frozenset[str]] = {}
    for line in TABS_DOC.read_text().splitlines():
        if not line.startswith("| ") or "---" in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 7 or not cells[0].isdigit():
            continue
        title = cells[1]
        if title not in title_to_id:
            continue
        required = {
            requirement
            for requirement, cell in zip(REQUIREMENT_COLUMNS, cells[2:], strict=True)
            if cell == "R" or cell.startswith("R ") or cell.startswith("*")
        }
        matrix[title_to_id[title]] = frozenset(required)
    return matrix


def test_gate_training_requires_split_or_pipeline(ready_state: ProjectState) -> None:
    ready_state.split = None
    ready_state.pipeline = None
    decision = evaluate_gate("training", ready_state)
    assert decision.locked
    assert decision.rule_id == "GATE-01"
    assert "split_or_pipeline" in decision.missing
    assert "GATE-01" in decision.message

    ready_state.split = SplitInfo(strategy="stratified_kfold", fitted=True)
    assert not evaluate_gate("training", ready_state).locked


def test_gate_tabs_require_trained_model(ready_state: ProjectState) -> None:
    ready_state.models = []
    ready_state.active_model_id = None
    for tab_id in ("prediction", "error_analysis", "explainability"):
        decision = evaluate_gate(tab_id, ready_state)
        assert decision.locked, tab_id
        assert decision.rule_id == "GATE-02", tab_id
        assert "model" in decision.missing, tab_id

    ready_state.models = [ModelRun(run_id="r_001", model_id="logistic_regression", status="done")]
    for tab_id in ("prediction", "error_analysis", "explainability"):
        assert not evaluate_gate(tab_id, ready_state).locked, tab_id


def test_gate_matrix_conformance(empty_state: ProjectState) -> None:
    documented = _matrix_from_docs()
    assert set(documented) == set(TAB_ORDER)
    for tab_id, required in documented.items():
        assert required == REQUIREMENTS[tab_id], f"matrix mismatch for tab '{tab_id}'"


def test_gate_outcome_requires_model(ready_state: ProjectState) -> None:
    ready_state.models = []
    ready_state.active_model_id = None
    decision = evaluate_gate("outcome", ready_state)
    assert decision.locked
    assert decision.rule_id == "GATE-04"
    assert "model" in decision.missing

    ready_state.models = [ModelRun(run_id="r_001", model_id="logistic_regression", status="done")]
    ready_state.split = None
    ready_state.pipeline = None
    decision_without_split = evaluate_gate("outcome", ready_state)
    assert decision_without_split.locked
    assert decision_without_split.rule_id == "GATE-01"

    ready_state.split = SplitInfo(strategy="stratified_kfold", fitted=True)
    assert not evaluate_gate("outcome", ready_state).locked


def test_gate_data_tab_always_unlocked(empty_state: ProjectState) -> None:
    decisions = gates(empty_state)
    assert not decisions["data"].locked
    assert decisions["cleaning"].locked
    assert TITLES["data"] == "Data Insertion"
