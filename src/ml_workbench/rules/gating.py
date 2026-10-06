from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ml_workbench.state import ProjectState
from ml_workbench.tabs import TAB_ORDER

PROJECT_REQ: Final = "project"
TASK_REQ: Final = "task"
CLEANED_REQ: Final = "cleaned"
SPLIT_REQ: Final = "split_or_pipeline"
MODEL_REQ: Final = "model"

ALL_REQS: Final = (PROJECT_REQ, TASK_REQ, CLEANED_REQ, SPLIT_REQ, MODEL_REQ)

REQUIREMENTS: Final[dict[str, frozenset[str]]] = {
    "data": frozenset(),
    "cleaning": frozenset({PROJECT_REQ, TASK_REQ}),
    "preprocessing": frozenset({PROJECT_REQ, TASK_REQ, CLEANED_REQ}),
    "eda": frozenset({PROJECT_REQ, TASK_REQ, CLEANED_REQ}),
    "modelling": frozenset({PROJECT_REQ, TASK_REQ, CLEANED_REQ}),
    "training": frozenset({PROJECT_REQ, TASK_REQ, CLEANED_REQ, SPLIT_REQ}),
    "prediction": frozenset({PROJECT_REQ, TASK_REQ, CLEANED_REQ, SPLIT_REQ, MODEL_REQ}),
    "error_analysis": frozenset({PROJECT_REQ, TASK_REQ, CLEANED_REQ, SPLIT_REQ, MODEL_REQ}),
    "explainability": frozenset({PROJECT_REQ, TASK_REQ, CLEANED_REQ, SPLIT_REQ, MODEL_REQ}),
    "outcome": frozenset({PROJECT_REQ, TASK_REQ, CLEANED_REQ, SPLIT_REQ, MODEL_REQ}),
}

DESCRIPTIONS: Final[dict[str, str]] = {
    PROJECT_REQ: "load a dataset (Data Insertion)",
    TASK_REQ: "select a task (Data Insertion)",
    CLEANED_REQ: "run Data Cleaning",
    SPLIT_REQ: "fit a split or preprocessing pipeline",
    MODEL_REQ: "train at least one model",
}


@dataclass(frozen=True)
class GateDecision:
    tab_id: str
    locked: bool
    missing: tuple[str, ...] = ()
    rule_id: str | None = None
    message: str = ""


def _satisfied(requirement: str, state: ProjectState) -> bool:
    if requirement == PROJECT_REQ:
        return state.dataset is not None
    if requirement == TASK_REQ:
        return state.task is not None
    if requirement == CLEANED_REQ:
        return state.cleaned
    if requirement == SPLIT_REQ:
        return state.split_ready
    if requirement == MODEL_REQ:
        return state.has_trained_model
    raise ValueError(f"unknown requirement '{requirement}'")


def _rule_for_tab(tab_id: str, missing: tuple[str, ...]) -> str:
    if MODEL_REQ in missing:
        return "GATE-04" if tab_id == "outcome" else "GATE-02"
    if SPLIT_REQ in missing:
        return "GATE-01"
    return "GATE-03"


def evaluate_gate(tab_id: str, state: ProjectState) -> GateDecision:
    requirements = REQUIREMENTS[tab_id]
    missing = tuple(req for req in ALL_REQS if req in requirements and not _satisfied(req, state))
    if not missing:
        return GateDecision(tab_id=tab_id, locked=False)
    rule_id = _rule_for_tab(tab_id, missing)
    message = f"Locked: {', '.join(DESCRIPTIONS[req] for req in missing)} ({rule_id})"
    return GateDecision(
        tab_id=tab_id, locked=True, missing=missing, rule_id=rule_id, message=message
    )


def gates(state: ProjectState) -> dict[str, GateDecision]:
    return {tab_id: evaluate_gate(tab_id, state) for tab_id in TAB_ORDER}
