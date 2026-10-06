from __future__ import annotations

from ml_workbench.rules.gating import GateDecision, evaluate_gate, gates
from ml_workbench.rules.staleness import (
    complete_rerun,
    mark_downstream_stale,
    stale_banner,
    stale_tabs,
)

IMPLEMENTED_RULES = frozenset(
    {
        "GATE-01",
        "GATE-02",
        "GATE-03",
        "GATE-04",
        "STALE-01",
        "STALE-02",
        "STALE-03",
        "STALE-04",
    }
)

__all__ = [
    "IMPLEMENTED_RULES",
    "GateDecision",
    "complete_rerun",
    "evaluate_gate",
    "gates",
    "mark_downstream_stale",
    "stale_banner",
    "stale_tabs",
]
