from __future__ import annotations

from ml_workbench.rules.cleaning import (
    ClassBalance,
    CleanReport,
    NanDecision,
    OutlierDecision,
    apply_imputation,
    apply_outlier_treatment,
    class_balance,
    columns_with_missing,
    dedupe_and_fix_dtypes,
    drop_missing_target,
    fix_dtypes,
    nan_decision,
    outlier_decision,
    remove_duplicates,
)
from ml_workbench.rules.gating import GateDecision, evaluate_gate, gates
from ml_workbench.rules.staleness import (
    complete_rerun,
    mark_downstream_stale,
    stale_banner,
    stale_tabs,
)
from ml_workbench.rules.task import build_task, suggest_task_type

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
        "TASK-01",
        "TASK-02",
        "TASK-03",
        "CLEAN-01",
        "CLEAN-02",
        "CLEAN-03",
        "CLEAN-04a",
        "CLEAN-04b",
        "CLEAN-04c",
        "CLEAN-05",
        "CLEAN-06",
    }
)

__all__ = [
    "IMPLEMENTED_RULES",
    "CleanReport",
    "ClassBalance",
    "GateDecision",
    "NanDecision",
    "OutlierDecision",
    "apply_imputation",
    "apply_outlier_treatment",
    "build_task",
    "class_balance",
    "columns_with_missing",
    "complete_rerun",
    "dedupe_and_fix_dtypes",
    "drop_missing_target",
    "evaluate_gate",
    "fix_dtypes",
    "gates",
    "mark_downstream_stale",
    "nan_decision",
    "outlier_decision",
    "remove_duplicates",
    "stale_banner",
    "stale_tabs",
    "suggest_task_type",
]
