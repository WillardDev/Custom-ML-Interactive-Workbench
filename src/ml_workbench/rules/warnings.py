from __future__ import annotations

from typing import Any

from ml_workbench.rules.modelling import SMALL_DATA_ROWS


def viz_only_blocks_prediction(flags: dict[str, Any]) -> bool:
    """WARN-01/WARN-07: a viz-only model (t-SNE) is blocked from predicting new rows."""
    return bool(flags.get("viz_only"))


def viz_only_blocks_downstream(flags: dict[str, Any]) -> bool:
    """WARN-07: a viz-only model used in a pipeline meant for new rows blocks downstream use."""
    return bool(flags.get("family") == "manifold" and flags.get("viz_only"))


def density_needs_surrogate(flags: dict[str, Any]) -> bool:
    """WARN-02: DBSCAN (no predict) cannot assign new rows — a surrogate is required."""
    return bool(flags.get("family") == "density" and not flags.get("has_predict"))


def smote_blocked(task_type: str) -> bool:
    """WARN-03: SMOTE is blocked for regression."""
    return task_type == "regression"


def target_selection_blocked(learning_type: str) -> bool:
    """WARN-04: target-based feature selection is blocked for unsupervised tasks."""
    return learning_type == "unsupervised"


def outlier_removal_blocked(task_type: str) -> bool:
    """WARN-05: outlier removal is blocked when the task is anomaly detection."""
    return task_type == "anomaly_detection"


def neural_small_data(family: str, row_count: int) -> bool:
    """WARN-06: warn when a neural model is picked on small data."""
    return family == "neural" and row_count < SMALL_DATA_ROWS


def warn_message(
    flags: dict[str, Any], task_type: str, learning_type: str, row_count: int, family: str
) -> str | None:
    """Return the first applicable warning message for the model + task combination."""
    if viz_only_blocks_prediction(flags):
        return "This model is visualization-only (t-SNE): new-row prediction is blocked (WARN-01)."
    if density_needs_surrogate(flags):
        return "DBSCAN cannot assign new rows — a nearest-centroid surrogate is required "
    "(WARN-02/PRED-04)."
    if smote_blocked(task_type):
        return "SMOTE is blocked for regression targets (WARN-03)."
    if target_selection_blocked(learning_type):
        return "Target-based feature selection is blocked for unsupervised tasks (WARN-04)."
    if outlier_removal_blocked(task_type):
        return "Outlier removal is blocked for anomaly detection (WARN-05)."
    if neural_small_data(family, row_count):
        return "Neural model on small data: consider a boosting baseline first (WARN-06)."
    return None
