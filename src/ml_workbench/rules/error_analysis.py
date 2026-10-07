from __future__ import annotations

from typing import Any

_VIEWS: dict[str, tuple[str, ...]] = {
    "binary": (
        "confusion_matrix",
        "roc",
        "pr_curve",
        "threshold_analysis",
        "calibration",
    ),
    "multiclass": (
        "confusion_matrix_normalized",
        "per_class_precision_recall",
        "most_confused_pairs",
    ),
    "regression": (
        "residuals_vs_predicted",
        "residual_histogram",
        "qq_plot",
        "heteroscedasticity",
        "error_by_target_quantile",
    ),
}
DEFAULT_VIEWS: tuple[str, ...] = ("prediction_table",)

# ERR-04..ERR-06 (clustering / dim reduction / anomaly) arrive with Phase 6.
_PHASE_6_VIEWS: dict[str, tuple[str, ...]] = {
    "clustering": (
        "silhouette_per_sample",
        "low_silhouette_points",
        "cluster_size_imbalance",
        "stability_warning",
    ),
    "dim_reduction": (
        "reconstruction_error_per_row",
        "poorly_embedded_points",
    ),
    "anomaly": (
        "score_distribution",
        "top_flagged_rows",
        "false_positives_negatives",
    ),
}


def error_view_plan(task_type: str) -> tuple[dict[str, Any], ...]:
    """ERR-01/02/03: error-analysis views dispatched by the current task type."""
    views = (
        _VIEWS.get(task_type, _PHASE_6_VIEWS.get(task_type, DEFAULT_VIEWS))
        if task_type != "forecasting"
        else _VIEWS["regression"]
    )
    return tuple({"view": view} for view in views)


def common_error_tools() -> dict[str, bool]:
    """ERR-07: worst-N rows and segment slicing are available on any task."""
    return {"worst_n_rows": True, "segment_by_features": True}
