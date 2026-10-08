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

_PHASE_6_VIEWS: dict[str, tuple[str, ...]] = {
    "clustering": (
        "silhouette_per_sample",
        "low_silhouette_points",
        "cluster_size_imbalance",
        "stability_warning",
    ),
    "dimensionality_reduction": (
        "reconstruction_error_per_row",
        "poorly_embedded_points",
    ),
    "anomaly_detection": (
        "score_distribution",
        "top_flagged_rows",
        "false_positives_negatives",
    ),
}


def error_view_plan(task_type: str) -> tuple[dict[str, Any], ...]:
    """ERR-01..06: error-analysis views dispatched by the current task type.

    ERR-04: clustering → per-sample silhouette, low-silhouette points, size
        imbalance, stability warning.
    ERR-05: dimensionality reduction → reconstruction error per row, poorly
        embedded points.
    ERR-06: anomaly → score distribution, top flagged rows, FP/FN when labeled.
    """
    views = (
        _VIEWS.get(task_type, _PHASE_6_VIEWS.get(task_type, DEFAULT_VIEWS))
        if task_type != "forecasting"
        else _VIEWS["regression"]
    )
    return tuple({"view": view} for view in views)


def clustering_error_views() -> tuple[str, ...]:
    """ERR-04: the clustering error-analysis view set."""
    return _PHASE_6_VIEWS["clustering"]


def dimred_error_views() -> tuple[str, ...]:
    """ERR-05: the dimensionality-reduction error-analysis view set."""
    return _PHASE_6_VIEWS["dimensionality_reduction"]


def anomaly_error_views() -> tuple[str, ...]:
    """ERR-06: the anomaly-detection error-analysis view set."""
    return _PHASE_6_VIEWS["anomaly_detection"]


def common_error_tools() -> dict[str, bool]:
    """ERR-07: worst-N rows and segment slicing are available on any task."""
    return {"worst_n_rows": True, "segment_by_features": True}
