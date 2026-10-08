from __future__ import annotations

from typing import Any


def supervised_deliverables() -> tuple[dict[str, str], ...]:
    """OUT-01: a supervised outcome packages pipeline, predictions, threshold and card."""
    return (
        {"deliverable": "refit_pipeline", "path": "outcome/pipeline.joblib"},
        {"deliverable": "predictions", "path": "outcome/predictions.csv"},
        {"deliverable": "chosen_threshold_or_interval", "path": "outcome/threshold.json"},
        {"deliverable": "model_card", "path": "outcome/model_card.json"},
    )


def clustering_deliverables() -> tuple[dict[str, str], ...]:
    """OUT-02: a clustering outcome ships cluster IDs, profiles and personas."""
    return (
        {"deliverable": "refit_pipeline", "path": "outcome/pipeline.joblib"},
        {"deliverable": "dataset_with_cluster_ids", "path": "outcome/clusters.csv"},
        {"deliverable": "cluster_profiles", "path": "outcome/profiles.csv"},
        {"deliverable": "cluster_personas", "path": "outcome/personas.json"},
        {"deliverable": "model_card", "path": "outcome/model_card.json"},
    )


def dimred_deliverables() -> tuple[dict[str, str], ...]:
    """OUT-03: a dim-reduction outcome ships embeddings, a variance/reconstruction
    summary and the reusable transformer."""
    return (
        {"deliverable": "reusable_transformer", "path": "outcome/pipeline.joblib"},
        {"deliverable": "transformed_dataset", "path": "outcome/embeddings.csv"},
        {"deliverable": "variance_reconstruction_summary", "path": "outcome/summary.json"},
        {"deliverable": "model_card", "path": "outcome/model_card.json"},
    )


def anomaly_deliverables() -> tuple[dict[str, str], ...]:
    """OUT-04: an anomaly outcome ships flagged rows, scores and the threshold used."""
    return (
        {"deliverable": "refit_pipeline", "path": "outcome/pipeline.joblib"},
        {"deliverable": "flagged_rows_with_scores", "path": "outcome/flagged.csv"},
        {"deliverable": "score_distribution", "path": "outcome/score_distribution.json"},
        {"deliverable": "threshold_used", "path": "outcome/threshold.json"},
        {"deliverable": "model_card", "path": "outcome/model_card.json"},
    )


def outcome_deliverables(task_type: str) -> tuple[dict[str, str], ...]:
    """OUT-01..04: the deliverable manifest by task type."""
    if task_type == "clustering":
        return clustering_deliverables()
    if task_type == "dimensionality_reduction":
        return dimred_deliverables()
    if task_type == "anomaly_detection":
        return anomaly_deliverables()
    return supervised_deliverables()


def report_bundle_parts() -> dict[str, str]:
    """EXPORT-01: the report bundle ships a report, a reproducible script and a manifest."""
    return {
        "report": "report.html",
        "script": "reproduce.py",
        "manifest": "manifest.json",
    }


def export_plan(onnx_available: bool) -> dict[str, Any]:
    """EXPORT-02: native artifact export always; ONNX adds portable inference."""
    return {
        "native": "joblib",
        "onnx": "onnx" if onnx_available else None,
        "note": (
            "ONNX will be enabled when the onnx package is importable"
            if not onnx_available
            else "ONNX artifact exported for portable inference"
        ),
    }
