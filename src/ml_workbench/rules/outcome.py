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
