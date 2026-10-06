from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm as _norm


def apply_threshold(proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """PRED-01: probability threshold maps soft outputs to hard labels."""
    return np.where(proba >= threshold, 1, 0)


def threshold_metrics(y_true: Any, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """PRED-01: confusion-matrix based metrics at an adjustable threshold."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = apply_threshold(proba, threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = float(np.mean(y_true == y_pred))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "threshold": threshold,
    }


def calibration_curve(
    proba: np.ndarray, y_true: Any, bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """PRED-01: expected-vs-observed calibration across probability bins."""
    proba = np.asarray(proba, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    observed: list[float] = []
    for i in range(bins):
        mask = (proba >= edges[i]) & (proba < edges[i + 1])
        observed.append(float(np.mean(y_true[mask])) if mask.any() else 0.0)
    return centers, np.asarray(observed, dtype=float)


def prediction_interval(
    y_pred: np.ndarray, residual_std: float, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """PRED-02: naive gaussian residual band around regression predictions."""
    z = -float(_quantile_normal(alpha / 2.0))
    width = z * residual_std
    return np.asarray(y_pred) - width, np.asarray(y_pred) + width


def _quantile_normal(q: float) -> float:
    # Inverse CDF of the standard normal.
    return float(_norm.ppf(q))


def needs_surrogate(flags: dict[str, Any]) -> str | None:
    """PRED-04: density models without a predict method need a nearest-centroid surrogate."""
    if flags.get("family") == "density" and not flags.get("has_predict"):
        return "nearest_centroid"
    return None


def projection_blocked(flags: dict[str, Any]) -> bool:
    """PRED-05: viz-only projections are blocked from the prediction pipeline."""
    return bool(flags.get("viz_only"))
