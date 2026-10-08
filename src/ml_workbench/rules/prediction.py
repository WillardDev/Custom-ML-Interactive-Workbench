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


NOVELTY_ONLY_MODELS = frozenset({"lof"})


def anomaly_scoring_plan(model_id: str) -> dict[str, Any]:
    """PRED-06: anomaly rows get a score and a flag at an adjustable threshold;
    LOF only scores unseen rows in novelty mode."""
    return {
        "score_by": "decision_function",
        "score_sign": "higher_is_more_anomalous",
        "threshold": 0.0,
        "novelty_mode": model_id in NOVELTY_ONLY_MODELS,
    }


def novelty_kwargs(model_id: str) -> dict[str, Any]:
    """PRED-06: estimator kwargs that keep novelty-only models in novelty mode."""
    if model_id in NOVELTY_ONLY_MODELS:
        return {"novelty": True}
    return {}


MAX_FORECAST_HORIZON: int = 48
FORECAST_ALPHA: float = 0.05


def forecast_horizon_plan(
    horizon: int = 12, *, max_horizon: int = MAX_FORECAST_HORIZON
) -> dict[str, Any]:
    """PRED-03: horizon selector with residual-quantile bands and an expanding backtest."""
    return {
        "horizon": max(1, min(int(horizon), int(max_horizon))),
        "max_horizon": int(max_horizon),
        "alpha": FORECAST_ALPHA,
        "bands": "residual_quantiles",
        "backtest": "expanding_window",
    }


def confidence_bands(
    point_forecast: np.ndarray,
    residuals: np.ndarray,
    alpha: float = FORECAST_ALPHA,
) -> tuple[np.ndarray, np.ndarray]:
    """PRED-03: constant-width bands from residual quantiles (lower, upper)."""
    point = np.asarray(point_forecast, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    if residuals.size == 0:
        return point.copy(), point.copy()
    lower = point + float(np.quantile(residuals, alpha / 2.0))
    upper = point + float(np.quantile(residuals, 1.0 - alpha / 2.0))
    return lower, upper


def backtest_folds(
    n_rows: int,
    *,
    n_splits: int = 3,
    min_train_fraction: float = 0.4,
) -> list[tuple[int, int]]:
    """PRED-03: expanding-window backtest cuts (train_end, test_end) over the tail."""
    if n_rows < 4 or n_splits < 1:
        return []
    first = max(2, int(n_rows * min_train_fraction))
    usable = n_rows - first
    if usable < n_splits:
        return []
    test_size = max(1, usable // n_splits)
    cuts: list[tuple[int, int]] = []
    for fold in range(n_splits):
        start = first + fold * test_size
        end = n_rows if fold == n_splits - 1 else start + test_size
        if end > start:
            cuts.append((start, end))
    return cuts
