from __future__ import annotations

import numpy as np
import pytest

from ml_workbench.rules.prediction import (
    anomaly_scoring_plan,
    apply_threshold,
    calibration_curve,
    needs_surrogate,
    novelty_kwargs,
    prediction_interval,
    projection_blocked,
    threshold_metrics,
)


def test_pred_binary_threshold_calibration() -> None:
    proba = np.array([0.9, 0.6, 0.4, 0.1])
    hard = apply_threshold(proba, 0.5)
    assert hard.tolist() == [1, 1, 0, 0]

    y_true = np.array([1, 1, 0, 0])
    metrics = threshold_metrics(y_true, proba, 0.5)
    assert metrics["threshold"] == 0.5
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0

    centers, observed = calibration_curve(proba, y_true, bins=10)
    assert len(centers) == 10
    assert len(observed) == 10
    assert observed.min() >= 0.0 and observed.max() <= 1.0


def test_pred_regression_intervals() -> None:
    predictions = np.array([10.0, 20.0, 30.0])
    lower, upper = prediction_interval(predictions, residual_std=2.0, alpha=0.05)
    width = upper - lower
    assert np.allclose(width, 2 * 1.959963984540054 * 2.0)
    assert np.all(lower < predictions)
    assert np.all(upper > predictions)


@pytest.mark.skip(reason="forecasting horizon/backtest arrives in Phase 8 (docs/rules.md PRED-03)")
def test_pred_forecast_horizon_backtest() -> None:
    raise NotImplementedError("docs/rules.md PRED-03")


def test_pred_clustering_surrogate_fallback() -> None:
    assert needs_surrogate({"family": "density", "has_predict": False}) == "nearest_centroid"
    assert needs_surrogate({"family": "density", "has_predict": True}) is None
    assert needs_surrogate({"family": "linear", "has_predict": True}) is None


def test_pred_tsne_projection_blocked() -> None:
    assert projection_blocked({"viz_only": True}) is True
    assert projection_blocked({"viz_only": False}) is False


def test_pred_anomaly_lof_novelty_only() -> None:
    plan = anomaly_scoring_plan("isolation_forest")
    assert plan["score_by"] == "decision_function"
    assert plan["score_sign"] == "higher_is_more_anomalous"
    assert plan["threshold"] == 0.0
    assert plan["novelty_mode"] is False
    assert novelty_kwargs("isolation_forest") == {}
    lof = anomaly_scoring_plan("lof")
    assert lof["novelty_mode"] is True
    assert lof["threshold"] == 0.0
    assert novelty_kwargs("lof") == {"novelty": True}


@pytest.mark.skip(reason="association rules arrive in Phase 8 (docs/rules.md PRED-07)")
def test_pred_basket_recommendations() -> None:
    raise NotImplementedError("docs/rules.md PRED-07")
