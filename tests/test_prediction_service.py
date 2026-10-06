from __future__ import annotations

import pandas as pd
import pytest

from ml_workbench.services.model_cache import ModelCache
from ml_workbench.services.prediction_service import (
    calibration,
    evaluate_test,
    get_pipeline,
    predict_single,
    threshold_eval,
)
from ml_workbench.services.training_service import train_model

LOGISTIC_PARAMS = {"C": 1.0, "penalty": "l2", "solver": "lbfgs"}


def _train(state, ws, model_id: str, params: dict[str, object]) -> str:
    return train_model(state, ws, model_id, params).run_id


def test_predict_single_returns_prediction_and_proba(
    prepared_workspace, binary_frame: pd.DataFrame
) -> None:
    state, ws = prepared_workspace(binary_frame, target="target", task_type="binary")
    run_id = _train(state, ws, "logistic_regression", LOGISTIC_PARAMS)
    row = {"units": 3.0, "band": "a", "score": 0.5}
    output = predict_single(state, ws, run_id, row)
    assert "prediction" in output
    assert any(key.startswith("p_") for key in output)


def test_evaluate_test_threshold_and_calibration(
    prepared_workspace, binary_frame: pd.DataFrame
) -> None:
    state, ws = prepared_workspace(binary_frame, target="target", task_type="binary")
    run_id = _train(state, ws, "decision_tree", {"max_depth": 4})
    evaluation = evaluate_test(state, ws, run_id)
    assert evaluation.y_true.shape == evaluation.y_pred.shape
    assert evaluation.y_proba is not None
    assert {"accuracy", "f1"}.issubset(evaluation.scores)

    at_50 = threshold_eval(evaluation, threshold=0.5)
    assert "threshold" in at_50 and "f1" in at_50

    expected, observed = calibration(evaluation, bins=4)
    assert len(expected) == len(observed)
    assert float(observed.min()) >= 0.0 and float(observed.max()) <= 1.0


def test_get_pipeline_uses_lru_cache_and_evicts(
    prepared_workspace, binary_frame: pd.DataFrame
) -> None:
    state, ws = prepared_workspace(binary_frame, target="target", task_type="binary")
    run_a = _train(state, ws, "logistic_regression", LOGISTIC_PARAMS)
    run_b = _train(state, ws, "decision_tree", {"max_depth": 3})
    run_c = _train(state, ws, "random_forest", {"n_estimators": 20, "max_depth": 4})

    cache = ModelCache(capacity=2)
    first = get_pipeline(ws, run_a, cache)
    assert get_pipeline(ws, run_a, cache) is first, "second load is a cache hit (PERF-02)"
    assert cache.hits == 1 and cache.misses == 1

    get_pipeline(ws, run_b, cache)
    get_pipeline(ws, run_c, cache)
    assert cache.size == 2
    assert cache.evicted() == [run_a], "capacity 2 evicts the least recently used"
    assert cache.get(run_b) is not None


def test_unknown_run_raises(prepared_workspace, binary_frame: pd.DataFrame) -> None:
    from ml_workbench.services.prediction_service import PredictionError

    state, ws = prepared_workspace(binary_frame, target="target", task_type="binary")
    with pytest.raises(PredictionError, match="unknown run"):
        predict_single(state, ws, "run_missing1", {})
