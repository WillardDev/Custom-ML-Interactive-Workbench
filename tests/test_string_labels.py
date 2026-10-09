from __future__ import annotations

import pandas as pd

from ml_workbench.services.error_service import error_views
from ml_workbench.services.explain_service import explain_model
from ml_workbench.services.outcome_service import build_supervised_outcome
from ml_workbench.services.prediction_service import (
    calibration,
    evaluate_test,
    predict_single,
    threshold_eval,
)
from ml_workbench.services.training_service import train_model

LOGISTIC = {"C": 1.0, "penalty": "l2", "solver": "lbfgs"}


def test_string_labels_full_path(prepared_workspace, string_binary_frame: pd.DataFrame) -> None:
    state, ws = prepared_workspace(string_binary_frame, target="target", task_type="binary")
    run_id = train_model(state, ws, "logistic_regression", LOGISTIC).run_id

    evaluation = evaluate_test(state, ws, run_id)
    assert evaluation.positive_label == "Yes"
    threshold_eval(evaluation, 0.5)
    calibration(evaluation, bins=4)
    predict_single(state, ws, run_id, {"units": 3.0, "band": "a", "score": 0.5})
    error_views(state, ws, run_id)

    explain_model(state, ws, run_id, force_recompute=True)
    build_supervised_outcome(state, ws, run_id)
