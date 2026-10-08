from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_workbench.services.error_service import error_views
from ml_workbench.services.explain_service import explain_model
from ml_workbench.services.outcome_service import build_unsupervised_outcome
from ml_workbench.services.prediction_service import evaluate_test, predict_frame
from ml_workbench.services.training_service import train_model

NEURAL_PARAMS = {"preset": "small", "epochs": 20, "patience": 5, "batch_size": 32}


@pytest.fixture
def anomaly_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    normal = rng.normal(0, 1, size=(180, 2))
    outliers = rng.normal(8, 1, size=(20, 2))
    frame = pd.DataFrame(np.vstack([normal, outliers]), columns=["x", "y"])
    frame["label"] = 0
    frame.loc[180:, "label"] = 1
    return frame


def test_neural_classification_full_flow(prepared_workspace, binary_frame: pd.DataFrame) -> None:
    state, ws = prepared_workspace(binary_frame, target="target", task_type="binary")
    run_id = train_model(state, ws, "mlp", NEURAL_PARAMS).run_id

    neural = ws.read_run_json(run_id, "metrics.json")["neural"]
    assert neural["device"] == "cpu"
    assert 1 <= len(neural["loss_curve"]) == len(neural["val_loss_curve"]) <= 20
    assert 1 <= neural["best_epoch"] <= neural["epochs_run"] <= 20
    assert neural["patience"] == 5
    assert neural["stopped_early"] is (neural["epochs_run"] < neural["epochs"])

    evaluation = evaluate_test(state, ws, run_id)
    assert "accuracy" in evaluation.scores and "f1" in evaluation.scores

    report = error_views(state, ws, run_id)
    names = {str(view["view"]) for view in report.views}
    assert {"learning_curve", "overfitting_diagnostics", "per_epoch_metrics"} <= names
    diagnostics = next(v for v in report.views if str(v["view"]) == "overfitting_diagnostics")
    assert diagnostics["epochs"] == 20
    assert diagnostics["patience"] == 5
    assert diagnostics["best_epoch"] == neural["best_epoch"]
    assert diagnostics["final_train_loss"] == pytest.approx(neural["loss_curve"][-1])

    explanation = explain_model(state, ws, run_id)
    assert explanation.method == "integrated_gradients"
    panels = {panel["panel"] for panel in explanation.panels}
    assert {"gradient_attributions", "gradient_local"} <= panels
    attributions = next(p for p in explanation.panels if p["panel"] == "gradient_attributions")
    assert attributions["steps"] == 32
    assert attributions["table"] and {"feature", "mean_abs"} <= set(attributions["table"][0])
    local = next(p for p in explanation.panels if p["panel"] == "gradient_local")
    assert local["attribution"]

    result = predict_frame(state, ws, run_id, binary_frame).df
    assert any(str(column).startswith("p_") for column in result.columns)
    assert len(result) == len(binary_frame)


def test_neural_regression_gradient(prepared_workspace, binary_frame: pd.DataFrame) -> None:
    state, ws = prepared_workspace(binary_frame, target="units", task_type="regression")
    run_id = train_model(state, ws, "mlp", NEURAL_PARAMS).run_id

    neural = ws.read_run_json(run_id, "metrics.json")["neural"]
    assert neural["loss_curve"]

    evaluation = evaluate_test(state, ws, run_id)
    assert any(name in evaluation.scores for name in ("rmse", "mae", "r2"))

    explanation = explain_model(state, ws, run_id)
    assert explanation.method == "integrated_gradients"
    panels = {panel["panel"] for panel in explanation.panels}
    assert {"gradient_attributions", "gradient_local"} <= panels


def test_anomaly_scores_flag_outliers(prepared_workspace, anomaly_frame: pd.DataFrame) -> None:
    state, ws = prepared_workspace(
        anomaly_frame,
        learning_type="unsupervised",
        task_type="anomaly_detection",
        eval_labels="label",
    )
    run_id = train_model(
        state, ws, "isolation_forest", {"n_estimators": 100, "contamination": 0.1}
    ).run_id

    result = predict_frame(state, ws, run_id, anomaly_frame).df
    assert {"score", "prediction"} <= set(result.columns)
    # higher score = more anomalous: outliers outrank inliers on average.
    assert result["score"].iloc[180:].mean() > result["score"].iloc[:180].mean()
    # the default threshold marks the outliers, not the normal rows.
    assert int(result.loc[:179, "prediction"].sum()) <= 25
    assert int(result.loc[180:, "prediction"].sum()) >= 12

    evaluation = evaluate_test(state, ws, run_id)
    assert "roc_auc" in evaluation.scores and evaluation.scores["roc_auc"] > 0.9

    report = error_views(state, ws, run_id)
    confusion = next(v for v in report.views if str(v["view"]) == "false_positives_negatives")
    # holdout is small (stratified eval split), so only check the direction.
    assert confusion["tp"] >= 2
    assert confusion["fp"] <= 5
    assert confusion["tp"] + confusion["fp"] + confusion["fn"] + confusion["tn"] == len(
        report.predicted
    )

    build_unsupervised_outcome(state, ws, run_id)
    flagged = pd.read_csv(ws.outcome_dir / "flagged.csv")
    assert {"score", "flagged"} <= set(flagged.columns)
    flagged_rows = flagged[flagged["flagged"] == 1]
    assert 10 <= len(flagged_rows) <= 40
    assert flagged_rows["label"].mean() >= 0.7


def test_lof_novelty_scores_rows(prepared_workspace, anomaly_frame: pd.DataFrame) -> None:
    state, ws = prepared_workspace(
        anomaly_frame,
        learning_type="unsupervised",
        task_type="anomaly_detection",
        eval_labels="label",
    )
    run_id = train_model(state, ws, "lof", {"n_neighbors": 30, "contamination": 0.1}).run_id

    # novelty mode: decision_function scores rows after fit (k > outliers forces
    # each far point to compare against the dense core).
    result = predict_frame(state, ws, run_id, anomaly_frame).df
    assert {"score", "prediction"} <= set(result.columns)
    assert result["score"].iloc[180:].mean() > result["score"].iloc[:180].mean()
    assert int(result.loc[180:, "prediction"].sum()) >= 15
    assert int(result.loc[:179, "prediction"].sum()) <= 5

    evaluation = evaluate_test(state, ws, run_id)
    assert "roc_auc" in evaluation.scores and evaluation.scores["roc_auc"] > 0.8

    explanation = explain_model(state, ws, run_id)
    panels = {panel["panel"] for panel in explanation.panels}
    assert "per_feature_deviation" in panels
