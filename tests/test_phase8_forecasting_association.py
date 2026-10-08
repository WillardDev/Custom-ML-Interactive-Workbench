from __future__ import annotations

import pandas as pd
import pytest

from ml_workbench.services.eda_service import eda_report
from ml_workbench.services.explain_service import explain_model
from ml_workbench.services.outcome_service import build_unsupervised_outcome
from ml_workbench.services.prediction_service import evaluate_test, forecast_frame, predict_frame
from ml_workbench.services.training_service import train_model

LAG_PARAMS = {"lags": 12, "learning_rate": 0.1, "max_iter": 100}
APRIORI_PARAMS = {"min_support": 0.15, "min_confidence": 0.3, "min_lift": 1.0, "max_len": 3}


@pytest.fixture
def forecasting_frame() -> pd.DataFrame:
    return pd.read_csv("data/samples/forecasting.csv")


@pytest.fixture
def association_frame() -> pd.DataFrame:
    return pd.read_csv("data/samples/association.csv")


def test_forecasting_full_flow(prepared_workspace, forecasting_frame: pd.DataFrame) -> None:
    state, ws = prepared_workspace(
        forecasting_frame,
        target="value",
        task_type="forecasting",
        time_column="date",
        confirm_forecasting=True,
    )
    naive_id = train_model(state, ws, "naive", {}).run_id
    run_id = train_model(state, ws, "lag_boosting", LAG_PARAMS).run_id

    payload = ws.read_run_json(naive_id, "metrics.json")
    assert "forecast" in payload
    forecast = payload["forecast"]
    assert forecast["backtest"] and set(forecast["backtest"][0]) >= {"fold", "mae"}
    assert "residual_q05" in forecast and "residual_q95" in forecast

    evaluation = evaluate_test(state, ws, run_id)
    assert {"mae", "rmse", "smape", "mase"} <= set(evaluation.scores)

    horizon = forecast_frame(state, ws, run_id, 6)
    assert list(horizon.columns) == ["step", "forecast", "lower", "upper"]
    assert len(horizon) == 6
    assert (horizon["lower"] <= horizon["upper"]).all()

    explanation = explain_model(state, ws, run_id)
    assert explanation.method == "history_trace"
    panels = {panel["panel"] for panel in explanation.panels}
    assert "history_trace" in panels

    report = eda_report(forecasting_frame, state.task)
    assert report.timeseries is not None
    assert report.timeseries.period >= 1
    assert {"decomposition", "acf_pacf", "rolling_stats", "stationarity"} <= set(report.views)


def test_forecasting_metrics_in_metrics_json(
    prepared_workspace, forecasting_frame: pd.DataFrame
) -> None:
    state, ws = prepared_workspace(
        forecasting_frame,
        target="value",
        task_type="forecasting",
        time_column="date",
        confirm_forecasting=True,
    )
    run_id = train_model(state, ws, "seasonal_naive", {"period": 12}).run_id
    evaluation = evaluate_test(state, ws, run_id)
    assert evaluation.scores["mae"] >= 0.0
    assert evaluation.scores["mase"] >= 0.0


def test_association_full_flow(prepared_workspace, association_frame: pd.DataFrame) -> None:
    state, ws = prepared_workspace(
        association_frame,
        learning_type="unsupervised",
        task_type="association",
    )
    run_id = train_model(state, ws, "apriori", APRIORI_PARAMS).run_id

    evaluation = evaluate_test(state, ws, run_id)
    assert {"support", "confidence", "lift"} <= set(evaluation.scores)
    assert evaluation.scores["lift"] >= 1.0

    result = predict_frame(state, ws, run_id, association_frame).df
    assert "prediction" in result.columns
    assert len(result) == len(association_frame)

    explanation = explain_model(state, ws, run_id)
    assert explanation.method == "rules"
    panels = {panel["panel"] for panel in explanation.panels}
    assert {"rule_network", "lift_vs_confidence", "rule_table"} <= panels
    rules_panel = next(p for p in explanation.panels if p["panel"] == "rule_table")
    assert rules_panel["table"]

    outcome = build_unsupervised_outcome(state, ws, run_id)
    rules_path = ws.project_dir / "outcome" / "rules.csv"
    assert rules_path.is_file()
    rules = pd.read_csv(rules_path)
    assert {"antecedent", "consequent", "support", "confidence", "lift"} <= set(rules.columns)
    assert len(rules) >= 1
    assert "outcome/rules.csv" in outcome.files

    report = eda_report(association_frame, state.task)
    assert report.association is not None
    # one sample row is all-NaN (an empty basket) — 39 real baskets remain.
    assert report.association.baskets == 39
    assert report.association.unique_items >= 2
    assert {"item_frequency", "basket_size"} <= set(report.views)


def test_association_rules_are_mined(prepared_workspace, association_frame: pd.DataFrame) -> None:
    state, ws = prepared_workspace(
        association_frame,
        learning_type="unsupervised",
        task_type="association",
    )
    run_id = train_model(state, ws, "apriori", APRIORI_PARAMS).run_id
    metrics = ws.read_run_json(run_id, "metrics.json")
    assert metrics["rule_id"] == "METRIC-08"
    assert metrics["primary"] == "lift"
    assert metrics["mean"] >= 1.0
    fold = metrics["folds"][0]
    assert fold["support"] > 0.0
    assert fold["confidence"] > 0.0
    assert fold["lift"] >= 1.0
