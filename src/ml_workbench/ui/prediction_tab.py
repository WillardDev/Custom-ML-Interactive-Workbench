from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pandas.api.types import is_numeric_dtype

from ml_workbench.services.model_cache import ModelCache
from ml_workbench.services.prediction_service import (
    PredictionError,
    calibration,
    evaluate_test,
    forecast_frame,
    predict_frame,
    predict_single,
    threshold_eval,
)
from ml_workbench.services.workspace import Workspace
from ml_workbench.state import ProjectState

MODEL_CACHE = ModelCache(capacity=2)


def render_prediction_tab(state: ProjectState, workspace: Workspace) -> None:
    trained = state.trained_models
    if not trained:
        st.info("No trained models yet. Train one in the Training tab.")
        return
    if state.frame is None or state.task is None:
        st.caption("Reload the dataset first (Data Insertion).")
        return

    run_id = st.selectbox(
        "Model",
        [run.run_id for run in trained],
        format_func=lambda run_id: _format_model(trained, run_id),
        key="pred_model",
    )

    binary = state.task.task_type == "binary"
    st.subheader("Test-set evaluation (PRED-01/PRED-02)")
    try:
        evaluation = evaluate_test(state, workspace, run_id, cache=MODEL_CACHE)
    except PredictionError as exc:
        st.error(str(exc))
        return
    st.caption(
        ", ".join(f"`{name} = {value:.4f}`" for name, value in sorted(evaluation.scores.items()))
    )

    if binary and evaluation.y_proba is not None:
        _render_binary(state, evaluation, run_id)
    elif state.task.task_type == "anomaly_detection":
        _render_anomaly(state, workspace, run_id)
    elif state.task.task_type == "forecasting":
        _render_forecasting(state, workspace, run_id, evaluation)
    elif state.task.learning_type == "unsupervised":
        _render_unsupervised(state, workspace, run_id, evaluation)
    else:
        _render_regression(state, evaluation, run_id)

    _render_single_row(state, workspace, run_id)
    _render_batch(state, workspace, run_id)


def _render_unsupervised(
    state: ProjectState,
    workspace: Workspace,
    run_id: str,
    evaluation: Any,
) -> None:
    st.markdown("##### Holdout assignments (first 200 test rows)")
    if state.split is not None and state.split.indices_path is not None:
        import json
        from pathlib import Path

        indices = json.loads(Path(state.split.indices_path).read_text())
        test_idx = np.asarray(indices["test"], dtype=int)
        if len(test_idx) > 0:
            assert state.frame is not None
            result = predict_frame(state, workspace, run_id, state.frame.iloc[test_idx])
            columns = [
                name
                for name in result.df.columns
                if str(name).startswith("pc") or name == "prediction"
            ]
            st.dataframe(
                result.df[columns].head(200),
                use_container_width=True,
            )
    if evaluation.y_true is not None and evaluation.y_pred is not None:
        st.markdown("##### Predicted vs evaluation labels")
        st.dataframe(
            pd.DataFrame({"label": evaluation.y_true, "prediction": evaluation.y_pred}).head(200),
            use_container_width=True,
        )


def _render_anomaly(state: ProjectState, workspace: Workspace, run_id: str) -> None:
    """PRED-06: score new rows, adjustable flag threshold, flagged-rows table."""
    assert state.frame is not None
    result = predict_frame(state, workspace, run_id, state.frame, cache=MODEL_CACHE)
    scores = np.asarray(result.df["score"], dtype=float)
    lo, hi = float(scores.min()), float(scores.max())
    default = float(np.clip(0.0, lo, hi))
    st.markdown("##### Scored rows (first 200)")
    st.dataframe(result.df.head(200), use_container_width=True)
    threshold = st.slider(
        "Flag threshold (higher = more anomalous)", lo, hi, default, key="anomaly_threshold"
    )
    flagged = result.df[result.df["score"] > threshold].sort_values("score", ascending=False)
    st.markdown(f"##### Flagged rows ({len(flagged)} of {len(result.df)})")
    if flagged.empty:
        st.info("No rows exceed the current threshold.")
    else:
        st.dataframe(flagged.head(200), use_container_width=True)


def _render_binary(state: ProjectState, evaluation: Any, run_id: str) -> None:
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, key="pred_threshold")
    metrics = threshold_eval(evaluation, threshold)
    st.table(
        pd.DataFrame(
            [
                {
                    "threshold": threshold,
                    **{
                        name: round(value, 4)
                        for name, value in metrics.items()
                        if name != "threshold"
                    },
                }
            ]
        )
    )
    centers, observed = calibration(evaluation, bins=10)
    st.markdown("##### Calibration (expected-vs-observed)")
    st.table(pd.DataFrame({"expected": centers, "observed": observed}).round(4))


def _render_regression(state: ProjectState, evaluation: Any, run_id: str) -> None:
    table = pd.DataFrame({"true": evaluation.y_true, "predicted": evaluation.y_pred})[
        ["true", "predicted"]
    ].round(4)
    st.markdown("##### Predicted vs actual (first 200 test rows)")
    st.dataframe(table.head(200), use_container_width=True)


def _render_forecasting(
    state: ProjectState,
    workspace: Workspace,
    run_id: str,
    evaluation: Any,
) -> None:
    st.markdown("##### Holdout forecast accuracy (METRIC-04)")
    st.table(pd.DataFrame([evaluation.scores], index=["value"]).T.round(4))

    run = next((model for model in state.models if model.run_id == run_id), None)
    payload: dict[str, Any] = {}
    if run is not None and run.metrics_path:
        payload = json.loads(Path(run.metrics_path).read_text()).get("forecast", {})
    if payload.get("backtest"):
        st.markdown("##### Expanding-window backtest (PRED-03)")
        st.dataframe(pd.DataFrame(payload["backtest"]), hide_index=True)
        st.caption(
            f"90% bands from backtest residual quantiles "
            f"(q05={payload.get('residual_q05')}, q95={payload.get('residual_q95')})"
        )

    horizon = st.slider("Forecast horizon (PRED-03)", 1, 48, 12, key="forecast_horizon")
    if st.button("Forecast ahead", key="forecast_run"):
        try:
            frame = forecast_frame(state, workspace, run_id, horizon, cache=MODEL_CACHE)
        except PredictionError as exc:
            st.error(str(exc))
        else:
            st.markdown(f"##### Next {horizon} steps with 90% residual bands")
            st.dataframe(frame, hide_index=True)
            figure = go.Figure()
            figure.add_scatter(
                x=frame["step"], y=frame["forecast"], mode="lines+markers", name="forecast"
            )
            figure.add_scatter(x=frame["step"], y=frame["upper"], mode="lines", name="upper")
            figure.add_scatter(
                x=frame["step"],
                y=frame["lower"],
                mode="lines",
                name="lower",
                fill="tonexty",
            )
            figure.update_layout(xaxis_title="step ahead", yaxis_title="forecast")
            st.plotly_chart(figure, use_container_width=True)


def _render_single_row(state: ProjectState, workspace: Workspace, run_id: str) -> None:
    frame = state.frame
    if frame is None:
        return
    st.subheader("Single-row prediction")
    row: dict[str, Any] = {}
    for column in frame.columns:
        if is_numeric_dtype(frame[column]):
            row[column] = st.number_input(
                column,
                value=float(frame[column].iloc[0]),
                key=f"pred_row_{column}",
            )
        else:
            unique = sorted(frame[column].dropna().unique().tolist())
            row[column] = st.selectbox(column, unique, key=f"pred_row_{column}")
    if st.button("Predict this row", key="pred_single"):
        try:
            output = predict_single(state, workspace, run_id, row, cache=MODEL_CACHE)
        except PredictionError as exc:
            st.error(str(exc))
        else:
            st.json(output, expanded=True)


def _render_batch(state: ProjectState, workspace: Workspace, run_id: str) -> None:
    st.subheader("Batch CSV prediction")
    uploaded = st.file_uploader(
        "Upload a CSV with the same feature columns", type=["csv"], key="pred_batch_file"
    )
    if uploaded is not None:
        frame = pd.read_csv(uploaded)
        if st.button("Predict batch", key="pred_batch"):
            try:
                result = predict_frame(state, workspace, run_id, frame, cache=MODEL_CACHE)
                st.dataframe(result.df, use_container_width=True)
                if result.has_proba:
                    st.caption("Probability columns `p_<class>` are included.")
            except PredictionError as exc:
                st.error(str(exc))


def _format_model(trained: list[Any], run_id: str) -> str:
    run = next((model for model in trained if model.run_id == run_id), None)
    return f"{run.model_id} ({run.run_id})" if run else run_id
