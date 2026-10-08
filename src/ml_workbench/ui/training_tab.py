from __future__ import annotations

from typing import Any

import streamlit as st

from ml_workbench.services.training_service import (
    TrainingError,
    leaderboard,
    train_model,
    tune_hyperparameters,
)
from ml_workbench.services.workspace import Workspace
from ml_workbench.state import ModelRun, ProjectState
from ml_workbench.ui.theme import card

QUEUE_KEY = "training_queue"


def render_training_tab(state: ProjectState, workspace: Workspace) -> None:
    queue = list(st.session_state.get(QUEUE_KEY, []))
    if queue:
        with card("Queued jobs"):
            rows = [
                {"run_id": entry["run_id"], "model": entry["model_id"], "mode": entry["mode"]}
                for entry in queue
            ]
            st.table(rows)
            if st.button(f"Run {len(queue)} queued job(s)", key="train_run"):
                remaining: list[dict[str, Any]] = []
                for entry in queue:
                    run_id = str(entry["run_id"])
                    try:
                        if entry["mode"] == "auto_tuning":
                            tuned = tune_hyperparameters(
                                state, workspace, str(entry["model_id"]), num_trials=4
                            )
                            tri_result = train_model(
                                state,
                                workspace,
                                str(entry["model_id"]),
                                tuned.best_params,
                                run_id=run_id,
                                mode="auto_tuning",
                            )
                            st.success(
                                f"{tri_result.model_id} trained with tuned params "
                                f"({tuned.primary}={tri_result.mean:.4f})."
                            )
                        else:
                            result = train_model(
                                state,
                                workspace,
                                str(entry["model_id"]),
                                dict(entry["params"] or {}),
                                run_id=run_id,
                                mode=str(entry["mode"]),
                            )
                            st.success(
                                f"{result.model_id} ({result.primary}={result.mean:.4f}, "
                                f"gap={result.gap:.4f})."
                            )
                    except TrainingError as exc:
                        st.error(f"{entry['model_id']}: {exc}")
                st.session_state[QUEUE_KEY] = remaining
    else:
        st.caption("No training jobs queued. Configure models in the Modelling tab.")
        if state.models:
            st.caption("Trained runs are listed below.")

    with card("Leaderboard (TRAIN-05, PERF-01)"):
        if state.has_trained_model:
            table = leaderboard(workspace, state.models)
            st.dataframe(table, use_container_width=True)
            _render_run_details(state, workspace)
        else:
            st.info("No trained models yet.")


def _render_run_details(state: ProjectState, workspace: Workspace) -> None:
    done = state.trained_models
    run_id = st.selectbox(
        "Run details",
        [run.run_id for run in done],
        format_func=lambda run_id: _format_run(done, run_id),
        key="train_run_select",
    )
    run: ModelRun = next((model for model in done if model.run_id == run_id), done[0])
    payload = workspace.read_run_json(run.run_id, "metrics.json")
    raw_folds = payload.get("folds", [])
    if isinstance(raw_folds, list) and all(isinstance(fold, dict) for fold in raw_folds):
        folds: list[dict[str, Any]] = raw_folds
    else:
        folds = []
    st.caption(
        f"`{run.model_id}` · primary `{payload.get('primary')}` · "
        f"rule `{payload.get('rule_id')}` · {len(folds)} folds"
    )
    if folds:
        st.markdown("##### Per-fold scores")
        st.dataframe(
            _fold_table(folds, str(payload.get("primary"))),
            use_container_width=True,
        )
    meta = workspace.read_run_json(run.run_id, "meta.json")
    st.code(str(meta.get("params", {})), language=None)


def _format_run(done: list[ModelRun], run_id: str) -> str:
    run = next((model for model in done if model.run_id == run_id), None)
    return f"{run.model_id} ({run.run_id})" if run else run_id


def _fold_table(folds: list[dict[str, Any]], primary: str) -> Any:
    import pandas as pd

    rows = [
        {
            "fold": int(fold.get("fold", index)),
            primary: round(float(fold.get(primary, float("nan"))), 4),
            "training_metric": round(float(fold.get("training_metric", float("nan"))), 4),
        }
        for index, fold in enumerate(folds)
    ]
    return pd.DataFrame(rows)
