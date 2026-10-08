from __future__ import annotations

import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype

from ml_workbench.rules import (
    choose_split,
    default_holdout_fraction,
    encoding_plan,
    scaler_fits_inside_fold,
    skew,
    target_transform_options,
)
from ml_workbench.services.cleaning_service import model_policy
from ml_workbench.services.preprocessing_service import (
    PreprocessingError,
    PreprocessingOptions,
    run_preprocessing,
)
from ml_workbench.services.workspace import Workspace, WorkspaceError
from ml_workbench.state import ProjectState, TaskDefinition, TaskError
from ml_workbench.ui.theme import card

NO_SCALER = "(no scaling)"


def render_preprocessing_tab(state: ProjectState, workspace: Workspace) -> None:
    frame = state.frame
    task = state.task
    if frame is None or task is None:
        st.caption("Load a dataset and set a task first (Data Insertion).")
        return
    _render_split(state, task)
    _render_encoding(state, frame, task)
    _render_scaling(state, frame, task)
    _render_target_transform(state, task)
    _render_build(state, workspace, task)


def _render_split(state: ProjectState, task: TaskDefinition) -> None:
    with card("Split (SPLIT-01 … SPLIT-09)"):
        rule = choose_split(task)
        test_size = default_holdout_fraction()
        if rule.strategy != "none":
            test_size = st.slider(
                "Holdout fraction (test size)",
                0.05,
                0.5,
                float(test_size),
                0.05,
                key="prep_test_size",
            )
        st.caption(f"Strategy: {rule.strategy} — {rule.message}")
        st.markdown(f"`split_strategy = {rule.strategy}`")
        st.markdown(f"`holdout_fraction = {test_size}`")


def _render_encoding(state: ProjectState, frame: pd.DataFrame, task: TaskDefinition) -> None:
    with card("Encoding (ENC-01 … ENC-05)"):
        _, family = model_policy(state)
        plan = encoding_plan(family)
        features = [
            column
            for column in frame.columns
            if not is_numeric_dtype(frame[column]) and column != task.target
        ]
        st.caption(f"Encoder by model family '{family}': default {plan.default} ({plan.note})")
        if not features:
            st.caption("No categorical features detected; encoding is a no-op.")
            return
        options = ["ordinal", "one_hot"]
        if plan.default in options:
            options = [plan.default, *[item for item in options if item != plan.default]]
        encoder = st.selectbox("Categorical encoder", options, key="prep_encoder")
        st.markdown(f"`encoder = {encoder}`")


def _render_scaling(state: ProjectState, frame: pd.DataFrame, task: TaskDefinition) -> None:
    with card("Scaling (SCALE-01 … SCALE-04)"):
        handles_nan, family = model_policy(state)
        allowed = scaler_fits_inside_fold()
        st.caption(f"Model family '{family}' needs_scaling={not handles_nan or family == 'neural'}")
        scaler = st.selectbox(
            "Scaler",
            ["standard", "robust", "minmax", NO_SCALER],
            key="prep_scaler",
        )
        st.markdown(f"`scaler = {scaler}`")
        if not allowed:
            st.warning("Fitted steps must stay inside the pipeline (SCALE-04/PIPE-01).")


def _render_target_transform(state: ProjectState, task: TaskDefinition) -> None:
    with card("Target transform (TT-01)"):
        _, family = model_policy(state)
        if task.task_type != "regression":
            st.caption("Target transforms apply to regression targets only.")
            return
        target_skew = skew(state.frame[task.target]) if state.frame is not None else 0.0
        options = target_transform_options(task.task_type, target_skew, family=family)
        if not options:
            level = f"|skew| = {abs(target_skew):.2f}"
            st.caption(f"Target skew is below the threshold ({level}); no transform offered.")
            return
        chosen = st.selectbox("Target transform", ["none", *options], key="prep_target_transform")
        st.markdown(f"`target_transform = {chosen}`")


def _render_build(state: ProjectState, workspace: Workspace, task: TaskDefinition) -> None:
    with card("Build pipeline (PIPE-01)"):
        if st.button("Build leakage-safe preprocessing pipeline", key="prep_build"):
            frame = state.frame
            if frame is None:
                return
            scaler = st.session_state.get("prep_scaler", "standard")
            encoder = st.session_state.get("prep_encoder", "ordinal")
            target_transform = st.session_state.get("prep_target_transform", None)
            test_size = float(st.session_state.get("prep_test_size", default_holdout_fraction()))
            options = PreprocessingOptions(
                scaler=None if scaler == NO_SCALER else scaler,
                encoder=encoder,
                target_transform=None if target_transform == "none" else target_transform,
                test_size=test_size,
            )
            try:
                result = run_preprocessing(state, workspace, options)
            except (PreprocessingError, TaskError, WorkspaceError) as exc:
                st.error(str(exc))
            else:
                st.success(
                    f"Split {result.strategy} ({result.rule_id}): {result.train_rows} train / "
                    f"{result.test_rows} test rows. Pipeline fitted on the training fold only "
                    f"({result.feature_count} features, encoder={result.encoder}, "
                    f"scaler={result.scaler})."
                )
