from __future__ import annotations

from typing import Any

import streamlit as st

from ml_workbench.registry import CURRENT_PHASE, ModelSpec, available_models, load_registry
from ml_workbench.rules import (
    conditional_prompts as model_prompts,
)
from ml_workbench.rules import (
    hyperparameter_form as model_form,
)
from ml_workbench.rules import (
    incompatible_options,
    small_data_hint,
)
from ml_workbench.rules.task import registry_task_type
from ml_workbench.rules.warnings import warn_message
from ml_workbench.services.training_service import new_run_id
from ml_workbench.services.workspace import Workspace
from ml_workbench.state import ModelRun, ProjectState
from ml_workbench.ui.theme import card

QUEUE_KEY = "training_queue"

MODES = ("auto_compare", "manual", "auto_tuning")


def render_modelling_tab(state: ProjectState, workspace: Workspace) -> None:
    if state.frame is None or state.task is None:
        st.caption("Load a dataset and set a task first (Data Insertion).")
        return
    frame = state.frame
    task = state.task

    hint = small_data_hint(len(frame))
    if hint:
        st.info(hint)

    with card("Model registry (MODEL-01)"):
        registry = load_registry()
        family = registry_task_type(task.task_type)
        specs = available_models(registry, task=family, phase=CURRENT_PHASE)
        if not specs:
            st.info("No models are available for this task at the current phase yet.")
            return
        st.caption(
            "Filtered to phase-budget and library-available models: "
            + ", ".join(f"`{spec.id}`" for spec in specs)
        )

        mode = st.radio("Mode", MODES, key="mod_mode")
        if mode == "auto_compare":
            _render_auto_compare(state, specs, family)
        elif mode == "manual":
            _render_manual(state, specs, family)
        else:
            _render_tuning(state, specs, family)


def _render_auto_compare(state: ProjectState, specs: list[ModelSpec], task_type: str) -> None:
    with card("Auto-compare (MODEL-02)"):
        ids = [spec.id for spec in specs]
        selected = st.multiselect(
            "Models to compare", ids, default=[ids[0]], key="mod_compare_models"
        )
        st.caption(
            "Queues each selected model with its registry defaults; the Training tab runs them."
        )
        if st.button("Queue comparison", key="mod_queue_all"):
            for model_id in selected:
                _enqueue(
                    state,
                    model_id,
                    _default_params(_spec(model_id), task_type),
                    mode="auto_compare",
                )
            st.success(f"Queued {len(selected)} model(s). Open the Training tab and run them.")


def _render_manual(state: ProjectState, specs: list[ModelSpec], task_type: str) -> None:
    with card("Manual configuration (MODEL-02 … MODEL-04)"):
        ids = [spec.id for spec in specs]
        model_id = st.selectbox("Model", ids, key="mod_model")
        spec = _spec(model_id)
        st.caption(f"Family `{spec.family}` · flags: {spec.flags}")
        warning = warn_message(
            {**spec.flags, "family": spec.family},
            state.task.task_type if state.task else "",
            state.task.learning_type if state.task else "",
            len(state.frame) if state.frame is not None else 0,
            spec.family,
        )
        if warning and state.task is not None and state.task.learning_type == "supervised":
            st.warning(warning)

        prompts = model_prompts(spec, task_type)
        if prompts:
            st.caption(
                "Conditional prompts (MODEL-03): " + "; ".join(item["prompt"] for item in prompts)
            )

        form = model_form(spec, task_type)
        st.markdown("#### Hyperparameters")
        for hyperparameter in form:
            _render_hyperparameter(spec.id, hyperparameter)
        if st.button("Queue manual run", key="mod_queue_manual"):
            params = _read_params(form, spec.id)
            rejected = incompatible_options(spec, params, task_type)
            if rejected:
                st.error(f"MODEL-04: incompatible options not applied: {', '.join(rejected)}")
            else:
                _enqueue(state, spec.id, params, mode="manual")
                st.success("Queued. Open the Training tab and run it.")


def _render_tuning(state: ProjectState, specs: list[ModelSpec], task_type: str) -> None:
    with card("Auto with tuning (TRAIN-04)"):
        ids = [spec.id for spec in specs]
        model_id = st.selectbox("Model to tune", ids, key="mod_tune_model")
        st.caption("Random search with pruning over the registry hyperparameter space.")
        if st.button("Queue tuning", key="mod_queue_tune"):
            _enqueue(state, model_id, {}, mode="auto_tuning")
            st.success("Tuning queued. Open the Training tab and run it.")


def _spec(model_id: str) -> ModelSpec:
    return load_registry()[model_id]


def _default_params(spec: ModelSpec, task_type: str) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for hyperparameter in model_form(spec, task_type):
        params[str(hyperparameter["name"])] = hyperparameter.get("default")
    return params


def _render_hyperparameter(model_id: str, hyperparameter: dict[str, Any]) -> None:
    name = str(hyperparameter["name"])
    kind = str(hyperparameter["type"])
    default = hyperparameter.get("default")
    key = f"hp_{model_id}_{name}"
    label = name.replace("_", " ").title()
    options = hyperparameter.get("options")
    if kind == "enum" and options is not None:
        st.selectbox(
            label,
            list(options),
            index=list(options).index(default) if default in options else 0,
            key=key,
        )
    elif kind == "bool":
        st.checkbox(label, bool(default), key=key)
    elif kind in {"int", "float"}:
        kwargs: dict[str, Any] = {
            "min_value": hyperparameter.get("min"),
            "max_value": hyperparameter.get("max"),
        }
        component = st.number_input if kind == "float" else st.number_input
        value = float(default) if default is not None else hyperparameter.get("min", 0)
        if kind == "int":
            kwargs["step"] = 1
            component(label, int(value), **{**kwargs, "step": 1, "key": key})
        else:
            component(label, value=value, **{**kwargs, "key": key})
    elif kind == "nullable_int":
        col_left, col_right = st.columns(2)
        with col_left:
            st.checkbox(
                f"{label}: None (auto)", True if default is None else False, key=f"{key}_none"
            )
        with col_right:
            fallback = int(default) if default is not None else int(hyperparameter.get("min", 0))
            st.number_input(
                label,
                min_value=hyperparameter.get("min"),
                max_value=hyperparameter.get("max"),
                value=fallback,
                step=1,
                key=key,
            )
    else:
        st.number_input(label, value=float(default) if default is not None else 0.0, key=key)


def _read_params(form: tuple[dict[str, Any], ...], model_id: str) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for hyperparameter in form:
        name = str(hyperparameter["name"])
        key = f"hp_{model_id}_{name}"
        kind = str(hyperparameter["type"])
        if kind == "nullable_int":
            params[name] = (
                None
                if st.session_state.get(f"{key}_none", False)
                else int(st.session_state.get(key, 0))
            )
        elif kind == "bool":
            params[name] = bool(st.session_state.get(key, False))
        elif kind in {"int", "float"}:
            params[name] = st.session_state.get(key, hyperparameter.get("default"))
        else:
            params[name] = st.session_state.get(key, hyperparameter.get("default"))
    return params


def _enqueue(state: ProjectState, model_id: str, params: dict[str, Any], *, mode: str) -> None:
    run_id = new_run_id()
    queue = list(st.session_state.get(QUEUE_KEY, []))
    queue.append({"run_id": run_id, "model_id": model_id, "params": params, "mode": mode})
    st.session_state[QUEUE_KEY] = queue
    state.models.append(ModelRun(run_id=run_id, model_id=model_id, status="queued"))
    st.session_state["state"] = state
