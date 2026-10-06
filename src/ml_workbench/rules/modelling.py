from __future__ import annotations

from typing import Any

from ml_workbench.registry import ModelSpec, available_models
from ml_workbench.rules.task import registry_task_type

SMALL_DATA_ROWS: int = 50_000  # HINT-01
MINORITY_FRACTION_THRESHOLD: float = 0.2  # METRIC-02


def filtered_models(
    models: dict[str, ModelSpec],
    task_type: str,
    *,
    phase: int,
    registry: dict[str, ModelSpec] | None = None,
) -> list[ModelSpec]:
    """MODEL-01: only models compatible with the task and the current phase are offered."""
    if registry is not None:
        models = registry
    return available_models(models, task=registry_task_type(task_type), phase=phase)


def _visible(hyperparameter: dict[str, Any], spec: ModelSpec, task_type: str) -> bool:
    condition = hyperparameter.get("visible_when")
    if not condition:
        return True
    task_family = registry_task_type(task_type)
    if "task" in condition and condition["task"] != task_family:
        return False
    for flag_name, expected in condition.items():
        if flag_name == "task":
            continue
        if spec.flags.get(flag_name) != expected:
            return False
    return True


def hyperparameter_form(spec: ModelSpec, task_type: str) -> tuple[dict[str, Any], ...]:
    """MODEL-02: the hyperparameter form is generated from the registry schema."""
    family = registry_task_type(task_type)
    return tuple(
        hyperparameter
        for hyperparameter in spec.hyperparameters
        if _visible(hyperparameter, spec, family)
    )


def conditional_prompts(spec: ModelSpec, task_type: str) -> tuple[dict[str, str], ...]:
    """MODEL-03: conditional prompts triggered by model capabilities."""
    family = registry_task_type(task_type)
    prompts: list[dict[str, str]] = []
    if spec.flags.get("needs_k"):
        prompts.append(
            {"condition": "needs_k", "param": "n_clusters", "prompt": "ask for cluster count"}
        )
    if spec.flags.get("has_transform"):
        prompts.append(
            {
                "condition": "has_transform",
                "param": "n_components",
                "prompt": "PCA asks for components",
            }
        )
    if family == "anomaly_detection" and not spec.flags.get("handles_nan"):
        prompts.append(
            {"condition": "anomaly", "param": "contamination", "prompt": "ask for contamination"}
        )
    names = [entry.get("name") for entry in spec.hyperparameters]
    if family == "classification" and "class_weight" in names:
        prompts.append(
            {"condition": "imbalanced", "param": "class_weight", "prompt": "offer class weights"}
        )
    return tuple(prompts)


def incompatible_options(
    spec: ModelSpec, params: dict[str, Any], task_type: str
) -> tuple[str, ...]:
    """MODEL-04: options the form must not silently apply — off-schema or hidden ones."""
    form = hyperparameter_form(spec, task_type)
    allowed = {entry["name"] for entry in form}
    rejected: list[str] = []
    for name in params:
        if name not in allowed:
            rejected.append(name)
            continue
        entry = next(item for item in form if item["name"] == name)
        options = entry.get("options")
        value = params[name]
        if options is not None and value is not None and value not in options:
            rejected.append(name)
    return tuple(sorted(rejected))


def small_data_hint(row_count: int) -> str | None:
    """HINT-01: small tabular data → recommend a boosting baseline over a neural net."""
    if row_count < SMALL_DATA_ROWS:
        return (
            f"{row_count} rows is small for tabular learning "
            "(< ~10–50k): a boosting baseline usually matches or beats a neural "
            "network with far less tuning — add one boosted baseline per comparison."
        )
    return None
