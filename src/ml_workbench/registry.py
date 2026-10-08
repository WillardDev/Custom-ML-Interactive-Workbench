from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CURRENT_PHASE: int = 8

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = REPO_ROOT / "registry" / "models.yaml"

REQUIRED_FLAGS = frozenset(
    {
        "needs_scaling",
        "handles_nan",
        "needs_k",
        "has_predict",
        "has_transform",
        "has_proba",
        "early_stopping",
        "uses_gpu",
        "viz_only",
        "explain_method",
    }
)
EXPLAIN_METHODS = frozenset(
    {"linear", "tree_shap", "kernel_shap", "gradient", "surrogate", "loadings", "forecast", "rules"}
)
FAMILIES = frozenset(
    {
        "linear",
        "tree",
        "kernel",
        "distance",
        "centroid",
        "density",
        "hierarchical",
        "neural",
        "forecast",
        "association",
    }
)


class RegistryError(ValueError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    library: str
    tasks: tuple[str, ...]
    family: str
    enabled_phase: int
    flags: dict[str, Any]
    hyperparameters: tuple[dict[str, Any], ...]


def load_registry(path: Path | None = None) -> dict[str, ModelSpec]:
    registry_path = path or DEFAULT_REGISTRY_PATH
    raw = yaml.safe_load(registry_path.read_text())
    if not isinstance(raw, dict) or "models" not in raw:
        raise RegistryError(f"{registry_path}: expected a mapping with a 'models' list")
    models: dict[str, ModelSpec] = {}
    for entry in raw["models"]:
        spec = _parse_entry(entry, registry_path)
        if spec.id in models:
            raise RegistryError(f"{registry_path}: duplicate model id '{spec.id}'")
        models[spec.id] = spec
    return models


def enabled_models(
    models: dict[str, ModelSpec],
    task: str | None = None,
    phase: int = CURRENT_PHASE,
) -> list[ModelSpec]:
    selected = [
        spec
        for spec in models.values()
        if spec.enabled_phase <= phase and (task is None or task in spec.tasks)
    ]
    return sorted(selected, key=lambda spec: (spec.enabled_phase, spec.id))


def library_available(library: str) -> bool:
    """True when the model's top-level library can be imported (e.g. xgboost, torch)."""
    package = library.split(".")[0]
    return importlib.util.find_spec(package) is not None


def available_models(
    models: dict[str, ModelSpec],
    task: str | None = None,
    phase: int = CURRENT_PHASE,
) -> list[ModelSpec]:
    """MODEL-01: enabled-by-phase models whose runtime library is actually importable."""
    return [
        spec
        for spec in enabled_models(models, task=task, phase=phase)
        if library_available(spec.library)
    ]


def _parse_entry(entry: Any, path: Path) -> ModelSpec:
    if not isinstance(entry, dict):
        raise RegistryError(f"{path}: model entry must be a mapping, got {type(entry).__name__}")
    for key in ("id", "name", "library", "tasks", "family", "enabled_phase", "flags"):
        if key not in entry:
            raise RegistryError(f"{path}: model entry missing required key '{key}'")
    flags = entry["flags"]
    if not isinstance(flags, dict):
        raise RegistryError(f"{path}: '{entry['id']}.flags' must be a mapping")
    missing = REQUIRED_FLAGS - flags.keys()
    if missing:
        raise RegistryError(f"{path}: '{entry['id']}' missing flags: {sorted(missing)}")
    if flags["explain_method"] not in EXPLAIN_METHODS:
        raise RegistryError(
            f"{path}: '{entry['id']}' has invalid explain_method '{flags['explain_method']}'"
        )
    if entry["family"] not in FAMILIES:
        raise RegistryError(f"{path}: '{entry['id']}' has invalid family '{entry['family']}'")
    tasks = entry["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise RegistryError(f"{path}: '{entry['id']}' tasks must be a non-empty list")
    return ModelSpec(
        id=str(entry["id"]),
        name=str(entry["name"]),
        library=str(entry["library"]),
        tasks=tuple(str(task) for task in tasks),
        family=str(entry["family"]),
        enabled_phase=int(entry["enabled_phase"]),
        flags=dict(flags),
        hyperparameters=tuple(entry.get("hyperparameters", ())),
    )
