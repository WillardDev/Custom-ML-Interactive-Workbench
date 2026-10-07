from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence, permutation_importance

from ml_workbench.registry import ModelSpec, load_registry
from ml_workbench.rules.explainability import (
    auto_explain_warnings,
    kernel_explanation,
    linear_explanation,
    tree_explanation,
)
from ml_workbench.rules.performance import disk_cache_key
from ml_workbench.services.model_cache import ModelCache
from ml_workbench.services.prediction_service import get_pipeline
from ml_workbench.services.workspace import Workspace, utc_now
from ml_workbench.state import ProjectState


class ExplainError(ValueError):
    pass


ProgressUpdate = Callable[[float, str], None]
BACKGROUND_MAX: int = 100
PDP_MAX_FEATURES: int = 2
ICE_MAX_ROWS: int = 20


@dataclass(frozen=True)
class ExplanationReport:
    model_id: str
    method: str
    background_rows: int
    panels: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]
    cached: bool
    cache_file: str


def shap_available() -> bool:
    return importlib.util.find_spec("shap") is not None


def explain_model(
    state: ProjectState,
    workspace: Workspace,
    run_id: str,
    *,
    cache: ModelCache | None = None,
    on_progress: ProgressUpdate | None = None,
    force_recompute: bool = False,
) -> ExplanationReport:
    """EXPL-01/02/03/05/10/11: dispatch explanations, writing a disk cache."""
    if state.frame is None or state.task is None or state.task.target is None:
        raise ExplainError("load a dataset and set a task first")
    run = next((model for model in state.models if model.run_id == run_id), None)
    if run is None:
        raise ExplainError(f"unknown run '{run_id}' — train a model first")
    spec = load_registry().get(run.model_id)
    if spec is None:
        raise ExplainError(f"unknown model '{run.model_id}'")
    meta = workspace.read_run_json(run_id, "meta.json")
    params = dict(meta.get("params", {}) or {})
    data_hash = str(meta.get("data_hash", state.dataset.data_hash if state.dataset else ""))
    cache_path = workspace.explain_dir / f"{disk_cache_key(data_hash, spec.id, params)}.json"
    if cache_path.is_file() and not force_recompute:
        payload = json.loads(cache_path.read_text())
        return ExplanationReport(
            model_id=spec.id,
            method=str(payload["method"]),
            background_rows=int(payload["background_rows"]),
            panels=tuple(payload["panels"]),
            warnings=tuple(payload["warnings"]),
            cached=True,
            cache_file=str(cache_path),
        )

    if on_progress:
        on_progress(0.05, "sampling background rows")
    frame, target = state.frame, state.task.target
    features = [column for column in frame.columns if column != target]
    x = frame[features]
    if len(x) > BACKGROUND_MAX:
        x = x.sample(n=BACKGROUND_MAX, random_state=state.seed)
    background_rows = len(x)
    y = frame[target].iloc[x.index]
    pipeline = get_pipeline(workspace, run_id, cache)

    if on_progress:
        on_progress(0.2, f"running {spec.flags.get('explain_method')} explanation")
    method = _method_for(spec, x, y, pipeline, on_progress)

    if on_progress:
        on_progress(0.85, "building PDP / ICE probes")
    panels, warnings = _explanations(spec, pipeline, features, x, y, method, background_rows, state)

    workspace.ensure()
    payload = {
        "model_id": spec.id,
        "method": method,
        "background_rows": background_rows,
        "panels": panels,
        "warnings": warnings,
        "computed_at": utc_now(),
    }
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if on_progress:
        on_progress(1.0, "done")
    return ExplanationReport(
        model_id=spec.id,
        method=method,
        background_rows=background_rows,
        panels=tuple(panels),
        warnings=tuple(warnings),
        cached=False,
        cache_file=str(cache_path),
    )


def _method_for(
    spec: ModelSpec,
    x: pd.DataFrame,
    y: pd.Series,
    pipeline: Any,
    on_progress: ProgressUpdate | None,
) -> str:
    desired = str(spec.flags.get("explain_method"))
    if desired == "linear":
        return "linear"
    if desired == "tree_shap":
        return "tree_shap" if shap_available() else "tree_importances"
    if desired == "kernel_shap":
        return "kernel_shap" if shap_available() else "permutation_importance"
    return "permutation_importance" if desired in {"gradient", "surrogate", "loadings"} else desired


def _explanations(
    spec: ModelSpec,
    pipeline: Any,
    feature_names: list[str],
    x: pd.DataFrame,
    y: pd.Series,
    method: str,
    background_rows: int,
    state: ProjectState,
) -> tuple[list[dict[str, Any]], list[str]]:
    panels: list[dict[str, Any]] = []
    names = _feature_names(pipeline) or feature_names
    model_step = pipeline.named_steps.get("model")
    binary = state.task is not None and state.task.task_type in {"binary", "multilabel"}
    importance_spread = False

    if method == "linear":
        coefs = np.ravel(np.asarray(getattr(model_step, "coef_", [])))
        if len(coefs) != len(names):
            coefs = np.resize(coefs, len(names))
        panels.append(
            {"panel": "coefficients", **linear_explanation(coefs.tolist(), names, binary=binary)}
        )
        local = _linear_local(pipeline, names, coefs, x)
        if local:
            panels.append({"panel": "local_attribution", **local})
    elif method in {"tree_shap", "tree_importances"}:
        importances = np.asarray(getattr(model_step, "feature_importances_", []))
        importance_spread = _importance_spread(importances)
        panels.append(
            {
                "panel": "importances",
                "method": method,
                "table": [
                    {"feature": feature, "importance": float(importance)}
                    for feature, importance in zip(names, importances, strict=False)
                ],
                **tree_explanation(shap_available=method == "tree_shap"),
            }
        )
    elif method in {"kernel_shap", "permutation_importance"}:
        rows = permutation_importance(pipeline, x, y, n_repeats=3, random_state=state.seed)
        importance_spread = _importance_spread(rows.importances_mean)
        panels.append(
            {
                "panel": "permutation_importance",
                "rows": [
                    {"feature": feature, "mean": float(v), "std": float(s)}
                    for feature, v, s in zip(
                        names, rows.importances_mean, rows.importances_std, strict=False
                    )
                ],
                **kernel_explanation(background_rows, shap_available=method == "kernel_shap"),
            }
        )

    pdp = _pdp_ice(pipeline, names, x)
    if pdp:
        panels.append({"panel": "pdp_ice", **pdp})

    warnings = list(
        auto_explain_warnings(
            importance_spread=importance_spread,
            on_sample=background_rows < len(y),
        )
    )
    return panels, warnings


def _feature_names(pipeline: Any) -> list[str]:
    preprocess = pipeline.named_steps.get("preprocess")
    getter = getattr(preprocess, "get_feature_names_out", None)
    if getter is None:
        return []
    try:
        return [str(name) for name in getter()]
    except (AttributeError, ValueError):
        return []


def _linear_local(
    pipeline: Any, names: list[str], coefs: np.ndarray, x: pd.DataFrame
) -> dict[str, Any] | None:
    preprocess = pipeline.named_steps.get("preprocess")
    if preprocess is None or not hasattr(preprocess, "transform"):
        return None
    transformed = np.asarray(preprocess.transform(x.iloc[[0]])).ravel()
    if len(transformed) == 0:
        return None
    width = min(len(names), len(transformed))
    contributions = coefs[:width] * transformed[:width]
    return {
        "row_index": 0,
        "attribution": [
            {"feature": feature, "contribution": float(c)}
            for feature, c in zip(names[:width], contributions, strict=False)
        ],
    }


def _pdp_ice(pipeline: Any, names: list[str], x: pd.DataFrame) -> dict[str, Any] | None:
    if len(names) < 2:
        return None
    x_casts = x.copy()
    for column in x_casts.columns:
        if pd.api.types.is_numeric_dtype(x_casts[column]):
            x_casts[column] = x_casts[column].astype(float)
    average: list[dict[str, Any]] = []
    ice_curves: list[dict[str, Any]] = []
    for idx in range(min(PDP_MAX_FEATURES, len(names))):
        try:
            global_pdp = partial_dependence(pipeline, x_casts, features=[idx], kind="average")
            grid_values = first_squeeze(np.asarray(global_pdp["grid_values"][0]))
            if not _numeric_grid(grid_values):
                continue
            average_curve = first_squeeze(np.asarray(global_pdp["average"]))
            ice_pdp = partial_dependence(
                pipeline, x_casts.iloc[:ICE_MAX_ROWS], features=[idx], kind="individual"
            )
            ice_rows = first_squeeze(np.asarray(ice_pdp["individual"]))
            average_grid = [float(value) for value in np.ravel(grid_values)]
            average.append(
                {
                    "feature": names[idx],
                    "grid": average_grid,
                    "average": [float(value) for value in np.ravel(average_curve)],
                }
            )
            ice_curves.append(
                {
                    "feature": names[idx],
                    "curves": [[float(value) for value in np.ravel(row)] for row in ice_rows],
                }
            )
        except (TypeError, ValueError):
            continue
    if not average:
        return None
    return {"average": average, "ice": ice_curves}


def first_squeeze(array: np.ndarray) -> np.ndarray:
    while array.ndim > 1 and array.shape[0] == 1:
        array = array[0]
    return array


def _numeric_grid(grid: np.ndarray) -> bool:
    try:
        floats = tuple(float(value) for value in grid)
    except (TypeError, ValueError):
        return False
    return len(floats) > 0


def _importance_spread(importances: np.ndarray) -> bool:
    if importances is None or len(importances) < 2 or float(np.sum(importances)) <= 0:
        return False
    ordered = np.sort(np.abs(importances))[::-1]
    return bool(ordered[1] / ordered[0] > 0.85 if ordered[0] > 0 else False)
