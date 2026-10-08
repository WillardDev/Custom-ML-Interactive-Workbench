from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.tree import DecisionTreeClassifier

from ml_workbench.registry import ModelSpec, load_registry
from ml_workbench.rules.explainability import (
    GRADIENT_STEPS,
    auto_explain_warnings,
    gradient_explanation,
    kernel_explanation,
    linear_explanation,
    tree_explanation,
)
from ml_workbench.rules.performance import disk_cache_key
from ml_workbench.services.model_cache import ModelCache
from ml_workbench.services.prediction_service import get_pipeline
from ml_workbench.services.training_service import _anomaly_decision
from ml_workbench.services.workspace import Workspace, utc_now
from ml_workbench.state import ProjectState, feature_columns


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
    """EXPL-01/02/03/05/06/07/08/10/11: dispatch explanations, writing a disk cache."""
    if state.frame is None or state.task is None:
        raise ExplainError("load a dataset and set a task first")
    if state.task.learning_type == "supervised" and state.task.target is None:
        raise ExplainError("a supervised task needs a target column")
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
    frame, task = state.frame, state.task
    if task.learning_type == "unsupervised":
        features = list(feature_columns(task, frame))
    else:
        features = [column for column in frame.columns if column != task.target]
    x = frame[features]
    if len(x) > BACKGROUND_MAX:
        x = x.sample(n=BACKGROUND_MAX, random_state=state.seed)
    background_rows = len(x)
    pipeline = get_pipeline(workspace, run_id, cache)

    if task.learning_type == "unsupervised":
        method = _unsupervised_method(spec, task.task_type, pipeline.named_steps["model"])
        panels, warnings = _unsupervised_explanations(
            task.task_type, pipeline, list(x.columns), x, state
        )
    else:
        if spec.family == "forecast":
            # Forecast models have no input features — the history trace is the explanation.
            method = "history_trace"
            panels = _forecast_explanations(state)
            warnings = list(auto_explain_warnings(on_sample=len(x) < len(frame)))
        else:
            if on_progress:
                on_progress(0.2, f"running {spec.flags.get('explain_method')} explanation")
            y = frame[task.target].iloc[x.index]
            method = _method_for(spec, x, y, pipeline, on_progress)
            panels, warnings = _explanations(
                spec, pipeline, list(x.columns), x, y, method, background_rows, state
            )

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
    if desired == "gradient":
        return "integrated_gradients"
    return "permutation_importance" if desired in {"surrogate", "loadings"} else desired


def _unsupervised_method(spec: ModelSpec, task_type: str, model: Any) -> str:
    if task_type == "clustering":
        return "surrogate"
    if task_type == "dimensionality_reduction":
        return "loadings"
    if task_type == "association":
        return "rules"
    desired = str(spec.flags.get("explain_method"))
    if desired == "tree_shap" and shap_available():
        return "tree_shap"
    if hasattr(model, "feature_importances_"):
        return "tree_importances"
    return "deviation"


def _unsupervised_explanations(
    task_type: str,
    pipeline: Any,
    feature_names: list[str],
    x: pd.DataFrame,
    state: ProjectState,
) -> tuple[list[dict[str, Any]], list[str]]:
    preprocess = pipeline.named_steps.get("preprocess")
    x_scaled = (
        np.asarray(preprocess.transform(x))
        if preprocess is not None and hasattr(preprocess, "transform")
        else np.asarray(x)
    )
    if task_type == "clustering":
        panels = _clustering_explanation_panels(pipeline, x, x_scaled, feature_names, state)
    elif task_type == "dimensionality_reduction":
        panels = _dimred_explanation_panels(pipeline, x_scaled, feature_names)
    elif task_type == "association":
        panels = _association_explanation_panels(pipeline.named_steps["model"])
    else:
        panels = _anomaly_explanation_panels(pipeline, x, x_scaled, feature_names)
    warnings = list(auto_explain_warnings(importance_spread=False, on_sample=True))
    return panels, warnings


def _clustering_explanation_panels(
    pipeline: Any,
    x: pd.DataFrame,
    x_scaled: np.ndarray,
    feature_names: list[str],
    state: ProjectState,
) -> list[dict[str, Any]]:
    model = pipeline.named_steps["model"]
    labels = np.asarray(model.predict(x_scaled))
    clusters = sorted(int(value) for value in np.unique(labels))
    frame = x.reset_index(drop=True)
    frame["_cluster_"] = labels
    if len(clusters) > 1:
        centroids = frame.groupby("_cluster_", observed=True)[feature_names].mean(numeric_only=True)
        centroid_rows = [
            {"cluster": cluster_index, **row.round(4).to_dict()}
            for cluster_index, (_, row) in enumerate(centroids.iterrows())
        ]
    else:
        centroid_rows = [
            {
                "cluster": 0,
                **{feature: round(float(frame[feature].mean()), 4) for feature in feature_names},
            }
        ]
    table = pd.DataFrame(centroid_rows)
    anova = _anova_f_ratios(frame, labels, clusters, feature_names)
    personas = _cluster_personas(frame, labels, clusters, feature_names)
    tree = DecisionTreeClassifier(max_depth=4, random_state=state.seed)
    tree.fit(np.asarray(frame[feature_names]), labels)
    surrogate_table = [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in zip(feature_names, tree.feature_importances_, strict=False)
    ]
    return [
        {
            "panel": "centroid_heatmap",
            "table": (table.round(4) if not table.empty else table).to_dict("records"),
            "note": "per-cluster feature means — interpret with care after preprocessing",
        },
        {"panel": "anova", "table": anova},
        {
            "panel": "surrogate_tree",
            "table": surrogate_table,
            **tree_explanation(shap_available=shap_available()),
        },
        {"panel": "personas", "personas": personas},
    ]


def _anova_f_ratios(
    frame: pd.DataFrame, labels: np.ndarray, clusters: list[int], features: list[str]
) -> list[dict[str, Any]]:
    if len(clusters) < 2:
        return [{"feature": feature, "f_ratio": None} for feature in features]
    rows: list[dict[str, Any]] = []
    for feature in features:
        column = frame[feature].to_numpy(dtype=float)
        groups = [column[labels == cluster_index] for cluster_index in clusters]
        if any(len(group) < 2 for group in groups):
            rows.append({"feature": feature, "f_ratio": None})
            continue
        grand = float(column.mean())
        between = sum(len(group) * (float(group.mean()) - grand) ** 2 for group in groups)
        within = sum(float(((group - group.mean()) ** 2).sum()) for group in groups)
        within_df = len(column) - len(clusters)
        if within <= 0 or within_df <= 0:
            rows.append({"feature": feature, "f_ratio": None})
            continue
        rows.append(
            {
                "feature": feature,
                "f_ratio": round((between / (len(clusters) - 1)) / (within / within_df), 4),
            }
        )
    return rows


def _cluster_personas(
    frame: pd.DataFrame, labels: np.ndarray, clusters: list[int], features: list[str]
) -> list[dict[str, Any]]:
    overall = frame[features].mean()
    personas = []
    for cluster_index in clusters:
        members = frame.loc[frame["_cluster_"] == cluster_index, features]
        centroid = members.mean()
        deviations = (centroid - overall).abs().sort_values(ascending=False)
        top = [
            {
                "feature": str(feature),
                "mean": round(float(centroid[feature]), 4),
                "deviation": round(float(centroid[feature] - overall[feature]), 4),
            }
            for feature in deviations.index[:3]
        ]
        personas.append(
            {
                "cluster": int(cluster_index),
                "size": int(len(members)),
                "top_features": top,
            }
        )
    return personas


def _dimred_explanation_panels(
    pipeline: Any, x_scaled: np.ndarray, feature_names: list[str]
) -> list[dict[str, Any]]:
    model = pipeline.named_steps["model"]
    components = np.asarray(getattr(model, "components_", []))
    rows = []
    for row_index in range(components.shape[0]):
        rows.append(
            {
                "pc": f"PC{row_index + 1}",
                **{
                    feature: round(float(value), 4)
                    for feature, value in zip(feature_names, components[row_index], strict=False)
                },
            }
        )
    panels: list[dict[str, Any]] = [
        {"panel": "loadings_table", "table": rows, "note": "component loadings per feature"}
    ]
    explained = getattr(model, "explained_variance_ratio_", [])
    if len(explained):
        panels.append(
            {
                "panel": "variance_explained",
                "explained": [round(float(value), 4) for value in explained],
            }
        )
    biplot: list[dict[str, Any]] = []
    if components.shape[0] >= 2:
        for feature, x_coord, y_coord in zip(
            feature_names, components[0], components[1], strict=False
        ):
            biplot.append(
                {
                    "feature": feature,
                    "x": round(float(x_coord), 4),
                    "y": round(float(y_coord), 4),
                }
            )
    if biplot:
        panels.append({"panel": "biplot_coordinates", "table": biplot})
    if hasattr(model, "inverse_transform"):
        embedding = np.asarray(model.transform(x_scaled))
        reconstructed = np.asarray(model.inverse_transform(embedding))
        per_feature = ((x_scaled - reconstructed) ** 2).mean(axis=0)
        panels.append(
            {
                "panel": "reconstruction_error_per_feature",
                "table": [
                    {"feature": feature, "mean_squared_error": round(float(value), 4)}
                    for feature, value in zip(feature_names, per_feature, strict=False)
                ],
            }
        )
    return panels


def _association_explanation_panels(model: Any) -> list[dict[str, Any]]:
    """EXPL-09: rule network, lift-vs-confidence scatter and a rules table."""
    rules = list(getattr(model, "rules_", []))
    nodes = sorted({item for rule in rules for item in (*rule["antecedent"], *rule["consequent"])})
    edges = [
        {
            "source": " + ".join(rule["antecedent"]),
            "target": " + ".join(rule["consequent"]),
            "lift": rule["lift"],
            "confidence": rule["confidence"],
        }
        for rule in rules[:100]
    ]
    scatter = [
        {
            "confidence": rule["confidence"],
            "lift": rule["lift"],
            "support": rule["support"],
            "rule": " => ".join((" + ".join(rule["antecedent"]), " + ".join(rule["consequent"]))),
        }
        for rule in rules
    ]
    table = [
        {
            "antecedent": " + ".join(rule["antecedent"]),
            "consequent": " + ".join(rule["consequent"]),
            "support": rule["support"],
            "confidence": rule["confidence"],
            "lift": rule["lift"],
        }
        for rule in rules[:20]
    ]
    return [
        {"panel": "rule_network", "nodes": nodes, "edges": edges},
        {"panel": "lift_vs_confidence", "points": scatter},
        {"panel": "rule_table", "table": table},
    ]


def _forecast_explanations(state: ProjectState) -> list[dict[str, Any]]:
    frame, task = state.frame, state.task
    assert frame is not None and task is not None and task.target is not None
    series = pd.to_numeric(frame[task.target], errors="coerce").dropna()
    trace = [
        {"index": str(index), "value": round(float(value), 6)}
        for index, value in series.tail(500).items()
    ]
    return [
        {
            "panel": "history_trace",
            "table": trace,
            "note": (
                "forecast models have no input features — the target history is the explanation"
            ),
        }
    ]


def _anomaly_explanation_panels(
    pipeline: Any, x: pd.DataFrame, x_scaled: np.ndarray, feature_names: list[str]
) -> list[dict[str, Any]]:
    model = pipeline.named_steps["model"]
    panels: list[dict[str, Any]] = []
    importances = np.asarray(getattr(model, "feature_importances_", []))
    if len(importances) == len(feature_names):
        panels.append(
            {
                "panel": "importances",
                "method": "tree_shap" if shap_available() else "tree_importances",
                "table": [
                    {"feature": feature, "importance": float(importance)}
                    for feature, importance in zip(feature_names, importances, strict=False)
                ],
                **tree_explanation(shap_available=shap_available()),
            }
        )
    decision = _anomaly_decision(model, x_scaled)
    flagged = x.loc[decision > 0.0]
    deviation_rows: list[dict[str, Any]] = []
    if len(flagged) > 0:
        overall_mean = x[feature_names].mean()
        overall_std = x[feature_names].std(ddof=0).replace(0.0, 1.0)
        flagged_mean = flagged[feature_names].mean()
        deviations = (flagged_mean - overall_mean) / overall_std
        for feature in deviations.abs().sort_values(ascending=False).index[:10]:
            deviation_rows.append(
                {
                    "feature": feature,
                    "flagged_mean": round(float(flagged_mean[feature]), 4),
                    "base_mean": round(float(overall_mean[feature]), 4),
                    "deviation": round(float(deviations[feature]), 4),
                }
            )
    if deviation_rows:
        panels.append(
            {
                "panel": "per_feature_deviation",
                "table": deviation_rows,
                "note": "feature z-scores of flagged rows relative to the base population",
            }
        )
    return panels


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
    elif method == "integrated_gradients" and hasattr(model_step, "coefs_"):
        gradient = _integrated_gradients(pipeline, x, names)
        if gradient is not None:
            importance_spread = _importance_spread(
                np.asarray([row["mean_abs"] for row in gradient["table"]], dtype=float)
            )
            panels.append(
                {
                    "panel": "gradient_attributions",
                    "table": gradient["table"],
                    **gradient_explanation(),
                }
            )
            panels.append(
                {"panel": "gradient_local", "row_index": 0, "attribution": gradient["local"]}
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


_MLP_ACTIVATIONS = frozenset({"relu", "tanh", "logistic"})


def _activation_prime(kind: str, z: np.ndarray) -> np.ndarray:
    if kind == "relu":
        return (z > 0.0).astype(float)
    if kind == "tanh":
        return 1.0 - np.tanh(z) ** 2
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
    return np.asarray(sigmoid * (1.0 - sigmoid), dtype=float)


def _output_delta(model: Any, logits: np.ndarray) -> np.ndarray:
    """d(target)/d(last pre-activation): predicted-class probability, or the output value."""
    out = str(getattr(model, "out_activation_", "identity"))
    if out == "identity":
        return np.ones_like(logits)
    if out == "logistic":
        # binary: one unit, probability of classes_[1].
        proba = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        proba = proba.ravel()
        sign = np.where(proba >= 0.5, 1.0, -1.0)
        return np.asarray((proba * (1.0 - proba) * sign)[:, None], dtype=float)
    # softmax: gradient of the predicted class probability per row.
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    proba = exp / exp.sum(axis=1, keepdims=True)
    target = np.argmax(proba, axis=1)
    target_p = proba[np.arange(len(proba)), target]
    delta = -proba * target_p[:, None]
    delta[np.arange(len(proba)), target] += target_p
    return np.asarray(delta, dtype=float)


def _mlp_input_gradients(model: Any, xs: np.ndarray) -> np.ndarray:
    """Analytic backprop through an sklearn MLP: d(output)/d(input) per row."""
    coefs = list(model.coefs_)
    intercepts = list(model.intercepts_)
    hidden_kind = str(model.activation)
    hidden_kind = hidden_kind if hidden_kind in _MLP_ACTIVATIONS else "relu"
    hidden_z: list[np.ndarray] = []
    activated = np.asarray(xs, dtype=float)
    for layer, (weights, bias) in enumerate(zip(coefs, intercepts, strict=False)):
        z = activated @ weights + bias
        if layer < len(coefs) - 1:
            hidden_z.append(z)
            if hidden_kind == "relu":
                activated = np.maximum(z, 0.0)
            elif hidden_kind == "tanh":
                activated = np.tanh(z)
            else:
                activated = 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
        else:
            activated = z
    delta = _output_delta(model, activated)
    for layer in range(len(coefs) - 1, 0, -1):
        delta = (delta @ coefs[layer].T) * _activation_prime(hidden_kind, hidden_z[layer - 1])
    return np.asarray(delta @ coefs[0].T, dtype=float)


def _integrated_gradients(
    pipeline: Any, x: pd.DataFrame, names: list[str]
) -> dict[str, Any] | None:
    """EXPL-04: integrated gradients from a zero baseline over analytic MLP gradients."""
    model = pipeline.named_steps.get("model")
    preprocess = pipeline.named_steps.get("preprocess")
    if model is None or preprocess is None or not hasattr(model, "coefs_"):
        return None
    xs = np.asarray(preprocess.transform(x), dtype=float)
    grads = np.zeros_like(xs)
    for step in range(GRADIENT_STEPS):
        alpha = (step + 0.5) / GRADIENT_STEPS
        grads += _mlp_input_gradients(model, alpha * xs)
    attributions = (grads / GRADIENT_STEPS) * xs
    width = min(len(names), attributions.shape[1])
    mean_abs = np.abs(attributions[:, :width]).mean(axis=0)
    order = np.argsort(-mean_abs)
    local_order = np.argsort(-np.abs(attributions[0, :width]))
    return {
        "table": [
            {
                "feature": names[index],
                "mean_abs": round(float(mean_abs[index]), 6),
                "mean": round(float(attributions[:, index].mean()), 6),
            }
            for index in order
        ],
        "local": [
            {
                "feature": names[index],
                "attribution": round(float(attributions[0, index]), 6),
            }
            for index in local_order
        ],
    }


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
