from __future__ import annotations

import json
import math
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    TimeSeriesSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ml_workbench.registry import ModelSpec, load_registry
from ml_workbench.rules.eda import skew
from ml_workbench.rules.metrics import MetricPlan, metric_plan
from ml_workbench.rules.split import cv_strategy
from ml_workbench.rules.staleness import mark_downstream_stale
from ml_workbench.rules.training import leaderboard_stats, log_fields
from ml_workbench.services.preprocessing_service import apply_target_transform, build_pipeline
from ml_workbench.services.workspace import Workspace, utc_now
from ml_workbench.state import ModelRun, ProjectState, StepEntry


class TrainingError(ValueError):
    pass


CLASSIFICATION_TASK_TYPES = frozenset({"binary", "multiclass", "multilabel"})


@dataclass(frozen=True)
class FoldScore:
    fold: int
    train_metric: float
    val_metric: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class TrainingResult:
    run_id: str
    model_id: str
    primary: str
    mean: float
    std: float
    train_metric: float
    val_metric: float
    gap: float
    n_splits: int
    params: dict[str, object]
    model_status: str
    metrics_path: str
    meta_path: str


@dataclass(frozen=True)
class TuningResult:
    model_id: str
    best_params: dict[str, object]
    primary: str
    best_score: float
    trials: int
    pruned: int


def _estimator_factory(spec: ModelSpec, task_type: str) -> Any:
    classifier = task_type in CLASSIFICATION_TASK_TYPES
    if spec.id == "logistic_regression":
        return LogisticRegression
    if spec.id == "linear_regression":
        return LinearRegression
    if spec.id == "ridge":
        return Ridge
    if spec.id == "lasso":
        return Lasso
    if spec.id == "decision_tree":
        return DecisionTreeClassifier if classifier else DecisionTreeRegressor
    if spec.id == "random_forest":
        return RandomForestClassifier if classifier else RandomForestRegressor
    if spec.id == "hist_gradient_boosting":
        return HistGradientBoostingClassifier if classifier else HistGradientBoostingRegressor
    raise TrainingError(f"model '{spec.id}' has no training implementation yet")


def _accepts_random_state(spec: ModelSpec) -> bool:
    return spec.id in {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "hist_gradient_boosting",
    }


def _translate_params(spec: ModelSpec, params: dict[str, object]) -> dict[str, Any]:
    cleaned: dict[str, Any] = dict(params)
    if spec.id == "logistic_regression" and cleaned.get("penalty") is not None:
        # sklearn >= 1.8 replaced `penalty` with `l1_ratio` (penalty is deprecated).
        penalty = cleaned.pop("penalty")
        cleaned["l1_ratio"] = {"l2": 0.0, "l1": 1.0, "elasticnet": 0.5}.get(str(penalty), 0.0)
    if spec.id in {"decision_tree", "random_forest", "hist_gradient_boosting"}:
        if cleaned.get("max_depth") is None:
            cleaned.pop("max_depth", None)
    if spec.id == "hist_gradient_boosting" and "l1_ratio" in cleaned:
        cleaned["l1_ratio"] = float(cleaned["l1_ratio"])
    if spec.id == "hist_gradient_boosting" and cleaned.get("early_stopping_rounds") is not None:
        cleaned["early_stopping"] = True
        cleaned["n_iter_no_change"] = cleaned.pop("early_stopping_rounds")
    return cleaned


def build_estimator(spec: ModelSpec, params: dict[str, object], task_type: str, seed: int) -> Any:
    factory = _estimator_factory(spec, task_type)
    kwargs = _translate_params(spec, params)
    if _accepts_random_state(spec):
        kwargs["random_state"] = seed
    return factory(**kwargs)


def _preprocessing_choices(state: ProjectState) -> dict[str, Any]:
    encoder: str = "ordinal"
    scaler: str | None = "standard"
    target_transform: str | None | object = None
    for entry in reversed(state.steps):
        if entry.tab != "preprocessing":
            continue
        if entry.op == "encode":
            encoder = str(entry.params.get("encoder") or "ordinal")
        elif entry.op == "scale":
            value = entry.params.get("scaler")
            scaler = str(value) if value is not None else None
        elif entry.op == "target_transform":
            value = entry.params.get("transform")
            target_transform = None if value is None else str(value)
    return {"encoder": encoder, "scaler": scaler, "target_transform": target_transform}


def inverse_target(transformer: Any, values: np.ndarray) -> np.ndarray:
    if transformer is None:
        return np.asarray(values)
    if isinstance(transformer, FunctionTransformer):
        return np.asarray(np.expm1(np.asarray(values, dtype=float)))
    return np.ravel(transformer.inverse_transform(np.asarray(values, dtype=float).reshape(-1, 1)))


def _classification_predict(estimator: Any, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    y_pred = np.asarray(estimator.predict(x))
    proba: np.ndarray | None = None
    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(x))
    return y_pred, proba


def _score(
    plan: MetricPlan, task_type: str, y_true: Any, y_pred: np.ndarray, proba: np.ndarray | None
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    scores: dict[str, float] = {}
    if task_type in CLASSIFICATION_TASK_TYPES:
        binary = len(np.unique(y_true)) <= 2
        score_map: dict[str, Callable[[], float]] = {
            "accuracy": lambda: float(accuracy_score(y_true, y_pred)),
            "f1": lambda: float(f1_score(y_true, y_pred, average="binary" if binary else "macro")),
            "macro_f1": lambda: float(f1_score(y_true, y_pred, average="macro")),
            "mcc": lambda: float(matthews_corrcoef(y_true, y_pred)),
            "balanced_accuracy": lambda: float(balanced_accuracy_score(y_true, y_pred)),
        }
        if binary and proba is not None:
            score_map["pr_auc"] = lambda: float(average_precision_score(y_true, proba[:, 1]))
            score_map["roc_auc"] = lambda: float(average_precision_score(y_true, proba[:, 1]))
    elif task_type == "regression":
        score_map = {
            "rmse": lambda: float(math.sqrt(float(mean_squared_error(y_true, y_pred)))),
            "mae": lambda: float(mean_absolute_error(y_true, y_pred)),
            "r2": lambda: float(r2_score(y_true, y_pred)),
        }
    else:
        return {}
    for name in plan.metrics:
        if name in score_map:
            try:
                scores[name] = round(score_map[name](), 6)
            except Exception:
                scores[name] = float("nan")
    return scores


def classification_scores(
    y_true: Any, y_pred: np.ndarray, proba: np.ndarray | None = None
) -> dict[str, float]:
    """Public per-test-set classification scoreboard for the prediction tab."""
    plan = metric_plan("binary")
    return _score(plan, "binary", y_true, y_pred, proba)


def regression_scores(y_true: Any, y_pred: np.ndarray) -> dict[str, float]:
    """Public per-test-set regression scoreboard for the prediction tab."""
    plan = metric_plan("regression")
    return _score(plan, "regression", y_true, y_pred, None)


def _make_splitter(task_type: str, frame: pd.DataFrame, task: Any, n_splits: int, seed: int) -> Any:
    rule = cv_strategy(task)
    if rule.strategy == "group_kfold":
        return GroupKFold(n_splits=n_splits).split(groups=frame[task.group].to_numpy())
    if rule.strategy == "time_series_split":
        return TimeSeriesSplit(n_splits=n_splits).split(frame)
    if rule.strategy == "shuffle_split":
        return ShuffleSplit(n_splits=n_splits, train_size=0.7, random_state=seed).split(frame)
    if rule.strategy == "stratified_kfold":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
            frame, frame[task.target].to_numpy()
        )
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(frame)


def _target_stats(state: ProjectState, task_type: str) -> tuple[float | None, float]:
    frame, task = state.frame, state.task
    assert frame is not None and task is not None and task.target is not None
    target = pd.to_numeric(frame[task.target], errors="coerce").dropna()
    if task_type in CLASSIFICATION_TASK_TYPES:
        counts = frame[task.target].value_counts(dropna=True)
        if counts.empty:
            return None, 0.0
        return float(counts.min() / counts.sum()), 0.0
    return None, float(skew(target)) if len(target) else 0.0


def _cv_scores(
    state: ProjectState,
    spec: ModelSpec,
    estimator_factory: Any,
    plan: MetricPlan,
    choices: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, float]], list[FoldScore]]:
    frame, task = state.frame, state.task
    assert frame is not None and task is not None and task.target is not None
    task_type = task.task_type
    splitter = _make_splitter(task_type, frame, task, n_splits=5, seed=seed)
    fold_scores: list[FoldScore] = []
    raw_metrics: list[dict[str, float]] = []
    for fold, (train_idx, test_idx) in enumerate(splitter):
        train_idx = np.asarray(train_idx)
        test_idx = np.asarray(test_idx)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        fold_encoder = choices.get("encoder") or "ordinal"
        fold_scaler = choices.get("scaler")
        reporter = build_pipeline(frame, train_idx, task, encoder=fold_encoder, scaler=fold_scaler)
        estimator = estimator_factory
        x_train = reporter.pipeline.transform(frame.iloc[train_idx])
        x_test = reporter.pipeline.transform(frame.iloc[test_idx])
        y_train = frame[task.target].iloc[train_idx]
        y_test = frame[task.target].iloc[test_idx]
        target_transform: object = None
        if task_type == "regression" and choices.get("target_transform") is not None:
            y_to_fit, target_transform = apply_target_transform(
                y_train, str(choices["target_transform"])
            )
        else:
            y_to_fit = y_train
        estimator.fit(x_train, np.asarray(y_to_fit))
        y_pred_val, proba_val = _classification_predict(estimator, x_test)
        if task_type == "regression" and target_transform is not None:
            y_pred_val = inverse_target(target_transform, y_pred_val)
        val_metrics = _score(plan, task_type, y_test, y_pred_val, proba_val)
        y_pred_train, proba_train = _classification_predict(estimator, x_train)
        if task_type == "regression" and target_transform is not None:
            y_pred_train = inverse_target(target_transform, y_pred_train)
        train_metrics = _score(plan, task_type, y_train, y_pred_train, proba_train)
        if all(math.isnan(v) for v in val_metrics.values()):
            continue
        fold_scores.append(
            FoldScore(
                fold=fold,
                train_metric=train_metrics.get(plan.primary, float("nan")),
                val_metric=val_metrics.get(plan.primary, float("nan")),
                metrics=val_metrics,
            )
        )
        fold_row = dict(val_metrics)
        fold_row["training_metric"] = train_metrics.get(plan.primary, float("nan"))
        fold_row["fold"] = fold
        raw_metrics.append(fold_row)
    return raw_metrics, fold_scores


def new_run_id() -> str:
    return f"run_{secrets.token_hex(4)}"


def _read_split_indices(workspace: Workspace, state: ProjectState) -> dict[str, np.ndarray]:
    assert state.split is not None and state.split.indices_path is not None
    path = Path(state.split.indices_path)
    if not path.is_file():
        raise TrainingError(f"split indices file not found: {path}")
    payload = json.loads(path.read_text())
    return {
        "train": np.asarray(payload["train"], dtype=int),
        "test": np.asarray(payload["test"], dtype=int),
    }


def _fit_final_artifact(
    state: ProjectState,
    workspace: Workspace,
    spec: ModelSpec,
    estimator: object,
    choices: dict[str, Any],
    run_id: str,
    seed: int,
) -> Path:
    frame, task = state.frame, state.task
    assert frame is not None and task is not None and task.target is not None
    split_indices = _read_split_indices(workspace, state)
    train_idx = split_indices["train"]
    if len(train_idx) == 0:
        train_idx = np.arange(len(frame))
    reporter = build_pipeline(
        frame,
        train_idx,
        task,
        encoder=choices.get("encoder") or "ordinal",
        scaler=choices.get("scaler"),
    )
    target_transform: object = None
    if task.task_type == "regression" and choices.get("target_transform") is not None:
        y_to_fit, target_transform = apply_target_transform(
            frame[task.target].iloc[train_idx], str(choices["target_transform"])
        )
    else:
        y_to_fit = frame[task.target].iloc[train_idx]
    combined = Pipeline([("preprocess", reporter.pipeline), ("model", estimator)])
    combined.fit(frame.iloc[train_idx], np.asarray(y_to_fit))
    run_dir = workspace.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact = run_dir / "pipeline.joblib"
    joblib.dump(combined, artifact)
    if target_transform is not None:
        joblib.dump(target_transform, run_dir / "target_transform.joblib")
    return artifact


def train_model(
    state: ProjectState,
    workspace: Workspace,
    model_id: str,
    params: dict[str, object],
    *,
    run_id: str | None = None,
    mode: str = "manual",
    seed: int | None = None,
) -> TrainingResult:
    if state.frame is None or state.task is None:
        raise TrainingError("load a dataset and set a task")
    if not state.cleaned:
        raise TrainingError("run Data Cleaning first")
    if not state.split_ready:
        raise TrainingError("run Preprocessing first (a split/pipeline is required)")
    spec = load_registry().get(model_id)
    if spec is None:
        raise TrainingError(f"unknown model '{model_id}'")
    run_seed = seed if seed is not None else state.seed
    frame, task = state.frame, state.task
    assert frame is not None and task is not None
    task_type = task.task_type
    choices = _preprocessing_choices(state)
    minority_fraction, target_skew = _target_stats(state, task_type)
    plan = metric_plan(task_type, minority_fraction=minority_fraction, target_skew=target_skew)
    if (
        minority_fraction is not None
        and minority_fraction < 0.2
        and not spec.flags.get("has_proba")
    ):
        raise TrainingError(
            f"imbalanced classification needs a probability-capable model; "
            f"'{spec.id}' has no probabilities"
        )
    estimator = build_estimator(spec, params, task_type, run_seed)
    raw_metrics, fold_scores = _cv_scores(state, spec, estimator, plan, choices, run_seed)
    if not fold_scores:
        raise TrainingError("cross-validation produced no usable folds")
    stats = leaderboard_stats(raw_metrics, plan.primary)
    assert state.dataset is not None
    data_hash = state.dataset.data_hash
    run_id = run_id if run_id is not None else new_run_id()
    artifact = _fit_final_artifact(state, workspace, spec, estimator, choices, run_id, run_seed)
    trained_at = utc_now()
    meta = {
        **_log_meta(state, spec, params, run_seed, plan, data_hash),
        "mode": mode,
        "library_versions": {"sklearn": sklearn.__version__},
        "trained_at": trained_at,
    }
    metrics = {
        "model_id": spec.id,
        "task_type": task_type,
        "primary": plan.primary,
        "rule_id": plan.rule_id,
        "mean": stats["mean"],
        "std": stats["std"],
        "train_metric": stats["train_metric"],
        "val_metric": stats["val_metric"],
        "gap": stats["gap"],
        "params": params,
        "folds": raw_metrics,
        "metrics": plan.metrics,
        "data_hash": data_hash,
    }
    meta_path = workspace.write_run_json(run_id, "meta.json", meta)
    metrics_path = workspace.write_run_json(run_id, "metrics.json", metrics)
    state.models = [model for model in state.models if model.run_id != run_id]
    state.models.append(
        ModelRun(
            run_id=run_id,
            model_id=spec.id,
            status="done",
            artifact_path=str(artifact),
            meta_path=str(meta_path),
            metrics_path=str(metrics_path),
            trained_at=trained_at,
        )
    )
    state.active_model_id = run_id
    mark_downstream_stale(state, "training")
    steps = workspace.read_steps()
    workspace.write_steps(
        steps
        + [
            StepEntry(
                id=max((e.id for e in steps), default=0) + 1,
                tab="training",
                op="train",
                params={
                    "run_id": run_id,
                    "model": spec.id,
                    "primary": plan.primary,
                    "mean": stats["mean"],
                    "gap": stats["gap"],
                    "mode": mode,
                },
                dataset_hash_before=data_hash,
                dataset_hash_after=data_hash,
                created_at=trained_at,
            )
        ]
    )
    state.steps = workspace.read_steps()
    state.job = None
    return TrainingResult(
        run_id=run_id,
        model_id=spec.id,
        primary=plan.primary,
        mean=stats["mean"],
        std=stats["std"],
        train_metric=stats["train_metric"],
        val_metric=stats["val_metric"],
        gap=stats["gap"],
        n_splits=len(fold_scores),
        params=params,
        model_status="done",
        metrics_path=str(metrics_path),
        meta_path=str(meta_path),
    )


def _log_meta(
    state: ProjectState,
    spec: ModelSpec,
    params: dict[str, object],
    seed: int,
    plan: MetricPlan,
    data_hash: str,
) -> dict[str, object]:
    log = log_fields(state, spec, params, data_hash)
    log["primary"] = plan.primary
    log["rule_id"] = plan.rule_id
    return log


def leaderboard(workspace: Workspace, runs: list[ModelRun]) -> pd.DataFrame:
    """PERF-01: leaderboard reads metrics sidecars only — never the fitted pipeline."""
    rows: list[dict[str, object]] = []
    for run in sorted(runs, key=lambda model: model.trained_at or ""):
        if run.status != "done" or run.metrics_path is None:
            continue
        payload = workspace.read_run_json(run.run_id, "metrics.json")
        rows.append(
            {
                "run_id": run.run_id,
                "model": run.model_id,
                "primary": str(payload.get("primary", "")),
                "mean": float(payload.get("mean", float("nan"))),
                "std": float(payload.get("std", float("nan"))),
                "train_metric": float(payload.get("train_metric", float("nan"))),
                "val_metric": float(payload.get("val_metric", float("nan"))),
                "gap": float(payload.get("gap", float("nan"))),
                "rule_id": str(payload.get("rule_id", "")),
            }
        )
    return pd.DataFrame(rows)


def _sample_params(spec: ModelSpec, task_type: str, rng: np.random.Generator) -> dict[str, Any]:
    from ml_workbench.rules.modelling import hyperparameter_form

    params: dict[str, Any] = {}
    for hyperparameter in hyperparameter_form(spec, task_type):
        name = str(hyperparameter["name"])
        kind = str(hyperparameter["type"])
        default = hyperparameter.get("default")
        if kind == "float":
            low = float(hyperparameter["min"])
            high = float(hyperparameter["max"])
            if hyperparameter.get("log"):
                params[name] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
            else:
                params[name] = float(rng.uniform(low, high))
        elif kind == "int":
            params[name] = int(
                rng.integers(int(hyperparameter["min"]), int(hyperparameter["max"]) + 1)
            )
        elif kind == "nullable_int":
            if rng.random() < 0.5:
                params[name] = None
            else:
                params[name] = int(
                    rng.integers(int(hyperparameter["min"]), int(hyperparameter["max"]) + 1)
                )
        elif kind == "bool":
            params[name] = bool(rng.integers(0, 2))
        elif kind == "enum":
            options = [option for option in (hyperparameter.get("options") or [])]
            if not options:
                params[name] = default
            else:
                params[name] = options[int(rng.integers(0, len(options)))]
        elif kind == "str":
            params[name] = default
        else:
            params[name] = default
    return params


def tune_hyperparameters(
    state: ProjectState,
    workspace: Workspace,
    model_id: str,
    *,
    num_trials: int = 6,
    seed: int | None = None,
) -> TuningResult:
    """TRAIN-04: random search with budget-based pruning, logged per TRAIN-01."""
    if state.frame is None or state.task is None or not state.split_ready:
        raise TrainingError("a cleaned dataset with a split is required for tuning")
    spec = load_registry().get(model_id)
    if spec is None:
        raise TrainingError(f"unknown model '{model_id}'")
    task_type = state.task.task_type
    minority_fraction, target_skew = _target_stats(state, task_type)
    plan = metric_plan(task_type, minority_fraction=minority_fraction, target_skew=target_skew)
    rng = np.random.default_rng(seed if seed is not None else state.seed)
    splitter = ShuffleSplit(n_splits=1, train_size=0.75, random_state=int(rng.integers(0, 2**31)))
    split_indices = _read_split_indices(workspace, state)
    pool = split_indices["train"]
    if len(pool) == 0:
        assert state.frame is not None
        pool = np.arange(len(state.frame))
    inner_train, inner_val = next(splitter.split(pool))
    choices = _preprocessing_choices(state)
    best_params: dict[str, object] = {}
    best_score = float("-inf")
    completed: list[float] = []
    pruned = 0
    for _trial in range(max(1, num_trials)):
        params = _sample_params(spec, task_type, rng)
        estimator = build_estimator(spec, params, task_type, int(rng.integers(0, 2**31)))
        if len(completed) >= 3 and _probe_score(
            state, spec, plan, task_type, params, choices, inner_train, inner_val
        ) < float(np.median(completed)):
            pruned += 1
            continue
        try:
            reporter = build_pipeline(
                state.frame,
                np.asarray(inner_train),
                state.task,
                encoder=choices.get("encoder") or "ordinal",
                scaler=choices.get("scaler"),
            )
            target_transform: object = None
            if task_type == "regression" and choices.get("target_transform") is not None:
                y_fit, target_transform = apply_target_transform(
                    state.frame[state.task.target].iloc[inner_train],
                    str(choices["target_transform"]),
                )
            else:
                y_fit = state.frame[state.task.target].iloc[inner_train]
            estimator.fit(
                reporter.pipeline.transform(state.frame.iloc[inner_train]), np.asarray(y_fit)
            )
            y_val, proba_val = _classification_predict(
                estimator, reporter.pipeline.transform(state.frame.iloc[inner_val])
            )
            if task_type == "regression" and target_transform is not None:
                y_val = inverse_target(target_transform, y_val)
            scores = _score(
                plan, task_type, state.frame[state.task.target].iloc[inner_val], y_val, proba_val
            )
            trial_score = scores.get(plan.primary, float("nan"))
            if not math.isnan(trial_score):
                completed.append(trial_score)
                if trial_score > best_score:
                    best_score = trial_score
                    best_params = params
        except Exception:
            continue
    return TuningResult(
        model_id=spec.id,
        best_params=best_params,
        primary=plan.primary,
        best_score=best_score,
        trials=num_trials,
        pruned=pruned,
    )


def _probe_score(
    state: ProjectState,
    spec: ModelSpec,
    plan: MetricPlan,
    task_type: str,
    params: dict[str, Any],
    choices: dict[str, Any],
    inner_train: np.ndarray,
    inner_val: np.ndarray,
) -> float:
    """TRAIN-04: a reduced-budget probe (halved max_iter / n_estimators) estimates a
    trial's region; below-median regions are pruned before a full-budget fit."""
    assert state.frame is not None and state.task is not None and state.task.target is not None
    probe_params = dict(params)
    for budget_param in ("max_iter", "n_estimators"):
        if budget_param in probe_params:
            probe_params[budget_param] = max(20, int(probe_params[budget_param]) // 2)
            break
    probe = build_estimator(spec, probe_params, task_type, seed=state.seed)
    rng = np.random.default_rng(state.seed)
    subset_size = min(len(inner_train), max(50, len(inner_train) // 2))
    subset = rng.choice(len(inner_train), size=subset_size, replace=False)
    fold_train = np.asarray(inner_train)[subset]
    probe_val = np.asarray(inner_train)[
        rng.choice(len(inner_train), size=min(200, len(inner_train)), replace=False)
    ]
    try:
        reporter = build_pipeline(
            state.frame,
            fold_train,
            state.task,
            encoder=choices.get("encoder") or "ordinal",
            scaler=choices.get("scaler"),
        )
        target_transform: object = None
        if task_type == "regression" and choices.get("target_transform") is not None:
            y_fit, target_transform = apply_target_transform(
                state.frame[state.task.target].iloc[fold_train], str(choices["target_transform"])
            )
        else:
            y_fit = state.frame[state.task.target].iloc[fold_train]
        probe.fit(reporter.pipeline.transform(state.frame.iloc[fold_train]), np.asarray(y_fit))
        y_val, proba_val = _classification_predict(
            probe, reporter.pipeline.transform(state.frame.iloc[probe_val])
        )
        if task_type == "regression" and target_transform is not None:
            y_val = inverse_target(target_transform, y_val)
        scores = _score(
            plan, task_type, state.frame[state.task.target].iloc[probe_val], y_val, proba_val
        )
        score = scores.get(plan.primary, float("nan"))
        return score if not math.isnan(score) else float("-inf")
    except Exception:
        return float("-inf")
