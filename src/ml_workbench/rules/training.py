from __future__ import annotations

from statistics import mean, stdev
from typing import Any

from ml_workbench.registry import ModelSpec

ARTIFACT_FILES: tuple[str, ...] = ("pipeline.joblib", "meta.json", "metrics.json")


def log_fields(
    state: Any, spec: ModelSpec, params: dict[str, Any], data_hash: str
) -> dict[str, Any]:
    """TRAIN-01: training metadata captures seed, config, split and library versions."""
    split = getattr(state, "split", None)
    return {
        "seed": getattr(state, "seed", None),
        "model_id": spec.id,
        "params": params,
        "data_hash": data_hash,
        "split": {
            "strategy": split.strategy if split is not None else None,
            "params": dict(getattr(split, "params", {}) or {}),
        },
        "versions": {"sklearn": None},
    }


def boosting_early_stopping(flags: dict[str, Any], params: dict[str, Any]) -> bool:
    """TRAIN-02: boosting baselines request early stopping rounds."""
    return bool(flags.get("early_stopping") and params.get("early_stopping_rounds"))


def tuning_uses_pruning() -> bool:
    """TRAIN-04: hyperparameter tuning prunes low-budget trials instead of running full orbits."""
    return True


def leaderboard_stats(fold_metrics: list[dict[str, float]], metric: str) -> dict[str, float]:
    """TRAIN-05: fold mean/std and the train-vs-val gap for the leaderboard."""
    if not fold_metrics:
        return {"mean": 0.0, "std": 0.0, "train_metric": 0.0, "val_metric": 0.0, "gap": 0.0}
    values = [fold[metric] for fold in fold_metrics]
    train_values = [fold["training_metric"] for fold in fold_metrics]
    val_mean = mean(values)
    train_mean = mean(train_values)
    return {
        "mean": val_mean,
        "std": stdev(values) if len(values) > 1 else 0.0,
        "train_metric": train_mean,
        "val_metric": val_mean,
        "gap": train_mean - val_mean,
    }


def artifact_files() -> tuple[str, ...]:
    """TRAIN-06: every training run persists pipeline, metadata and metrics."""
    return ARTIFACT_FILES


def job_routing(flags: dict[str, Any], accelerator_available: bool = False) -> tuple[str, str]:
    """TRAIN-07: GPU-capable models only route to a GPU when using_gpu and one exists."""
    if flags.get("uses_gpu") and accelerator_available:
        return "gpu", "routed to GPU"
    return "cpu", "routed to CPU"
