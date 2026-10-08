from __future__ import annotations

from typing import Any

from ml_workbench.rules.training import (
    artifact_files,
    boosting_early_stopping,
    job_routing,
    leaderboard_stats,
    log_fields,
    neural_training_plan,
    tuning_uses_pruning,
)


def _state(**kwargs: Any) -> Any:
    state = type("State", (), {})()
    for key, value in kwargs.items():
        setattr(state, key, value)
    return state


def test_train_logs_seed_config_hash() -> None:
    state = _state(
        seed=7,
        split=type("Split", (), {"strategy": "stratified_kfold", "params": {"n_splits": 5}})(),
    )
    spec = type("Spec", (), {"id": "logistic_regression"})()
    log = log_fields(state, spec, {"C": 1.0}, "sha256:abc")
    assert log["seed"] == 7
    assert log["model_id"] == "logistic_regression"
    assert log["params"] == {"C": 1.0}
    assert log["data_hash"] == "sha256:abc"
    assert log["split"]["strategy"] == "stratified_kfold"
    assert "versions" in log


def test_boosting_early_stopping() -> None:
    assert boosting_early_stopping({"early_stopping": True}, {"early_stopping_rounds": 50}) is True
    assert (
        boosting_early_stopping({"early_stopping": False}, {"early_stopping_rounds": 50}) is False
    )
    assert boosting_early_stopping({"early_stopping": True}, {}) is False


def test_neural_training_loop() -> None:
    plan = neural_training_plan({"batch_size": 32, "epochs": 50, "patience": 5})
    assert plan["early_stopping"] == "validation_loss"
    assert plan["checkpoint"] == "best_validation_loss"
    assert plan["loss_curve"] is True
    assert plan["validation_fraction"] == 0.1
    assert plan["batch_size"] == 32
    assert plan["epochs"] == 50
    assert plan["patience"] == 5
    assert plan["device"] == "cpu"
    defaults = neural_training_plan({})
    assert defaults["batch_size"] == 64
    assert defaults["epochs"] == 100
    assert defaults["patience"] == 10
    gpu = neural_training_plan({}, uses_gpu=True, accelerator_available=True)
    assert gpu["device"] == "gpu"


def test_tuning_uses_pruning() -> None:
    assert tuning_uses_pruning() is True


def test_leaderboard_fold_stats() -> None:
    folds = [
        {"accuracy": 0.80, "training_metric": 0.86},
        {"accuracy": 0.82, "training_metric": 0.88},
        {"accuracy": 0.78, "training_metric": 0.84},
    ]
    stats = leaderboard_stats(folds, "accuracy")
    assert round(stats["mean"], 4) == 0.8
    assert round(stats["train_metric"], 4) == 0.86
    assert round(stats["gap"], 4) == 0.06
    assert stats["std"] > 0


def test_train_persists_artifacts() -> None:
    assert artifact_files() == ("pipeline.joblib", "meta.json", "metrics.json")


def test_job_routing_gpu_vs_cpu() -> None:
    kind, _message = job_routing({"uses_gpu": True}, accelerator_available=True)
    assert kind == "gpu"
    kind, _message = job_routing({"uses_gpu": True}, accelerator_available=False)
    assert kind == "cpu"
    kind, _message = job_routing({"uses_gpu": False}, accelerator_available=True)
    assert kind == "cpu"
