from __future__ import annotations

from ml_workbench.registry import load_registry
from ml_workbench.rules.modelling import (
    conditional_prompts,
    filtered_models,
    hyperparameter_form,
    incompatible_options,
)


def test_registry_filtered_by_task() -> None:
    registry = load_registry()
    ids = {spec.id for spec in filtered_models(registry, "classification", phase=4)}
    assert "logistic_regression" in ids
    assert "decision_tree" in ids
    assert "random_forest" in ids
    assert "hist_gradient_boosting" in ids
    assert "kmeans" not in ids
    assert "mlp" not in ids
    assert "lightgbm" not in ids, "uninstalled libraries are filtered out (MODEL-01)"

    regression = {spec.id for spec in filtered_models(registry, "regression", phase=4)}
    assert "linear_regression" in regression
    assert "ridge" in regression
    assert "logistic_regression" not in regression


def test_hyperparameter_form_from_schema() -> None:
    registry = load_registry()
    spec = registry["logistic_regression"]
    form = hyperparameter_form(spec, "classification")
    names = {entry["name"] for entry in form}
    assert names == {"C", "penalty", "solver", "class_weight", "max_iter"}
    defaults = {entry["name"]: entry["default"] for entry in form}
    assert defaults["C"] == 1.0
    assert defaults["penalty"] == "l2"
    assert defaults["class_weight"] is None

    overt = {entry["name"]: entry for entry in form}
    assert overt["class_weight"]["options"] == [None, "balanced"]
    assert overt["C"]["log"] is True

    boosting = registry["hist_gradient_boosting"]
    form = hyperparameter_form(boosting, "classification")
    assert "early_stopping_rounds" in {entry["name"] for entry in form}
    assert any(entry.get("visible_when") == {"early_stopping": True} for entry in form)


def test_conditional_prompts_by_capability() -> None:
    registry = load_registry()
    prompts = conditional_prompts(registry["kmeans"], "clustering")
    assert any(item["param"] == "n_clusters" for item in prompts)

    pca = conditional_prompts(registry["pca"], "dimensionality_reduction")
    assert any(item["param"] == "n_components" for item in pca)

    logistic = conditional_prompts(registry["logistic_regression"], "classification")
    assert any(item["condition"] == "imbalanced" for item in logistic)


def test_incompatible_options_not_submittable() -> None:
    registry = load_registry()
    spec = registry["logistic_regression"]
    rejected = incompatible_options(spec, {"C": 1.0, "penalty": "not_an_option"}, "classification")
    assert "penalty" in rejected

    tree = registry["decision_tree"]
    rejected = incompatible_options(tree, {"alpha": 0.5}, "classification")
    assert "alpha" in rejected

    assert incompatible_options(spec, {"C": 1.0, "solver": "lbfgs"}, "classification") == ()
