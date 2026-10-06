from __future__ import annotations

from pathlib import Path

import pytest

from ml_workbench.registry import (
    DEFAULT_REGISTRY_PATH,
    EXPLAIN_METHODS,
    FAMILIES,
    RegistryError,
    enabled_models,
    load_registry,
)


def test_registry_loads_and_validates() -> None:
    models = load_registry()
    assert len(models) >= 13
    for spec in models.values():
        assert spec.family in FAMILIES
        assert spec.flags["explain_method"] in EXPLAIN_METHODS
        assert spec.tasks
        assert spec.enabled_phase >= 1
    assert "logistic_regression" in models
    assert "lightgbm" in models
    assert models["lightgbm"].flags["handles_nan"] is True
    assert models["logistic_regression"].flags["needs_scaling"] is True


def test_enabled_models_filter_by_task_and_phase() -> None:
    models = load_registry()

    assert enabled_models(models) == []

    classification = {spec.id for spec in enabled_models(models, task="classification", phase=4)}
    assert "logistic_regression" in classification
    assert "kmeans" not in classification
    assert "mlp" not in classification

    clustering = {spec.id for spec in enabled_models(models, task="clustering", phase=6)}
    assert clustering == {"kmeans"}

    phase7 = {spec.id for spec in enabled_models(models, phase=7)}
    assert "mlp" in phase7
    assert "pca" in phase7


def test_registry_rejects_missing_flags(tmp_path: Path) -> None:
    broken = tmp_path / "models.yaml"
    broken.write_text(
        "models:\n"
        "  - id: bad\n"
        "    name: Bad\n"
        "    library: sklearn\n"
        "    tasks: [classification]\n"
        "    family: tree\n"
        "    enabled_phase: 1\n"
        "    flags: {needs_scaling: false}\n"
    )
    with pytest.raises(RegistryError, match="missing flags"):
        load_registry(broken)


def test_registry_rejects_duplicate_ids(tmp_path: Path) -> None:
    entry = (
        "  - id: dup\n"
        "    name: Dup\n"
        "    library: sklearn\n"
        "    tasks: [classification]\n"
        "    family: tree\n"
        "    enabled_phase: 1\n"
        "    flags: {needs_scaling: false, handles_nan: false, needs_k: false,"
        " has_predict: true, has_transform: false, has_proba: true,"
        " early_stopping: false, uses_gpu: false, viz_only: false,"
        " explain_method: tree_shap}\n"
    )
    broken = tmp_path / "models.yaml"
    broken.write_text(f"models:\n{entry}{entry}")
    with pytest.raises(RegistryError, match="duplicate model id"):
        load_registry(broken)


def test_default_registry_path_exists() -> None:
    assert DEFAULT_REGISTRY_PATH.is_file()
