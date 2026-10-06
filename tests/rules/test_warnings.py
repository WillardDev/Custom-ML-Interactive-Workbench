from __future__ import annotations

from ml_workbench.rules.warnings import (
    density_needs_surrogate,
    neural_small_data,
    outlier_removal_blocked,
    smote_blocked,
    target_selection_blocked,
    viz_only_blocks_downstream,
    viz_only_blocks_prediction,
)


def test_warn_tsne_with_prediction() -> None:
    assert viz_only_blocks_prediction({"viz_only": True, "family": "manifold"}) is True
    assert viz_only_blocks_prediction({"viz_only": False}) is False


def test_warn_dbscan_new_rows() -> None:
    assert density_needs_surrogate({"family": "density", "has_predict": False}) is True
    assert density_needs_surrogate({"family": "density", "has_predict": True}) is False


def test_warn_smote_regression() -> None:
    assert smote_blocked("regression") is True
    assert smote_blocked("binary") is False


def test_warn_target_selection_unsupervised() -> None:
    assert target_selection_blocked("unsupervised") is True
    assert target_selection_blocked("supervised") is False


def test_warn_outlier_removal_anomaly() -> None:
    assert outlier_removal_blocked("anomaly_detection") is True
    assert outlier_removal_blocked("regression") is False


def test_warn_neural_small_data() -> None:
    assert neural_small_data("neural", 1_000) is True
    assert neural_small_data("neural", 1_000_000) is False
    assert neural_small_data("tree", 500) is False


def test_warn_viz_only_blocks_downstream() -> None:
    assert viz_only_blocks_downstream({"family": "manifold", "viz_only": True}) is True
    assert viz_only_blocks_downstream({"family": "manifold", "viz_only": False}) is False
    assert viz_only_blocks_downstream({"family": "tree", "viz_only": True}) is False
