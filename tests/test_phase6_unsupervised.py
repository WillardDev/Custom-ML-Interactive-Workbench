from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_workbench.services.error_service import ErrorAnalysisError, error_views
from ml_workbench.services.explain_service import explain_model
from ml_workbench.services.outcome_service import build_unsupervised_outcome
from ml_workbench.services.prediction_service import evaluate_test, predict_frame
from ml_workbench.services.training_service import train_model

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"


@pytest.fixture
def clustering_frame() -> pd.DataFrame:
    return pd.read_csv(SAMPLES / "clustering.csv")


@pytest.fixture
def anomaly_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    normal = rng.normal(0, 1, size=(180, 2))
    outliers = rng.normal(8, 1, size=(20, 2))
    frame = pd.DataFrame(np.vstack([normal, outliers]), columns=["x", "y"])
    frame["label"] = 0
    frame.loc[180:, "label"] = 1
    return frame


def _train(state, ws, model_id: str, params: dict) -> str:
    return train_model(state, ws, model_id, params).run_id


def test_clustering_full_flow(prepared_workspace, clustering_frame: pd.DataFrame) -> None:
    state, ws = prepared_workspace(
        clustering_frame,
        learning_type="unsupervised",
        task_type="clustering",
        eval_labels="true_cluster",
    )
    run_id = _train(state, ws, "kmeans", {"n_clusters": 3, "n_init": 10})

    evaluation = evaluate_test(state, ws, run_id)
    assert "adjusted_rand" in evaluation.scores
    assert evaluation.y_pred is not None and evaluation.y_pred.shape == evaluation.y_true.shape

    assignment = predict_frame(state, ws, run_id, clustering_frame).df
    assert "prediction" in assignment.columns

    report = error_views(state, ws, run_id)
    names = {str(view["view"]) for view in report.views}
    assert {
        "silhouette_per_sample",
        "low_silhouette_points",
        "cluster_size_imbalance",
        "stability_warning",
    } <= names
    assert "true" in report.predicted.columns and "prediction" in report.predicted.columns

    explanation = explain_model(state, ws, run_id)
    panels = {panel["panel"] for panel in explanation.panels}
    assert {"centroid_heatmap", "anova", "surrogate_tree", "personas"} <= panels
    assert explanation.method == "surrogate"

    outcome = build_unsupervised_outcome(state, ws, run_id)
    assert any(name.endswith("clusters.csv") for name in outcome.files)
    assert (ws.outcome_dir / "profiles.csv").is_file()
    assert (ws.outcome_dir / "personas.json").is_file()
    profiles = pd.read_csv(ws.outcome_dir / "profiles.csv")
    assert set(profiles["cluster_id"]) == {0, 1, 2}
    card = json.loads((ws.outcome_dir / "model_card.json").read_text())
    assert card["deliverables"][0]["deliverable"] == "refit_pipeline"


def test_dimred_full_flow(prepared_workspace, clustering_frame: pd.DataFrame) -> None:
    state, ws = prepared_workspace(
        clustering_frame,
        learning_type="unsupervised",
        task_type="dimensionality_reduction",
        eval_labels="true_cluster",
    )
    run_id = _train(state, ws, "pca", {"variance_target": 0.9})

    evaluation = evaluate_test(state, ws, run_id)
    assert "explained_variance" in evaluation.scores
    assert 0.0 <= evaluation.scores["explained_variance"] <= 1.0

    projected = predict_frame(state, ws, run_id, clustering_frame).df
    assert "pc1" in projected.columns and "pc2" in projected.columns

    report = error_views(state, ws, run_id)
    names = {str(view["view"]) for view in report.views}
    assert {"reconstruction_error_per_row", "poorly_embedded_points"} <= names

    explanation = explain_model(state, ws, run_id)
    panels = {panel["panel"] for panel in explanation.panels}
    assert {"loadings_table", "biplot_coordinates", "reconstruction_error_per_feature"} <= panels

    build_unsupervised_outcome(state, ws, run_id)
    assert (ws.outcome_dir / "embeddings.csv").is_file()
    summary = json.loads((ws.outcome_dir / "summary.json").read_text())
    assert "mean_reconstruction_error" in summary
    assert summary["n_components"] >= 1


def test_anomaly_labeled_flow(prepared_workspace, anomaly_frame: pd.DataFrame) -> None:
    state, ws = prepared_workspace(
        anomaly_frame,
        learning_type="unsupervised",
        task_type="anomaly_detection",
        eval_labels="label",
    )
    run_id = _train(state, ws, "isolation_forest", {"n_estimators": 100, "contamination": 0.1})

    evaluation = evaluate_test(state, ws, run_id)
    assert "roc_auc" in evaluation.scores
    assert 0.0 <= evaluation.scores["roc_auc"] <= 1.0

    report = error_views(state, ws, run_id)
    names = {str(view["view"]) for view in report.views}
    assert {"score_distribution", "top_flagged_rows", "false_positives_negatives"} <= names
    confusion = next(v for v in report.views if str(v["view"]) == "false_positives_negatives")
    total = confusion["tp"] + confusion["fp"] + confusion["fn"] + confusion["tn"]
    assert total == len(report.predicted)

    explanation = explain_model(state, ws, run_id)
    panels = {panel["panel"] for panel in explanation.panels}
    assert "per_feature_deviation" in panels

    build_unsupervised_outcome(state, ws, run_id)
    assert (ws.outcome_dir / "flagged.csv").is_file()
    flagged = pd.read_csv(ws.outcome_dir / "flagged.csv")
    assert {"score", "flagged"} <= set(flagged.columns)
    distribution = json.loads((ws.outcome_dir / "score_distribution.json").read_text())
    assert distribution["count"] == len(anomaly_frame)
    threshold = json.loads((ws.outcome_dir / "threshold.json").read_text())
    assert threshold["threshold"] == 0.0 and threshold["applied"] is True


def test_clustering_without_labels_raises_no_split(
    prepared_workspace, clustering_frame: pd.DataFrame
) -> None:
    state, ws = prepared_workspace(
        clustering_frame, learning_type="unsupervised", task_type="clustering"
    )
    run_id = _train(state, ws, "kmeans", {"n_clusters": 3})
    assert state.split is not None and state.split.strategy == "none"
    with pytest.raises(ErrorAnalysisError):
        error_views(state, ws, run_id)


def test_outcome_requires_known_run(prepared_workspace, clustering_frame: pd.DataFrame) -> None:
    state, ws = prepared_workspace(
        clustering_frame,
        learning_type="unsupervised",
        task_type="clustering",
        eval_labels="true_cluster",
    )
    from ml_workbench.services.outcome_service import OutcomeError

    with pytest.raises(OutcomeError):
        build_unsupervised_outcome(state, ws, "run_00000000")
