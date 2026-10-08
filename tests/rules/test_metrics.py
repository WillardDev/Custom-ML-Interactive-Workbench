from __future__ import annotations

import numpy as np
import pytest

from ml_workbench.rules.metrics import (
    anomaly_labeled_scores,
    labeled_cluster_scores,
    metric_plan,
    trustworthiness_coef,
)


def test_metrics_balanced_classification() -> None:
    plan = metric_plan("binary", minority_fraction=0.4)
    assert plan.primary == "accuracy"
    assert set(plan.metrics) == {"accuracy", "f1"}
    assert plan.rule_id == "METRIC-01"


def test_metrics_imbalanced_classification() -> None:
    plan = metric_plan("binary", minority_fraction=0.08)
    assert plan.primary == "pr_auc"
    assert set(plan.metrics) == {"pr_auc", "macro_f1", "mcc", "balanced_accuracy"}
    assert plan.rule_id == "METRIC-02"


def test_metrics_regression_skew_prefers_mae() -> None:
    plan = metric_plan("regression")
    assert plan.primary == "rmse"
    assert plan.rule_id == "METRIC-03"

    skewed = metric_plan("regression", target_skew=3.2)
    assert skewed.primary == "mae"
    assert "rmse" in skewed.metrics and "r2" in skewed.metrics


@pytest.mark.skip(reason="time series metrics arrive in Phase 7 (docs/rules.md METRIC-04)")
def test_metrics_time_series() -> None:
    raise NotImplementedError("docs/rules.md METRIC-04")


def test_metrics_clustering() -> None:
    plan = metric_plan("clustering")
    assert plan.primary == "silhouette"
    assert set(plan.metrics) == {"silhouette", "davies_bouldin", "calinski_harabasz"}
    assert plan.rule_id == "METRIC-05"

    labeled = metric_plan("clustering", labeled=True)
    assert labeled.primary == "adjusted_rand"
    assert set(labeled.metrics) == {"adjusted_rand", "nmi", "silhouette"}

    labels_true = np.array([0, 0, 1, 1, 2, 2])
    labels_pred = np.array([0, 0, 0, 1, 1, 1])
    scores = labeled_cluster_scores(labels_true, labels_pred)
    assert set(scores) == {"adjusted_rand", "nmi"}
    assert -1.0 <= scores["adjusted_rand"] <= 1.0


def test_metrics_dim_reduction() -> None:
    plan = metric_plan("dimensionality_reduction")
    assert plan.primary == "explained_variance"
    assert set(plan.metrics) == {
        "explained_variance",
        "reconstruction_error",
        "trustworthiness",
    }
    assert plan.rule_id == "METRIC-06"

    rng = np.random.default_rng(0)
    high = rng.normal(size=(60, 5))
    low = high[:, :2]
    trustworthy = trustworthiness_coef(high, low, n_neighbors=5)
    assert 0.0 <= trustworthy <= 1.0


def test_metrics_anomaly_labeled_vs_not() -> None:
    plan = metric_plan("anomaly_detection")
    assert plan.primary == "score_distribution"
    assert plan.metrics == ("score_distribution",)
    assert plan.rule_id == "METRIC-07"

    labeled = metric_plan("anomaly_detection", labeled=True)
    assert labeled.primary == "roc_auc"
    assert set(labeled.metrics) == {"roc_auc", "pr_auc"}

    rng = np.random.default_rng(1)
    labels = np.array([0] * 40 + [1] * 10)
    scores = rng.normal(size=50) + labels
    labeled_scores = anomaly_labeled_scores(labels, scores)
    assert 0.0 < labeled_scores["roc_auc"] <= 1.0
    assert 0.0 < labeled_scores["pr_auc"] <= 1.0

    single_class = anomaly_labeled_scores(np.zeros(10), np.zeros(10))
    assert np.isnan(single_class["roc_auc"])


@pytest.mark.skip(reason="association-rule metrics arrive in Phase 7 (docs/rules.md METRIC-08)")
def test_metrics_association_filters() -> None:
    raise NotImplementedError("docs/rules.md METRIC-08")
