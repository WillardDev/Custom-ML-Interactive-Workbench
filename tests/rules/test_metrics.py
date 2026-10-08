from __future__ import annotations

import numpy as np
import pytest

from ml_workbench.rules.association import association_rule_stats, filter_rules
from ml_workbench.rules.metrics import (
    anomaly_labeled_scores,
    forecast_scores,
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


def test_metrics_time_series() -> None:
    plan = metric_plan("forecasting")
    assert plan.primary == "mae"
    assert set(plan.metrics) == {"mae", "rmse", "smape", "mase"}
    assert plan.rule_id == "METRIC-04"

    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 4.0, 3.0])
    scores = forecast_scores(y_true, y_pred)
    assert scores["mae"] == pytest.approx(0.5)
    assert scores["rmse"] == pytest.approx(np.sqrt(0.5))
    assert scores["smape"] > 0.0
    # MASE scales by the mean |y_t - y_{t-1}| of the holdout (= 1.0 here).
    assert scores["mase"] == pytest.approx(0.5)

    zeros = forecast_scores(np.zeros(4), np.zeros(4))
    assert np.isnan(zeros["mape"]) and np.isnan(zeros["mase"])


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


def test_metrics_association_filters() -> None:
    plan = metric_plan("association")
    assert plan.primary == "lift"
    assert set(plan.metrics) == {"support", "confidence", "lift"}
    assert plan.rule_id == "METRIC-08"

    rules = [
        {
            "antecedent": ("a",),
            "consequent": ("b",),
            "support": 0.4,
            "confidence": 0.8,
            "lift": 1.6,
        },
        {
            "antecedent": ("b",),
            "consequent": ("c",),
            "support": 0.1,
            "confidence": 0.2,
            "lift": 0.8,
        },
    ]
    strong = filter_rules(rules, min_support=0.2, min_confidence=0.5, min_lift=1.0)
    assert len(strong) == 1 and strong[0]["antecedent"] == ("a",)
    assert filter_rules(rules, min_lift=2.0) == []

    stats = association_rule_stats(rules)
    assert stats["support"] == pytest.approx(0.25)
    assert stats["confidence"] == pytest.approx(0.5)
    assert stats["lift"] == pytest.approx(1.2)
    assert association_rule_stats([]) == {"support": 0.0, "confidence": 0.0, "lift": 0.0}
