from __future__ import annotations

import pytest

from ml_workbench.rules.metrics import metric_plan


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


@pytest.mark.skip(reason="clustering metrics arrive in Phase 6 (docs/rules.md METRIC-05)")
def test_metrics_clustering() -> None:
    raise NotImplementedError("docs/rules.md METRIC-05")


@pytest.mark.skip(reason="dim reduction metrics arrive in Phase 6 (docs/rules.md METRIC-06)")
def test_metrics_dim_reduction() -> None:
    raise NotImplementedError("docs/rules.md METRIC-06")


@pytest.mark.skip(reason="anomaly metrics arrive in Phase 6 (docs/rules.md METRIC-07)")
def test_metrics_anomaly_labeled_vs_not() -> None:
    raise NotImplementedError("docs/rules.md METRIC-07")


@pytest.mark.skip(reason="association-rule metrics arrive in Phase 7 (docs/rules.md METRIC-08)")
def test_metrics_association_filters() -> None:
    raise NotImplementedError("docs/rules.md METRIC-08")
