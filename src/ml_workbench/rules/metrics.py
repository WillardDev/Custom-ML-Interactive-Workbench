from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ml_workbench.rules.modelling import MINORITY_FRACTION_THRESHOLD
from ml_workbench.rules.preprocessing import SKEW_THRESHOLD

CLASSIFICATION_TASK_TYPES = frozenset({"binary", "multiclass", "multilabel"})

DEFAULT_METRICS: tuple[str, ...] = ("accuracy", "f1")


@dataclass(frozen=True)
class MetricPlan:
    primary: str
    metrics: tuple[str, ...]
    rule_id: str
    note: str


def metric_plan(
    task_type: str,
    *,
    minority_fraction: float | None = None,
    target_skew: float = 0.0,
    labeled: bool = False,
) -> MetricPlan:
    """METRIC-01..08: scoreboard for the task (supervised and unsupervised phases)."""
    if task_type in CLASSIFICATION_TASK_TYPES:
        if minority_fraction is not None and minority_fraction < MINORITY_FRACTION_THRESHOLD:
            return MetricPlan(
                primary="pr_auc",
                metrics=("pr_auc", "macro_f1", "mcc", "balanced_accuracy"),
                rule_id="METRIC-02",
                note="imbalanced classification (minority < 20%): PR-AUC primary, "
                "macro-F1, MCC, balanced accuracy.",
            )
        return MetricPlan(
            primary="accuracy",
            metrics=("accuracy", "f1"),
            rule_id="METRIC-01",
            note="balanced classification: accuracy and F1.",
        )
    if task_type == "regression":
        if abs(target_skew) > SKEW_THRESHOLD:
            return MetricPlan(
                primary="mae",
                metrics=("mae", "rmse", "r2"),
                rule_id="METRIC-03",
                note="skewed/outlier-heavy target: primary metric is MAE (robust to skew).",
            )
        return MetricPlan(
            primary="rmse",
            metrics=("rmse", "mae", "r2"),
            rule_id="METRIC-03",
            note="regression: RMSE primary, MAE and R2 reported.",
        )
    if task_type == "forecasting":
        return MetricPlan(
            primary="mae",
            metrics=("mae", "rmse", "smape", "mase"),
            rule_id="METRIC-04",
            note="time series: MAE primary, RMSE, sMAPE and MASE reported.",
        )
    if task_type == "clustering":
        if labeled:
            return MetricPlan(
                primary="adjusted_rand",
                metrics=("adjusted_rand", "nmi", "silhouette"),
                rule_id="METRIC-05",
                note="clustering with evaluation labels: ARI and NMI on the holdout, "
                "silhouette alongside.",
            )
        return MetricPlan(
            primary="silhouette",
            metrics=("silhouette", "davies_bouldin", "calinski_harabasz"),
            rule_id="METRIC-05",
            note="clustering: silhouette, Davies-Bouldin, Calinski-Harabasz.",
        )
    if task_type == "dimensionality_reduction":
        return MetricPlan(
            primary="explained_variance",
            metrics=("explained_variance", "reconstruction_error", "trustworthiness"),
            rule_id="METRIC-06",
            note="dimensionality reduction: explained variance, reconstruction error "
            "and trustworthiness.",
        )
    if task_type == "anomaly_detection":
        if labeled:
            return MetricPlan(
                primary="roc_auc",
                metrics=("roc_auc", "pr_auc"),
                rule_id="METRIC-07",
                note="anomaly detection with labels: ROC-AUC and PR-AUC on the holdout.",
            )
        return MetricPlan(
            primary="score_distribution",
            metrics=("score_distribution",),
            rule_id="METRIC-07",
            note="anomaly detection: score distribution (ROC/PR-AUC when labeled).",
        )
    return MetricPlan(
        primary="lift",
        metrics=("support", "confidence", "lift"),
        rule_id="METRIC-08",
        note="association rules: support, confidence, lift filters.",
    )


def labeled_cluster_scores(labels_true: object, labels_pred: object) -> dict[str, float]:
    """METRIC-05: ARI and NMI when clustering has evaluation labels."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    ari = float(adjusted_rand_score(labels_true, labels_pred))
    nmi = float(normalized_mutual_info_score(labels_true, labels_pred))
    return {"adjusted_rand": round(ari, 6), "nmi": round(nmi, 6)}


def anomaly_labeled_scores(labels_true: object, decision_scores: object) -> dict[str, float]:
    """METRIC-07: ROC-AUC and PR-AUC when anomaly detection has evaluation labels."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    try:
        roc = float(roc_auc_score(labels_true, decision_scores))
    except ValueError:
        roc = float("nan")
    try:
        pr = float(average_precision_score(labels_true, decision_scores))
    except ValueError:
        pr = float("nan")
    return {"roc_auc": round(roc, 6), "pr_auc": round(pr, 6)}


def trustworthiness_coef(high_dim: object, low_dim: object, n_neighbors: int = 5) -> float:
    """METRIC-06: trustworthiness of an embedding vs the original space."""
    from sklearn.manifold import trustworthiness as _trustworthiness

    try:
        value = float(_trustworthiness(high_dim, low_dim, n_neighbors=n_neighbors))
    except ValueError:
        return float("nan")
    return round(value, 6)


def _rounded(values: dict[str, float]) -> dict[str, float]:
    return {name: round(value, 6) for name, value in values.items()}


def forecast_scores(
    y_true: object,
    y_pred: object,
    *,
    seasonal_period: int = 1,
) -> dict[str, float]:
    """METRIC-04: MAE, RMSE, MAPE/sMAPE and MASE for a holdout forecast.
    MASE scales by the mean absolute seasonal-naive error of the holdout series."""
    y = np.asarray(y_true, dtype=float).ravel().tolist()
    p = np.asarray(y_pred, dtype=float).ravel().tolist()
    n = min(len(y), len(p))
    if n == 0:
        return {name: float("nan") for name in ("mae", "rmse", "mape", "smape", "mase")}
    y, p = y[:n], p[:n]
    errors = [truth - pred for truth, pred in zip(y, p, strict=False)]
    mae = math.fsum(abs(error) for error in errors) / n
    rmse = math.sqrt(math.fsum(error * error for error in errors) / n)
    nonzero = [
        (abs(error), truth, pred)
        for error, truth, pred in zip(errors, y, p, strict=False)
        if abs(truth) > 1e-12
    ]
    if nonzero:
        mape = (
            100.0 * math.fsum(abs(error) / abs(truth) for error, truth, _ in nonzero) / len(nonzero)
        )
        smape = (
            100.0
            * math.fsum(
                2.0 * abs(error) / (abs(truth) + abs(pred)) for error, truth, pred in nonzero
            )
            / len(nonzero)
        )
    else:
        mape = float("nan")
        smape = float("nan")
    period = max(1, int(seasonal_period))
    if n > period:
        scale = math.fsum(abs(y[i] - y[i - period]) for i in range(period, n)) / (n - period)
        mase = mae / scale if scale > 1e-12 else float("nan")
    else:
        mase = float("nan")
    return _rounded({"mae": mae, "rmse": rmse, "mape": mape, "smape": smape, "mase": mase})
