from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from ml_workbench.rules.association import basket_size_distribution, item_frequency
from ml_workbench.rules.eda import (
    EDA_MAX_ROWS,
    eda_plan,
    numeric_columns,
    sample_frame_for_eda,
    skew,
    skew_hint,
    task_views,
)
from ml_workbench.state import TaskDefinition, feature_columns

ADF_CRITICAL_5PCT: float = -2.86  # large-sample 5% critical value, constant only (EDA-04)


@dataclass(frozen=True)
class EdaException(ValueError):
    pass


@dataclass(frozen=True)
class BaseReport:
    rows: int
    columns: int
    shape: tuple[int, int]
    sampled_rows: int
    dtypes: dict[str, str]
    missing: dict[str, int]
    numeric_describe: pd.DataFrame | None
    numeric_pearson: pd.DataFrame | None
    numeric_spearman: pd.DataFrame | None
    cramers_v: pd.DataFrame | None


@dataclass(frozen=True)
class ClassificationReport:
    class_counts: pd.DataFrame
    feature_by_class: dict[str, pd.DataFrame] = field(default_factory=dict)
    chi_square: pd.DataFrame | None = None


@dataclass(frozen=True)
class RegressionReport:
    target_skew: float
    skew_hint: str | None
    target_histogram: pd.Series
    feature_scatter: pd.DataFrame | None


@dataclass(frozen=True)
class ClusteringReport:
    hopkins: float
    components_for_90: int | None
    preview_columns: tuple[str, ...]
    scree: pd.DataFrame | None
    n_features: int


@dataclass(frozen=True)
class AnomalyReport:
    zscore_flags: pd.DataFrame | None
    iqr_flags: pd.DataFrame | None
    mahalanobis: pd.DataFrame | None
    n_features: int


@dataclass(frozen=True)
class DimReductionReport:
    correlation_groups: tuple[tuple[str, ...], ...]
    vif: pd.DataFrame | None
    n_features: int


@dataclass(frozen=True)
class TimeseriesReport:
    period: int
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series
    acf: pd.DataFrame
    pacf: pd.DataFrame
    rolling: pd.DataFrame
    stationarity_stat: float
    stationarity_critical: float
    stationary: bool


@dataclass(frozen=True)
class AssociationReport:
    item_frequency: pd.DataFrame
    basket_size: pd.Series
    baskets: int
    unique_items: int


@dataclass(frozen=True)
class EdaResult:
    plan: tuple[str, ...]
    base: BaseReport
    views: tuple[str, ...]
    classification: ClassificationReport | None = None
    regression: RegressionReport | None = None
    clustering: ClusteringReport | None = None
    anomaly: AnomalyReport | None = None
    dimred: DimReductionReport | None = None
    timeseries: TimeseriesReport | None = None
    association: AssociationReport | None = None


def _categorical(target: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [column for column in target.columns if is_numeric_dtype(target[column])]
    categorical = [column for column in target.columns if column not in numeric]
    return numeric, categorical


def _cramers_v(observed: np.ndarray) -> float:
    table = np.asarray(observed, dtype=float)
    total = float(table.sum())
    if total == 0:
        return 0.0
    row_tot = table.sum(axis=1, keepdims=True)
    col_tot = table.sum(axis=0, keepdims=True)
    expected = row_tot * col_tot / total
    chi2 = float(np.nansum(np.where(expected > 0, (table - expected) ** 2 / expected, 0.0)))
    denominator = total * (min(table.shape) - 1)
    if denominator == 0:
        return 0.0
    return float(np.sqrt(max(0.0, chi2 / denominator)))


def cramers_v_matrix(frame: pd.DataFrame, columns: list[str], cap: int = 20) -> pd.DataFrame | None:
    """Cramér's V for categorical column pairs (EDA-01)."""
    low = [column for column in columns if frame[column].nunique(dropna=True) <= cap]
    if len(low) == 0:
        return None
    matrix = pd.DataFrame(np.eye(len(low)), index=low, columns=low)
    for i, left in enumerate(low):
        for right in low[i + 1 :]:
            table = pd.crosstab(frame[left], frame[right])
            value = _cramers_v(table.to_numpy())
            matrix.loc[left, right] = value
            matrix.loc[right, left] = value
    return matrix


def _chi_square(frame: pd.DataFrame, target: str, feature_cols: list[str]) -> pd.DataFrame | None:
    rows: list[dict[str, object]] = []
    for feature in feature_cols:
        table = pd.crosstab(frame[feature], frame[target])
        observed = table.to_numpy()
        total = float(observed.sum())
        if total == 0:
            continue
        row_tot = observed.sum(axis=1, keepdims=True)
        col_tot = observed.sum(axis=0, keepdims=True)
        expected = row_tot * col_tot / total
        chi2 = float(np.nansum((observed - expected) ** 2 / np.where(expected > 0, expected, 1)))
        rows.append(
            {
                "feature": feature,
                "chi2": round(chi2, 4),
                "dof": int((observed.shape[0] - 1) * (observed.shape[1] - 1)),
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows)


def base_report(
    frame: pd.DataFrame,
    *,
    seed: int = 0,
    max_rows: int = EDA_MAX_ROWS,
) -> BaseReport:
    sampled_rows = len(frame)
    sample = sample_frame_for_eda(frame, limit=max_rows, seed=seed)
    sampled_rows = len(sample)
    numeric = numeric_columns(sample)

    numeric_describe = sample[numeric].describe().transpose() if numeric else None
    pearson = sample[numeric].corr(method="pearson") if numeric else None
    spearman = sample[numeric].corr(method="spearman") if numeric else None

    _, categorical = _categorical(sample)
    cramers = cramers_v_matrix(sample, categorical) if categorical else None

    return BaseReport(
        rows=len(frame),
        columns=len(frame.columns),
        shape=sample.shape,
        sampled_rows=sampled_rows,
        dtypes={str(column): str(dtype) for column, dtype in sample.dtypes.items()},
        missing={
            column: int(sample[column].isna().sum())
            for column in sample.columns
            if int(sample[column].isna().sum()) > 0
        },
        numeric_describe=numeric_describe,
        numeric_pearson=pearson,
        numeric_spearman=spearman,
        cramers_v=cramers,
    )


def classification_report(frame: pd.DataFrame, task: TaskDefinition) -> ClassificationReport:
    assert task.target is not None
    target = frame[task.target]
    counts = target.value_counts(dropna=True).rename("count")
    fractions = (target.value_counts(dropna=True, normalize=True)).rename("fraction")
    table = pd.DataFrame({"count": counts, "fraction": fractions})

    features = [column for column in frame.columns if column != task.target]
    feature_by_class: dict[str, pd.DataFrame] = {}
    for feature in features:
        if frame[feature].nunique(dropna=True) <= 20:
            feature_by_class[feature] = pd.crosstab(frame[feature], target)
    categorical_features = [column for column in features if not is_numeric_dtype(frame[column])]
    chi = _chi_square(frame, task.target, categorical_features)
    return ClassificationReport(
        class_counts=table, feature_by_class=feature_by_class, chi_square=chi
    )


def regression_report(frame: pd.DataFrame, task: TaskDefinition) -> RegressionReport:
    assert task.target is not None
    values = pd.to_numeric(frame[task.target], errors="coerce").dropna()
    histogram = values.value_counts(bins=30, sort=False).sort_index()
    features = [column for column in frame.columns if column != task.target]
    sampled = sample_frame_for_eda(frame, limit=EDA_MAX_ROWS)
    scatter = sampled[features + [task.target]].dropna() if len(sampled) <= 5000 else None
    return RegressionReport(
        target_skew=skew(values),
        skew_hint=skew_hint(values),
        target_histogram=histogram,
        feature_scatter=scatter,
    )


_UNSUPERVISED_SAMPLE = 10_000
_ABS_CORR_GROUP_THRESHOLD = 0.7
_ZSCORE_FLAG = 3.0
_MAHALANOBIS_QUANTILE = 0.999


def _numeric_feature_matrix(
    frame: pd.DataFrame, task: TaskDefinition, *, seed: int = 0
) -> tuple[np.ndarray, list[str]]:
    """Unsupervised EDA uses only the numeric task features, NaN-safe, capped."""
    columns = [column for column in feature_columns(task, frame) if is_numeric_dtype(frame[column])]
    if not columns:
        return np.empty((0, 0)), columns
    matrix = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    matrix = matrix[~np.isnan(matrix).any(axis=1)]
    if len(matrix) > _UNSUPERVISED_SAMPLE:
        rng = np.random.default_rng(seed)
        matrix = matrix[rng.choice(len(matrix), size=_UNSUPERVISED_SAMPLE, replace=False)]
    return matrix, columns


def _nearest_pairwise(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    distances = pairwise_distances(reference, values).min(axis=1)
    return np.asarray(distances)


def _hopkins_statistic(values: np.ndarray, *, seed: int, m: int | None = None) -> float:
    """EDA-05: Hopkins statistic — ~0.5 means random, >0.7 suggests clusterable structure."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 20 or values.shape[1] == 0 or not np.isfinite(values).all():
        return float("nan")
    m = m or min(100, max(5, n // 10))
    rng = np.random.default_rng(seed)
    observed = values[rng.choice(n, size=m, replace=False)]
    low = values.min(axis=0)
    high = values.max(axis=0)
    span = high - low
    if not np.isfinite(span).all() or float(span.sum()) == 0.0:
        return float("nan")
    uniform = rng.uniform(low, high, size=(m, values.shape[1]))
    to_uniform = _nearest_pairwise(uniform, values)
    to_observed = _nearest_pairwise(observed, values)
    denominator = float(to_uniform.sum() + to_observed.sum())
    if denominator == 0.0:
        return float("nan")
    return round(float((to_uniform.sum()) / denominator), 4)


def _scree(values: np.ndarray, columns: list[str]) -> pd.DataFrame | None:
    if len(values) < 2 or values.shape[1] == 0:
        return None
    pca = PCA().fit(values)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(pca.n_components_)],
            "eigenvalue": np.round(pca.explained_variance_, 4),
            "explained_variance": np.round(pca.explained_variance_ratio_, 4),
            "cumulative": np.round(cumulative, 4),
        }
    )


def clustering_report(
    frame: pd.DataFrame, task: TaskDefinition, *, seed: int = 0
) -> ClusteringReport:
    """EDA-05: Hopkins statistic, a PCA/UMAP preview, and a scree plot."""
    values, columns = _numeric_feature_matrix(frame, task, seed=seed)
    hopkins = _hopkins_statistic(values, seed=seed)
    scree = _scree(values, columns)
    components_for_90: int | None = None
    if scree is not None and len(scree):
        above = scree["cumulative"] >= 0.9
        target = float(scree["cumulative"].iloc[-1]) if len(scree) else 0.0
        reached = int(above.argmax()) if above.any() else len(scree)
        components_for_90 = reached if float(target) > 0.0 else None
    preview_columns: tuple[str, ...] = ("pc1", "pc2")
    return ClusteringReport(
        hopkins=hopkins,
        components_for_90=components_for_90,
        preview_columns=preview_columns,
        scree=scree,
        n_features=len(columns),
    )


def _zscore_iqr_flags(
    frame: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    zrows: list[dict[str, object]] = []
    irows: list[dict[str, object]] = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if len(values) < 3 or float(values.std()) == 0.0:
            continue
        z = (values - values.mean()) / values.std()
        zrows.append({"feature": column, "flagged": int((z.abs() > _ZSCORE_FLAG).sum())})
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0.0:
            continue
        irows.append(
            {
                "feature": column,
                "below": int((values < q1 - 1.5 * iqr).sum()),
                "above": int((values > q3 + 1.5 * iqr).sum()),
            }
        )
    zframe = pd.DataFrame(zrows) if zrows else None
    iframe = pd.DataFrame(irows) if irows else None
    return zframe, iframe


def _mahalanobis(mask: pd.DataFrame, columns: list[str]) -> pd.DataFrame | None:
    assert len(columns) == mask.shape[1]
    if len(mask) < 3 or len(columns) == 0:
        return None
    values = mask.to_numpy(dtype=float)
    values = values[~np.isnan(values).any(axis=1)]
    if len(values) < 3:
        return None
    centered = values - values.mean(axis=0)
    try:
        inv_cov = np.linalg.inv(np.cov(centered, rowvar=False) + np.eye(len(columns)) * 1e-9)
    except np.linalg.LinAlgError:
        return None
    distances = np.sqrt(np.einsum("ij,jk,ik->i", centered, inv_cov, centered))
    threshold = float(stats.chi2.ppf(_MAHALANOBIS_QUANTILE, len(columns)))
    return pd.DataFrame(
        {
            "mean_distance": round(float(distances.mean()), 4),
            "max_distance": round(float(distances.max()), 4),
            "flagged": int((distances > threshold).sum()),
            "threshold": round(threshold, 4),
        },
        index=[0],
    )


def anomaly_report(frame: pd.DataFrame, task: TaskDefinition) -> AnomalyReport:
    """EDA-06: univariate z-score/IQR flags plus a multivariate Mahalanobis view."""
    _, columns = _numeric_feature_matrix(frame, task)
    zflags, iflags = _zscore_iqr_flags(frame, columns)
    subset = frame[[column for column in columns if column in frame.columns]].copy()
    mahalanobis = _mahalanobis(subset[columns], columns) if columns else None
    return AnomalyReport(
        zscore_flags=zflags,
        iqr_flags=iflags,
        mahalanobis=mahalanobis,
        n_features=len(columns),
    )


def _correlation_groups(
    values: np.ndarray, columns: list[str], threshold: float
) -> tuple[tuple[str, ...], ...]:
    if values.shape[1] < 2 or len(values) < 3:
        return ()
    correlation = np.corrcoef(values, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0)
    used: set[int] = set()
    groups: list[list[str]] = []
    for i in range(len(columns)):
        if i in used:
            continue
        group = [columns[i]]
        for j in range(len(columns)):
            if i != j and j not in used and abs(correlation[i, j]) >= threshold:
                group.append(columns[j])
                used.add(j)
        used.add(i)
        if len(group) > 1:
            groups.append(group)
    return tuple(tuple(group) for group in groups)


def _vif(values: np.ndarray, columns: list[str]) -> pd.DataFrame | None:
    if values.shape[1] < 2 or len(values) < 3:
        return None
    covariance = np.corrcoef(values, rowvar=False)
    nonzero = np.array([abs(covariance[k, k]) > 1e-12 for k in range(covariance.shape[0])])
    reduced = covariance[np.ix_(nonzero, nonzero)]
    names = [columns[k] for k in range(len(columns)) if nonzero[k]]
    if len(names) < 2:
        return None
    try:
        inverse = np.linalg.inv(reduced)
    except np.linalg.LinAlgError:
        return None
    if inverse.shape[0] <= 2:
        return None
    rows = [
        {
            "feature": name,
            "vif": round(float(inverse[k, k]), 4),
        }
        for k, name in enumerate(names)
    ]
    return pd.DataFrame(rows)


def dimred_report(frame: pd.DataFrame, task: TaskDefinition) -> DimReductionReport:
    """EDA-07: correlation groups plus variance inflation factors."""
    values, columns = _numeric_feature_matrix(frame, task)
    groups = _correlation_groups(values, columns, _ABS_CORR_GROUP_THRESHOLD)
    vif = _vif(values, columns)
    return DimReductionReport(correlation_groups=groups, vif=vif, n_features=len(columns))


def _pacf_from_acf(acf: list[float], max_lag: int) -> list[float]:
    """Durbin–Levinson recursion: partial autocorrelations from the ACF."""
    pacf = [1.0]
    phi: list[float] = []
    for lag in range(1, max_lag + 1):
        numerator = acf[lag] - sum(phi[j - 1] * acf[lag - j] for j in range(1, lag))
        denominator = 1.0 - sum(phi[j - 1] * acf[j] for j in range(1, lag))
        phi_kk = numerator / denominator if abs(denominator) > 1e-12 else 0.0
        updated = [phi[j - 1] - phi_kk * phi[lag - 1 - j] for j in range(1, lag)]
        phi = [*updated, phi_kk]
        pacf.append(phi_kk)
    return pacf


def _adf_statistic(values: np.ndarray) -> float:
    """EDA-04: augmented Dickey–Fuller t-statistic on the lagged level (no trend)."""
    if len(values) < 8:
        return float("nan")
    delta = np.diff(values)
    lagged = values[:-1]
    design = np.column_stack([np.ones(len(lagged)), lagged])
    coefficients, *_ = np.linalg.lstsq(design, delta, rcond=None)
    residuals = delta - design @ coefficients
    sse = float(residuals @ residuals)
    dof = max(1, len(lagged) - design.shape[1])
    covariance = (sse / dof) * np.linalg.pinv(design.T @ design)
    std_err = math.sqrt(max(float(covariance[1, 1]), 1e-18))
    return float(coefficients[1] / std_err)


def timeseries_report(frame: pd.DataFrame, task: TaskDefinition) -> TimeseriesReport:
    """EDA-04: decomposition, ACF/PACF, rolling statistics and a stationarity test."""
    if task.target is None:
        raise EdaException("forecasting EDA needs a target column")
    series = pd.to_numeric(frame[task.target], errors="coerce").dropna().astype(float)
    values = series.to_numpy()
    if len(values) < 4:
        raise EdaException("forecasting EDA needs at least 4 rows")
    period = 12 if len(values) >= 24 else max(2, len(values) // 4)
    window = min(period if len(values) >= 2 * period else max(3, len(values) // 4), len(values))
    window = max(3, window)
    if window % 2 == 0:
        window += 1
    trend = series.rolling(window, center=True, min_periods=1).mean()
    detrended = series - trend
    phases = pd.Series(np.arange(len(values)) % period, index=series.index)
    seasonal_by_phase = detrended.groupby(phases).mean()
    seasonal = phases.map(seasonal_by_phase).astype(float).fillna(0.0)
    residual = series - trend - seasonal
    max_lag = min(40, max(1, len(values) // 2 - 1))
    mean = float(np.mean(values))
    centered = values - mean
    denominator = float(centered @ centered)
    acf_values = [
        1.0
        if lag == 0
        else (float(centered[lag:] @ centered[:-lag]) / denominator if denominator > 1e-12 else 0.0)
        for lag in range(max_lag + 1)
    ]
    pacf_values = _pacf_from_acf(acf_values, max_lag)
    acf_table = pd.DataFrame({"lag": range(1, max_lag + 1), "acf": acf_values[1:]}).round(4)
    pacf_table = pd.DataFrame({"lag": range(1, max_lag + 1), "pacf": pacf_values[1:]}).round(4)
    rolling = pd.DataFrame(
        {"rolling_mean": trend, "rolling_std": series.rolling(window, min_periods=2).std()}
    ).round(4)
    stat = _adf_statistic(values)
    return TimeseriesReport(
        period=period,
        trend=trend.round(4),
        seasonal=seasonal.round(4),
        residual=residual.round(4),
        acf=acf_table,
        pacf=pacf_table,
        rolling=rolling,
        stationarity_stat=round(stat, 4),
        stationarity_critical=ADF_CRITICAL_5PCT,
        stationary=bool(math.isfinite(stat) and stat < ADF_CRITICAL_5PCT),
    )


def association_report(frame: pd.DataFrame, task: TaskDefinition) -> AssociationReport:
    """EDA-08: item frequency and basket size distribution."""
    frequency = item_frequency(frame)
    sizes = basket_size_distribution(frame)
    return AssociationReport(
        item_frequency=frequency,
        basket_size=sizes,
        baskets=int(sizes.sum()) if not sizes.empty else 0,
        unique_items=int(len(frequency)),
    )


def eda_report(frame: pd.DataFrame, task: TaskDefinition, *, seed: int = 0) -> EdaResult:
    plan = eda_plan(task)
    base = base_report(frame, seed=seed)
    views: list[str] = []
    classification = None
    regression = None
    clustering = None
    anomaly = None
    dimred = None
    timeseries = None
    association = None
    if task.task_type in {"binary", "multiclass", "multilabel"}:
        classification = classification_report(frame, task)
        views = ["class_balance", "feature_by_class", "chi_square"]
    elif task.task_type == "regression":
        regression = regression_report(frame, task)
        views = ["target_histogram", "qq_plot", "feature_target_scatter"]
    elif task.task_type == "forecasting":
        timeseries = timeseries_report(frame, task)
        views = ["decomposition", "acf_pacf", "rolling_stats", "stationarity"]
    elif task.learning_type == "unsupervised":
        views = list(task_views(task))
        if task.task_type == "clustering":
            clustering = clustering_report(frame, task, seed=seed)
        elif task.task_type == "anomaly_detection":
            anomaly = anomaly_report(frame, task)
        elif task.task_type == "dimensionality_reduction":
            dimred = dimred_report(frame, task)
        elif task.task_type == "association":
            association = association_report(frame, task)
    return EdaResult(
        plan=plan,
        base=base,
        views=tuple(views),
        classification=classification,
        regression=regression,
        clustering=clustering,
        anomaly=anomaly,
        dimred=dimred,
        timeseries=timeseries,
        association=association,
    )
