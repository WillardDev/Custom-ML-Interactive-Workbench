from __future__ import annotations

from dataclasses import dataclass
from typing import Final

HIGH_CARDINALITY_THRESHOLD: Final = 20  # ENC-02
HIGH_DIM_FEATURE_THRESHOLD: Final = 30  # DR-01
SKEW_THRESHOLD: Final = 1.0  # TT-01, HINT-02

TREE_FAMILY: Final = "tree"
NEURAL_FAMILY: Final = "neural"
LINEAR_FAMILY: Final = "linear"
ONE_HOT_FAMILIES: Final[frozenset[str]] = frozenset({"linear", "kernel", "distance", "neural"})
SCALING_FAMILIES: Final[frozenset[str]] = frozenset(
    {"linear", "kernel", "distance", "centroid", "neural"}
)

CLASSIFICATION_TASK_TYPES: Final[frozenset[str]] = frozenset({"binary", "multiclass", "multilabel"})


@dataclass(frozen=True)
class EncodingPlan:
    options: tuple[str, ...]
    default: str
    target_encode_in_folds: bool
    rule_id: str
    note: str


@dataclass(frozen=True)
class ScalingPlan:
    needed: bool
    options: tuple[str, ...]
    default: str | None
    mandatory: bool
    rule_id: str
    note: str


def encoding_plan(
    family: str | None,
    *,
    supervised: bool = True,
    high_cardinality: bool = False,
) -> EncodingPlan:
    """ENC-01..ENC-04: encoding strategy from the model family."""
    if family == "tree":
        return EncodingPlan(
            options=("ordinal", "native"),
            default="ordinal",
            target_encode_in_folds=False,
            rule_id="ENC-01",
            note="tree family encodes categorically (ordinal) or uses native handling; "
            "never one-hot by default.",
        )
    if family == "neural":
        return EncodingPlan(
            options=("one_hot", "embedding"),
            default="one_hot",
            target_encode_in_folds=False,
            rule_id="ENC-03",
            note="neural family one-hot encodes, or learns embeddings for high cardinality.",
        )
    if supervised and high_cardinality and family in ONE_HOT_FAMILIES:
        return EncodingPlan(
            options=("one_hot", "target"),
            default="one_hot",
            target_encode_in_folds=True,
            rule_id="ENC-02",
            note="high-cardinality categoricals may use target encoding fitted inside CV folds.",
        )
    if family in ONE_HOT_FAMILIES:
        return EncodingPlan(
            options=("one_hot",),
            default="one_hot",
            target_encode_in_folds=False,
            rule_id="ENC-02",
            note="linear/distance/kernel families use one-hot encoding.",
        )
    return EncodingPlan(
        options=("ordinal", "one_hot"),
        default="ordinal",
        target_encode_in_folds=False,
        rule_id="ENC-01",
        note="no model selected yet — generic encoding; re-derived when a model is chosen.",
    )


def clustering_encoding(categorical_columns: tuple[str, ...]) -> str | None:
    """ENC-04: clustering with categorical features uses K-Prototypes or Gower distance."""
    if categorical_columns:
        return "kprototypes_or_gower"
    return None


def encoder_fits_inside_fold() -> bool:
    """ENC-05: encoders are always fitted on training folds only."""
    return True


def scaling_plan(needs_scaling: bool, *, family: str | None = None) -> ScalingPlan:
    """SCALE-01..SCALE-03: scaling strategy from the model's needs_scaling flag."""
    if needs_scaling:
        return ScalingPlan(
            needed=True,
            options=("standard", "robust", "minmax"),
            default="standard",
            mandatory=family == "neural",
            rule_id="SCALE-01" if family != "neural" else "SCALE-03",
            note="Standard (default), Robust, or MinMax scaling offered.",
        )
    return ScalingPlan(
        needed=False,
        options=(),
        default=None,
        mandatory=False,
        rule_id="SCALE-02",
        note="model does not need scaling — scaler options are hidden.",
    )


def scaler_fits_inside_fold() -> bool:
    """SCALE-04: scalers are always fitted on training folds only."""
    return True


def missing_indicators(family: str | None) -> bool:
    """FEAT-01: neural models impute and add missing-indicator columns."""
    return family == "neural"


def selection_menu(supervised: bool) -> tuple[str, ...]:
    """FEAT-02a/b: feature-selection menu by supervision type."""
    if supervised:
        return ("anova", "chi2", "mutual_info", "rfe", "l1", "model_based")
    return ("variance_threshold", "correlation_filter")


def selector_fits_inside_fold() -> bool:
    """FEAT-03: selectors are always fitted inside CV folds."""
    return True


def imbalance_options(task_type: str) -> tuple[str, ...]:
    """FEAT-04a: imbalance handling offered for classification."""
    if task_type in CLASSIFICATION_TASK_TYPES:
        return ("class_weight", "smote", "threshold_tuning")
    return ()


def smote_inside_folds_only() -> bool:
    """FEAT-04b: SMOTE resampling happens only inside CV folds, never before the split."""
    return True


def smote_allowed(task_type: str) -> bool:
    """FEAT-04c: SMOTE is rejected for regression."""
    return task_type in CLASSIFICATION_TASK_TYPES


def pca_suggestion(n_features: int, family: str, *, task_type: str = "clustering") -> str | None:
    """DR-01/DR-02: PCA is suggested only for high-dimensional clustering, never for trees."""
    if family == "tree":
        return None
    if n_features > HIGH_DIM_FEATURE_THRESHOLD and task_type == "clustering":
        return "pca_95pct"
    return None


def dr_as_clustering_input(viz_only: bool) -> bool:
    """DR-03: t-SNE/UMAP (viz_only) are never used as clustering input."""
    return not viz_only


def target_transform_options(
    task_type: str, skew: float, *, family: str | None = None
) -> tuple[str, ...]:
    """TT-01: offer target transforms for skewed regression targets."""
    if task_type != "regression":
        return ()
    if abs(skew) <= SKEW_THRESHOLD:
        return ()
    if family in {"kernel", "neural"} or family is None:
        return ("log", "boxcox", "yeojohnson")
    return ()


def pipeline_fit_leakage_safe() -> bool:
    """PIPE-01: every fitted step lives in one pipeline fitted on training data only."""
    return True
