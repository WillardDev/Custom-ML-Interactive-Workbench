from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ml_workbench.rules.task import SUPERVISED_TASK_TYPES
from ml_workbench.state import TaskDefinition

CLASSIFICATION_TASK_TYPES: Final[frozenset[str]] = frozenset({"binary", "multiclass", "multilabel"})

DEFAULT_HOLDOUT_FRACTION: Final = 0.2  # SPLIT-09
DEFAULT_N_SPLITS: Final = 5  # SPLIT-09


@dataclass(frozen=True)
class SplitRule:
    strategy: str
    rule_id: str
    message: str
    shuffle: bool = False
    n_splits: int = DEFAULT_N_SPLITS
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION


def choose_split(task: TaskDefinition) -> SplitRule:
    """SPLIT-01..07, SPLIT-09: pick the holdout strategy for the task."""
    if task.group is not None:
        return SplitRule(
            strategy="grouped",
            rule_id="SPLIT-07",
            shuffle=True,
            message="a group column is set; folds never split a group (SPLIT-07 / SPLIT-02).",
        )
    if task.learning_type == "unsupervised":
        if task.eval_labels is not None:
            return SplitRule(
                strategy="eval_labels",
                rule_id="SPLIT-05",
                message="evaluation labels select a stratified holdout used for scoring only.",
            )
        return SplitRule(
            strategy="none",
            rule_id="SPLIT-06",
            message="unsupervised without labels — optional stability holdout or no split.",
        )
    if task.task_type == "forecasting":
        return SplitRule(
            strategy="chronological",
            rule_id="SPLIT-01",
            message="time-ordered data is split chronologically, never shuffled.",
        )
    if task.task_type in CLASSIFICATION_TASK_TYPES:
        return SplitRule(
            strategy="stratified",
            rule_id="SPLIT-03",
            shuffle=True,
            message="classification uses a stratified holdout to preserve class proportions.",
        )
    if task.task_type in SUPERVISED_TASK_TYPES:
        return SplitRule(
            strategy="random",
            rule_id="SPLIT-04",
            shuffle=True,
            message="regression uses a random holdout split.",
        )
    return SplitRule(
        strategy="random",
        rule_id="SPLIT-09",
        shuffle=True,
        message=f"default holdout {int((1 - DEFAULT_HOLDOUT_FRACTION) * 100)}/"
        f"{int(DEFAULT_HOLDOUT_FRACTION * 100)} (SPLIT-09).",
    )


def cv_strategy(task: TaskDefinition) -> SplitRule:
    """SPLIT-08: cross-validation splitter by task (default 5 folds, SPLIT-09)."""
    if task.group is not None:
        return SplitRule(
            strategy="group_kfold",
            rule_id="SPLIT-08",
            message="grouped data uses GroupKFold so no group crosses folds.",
        )
    if task.task_type == "forecasting":
        return SplitRule(
            strategy="time_series_split",
            rule_id="SPLIT-08",
            message="time series uses TimeSeriesSplit (chronological, never shuffled).",
        )
    if task.learning_type == "unsupervised":
        return SplitRule(
            strategy="shuffle_split",
            rule_id="SPLIT-08",
            message="unsupervised tasks use repeated randomized holdouts for stability.",
        )
    if task.task_type in CLASSIFICATION_TASK_TYPES:
        return SplitRule(
            strategy="stratified_kfold",
            rule_id="SPLIT-08",
            message="classification uses StratifiedKFold to preserve class proportions.",
        )
    if task.task_type in SUPERVISED_TASK_TYPES:
        return SplitRule(
            strategy="kfold",
            rule_id="SPLIT-08",
            message="regression uses KFold.",
        )
    return SplitRule(strategy="kfold", rule_id="SPLIT-09", message="default KFold (SPLIT-09).")


def default_holdout_fraction() -> float:
    """SPLIT-09: 80% train / 20% holdout unless a rule overrides it."""
    return DEFAULT_HOLDOUT_FRACTION


def default_n_splits() -> int:
    """SPLIT-09: 5-fold cross-validation unless a rule overrides it."""
    return DEFAULT_N_SPLITS
