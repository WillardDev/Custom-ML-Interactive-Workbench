from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ml_workbench.tabs import TAB_ORDER

if TYPE_CHECKING:
    from pandas import DataFrame

LearningType = str
TaskType = str
ModelStatus = str


class TaskError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetInfo:
    version: str
    path: str
    data_hash: str
    row_count: int
    loaded_at: str


@dataclass(frozen=True)
class TaskDefinition:
    learning_type: LearningType
    task_type: TaskType
    target: str | None = None
    group: str | None = None
    time_column: str | None = None
    eval_labels: str | None = None

    def __post_init__(self) -> None:
        if self.learning_type == "unsupervised":
            if self.target is not None:
                raise TaskError("unsupervised tasks must not set a target column")
            if self.task_type in {
                "binary",
                "multiclass",
                "multilabel",
                "regression",
                "forecasting",
            }:
                raise TaskError(f"'{self.task_type}' is a supervised task type")
        if self.learning_type == "supervised" and self.target is None:
            raise TaskError("supervised tasks require a target column")
        if self.task_type == "forecasting" and self.time_column is None:
            raise TaskError("forecasting requires a time column")
        if self.eval_labels is not None and self.eval_labels == self.target:
            raise TaskError("evaluation labels must differ from the target")


@dataclass(frozen=True)
class SplitInfo:
    strategy: str
    params: dict[str, int | float | str] = field(default_factory=dict)
    fitted: bool = False


@dataclass(frozen=True)
class PipelineInfo:
    fitted: bool = False
    path: str | None = None


@dataclass(frozen=True)
class StepEntry:
    id: int
    tab: str
    op: str
    params: dict[str, object] = field(default_factory=dict)
    dataset_hash_before: str = ""
    dataset_hash_after: str = ""
    created_at: str = ""


@dataclass(frozen=True)
class ModelRun:
    run_id: str
    model_id: str
    status: ModelStatus
    artifact_path: str | None = None
    meta_path: str | None = None
    metrics_path: str | None = None
    trained_at: str | None = None


@dataclass
class ProjectState:
    project_id: str
    seed: int = 42
    dataset: DatasetInfo | None = None
    frame: DataFrame | None = None
    task: TaskDefinition | None = None
    split: SplitInfo | None = None
    pipeline: PipelineInfo | None = None
    steps: list[StepEntry] = field(default_factory=list)
    models: list[ModelRun] = field(default_factory=list)
    active_model_id: str | None = None
    stale: dict[str, bool] = field(default_factory=lambda: dict.fromkeys(TAB_ORDER, False))
    job: dict[str, object] | None = None

    @property
    def cleaned(self) -> bool:
        return any(step.tab == "cleaning" for step in self.steps)

    @property
    def split_ready(self) -> bool:
        split_fitted = self.split is not None and self.split.fitted
        pipeline_fitted = self.pipeline is not None and self.pipeline.fitted
        return split_fitted or pipeline_fitted

    @property
    def trained_models(self) -> list[ModelRun]:
        return [model for model in self.models if model.status == "done"]

    @property
    def has_trained_model(self) -> bool:
        return bool(self.trained_models)


def feature_columns(task: TaskDefinition, frame: DataFrame) -> list[str]:
    excluded = {col for col in (task.target, task.eval_labels, task.group) if col is not None}
    return [col for col in frame.columns if col not in excluded]
