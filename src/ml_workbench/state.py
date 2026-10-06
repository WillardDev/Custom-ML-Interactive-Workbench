from __future__ import annotations

from dataclasses import dataclass, field

from ml_workbench.tabs import TAB_ORDER

LearningType = str
TaskType = str
ModelStatus = str


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
