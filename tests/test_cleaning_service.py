from __future__ import annotations

from pathlib import Path

import pytest

from ml_workbench.rules.task import build_task
from ml_workbench.services.cleaning_service import (
    CleaningOptions,
    CleaningServiceError,
    model_policy,
    run_cleaning,
)
from ml_workbench.services.data_service import insert_dataset, read_sample
from ml_workbench.services.workspace import Workspace
from ml_workbench.state import ProjectState, TaskError

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"


def _workspace(tmp_path: Path) -> Workspace:
    return Workspace(project_id="p_clean", root=tmp_path)


def _loaded_messy(workspace: Workspace) -> ProjectState:
    state = ProjectState(project_id="p_clean")
    insert_dataset(
        state,
        workspace,
        read_sample("messy_classification.csv"),
        source="sample:messy_classification.csv",
    )
    state.task = build_task(state.frame, "supervised", target="churned")
    return state


def test_model_policy_without_active_model() -> None:
    state = ProjectState(project_id="p_clean")
    assert model_policy(state) == (False, None)
    state.active_model_id = "hist_gradient_boosting"
    assert model_policy(state) == (True, "tree")


def test_run_cleaning_records_steps_and_versions(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    state = _loaded_messy(workspace)

    result = run_cleaning(
        state,
        workspace,
        CleaningOptions(
            imputations=(("revenue", "median"),),
            outlier_columns=("revenue",),
            outlier_op="clip",
        ),
    )

    assert [report.op for report in result.ops] == [
        "dedupe",
        "fix_dtypes",
        "impute",
        "outlier_treatment",
    ]
    assert result.rows_before == 195
    assert result.rows_after < result.rows_before
    assert len(result.versions) == 4 and len(set(result.versions)) == 4
    assert state.stale["cleaning"] is False
    assert all(
        state.stale[tab]
        for tab in [
            "preprocessing",
            "eda",
            "modelling",
            "training",
            "prediction",
            "error_analysis",
            "explainability",
            "outcome",
        ]
    )
    assert state.dataset is not None and state.dataset.version == result.versions[-1]
    assert result.balance is not None
    for version in result.versions:
        assert workspace.read_version_info(version).version == version
        assert workspace.version_path(version).is_file()
    assert int(state.frame["revenue"].isna().sum()) == 0  # median imputation applied


def test_run_cleaning_never_imputes_target(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    state = _loaded_messy(workspace)
    loaded_version = state.dataset.version if state.dataset else ""

    with pytest.raises(TaskError):
        run_cleaning(state, workspace, CleaningOptions(imputations=(("churned", "median"),)))
    assert state.dataset is not None and state.dataset.version == loaded_version


def test_run_cleaning_requires_setup(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    state = ProjectState(project_id="p_clean")
    with pytest.raises(CleaningServiceError):
        run_cleaning(state, workspace, CleaningOptions())

    insert_dataset(
        state, workspace, read_sample("binary_classification.csv"), source="sample:b.csv"
    )
    with pytest.raises(CleaningServiceError):
        run_cleaning(state, workspace, CleaningOptions())


def test_anomaly_blocks_outlier_removal(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    state = ProjectState(project_id="p_clean")
    insert_dataset(state, workspace, read_sample("clustering.csv"), source="sample:clustering.csv")
    state.task = build_task(state.frame, "unsupervised", task_type="anomaly_detection")

    with pytest.raises(CleaningServiceError):
        run_cleaning(state, workspace, CleaningOptions(outlier_columns=("x",), outlier_op="drop"))


def test_tree_family_limits_outlier_choices(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    state = _loaded_messy(workspace)
    state.active_model_id = "hist_gradient_boosting"

    with pytest.raises(CleaningServiceError):
        run_cleaning(
            state, workspace, CleaningOptions(outlier_columns=("revenue",), outlier_op="clip")
        )
