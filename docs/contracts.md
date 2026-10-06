# Data contracts

Source: design document §3 (workflow/gating), §6.1 (data insertion), §9 (state and safety), §10 (storage).
Status: accepted (Phase 0). ⚠ = inferred, confirm at review.

## Tab identifiers

Tabs have stable IDs and a fixed order; order defines downstream direction for staleness.

| # | ID | Tab |
|---|---|---|
| 1 | `data` | Data Insertion |
| 2 | `cleaning` | Data Cleaning |
| 3 | `preprocessing` | Data Preprocessing |
| 4 | `eda` | Exploratory Data Analysis |
| 5 | `modelling` | Modelling |
| 6 | `training` | Training |
| 7 | `prediction` | Prediction |
| 8 | `error_analysis` | Error Analysis |
| 9 | `explainability` | Model Explainability |
| 10 | `outcome` | Outcome and Final Prediction |

```python
TAB_ORDER = ["data", "cleaning", "preprocessing", "eda", "modelling",
             "training", "prediction", "error_analysis", "explainability", "outcome"]
```

## ProjectState (lives in `st.session_state["state"]`)

Single source of truth (§2). Tabs read it; only the owning tab or service writes it.

| Field | Type | Notes |
|---|---|---|
| `project_id` | str | Matches `workspace/projects/<project_id>/` |
| `dataset` | object \| null | `{version, path, hash, row_count, schema_report, loaded_at}` |
| `task` | object \| null | `{learning_type: "supervised"\|"unsupervised", task_type, target, group, time_column, eval_labels}` — see task inference below |
| `split` | object \| null | `{strategy, params, indices_path, fitted: bool}` |
| `steps` | list | In-memory mirror of `steps.json` (step log, §6.2) |
| `pipeline` | object \| null | `{fitted: bool, path}` — preprocessing pipeline artifact |
| `models` | list | `{run_id, model_id, status: "queued"\|"running"\|"done"\|"failed", artifact_path, meta_path, metrics_path, trained_at}` |
| `active_model_id` | str \| null | Model used by Prediction/Error/Explain/Outcome |
| `stale` | dict[str, bool] | Per-tab stale flags, see staleness below |
| `seed` | int | Project-wide random seed |
| `job` | object \| null | `{job_id, kind, progress, message}` — current background job |

**Invariants**

- `task.learning_type == "unsupervised"` ⟹ `task.target is None`; `task.eval_labels` optional and used only for scoring (§6.1).
- `stale[tab]` is `False` for `data` (nothing upstream of it).
- Downstream tabs never read another tab's local widget values — only `ProjectState`.

## Task inference (§6.1)

Supervised: target column required.
- Binary target (2 unique values) → `binary`
- Integer target with few unique values (≤ 20 ⚠ threshold) → `multiclass`
- Numeric target with many unique values → `regression`
- Datetime column + user confirmation → `forecasting`
- Group column (patient/customer ID) offered to prevent leakage; once set, every split must respect it (rule `SPLIT-07`).

Unsupervised: no target; optional "evaluation labels" column held aside, never seen by fitting.

## `steps.json` — step log (§6.2, §9)

Powers undo, script export, and reports. Append-only within a version; each step records the
dataset hash before/after so undo and staleness are verifiable.

```json
{
  "version": 1,
  "project_id": "p_01H...",
  "entries": [
    {
      "id": 7,
      "tab": "cleaning",
      "op": "impute",
      "params": {"columns": ["age"], "strategy": "median"},
      "dataset_hash_before": "sha256:9f2a...",
      "dataset_hash_after": "sha256:c41d...",
      "created_at": "2026-10-06T10:42:11Z"
    }
  ]
}
```

- `op` vocabulary: `load`, `dedupe`, `fix_dtypes`, `drop_rows`, `impute`, `outlier_treatment`,
  `encode`, `scale`, `feature_select`, `split`, `target_transform`, `resample` (Phase 6),
  `window` (Phase 8).
- Script export (rule `EXPORT-01`) replays entries in order into a standalone Python script.

## Model run metadata — `models/<run_id>/meta.json` (§6.6, §6.10)

```json
{
  "run_id": "r_0007",
  "model_id": "lightgbm",
  "task": {"learning_type": "supervised", "task_type": "binary"},
  "params": {"learning_rate": 0.05, "n_estimators": 500},
  "seed": 42,
  "data_hash": "sha256:c41d...",
  "split": {"strategy": "stratified_kfold", "params": {"n_splits": 5}},
  "versions": {"python": "3.12.x", "scikit_learn": "1.5.x", "...": "..."},
  "trained_at": "2026-10-06T11:03:52Z"
}
```

This is also the **metadata sidecar** read by tabs that do not need the heavy artifact (§8):
leaderboard, gating checks, and registry views read `meta.json` only.

## Metrics — `models/<run_id>/metrics.json` (§6.6)

```json
{
  "primary": "pr_auc",
  "metrics": {"pr_auc": 0.81, "macro_f1": 0.77, "mcc": 0.64, "balanced_accuracy": 0.74},
  "cv": {"n_splits": 5, "fold_values": {"pr_auc": [0.79, 0.83, "..."]},
         "mean": {"pr_auc": 0.81}, "std": {"pr_auc": 0.02}},
  "train_vs_val": {"train": {"pr_auc": 0.91}, "val": {"pr_auc": 0.81}},
  "curves": {"loss": "...", "calibration": "..."}
}
```

- Which metrics appear is decided by rules `METRIC-01`…`METRIC-08`.
- Unsupervised runs may have no holdout; `cv` present only when CV ran (rule `TRAIN-05`).

## Data hash (§6.1, §9)

- `dataset_hash = "sha256:" + sha256(bytes of the versioned Parquet file)`
- Parquet written with pinned writer settings; library version recorded in `meta.json`.
- Used as: cache key component (§8), reproducibility record in reports (§6.10), step-log linkage
  (`dataset_hash_before/after`), stale detection (`stale` set when current hash ≠ hash a tab last saw).
- ⚠ Fallback if byte-hash proves unstable across library versions: content digest =
  `sha256(schema JSON + column-wise value digests)`. Decide at review (open question 5).

## Staleness model (§3)

- Editing tab *k* marks every tab with index > *k* stale (`STALE-01`).
- A stale tab shows a banner with what went stale and prompts a re-run (`STALE-02`).
- A successful re-run of tab *k* clears `stale[k]`; downstream flags remain until re-run (`STALE-03`).
- Gating is orthogonal: a locked tab cannot be edited even if not stale.

## Artifacts and formats (§6.10, §9)

| Artifact | Format | When |
|---|---|---|
| Fitted preprocessing pipeline | joblib (`models/<run_id>/pipeline`) | Phase 3 |
| Model | joblib / framework checkpoint | Phase 4 |
| Portable inference | ONNX | Phase 5 |
| Predictions | Parquet/CSV | Phase 5 |
| Reports | HTML, PDF | Phase 5 |
| Reproducible script | `.py` generated from step log | Phase 5 |

**Security:** only artifacts the app produced are ever loaded; uploads validated and size-limited;
one workspace directory per project (§9).
