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
TAB_ORDER = [
    "data",
    "cleaning",
    "preprocessing",
    "eda",
    "modelling",
    "training",
    "prediction",
    "error_analysis",
    "explainability",
    "outcome",
]
```

## ProjectState (lives in `st.session_state["state"]`)

Single source of truth (§2). Tabs read it; only the owning tab or service writes it.

| Field | Type | Notes |
|---|---|---|
| `project_id` | str | Matches `workspace/projects/<project_id>/` |
| `dataset` | object \| null | `{version, path, data_hash, row_count, loaded_at}` — pointer at the current versioned Parquet |
| `frame` | DataFrame \| null | In-memory copy of the current version; written back as a new version on every step (Phase 2) |
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

- `frame` always mirrors the Parquet file `dataset.path` points at (same `data_hash`).
- `task.learning_type == "unsupervised"` ⟹ `task.target is None`; `task.eval_labels` optional and used only for scoring (§6.1).
- `stale[tab]` is `False` for `data` (nothing upstream of it).
- Downstream tabs never read another tab's local widget values — only `ProjectState`.
- Loading a dataset (op `load`) resets the step log: prior entries describe a dataset that no longer exists.

## Task inference (§6.1)

Supervised: target column required; task type suggested from the target column (rule `TASK-01`)
and accepted/overridden by the user.
- Binary target (2 unique values) → `binary`
- Integer target with few unique values (≤ 20 ⚠ threshold) → `multiclass`
- Numeric target with many unique values → `regression`
- Non-numeric (string/category) target with >2 uniques → `multiclass` ⚠ (design doc silent: string
  class labels such as `"cat"/"dog"/"fish"` are class labels, not numbers)
- Target with <2 distinct values → rejected (`TaskError`)
- Datetime column + user confirmation → `forecasting` with a `time_column` (rule `TASK-02`);
  without confirmation a datetime column is an ordinary feature
- Group column (patient/customer ID) offered to prevent leakage; once set, every split must respect it (rule `SPLIT-07`).

Unsupervised: no target (setting one is rejected, rule `TASK-03`); the user picks the task type
from `clustering`, `dimensionality_reduction`, `anomaly_detection`, `association`; an optional
"evaluation labels" column is held aside and excluded from `feature_columns()` (never fitted).

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
      "params": {"columns": ["age"], "strategy": "median",
                 "version_before": "v00003", "version_after": "v00004"},
      "dataset_hash_before": "sha256:9f2a...",
      "dataset_hash_after": "sha256:c41d...",
      "created_at": "2026-10-06T10:42:11Z"
    }
  ]
}
```

- `op` vocabulary: `load`, `dedupe`, `fix_dtypes`, `drop_rows`, `impute`, `outlier_treatment`,
  `encode`, `scale`, `feature_select`, `split`, `target_transform`, `train`, `tune` (Phase 4),
  `resample` (Phase 6), `window` (Phase 8).
- Steps written by Phase 2 (data + cleaning) also record `params.version_before` and
  `params.version_after`; undo restores the Parquet of `version_before` and removes that entry.
  Only `tab == "cleaning"` entries are undoable — a `load` entry is the start of the log.
- Phase 3 preprocessing steps (`split`, `encode`, `scale`, `target_transform`) do **not** write a
  new dataset version: `params.version_before == params.version_after` and
  `dataset_hash_before == dataset_hash_after` because the data is unchanged — only a fitted
  pipeline artifact is produced. They are not undoable (no dataset restore target).
- Script export (rule `EXPORT-01`) replays entries in order into a standalone Python script.

## Preprocessing artifacts (§6.3)

Preprocessing produces three files under `workspace/projects/<project_id>/preprocessing/`, all
written by `preprocessing_service.run_preprocessing`:

1. `split.json` — the concrete holdout indices plus the rule that chose them:
   ```json
   {
     "strategy": "stratified",
     "rule_id": "SPLIT-03",
     "test_size": 0.2,
     "train": [1, 3, 6, 9, "..."],
     "test": [2, 4, "..."]
   }
   ```
   `strategy` is one of `grouped`, `chronological`, `eval_labels`, `none`, `stratified`, `random`
   (`SPLIT-01`…`SPLIT-07`). Train + test partition each frame row exactly once (no split for
   strategy `none`).
2. `pipeline.joblib` — the fitted, leakage-safe sklearn `ColumnTransformer` pipeline
   (`numeric` imputer + optional scaler; `categorical` one-hot or ordinal encoder with
   `handle_unknown="ignore"` / `unknown_value=-1`). Every fitted step is inside this single
   pipeline and is fitted on the training rows of `split.json` **only** (`PIPE-01`, `ENC-05`,
   `SCALE-04`).
3. `pipeline-meta.json` — human-readable summary written by the service for sidecar readers.
   *(Optional; the step-log entries are the authoritative record.)*

`ProjectState.split.indices_path` points at `split.json`; `ProjectState.pipeline.path` points at
`pipeline.joblib`. Training becomes unlocked once both exist (`GATE-01` → `split_or_pipeline`),
and any preprocessing re-run re-gates it.

EDA reports (review 6.4) are computed on the fly from the in-memory frame; no persisted artifact in
Phase 3.

## Model run metadata — `models/<run_id>/meta.json` (§6.6, §6.10)

Run ids are `run_` + 8 alphanumerics (`RUN_ID_RE`); directories live under
`workspace/projects/<project_id>/models/<run_id>/`.

```json
{
  "run_id": "run_4f8Kz2Mp",
  "model_id": "logistic_regression",
  "task": {"learning_type": "supervised", "task_type": "binary"},
  "params": {"C": 1.0, "penalty": "l2", "solver": "lbfgs"},
  "seed": 42,
  "data_hash": "sha256:c41d...",
  "split": {"strategy": "stratified_kfold", "params": {"n_splits": 5}},
  "library_versions": {"sklearn": "1.5.x"},
  "primary": "accuracy",
  "rule_id": "METRIC-01",
  "mode": "manual",
  "trained_at": "2026-10-06T11:03:52Z"
}
```

This is the **metadata sidecar** read by tabs that do not need the heavy artifact (§8):
leaderboard and gating checks read `meta.json` / `metrics.json` only — never the fitted pipeline
(rule `PERF-01`).

## Metrics — `models/<run_id>/metrics.json` (§6.6)

```json
{
  "model_id": "logistic_regression",
  "task_type": "binary",
  "primary": "accuracy",
  "rule_id": "METRIC-01",
  "mean": 0.81,
  "std": 0.02,
  "train_metric": 0.91,
  "val_metric": 0.81,
  "gap": 0.1,
  "params": {"C": 1.0, "penalty": "l2", "solver": "lbfgs"},
  "folds": [
    {"fold": 0, "accuracy": 0.79, "f1": 0.76, "training_metric": 0.91},
    {"fold": 1, "accuracy": 0.83, "f1": 0.78, "training_metric": 0.90}
  ],
  "metrics": ["accuracy", "f1"],
  "data_hash": "sha256:c41d..."
}
```

- Which metrics appear is decided by rules `METRIC-01`…`METRIC-08`; `primary` is the single
  headline metric (accuracy / pr_auc for classification, rmse / mae for regression, …).
- `mean`/`std` are across folds (CV rule `TRAIN-05`); `gap = train_metric - val_metric` signals
  overfit. The leaderboard (`leaderboard()`) renders rows purely from these sidecars.
- `folds` is the per-fold list consumed by the Training tab's per-fold table.

## Phase 4 training artifact set (§6.6, `TRAIN-06`)

`training_service.train_model` writes exactly three files to `models/<run_id>/`, plus a fourth
only for regression with a target transform:

1. `pipeline.joblib` — the final artifact: a fitted `Pipeline([("preprocess", ColumnTransformer),
   ("model", estimator)])` fit on the official holdout **train** rows of `split.json`. When a
   regression `target_transform` is active the estimator is fit on the transformed target.
2. `meta.json`, 3. `metrics.json` — the sidecars above.
4. `target_transform.joblib` (regression + target transform only) — the fitted transform; the
   Prediction service applies `inverse_transform` to bring predictions back to the original scale.

Training is wrapped by CV: per-fold leak-safe preprocessing is rebuilt on the fold's training rows
only, then the final artifact is re-fit on the full train split. `state.models` entries move from
`status="queued"` to `"done"` (a queued run id is reused by `train_model(run_id=...)`); each run
appends a step-log entry `{op: "train", params: {run_id, model, primary, mean, gap, mode}}`.

Model pipelines are loaded lazily through a bounded LRU cache (`ModelCache`, capacity 2,
rule `PERF-02`): `get_pipeline(workspace, run_id, cache=...)` tracks hit/miss counters and records
evicted keys, so Prediction never re-disk-loads the active model on every interaction.

## Phase 5 delivery artifacts (§6.8–6.10, `ERR-*`, `EXPL-*`, `OUT-01`, `EXPORT-01/02`)

Heavy explainer jobs (SHAP; PDP/ICE) run off the UI thread through a `JobQueue`
(`src/ml_workbench/services/jobs.py`, rule `PERF-05`): `submit(kind, fn)` returns a `JobHandle`
(`job_` + 8 hex chars) with `state ∈ {queued, running, done, failed}`, `progress` and `message`;
`poll` returns the in-place handle, `wait(job_id, timeout)` blocks on the underlying future. The
MVP backs the queue with an in-process `ThreadPoolExecutor(max_workers=2)`; the UI only depends on
submit/poll/wait, so the executor can later be swapped for a process pool or a broker without
changes (`ADR-003`).

- **Error views** (`services/error_service.py`): `error_views(state, workspace, run_id, cache=...)`
  returns an `ErrorReport` with `views` keyed per task type (binary → confusion matrix 2×2, ROC with
  AUC, PR curve, 9-row threshold analysis, 10-bin calibration; multiclass → normalized confusion
  matrix, per-class metrics, most-confused pairs; regression → residuals vs predicted, residual
  histogram, QQ plot, heteroscedasticity Spearman ρ, error by target quantile) plus `worst_n` (≤10
  rows sorted by error) and `segments` (≤8-row summaries by cardinality) (ERR-01/02/03/07).
- **Explainer cache** (`services/explain_service.py`): every `explain_model` writes JSON to
  `projects/<id>/explain/<disk_cache_key(data_hash, model_id, params)>.json` and returns
  `cached=True` on a subsequent hit for the same hash/model/params (EXPL-10/11, PERF-03). The JSON
  mirrors the report: method, background rows, panels, warnings, computed_at. Explain dispatch is by
  `explain_method` with shap optional (EXPL-01..03/05); SHAP/PDP/ICE jobs surface progress via the
  queue.
- **Outcome bundle** (`projects/<id>/outcome/`, OUT-01):
  1. `predictions.csv` — full-sample predictions; a `prediction@threshold` column is added for
     binary models with class probabilities (positive class from the trained estimator).
  2. `threshold.json` — `{threshold, applied}`.
  3. `model_card.json` — spec, task, metrics, params, data hash, seed, split, trained_at, library
     versions, explain method, deliverable list.
  4. `pipeline.joblib` — copy of the run's final artifact, or a fresh pipeline refit **on all data**
     when `refit_on_all=True` (preprocessing choices replayed from the logged step ops;
     `target_transform.joblib` re-emitted for regression).
  Packaging appends a step-log entry `{op: "package", tab: "outcome", params: {run_id, model,
  files}}` and marks downstream tabs stale.
- **Report bundle** (`projects/<id>/reports/`, EXPORT-01/02):
  1. `report.html` — human-readable model report (run, model, task, metrics, threshold, data hash).
  2. `reproduce.py` — a standalone script that copies the project into a throwaway root
     `reproduce-<hash-tail>`, reloads the raw v00001, **functionally replays the recorded cleaning
     steps in log order** (dedupe → dtype fixes → drop rows → impute → outlier treatment), logs the
     replay, persists the replayed frame, asserts the reproduced data hash equals `DATA_HASH`, then
     re-runs preprocessing and trains with the recorded `run_id`/`mode='reproduce'`/`SEED`.
  3. `manifest.json` — pins `data_hash`, `seed`, Python/platform, library versions, the step list,
     the outcome file list, threshold, and the model id/primary/mean.

## Dataset versions and data hash (§6.1, §9)

- Every dataset write creates a new immutable file
  `workspace/projects/<project_id>/datasets/v00001.parquet` — `v` + zero-padded 5 digits, next
  number = highest existing + 1 (ADR-004 layout).
- Written with pinned writer settings (pyarrow, `index=False`, snappy) so identical frames produce
  identical bytes (verified).
- Sidecar `v00001.schema.json` = `{version, data_hash, row_count, loaded_at, writer, schema: [...]}`
  — the schema report (dtypes, missing counts, cardinalities) plus the writer library version is
  readable without loading the Parquet.
- `dataset_hash = "sha256:" + sha256(bytes of the versioned Parquet file)`
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
| Dataset version | Parquet (`datasets/vNNNNN.parquet`) | data load + every cleaning step |
| Schema report sidecar | JSON (`datasets/vNNNNN.schema.json`) | with every dataset version |
| Split indices | JSON (`preprocessing/split.json`) | preprocessing run |
| Fitted preprocessing pipeline | joblib (`preprocessing/pipeline.joblib`) | preprocessing run |
| Fitted model pipeline | joblib (`models/<run_id>/pipeline.joblib`) | Phase 4 (`TRAIN-06`) |
| Model metadata sidecar | JSON (`models/<run_id>/meta.json`) | Phase 4 (`TRAIN-01`) |
| Model metrics sidecar | JSON (`models/<run_id>/metrics.json`) | Phase 4 (`PERF-01`) |
| Target transform | joblib (`models/<run_id>/target_transform.joblib`) | Phase 4 regression + target transform |
| Explainer cache | JSON (`explain/<cache_key>.json`) | Phase 5 (`EXPL-11`, `PERF-03`) |
| Predictions | CSV (`outcome/predictions.csv`) | Phase 5 (`OUT-01`) |
| Decision threshold payload | JSON (`outcome/threshold.json`) | Phase 5 (`OUT-01`) |
| Model card | JSON (`outcome/model_card.json`) | Phase 5 (`OUT-01`) |
| Outcome pipeline | joblib (`outcome/pipeline.joblib`) | Phase 5 (`OUT-01`) |
| HTML report | `reports/report.html` | Phase 5 (`EXPORT-01`) |
| Reproducible script | `reports/reproduce.py` generated from step log | Phase 5 (`EXPORT-01`) |
| Report manifest | `reports/manifest.json` | Phase 5 (`EXPORT-01/02`) |

**Security:** only artifacts the app produced are ever loaded; uploads validated and size-limited;
one workspace directory per project (§9).
