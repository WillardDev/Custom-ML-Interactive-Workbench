# Tab specifications and acceptance criteria

Source: design document §3 (workflow and gating), §6.1–§6.10.
Status: accepted (Phase 0). ⚠ = inferred, confirm at review.

## Workflow and gating matrix (§3)

```
1 Data → 2 Cleaning → 3 Preprocessing → 4 EDA → 5 Modelling → 6 Training
       → 7 Prediction → 8 Error Analysis → 9 Explainability → 10 Outcome
```
Editing any tab marks all downstream tabs stale and prompts a re-run (rules `STALE-*`).

Legend: **R** = editable/required, **view** = readable, **—** = not yet available, **†** = inferred.

| # | Tab | Project loaded | Task set | Cleaned data | Split/pipeline | ≥1 trained model |
|---|---|---|---|---|---|---|
| 1 | Data Insertion | — (upload) | — (sets task) | — | — | — |
| 2 | Data Cleaning | R | R | — | — | — |
| 3 | Data Preprocessing | R | R | R | — | — |
| 4 | Exploratory Data Analysis | R | R | R † | view † | — |
| 5 | Modelling | R | R | R | view | view (compare) |
| 6 | Training | R | R | R | **R (GATE-01)** | — (creates) |
| 7 | Prediction | R | R | R | R | **R (GATE-02)** |
| 8 | Error Analysis | R | R | R | R | **R (GATE-02)** |
| 9 | Model Explainability | R | R | R | R | **R (GATE-02)** |
| 10 | Outcome and Final Prediction | R | R | R | R | **R (GATE-04 ⚠)** |

---

## 6.1 Data Insertion (`data`)

- **Purpose:** get data in and define the task.
- **Inputs:** CSV, Excel, Parquet, SQL connection, sample datasets (§6.1).
- **Outputs:** versioned dataset (Parquet) with schema report; task definition in `ProjectState`.
- **Acceptance criteria:**
  - Upload succeeds for CSV/Excel/Parquet/SQL; file size and type validated (§9 security).
  - Schema report generated: dtypes inferred, missing counts, cardinalities; data hash computed and stored.
  - Dataset version written to `datasets/` with schema report (`ADR-004`).
  - Task selection: supervised requires target; task suggested from target (`TASK-01`), datetime + confirmation → forecasting (`TASK-02`).
  - Group column offered to prevent leakage; once set, enforced everywhere (`SPLIT-07`).
  - Unsupervised: no target; optional evaluation-labels column held aside (`TASK-03`).
- **Rules:** TASK-01, TASK-02, TASK-03, CLEAN-01 (dtypes), PERF-*.
- **Edit effect:** editing invalidates tabs 2–10 (`STALE-01`).

## 6.2 Data Cleaning (`cleaning`)

- **Purpose:** duplicates, dtypes, missing values, outliers, class balance — with undo.
- **Inputs:** loaded dataset + task + (optionally) model family from previous selection.
- **Outputs:** cleaned dataset version; step-log entries; before/after counts.
- **Acceptance criteria:**
  - Decision flow implemented as rules: duplicates/dtypes always (`CLEAN-01`); drop missing-target rows, never impute target (`CLEAN-02`); NaN policy from `handles_nan` (`CLEAN-03`); outlier policy from task/family (`CLEAN-04a/b/c`); class balance + minority <20% flag (`CLEAN-05`).
  - Every action appended to step log with before/after hashes; undo restores counts (`CLEAN-06`).
  - Completion screen shows before/after row and column counts.
- **Rules:** CLEAN-01…CLEAN-06.
- **Edit effect:** invalidates tabs 3–10.

## 6.3 Data Preprocessing (`preprocessing`)

- **Purpose:** build the fitted, leakage-safe pipeline: split, encoding, scaling, features.
- **Inputs:** cleaned dataset, task, model family (registry flags).
- **Outputs:** split definition; fitted preprocessing pipeline artifact; step-log entries.
- **Acceptance criteria:**
  - Split strategy chosen by rules (`SPLIT-01`…`SPLIT-09`), including group/time enforcement.
  - Encoding by family (`ENC-01`…`ENC-05`); scaling by `needs_scaling` (`SCALE-01`…`SCALE-04`); neural specifics (`FEAT-01`, `SCALE-03`).
  - Feature selection menu by supervision type (`FEAT-02a/b`), fitted inside folds (`FEAT-03`).
  - Imbalance handling offered for classification (`FEAT-04a/b`), SMOTE rejected for regression (`WARN-03`).
  - PCA suggested only per `DR-01`/`DR-02`; t-SNE/UMAP never pipeline inputs (`DR-03`).
  - Target transform offered per `TT-01`.
  - Pipeline fitted on training folds only — leakage test passes (`PIPE-01`).
- **Rules:** SPLIT-*, ENC-*, SCALE-*, FEAT-*, DR-*, TT-01, PIPE-01.
- **Edit effect:** invalidates tabs 4–10; fitted-pipeline flag change also re-gates Training (`GATE-01`).

## 6.4 Exploratory Data Analysis (`eda`)

- **Purpose:** task-adaptive views over the data.
- **Inputs:** cleaned dataset (preprocessing state for split-aware views †).
- **Outputs:** interactive plots/tests; optional cached profile.
- **Acceptance criteria:**
  - Base views for all tasks (`EDA-01`).
  - Task-specific views: classification (`EDA-02`), regression (`EDA-03`, skew hint `HINT-02`), time series (`EDA-04`), clustering (`EDA-05`), anomaly (`EDA-06`), dim. reduction (`EDA-07`), association (`EDA-08`).
  - Large data sampled per `EDA-09`/`PERF-04`.
- **Rules:** EDA-01…EDA-09, HINT-02, PERF-04.
- **Edit effect:** views are read-only; upstream edits mark it stale.

## 6.5 Modelling (`modelling`)

- **Purpose:** pick models and configure hyperparameters.
- **Inputs:** task + registry (flags).
- **Outputs:** selected model(s) and parameter sets queued for training; Auto-compare/Manual/Auto+tuning mode.
- **Acceptance criteria:**
  - Registry filtered by task (`MODEL-01`); only models with `enabled_phase ≤ current phase` shown (registry policy).
  - Mode selection: Auto-compare, Manual, Auto with tuning.
  - Hyperparameter form generated from registry schema (`MODEL-02`); conditional prompts per `MODEL-03`; incompatible options not expressible (`MODEL-04`).
  - Neural builder with presets/advanced fields shown only for `family: neural` (Phase 7).
  - Small-data boosting-baseline hint visible (`HINT-01`); small-data neural warning (`WARN-06`).
- **Rules:** MODEL-01…MODEL-04, HINT-01, WARN-06, WARN-07.
- **Edit effect:** invalidates tabs 6–10.

## 6.6 Training (`training`)

- **Purpose:** fit/tune models, report leaderboard, persist artifacts.
- **Inputs:** split + pipeline + model selection; background job queue.
- **Outputs:** trained artifacts (`pipeline`, `meta.json`, `metrics.json`); leaderboard; `ProjectState.models` entries.
- **Acceptance criteria:**
  - Locked until split or pipeline exists (`GATE-01`).
  - Seed, config, data hash logged on start (`TRAIN-01`).
  - Validation strategy by task (`SPLIT-08`); boosting early stopping (`TRAIN-02`); neural loop (Phase 7, `TRAIN-03`); Optuna/random search with pruning when tuning (`TRAIN-04`).
  - Leaderboard: fold mean/std + train-vs-val gap (`TRAIN-05`).
  - Artifacts saved and metadata sidecar readable without loading model (`TRAIN-06`, `PERF-01`).
  - GPU/CPU routing (`TRAIN-07`); jobs off the UI thread with live progress (`PERF-05`).
  - Metrics chosen per task (`METRIC-01`…`METRIC-08`).
- **Rules:** GATE-01, TRAIN-01…TRAIN-07, METRIC-01…METRIC-08, SPLIT-08, PERF-01/02/05.
- **Edit effect:** retraining invalidates tabs 7–10 (and leaves 6 current).

## 6.7 Prediction (`prediction`)

- **Purpose:** score data with the active model.
- **Inputs:** active model + new rows (form/batch) or test set.
- **Outputs:** predictions file, threshold/interval settings stored for Outcome.
- **Acceptance criteria:**
  - Locked until ≥1 trained model (`GATE-02`).
  - Task capabilities per `PRED-01`…`PRED-07`, including clustering surrogate fallback (`PRED-04`), t-SNE blocked (`PRED-05`, `WARN-01`), LOF novelty mode (`PRED-06`).
- **Rules:** GATE-02, PRED-01…PRED-07, WARN-01, WARN-02, WARN-07.
- **Edit effect:** read-only w.r.t. training; threshold choice feeds Outcome (`OUT-01`).

## 6.8 Error Analysis (`error_analysis`)

- **Purpose:** understand where and why the model fails.
- **Inputs:** active model + validation/test predictions.
- **Outputs:** error views; worst-N rows / segment slices referenced in reports.
- **Acceptance criteria:**
  - Locked until ≥1 trained model (`GATE-02`).
  - Task-specific views per `ERR-01`…`ERR-06`; common tools (`ERR-07`); neural diagnostics (`ERR-08`).
- **Rules:** GATE-02, ERR-01…ERR-08.
- **Edit effect:** read-only.

## 6.9 Model Explainability (`explainability`)

- **Purpose:** global and local explanations dispatched by `explain_method`.
- **Inputs:** active model + data (sampled background where needed).
- **Outputs:** explanation plots; cached SHAP arrays.
- **Acceptance criteria:**
  - Locked until ≥1 trained model (`GATE-02`).
  - Panels dispatched by `explain_method` and task per `EXPL-01`…`EXPL-09`.
  - Auto-warnings shown (`EXPL-10`); SHAP as background job with disk cache (`EXPL-11`, `PERF-03/05`).
- **Rules:** GATE-02, EXPL-01…EXPL-11, PERF-03, PERF-05.
- **Edit effect:** read-only.

## 6.10 Outcome and Final Prediction (`outcome`)

- **Purpose:** package the result: deliverables, report, reproducible script.
- **Inputs:** everything upstream (dataset, steps, active model, threshold/interval).
- **Outputs:** per-task deliverables (`OUT-01`…`OUT-05`); HTML/PDF report; script from step log; native + ONNX export (`EXPORT-01/02`).
- **Acceptance criteria:**
  - Locked without a trained model (`GATE-04 ⚠`).
  - Task-appropriate deliverables present.
  - Report bundle complete: report, script, data hash, library versions, seed (`EXPORT-01`).
  - Export parity: native artifact loads back and matches ONNX predictions within tolerance (`EXPORT-02`, Phase 5).
- **Rules:** GATE-04, OUT-01…OUT-05, EXPORT-01, EXPORT-02.
- **Edit effect:** read-only; re-run when any upstream tab changes (stale).

---

## Navigation and UI expectations (§3)

- Sidebar (or top nav) lists the 10 tabs in order; navigation labels are pure (number + title) so
  they never depend on mutable state.
- Lock and stale markers are listed in a "Tab status" block directly under the navigation;
  selecting a locked tab shows a lock panel stating which rule locked it and which requirements
  are missing (`GATE-*`).
- A stale tab shows a banner with what went stale and offers re-run (`STALE-02`).
- Only the active tab performs work (§8); background-job progress is visible app-wide.
