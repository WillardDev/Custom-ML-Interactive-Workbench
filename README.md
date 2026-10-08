# ML Workbench

An interactive, notebook-free dashboard that guides a user through the full machine learning
lifecycle: data in, model out, with explanation and reporting.

- **Supervised:** classification (binary, multiclass, multilabel), regression, time series forecasting
- **Unsupervised:** clustering, dimensionality reduction, anomaly detection, association rules
- **Models:** classical (scikit-learn, XGBoost, LightGBM, CatBoost) and neural networks (PyTorch)

Design source: *ML Workbench: Software Documentation v1.0 — Architecture and technique specification*.

## Design principles

1. **Model-aware** — every tab adapts to the task and the model through a capability registry, never hard-coded branches.
2. **Leakage-safe** — every fitted step lives inside a pipeline fitted on training data only.
3. **Reproducible** — every action is logged, exportable as a script, and versioned with seed and data hash.
4. **Responsive** — heavy work runs in background workers, models load lazily and are cached.

## Status

| Phase | State |
|---|---|
| Phase 0 — Design & contracts (no code) | **complete** |
| Phase 1 — Foundation (scaffolding) | **complete — exit criteria met** |
| Phase 2 — Walking skeleton | **complete — exit criteria met** |
| Phase 3 — Preprocessing + EDA | **complete — exit criteria met** |
| Phase 4 — Modelling, Training, Prediction | **complete — exit criteria met** |
| Phase 5 — Error Analysis, Explainability, Outcome | **complete — exit criteria met** |
| Phase 6 — Unsupervised (Clustering, Dim-Reduction, Anomaly) | **complete — exit criteria met** |
| Phase 7 — Neural networks + anomaly scoring | **complete — exit criteria met** |
| Phase 8 — Time series + association rules | **complete — exit criteria met** |
| Phase 9 — Production hardening | not started |

Current state: `docs/` and `registry/` hold the Phase 0 spec (124 rules, tab matrix, contracts);
`src/` holds the foundation (project state, rules engine for gating/staleness, registry loader,
Streamlit shell). Phases 2–6 are live: Data Insertion, Data Cleaning, Data Preprocessing, EDA,
Modelling, Training, Prediction, Error Analysis, Model Explainability, and Outcome tabs with
versioned Parquet storage, schema reports, the step log with undo, task inference (TASK-01..03),
cleaning rules (CLEAN-01..06), leakage-safe split/encode/scale preprocessing (SPLIT-01..09, ENC-*,
SCALE-*, FEAT-*, DR-*, TT-01, PIPE-01), task-adaptive EDA (EDA-01..03, EDA-09, HINT-02), the
Phase 4 modelling/training/prediction stack (MODEL-01..04, HINT-01, TRAIN-01..07, METRIC-01..03,
PRED-01..05, PERF-01..02, WARN-01..07), the Phase 5 error/explainability/outcome stack
(ERR-01..03/07, EXPL-01..03/05/10/11, OUT-01, EXPORT-01/02, PERF-05, GATE-02/04), and the Phase 6
unsupervised stack — KMeans clustering, PCA dimensionality reduction and Isolation-Forest anomaly
detection with holdout scoring (METRIC-05..07), unsupervised EDA (EDA-05/06/07), error views
(ERR-04/05/06), explanations (EXPL-06/07/08) and outcome packaging (OUT-02/03/04), and the Phase 7
neural/anomaly stack — MLP classifier + regressor with a per-epoch early-stopping training loop
(TRAIN-03), anomaly scoring with threshold flagging and LOF novelty mode (PRED-06), neural error
views (ERR-08) and Integrated-Gradients explanations (EXPL-04) — and the Phase 8
forecasting/association stack — naive and seasonal-naive baselines, lag-feature boosting,
pure-Python apriori rule mining, forecast metrics with residual-quantile bands and expanding
backtest (METRIC-04/08, PRED-03/07), time-series and association EDA (EDA-04/08), rule-network and
lift-vs-confidence explanations (EXPL-09) and rules-table outcome packaging (OUT-05) — all checks
green (ruff, mypy strict, rule-test sync, pytest).

## MVP scope (delivered in Phases 2–8)

- Tasks: binary and multiclass classification, regression, forecasting, clustering, dimensionality
  reduction, anomaly detection and association rules on tabular data
- Models: linear baselines and gradient boosting (scikit-learn, XGBoost, LightGBM), KMeans, PCA,
  Isolation Forest, MLP (classification/regression), Local Outlier Factor, naive and
  seasonal-naive forecasters, lag-feature boosting, pure-Python apriori
- All 10 workflow tabs present end-to-end: state, gating, staleness, step log, SHAP, report and script export
- Deferred: autoencoders, auto-compare, transformer time-series models (deeper backends),
  production hardening (Phase 9)

## Phases

### Pre-implementation — completed before any feature code

#### Phase 0 — Design & contracts (no code) ✅

- [x] Lock MVP scope (tasks, models, tab priorities)
- [x] Architecture decisions recorded as ADRs → [`docs/architecture.md`](docs/architecture.md)
- [x] Data contracts: project state, `steps.json`, `meta.json` / `metrics.json`, data hash, stale-flag propagation → [`docs/contracts.md`](docs/contracts.md)
- [x] Capability registry: flag reference, full model catalog, YAML schema with MVP entries → [`docs/registry.md`](docs/registry.md), [`registry/models.yaml`](registry/models.yaml)
- [x] Rules catalog: every rule from the design doc given an ID, a given/when/then statement, and a reserved pytest name → [`docs/rules.md`](docs/rules.md)
- [x] Tab acceptance criteria and gating matrix → [`docs/tabs.md`](docs/tabs.md)

**Exit criteria:** reviewed spec; every rule has an ID and a named test case. → *met (approved to proceed)*

#### Phase 1 — Foundation (scaffolding only) ✅

- [x] Repo layout mirroring the service layer; pinned dependencies (`pyproject.toml`)
- [x] ruff + mypy + pytest configured; CI pipeline (`.github/workflows/ci.yml`)
- [x] Sample datasets checked in for tests and demos (`data/samples/`, regenerable via `scripts/make_sample_data.py`)
- [x] App shell: 10 tab stubs, navigation, working gating driven by `docs/rules.md`
- [x] Rules-engine skeleton with the Phase 0 test names wired up (124 tests: 8 real for GATE/STALE,
  116 pending stubs kept in sync by `scripts/scaffold_rule_tests.py --check`)

**Exit criteria:** `streamlit run` boots, gated tabs behave correctly, CI green. → *met locally;
CI runs on first push*

#### Phase 2 — Walking skeleton ✅

- [x] Workspace storage per ADR-004: versioned Parquet datasets (`vNNNNN.parquet`) + schema sidecars,
      `steps.json` step log, models/cache/reports dirs (`src/ml_workbench/services/workspace.py`)
- [x] Data Insertion tab: sample / upload (CSV, TSV, Excel, Parquet) / SQLite, schema report,
      hashing, dataset versioning — resets task/split/models and propagates staleness
      (`src/ml_workbench/services/data_service.py`, `src/ml_workbench/ui/data_tab.py`)
- [x] Task inference rules: `suggest_task_type` + `build_task` with column validation (TASK-01..03)
- [x] Cleaning tab: duplicates/dtypes, missing-target handling, imputation (6 strategies, target never
      imputed), model-aware outlier decision (CLEAN-04a/b/c), class-balance warning (CLEAN-06), step
      log with per-step dataset versions and undo (CLEAN-01)
      (`src/ml_workbench/rules/cleaning.py`, `src/ml_workbench/services/cleaning_service.py`,
      `src/ml_workbench/ui/cleaning_tab.py`)
- [x] App shell rewired to the real tabs with gating and stale banners; rule coverage extended
      (TASK-01..03, CLEAN-01..06))

**Exit criteria:** Data → versioned cleaned Parquet; stale flags propagate. → *met: AppTest covers
the real flow (load sample → set task → clean → tabs unlock / stale on re-edit / undo)*

#### Phase 3 — Preprocessing + EDA ✅

- [x] Split decision rules: `choose_split` / `cv_strategy` / defaults (SPLIT-01..09)
      (`src/ml_workbench/rules/split.py`)
- [x] Preprocessing decision rules: encoding by family (ENC-01..05), scaling by `needs_scaling`
      (SCALE-01..04), feature-selection/imbalance menus (FEAT-01..04c), PCA suggestion (DR-01..03),
      target transforms (TT-01), leak-safety (PIPE-01) (`src/ml_workbench/rules/preprocessing.py`)
- [x] EDA decision rules: base + task-specific view plans (EDA-01..03, EDA-08/09 stubbed), skew
      hint (HINT-02), large-frame sampling (EDA-09) (`src/ml_workbench/rules/eda.py`)
- [x] Preprocessing service: materializes the split, fits a leak-safe scalar/pipeline on the
      training fold only, persists `preprocessing/split.json` + `pipeline.joblib`, appends
      step-log ops (`split`/`encode`/`scale`/`target_transform`), re-gates Training
      (`src/ml_workbench/services/preprocessing_service.py`, `preprocessing/` dir in workspace)
- [x] EDA service: shape/dtypes/missing, Pearson/Spearman + Cramér's V (numpy-only), class balance
      + chi-square, target histogram + skew hint (`src/ml_workbench/services/eda_service.py`)
- [x] Preprocessing + EDA tabs wired into `app.py` (`src/ml_workbench/ui/preprocessing_tab.py`,
      `src/ml_workbench/ui/eda_tab.py`)
- [x] Rule tests implemented (SPLIT-01..09, ENC/SCALE/FEAT/DR/TT/PIPE, EDA-01..03/09, HINT-02) +
      service/app tests (leak-safety: scaler fitted on train split only)

**Exit criteria:** leakage-safety tests pass; EDA adapts to task. → *met: `test_run_preprocessing_builds_leak_safe_pipeline`
proves the scaler statistic equals the train-fold value; `eda_report` returns task-specific views.*

#### Phase 4 — Modelling, Training, Prediction ✅

- [x] Model registry gated by **phase budget and installed libraries**: `enabled_models(task, phase)`
      filters `enabled_phase <= CURRENT_PHASE`, `available_models` also requires the model's library to be
      importable (sklearn present; xgboost/lightgbm/catboost/mlp filtered out) (MODEL-01)
- [x] Modelling rules: hyperparameter forms generated from the registry schema with
      `visible_when` flag/task conditions (MODEL-02), capability/condition prompts (MODEL-03),
      off-schema option rejection (MODEL-04), small-data boosting hint (HINT-01)
      (`src/ml_workbench/rules/modelling.py`)
- [x] Metric plans by task + data shape: balanced binary → accuracy/F1 (METRIC-01), imbalanced →
      PR-AUC/macro-F1/MCC/balanced-accuracy (METRIC-02), regression → RMSE or MAE by target skew
      (METRIC-03) (`src/ml_workbench/rules/metrics.py`)
- [x] Training rules: log fields with seed + data hash (TRAIN-01), boosting early stopping (TRAIN-02),
      pruned random-search tuning (TRAIN-04), CV leaderboard stats mean/std/gap (TRAIN-05),
      artifact sidecars (TRAIN-06), CPU/GPU job routing (TRAIN-07)
      (`src/ml_workbench/rules/training.py`)
- [x] Prediction rules: adjustable decision threshold + calibration for binary (PRED-01/02),
      surrogate requirement for non-predicting models (PRED-04/05)
      (`src/ml_workbench/rules/prediction.py`)
- [x] Performance rules: leaderboard reads sidecars only (PERF-01), bounding LRU model cache (PERF-02)
      (`src/ml_workbench/rules/performance.py`)
- [x] Training service: 7 sklearn estimators, per-fold leak-safe CV, final artifact fitted on the
      holdout train, `tune_hyperparameters` with reduced-budget probes + pruning, regression target
      transform saved alongside the pipeline (`src/ml_workbench/services/training_service.py`,
      `services/model_cache.py`)
- [x] Prediction service: lazy pipeline load through the cache, single-row / batch / test-set
      prediction, threshold + calibration evaluation (`src/ml_workbench/services/prediction_service.py`)
- [x] Modelling / Training / Prediction tabs wired into `app.py`: model forms, job queue
      (`st.session_state["training_queue"]`), leaderboard with per-fold details, prediction UI
      (`src/ml_workbench/ui/modelling_tab.py`, `ui/training_tab.py`, `ui/prediction_tab.py`)
- [x] Rule tests implemented (MODEL-01..04, HINT-01, TRAIN-01..07, METRIC-01..03, PRED-01..05,
      PERF-01..04, WARN-01..07) + service tests (artifacts/sidecars, LRU eviction, threshold/calibration)
      + an AppTest driving modelling → queue → training → prediction

**Exit criteria:** train → leaderboard → predict end-to-end on tabular classification and regression;
LRU lazy-load works. → *met: `test_phase4_train_and_predict_end_to_end` queues a job in the Modelling tab,
trains it in the Training tab, and predicts in the Prediction tab; `test_get_pipeline_uses_lru_cache_and_evicts`
proves capacity-2 eviction.*

#### Phase 5 — Error Analysis, Explainability, Outcome ✅

- [x] Error analysis views per task (ERR-01..03): binary confusion/ROC/PR/threshold/calibration,
      multiclass normalized matrix + per-class + most-confused, regression residual/QQ/heteroscedastic
      views; shared model-error tools (ERR-07) with worst-N rows and segment summaries
      (`src/ml_workbench/rules/error_analysis.py`, `services/error_service.py`, `ui/error_analysis_tab.py`)
- [x] Explainability dispatch by `explain_method` (EXPL-01..03/05): linear coefficients + local
      attribution, tree feature importances (tree SHAP when the optional `shap` extra is installed),
      permutation importance fallback, PDP/ICE panels, background-sample + spread warnings
      (`src/ml_workbench/rules/explainability.py`, `services/explain_service.py`, `ui/explainability_tab.py`)
- [x] Explain results cached by data hash + model + params in the workspace (EXPL-10/11, PERF-03 via
      `disk_cache_key`); SHAP and PDP/ICE run as background jobs through `JobQueue` (PERF-05, EXPL-10)
      (`src/ml_workbench/services/jobs.py` — submit/poll/wait interface behind a thread pool)
- [x] Outcome bundling (OUT-01): full-sample predictions, decision threshold payload, model card,
      and pipeline copy or refit-on-all; outcome step written to the step log
      (`src/ml_workbench/rules/outcome.py`, `services/outcome_service.py`, `ui/outcome_tab.py`)
- [x] Report + script export (EXPORT-01/02): HTML report, JSON manifest pinning data hash/seed/library
      versions, and `reproduce.py` that clones the project into a throwaway root and replays the
      recorded cleaning steps functionally, verifying the reproduced data hash before refitting
      (`src/ml_workbench/services/report_service.py`)
- [x] Rule tests implemented (ERR-01/02/03/07, EXPL-01/02/03/05/10/11, OUT-01, EXPORT-01/02,
      PERF-03/05) + service tests (error views per task, explain caching + job round-trip, outcome
      deliverables, report manifest/script) + an AppTest driving error → explain → outcome for the
      full 10-tab flow

**Exit criteria:** full 10-tab flow on sample data with a reproducible script; rule coverage green.
→ *met: `test_phase5_error_explain_outcome_end_to_end` computes error views, runs the explainer as a
background job, packages the outcome and writes the report bundle; `scaffold_rule_tests --check`
stays in sync.*

#### Phase 6 — Unsupervised: Clustering, Dim-Reduction, Anomaly ✅

- [x] Unsupervised task definitions (TASK-03): clustering, dimensionality reduction, anomaly
      detection registered 1:1 into the registry and gated by `learning_type=unsupervised`
      (`src/ml_workbench/rules/task.py`, `registry/models.yaml`)
- [x] Phase 6 models: KMeans (clustering, `has_predict`, surrogate explanations), PCA
      (dimensionality-reduction, `has_transform`, loadings explanations), Isolation Forest (anomaly,
      `has_predict`, tree importance/deviations) — buildable, trappable, trainable in the
      Training tab, and scoreable on holdouts (METRIC-05..07)
      (`src/ml_workbench/services/training_service.py`)
- [x] Unsupervised splits (SPLIT-05/06): stratified holdout when evaluation labels are set, one-step
      no-split otherwise (`src/ml_workbench/services/preprocessing_service.py`)
- [x] Unsupervised EDA (EDA-05/06/07): Hopkins statistic, PCA preview + scree plot (clustering),
      correlation clusters + VIF (dim-reduction), z-score/IQR + Mahalanobis flags (anomaly)
      (`src/ml_workbench/services/eda_service.py`, `ui/eda_tab.py`)
- [x] Unsupervised prediction: `pc1..pcN` projections for transformers, cluster/flag assignments
      otherwise, test-evaluation scores (silhouette/D-B/Calinski-H + ARI/NMI, explained
      variance/reconstruction/trustworthiness, ROC/PR-AUC)
      (`src/ml_workbench/services/prediction_service.py`, `ui/prediction_tab.py`)
- [x] Unsupervised error views (ERR-04/05/06): per-sample silhouette + low-score points + cluster
      size imbalance, reconstruction error per row + poorly embedded points, anomaly score
      distribution + top flagged + FP/FN vs labels
      (`src/ml_workbench/rules/error_analysis.py`, `services/error_service.py`, `ui/error_analysis_tab.py`)
- [x] Unsupervised explanations (EXPL-06/07/08): cluster centroids + per-feature F-ratios + surrogate
      tree + personas; PCA loadings + variance + biplot + per-feature reconstruction error; anomaly
      importances + flagged-row feature deviations (`src/ml_workbench/services/explain_service.py`)
- [x] Unsupervised outcome packaging (OUT-02/03/04): `clusters.csv` += profiles + personas,
      `embeddings.csv` += summary.json, `flagged.csv` += score distribution + threshold, each with a
      pipeline copy and model card (`src/ml_workbench/services/outcome_service.py`, `ui/outcome_tab.py`)
- [x] Rule tests implemented (METRIC-05/06/07, ERR-04/05/06, EXPL-06/07/08, OUT-02/03/04, EDA-05/06/07)
      + service tests per flow (`tests/test_phase6_unsupervised.py`) against
      `data/samples/clustering.csv` and a synthetic-labeled anomaly sample; unlabeled flows raise a
      clear no-test-split error

**Exit criteria:** clustering, dim-reduction and anomaly runs on sample data with explanations,
error views and packaged outcomes; rule coverage green.
→ *met: `test_phase6_unsupervised.py` drives train → evaluate → predict → error → explain → outcome
for all three task types; `scaffold_rule_tests --check` stays in sync.*

#### Phase 7 — Neural networks + anomaly scoring ✅

- [x] Neural builder presets (`TRAIN-03`): registry-driven `family: neural` form (preset → layer
      widths, activation, optimizer, learning rate, batch size, weight decay, epochs, patience)
      (`registry/models.yaml`, `ui/modelling_tab.py`)
- [x] Neural training loop (`TRAIN-03`): per-epoch mini-batch fits with warm start, validation-loss
      early stopping, best-epoch checkpoint restore, loss curves recorded in
      `metrics.json["neural"]` (`src/ml_workbench/services/training_service.py`)
- [x] MLP classifier + regressor (`enabled_phase: 7`): probability columns for classification only;
      RMSE/MAE/R² for regression (`src/ml_workbench/services/prediction_service.py`)
- [x] Anomaly scoring (PRED-06): `predict_frame` scores every row (`score`, higher = more anomalous)
      and flags at the plan threshold; Prediction tab gets a threshold slider + flagged-rows table;
      LOF trains in novelty mode (`novelty: true`) so it can score rows after fit
- [x] Anomaly sign contract: error/outcome/explain services all use the shared negated decision
      (higher = more anomalous), matching the OUT-04 contract in `docs/contracts.md`
- [x] Neural error views (`ERR-08`): learning curve, overfitting diagnostics (final gap, best
      epoch, early stop) and per-epoch metrics appended for neural runs on any task
- [x] Gradient explanations (`EXPL-04`): `explain_method: gradient` → Integrated Gradients over
      analytic MLP backprop (zero baseline, 32 steps) with global attributions + first-row local
      panel; Deep/GradientExplainer SHAP and transformer attention maps noted as deep-backend work
- [x] Rule tests implemented (TRAIN-03, PRED-06, ERR-08, EXPL-04) + service tests
      (`tests/test_phase7_neural_anomaly.py`): neural classification/regression flows, anomaly
      score/flag direction vs labels, LOF novelty scoring

**Exit criteria:** an MLP trains, evaluates, errors and explains end-to-end; anomaly rows score and
flag at an adjustable threshold; rule coverage green.
→ *met: `test_phase7_neural_anomaly.py` drives train → evaluate → predict → error → explain for
neural runs and anomaly/LOF scoring; `scaffold_rule_tests --check` stays in sync.*

#### Phase 8 — Time series + association rules ✅

- [x] Forecast baselines (`enabled_phase: 8`): naive (last-value), seasonal-naive (period-aware)
      and lag-feature boosting (HistGradientBoosting on lag windows) — all `library: ml_workbench`,
      no new dependencies (`src/ml_workbench/services/training_service.py`, `registry/models.yaml`)
- [x] Forecast metrics (`METRIC-04`): MAE, RMSE, sMAPE and MASE (scaled by the holdout's own naive
      diffs) with the expanding-window one-block-ahead backtest and residual-quantile bands
      (`PRED-03`) recorded in `metrics.json["forecast"]`
- [x] Forecast horizon (PRED-03): Prediction tab slider (1–48) → `forecast_frame` point forecast
      with 90% residual-quantile bands (`lower = point + q05`, `upper = point + q95`) plus the
      backtest fold table
- [x] Association mining (`METRIC-08`): pure-Python apriori — item frequencies, frequent itemsets
      (max length 3) and rules filtered by support/confidence/lift
      (`src/ml_workbench/rules/association.py`)
- [x] Basket recommendations (PRED-07): `recommend_items` ranks consequents by lift × confidence
      for a given basket, excluding items already held; exposed via `predict_frame`
- [x] Association EDA (`EDA-08`) and time-series EDA (`EDA-04`): item frequency + basket-size
      distributions; moving-average decomposition, ACF/PACF (Durbin–Levinson), rolling statistics
      and an ADF-lite stationarity check (`src/ml_workbench/services/eda_service.py`)
- [x] Association explanations (`EXPL-09`): rule network, lift-vs-confidence scatter and the mined
      rule table; forecasting explains via the target history trace
- [x] Association outcome (`OUT-05`): `outcome/rules.csv` (antecedent/consequent joined with ` + `)
      alongside the refit pipeline and model card
- [x] Rule tests implemented (METRIC-04/08, PRED-03/07, EDA-04/08, EXPL-09, OUT-05) + service tests
      (`tests/test_phase8_forecasting_association.py`): forecasting and association flows with the
      `data/samples/forecasting.csv` and `data/samples/association.csv` samples

**Exit criteria:** a forecasting model trains, evaluates, forecasts with bands and explains
end-to-end; apriori mines rules, recommends baskets, explains and packages the rules table; rule
coverage green.
→ *met: `test_phase8_forecasting_association.py` drives train → evaluate → forecast / recommend →
explain → outcome for both families; `scaffold_rule_tests --check` stays in sync (124 tests).*

### Implementation

| # | Phase | Status | Content | Exit criteria |
|---|---|---|---|---|
| 2 | Walking skeleton | ✅ | Project state manager, Data Insertion (schema report, hashing), Cleaning decision tree, step log + undo | Data → versioned cleaned Parquet; stale flags propagate |
| 3 | Preprocessing + EDA | ✅ | Split/encoding/scaling rules from registry, leakage-safe pipeline build, task-adaptive EDA views | Leakage-safety tests pass; EDA adapts to task |
| 4 | Modelling, Training, Prediction | ✅ | Registry filtering, hyperparameter forms, CV/tuning, leaderboard, artifacts + metadata sidecars, prediction tab | Train → leaderboard → predict on tabular cls/reg; LRU lazy-load works |
| 5 | Error Analysis, Explainability, Outcome | ✅ | Task-specific error views, SHAP as background jobs, report + script export + model card | **MVP complete:** full 10-tab flow on sample data with reproducible script |
| 6 | Unsupervised | ✅ | KMeans clustering, PCA dimensionality reduction, Isolation-Forest anomaly detection: splits, EDA, holdout scoring, error views, explanations, outcomes | Clustering/dim-reduction/anomaly runs with explanations + packaged outcomes on sample data |
| 7 | Neural networks | ✅ | MLP classifier + regressor, early-stopping loop, anomaly scoring/LOF novelty, Integrated Gradients | Neural runs train → evaluate → error → explain end-to-end |
| 8 | Time series + association | ✅ | Naive/seasonal-naive/lag boosting forecasters, pure-Python apriori, forecast metrics + bands + backtest, basket recommendations, ts/association EDA + explanations + outcomes | Forecasting and association flows complete end-to-end; rule coverage green |
| 9 | Production (optional) | — | Background job queue (Redis/Celery), multi-user projects, ONNX serving, monitoring | Design doc Phase 5 delivered |

## Repository layout

```
ml_workbench/
├── README.md               this file: overview + phase plan
├── pyproject.toml          pinned dependencies, ruff/mypy/pytest config
├── .github/workflows/ci.yml  CI: ruff, mypy, rule sync check, pytest
├── docs/
│   ├── architecture.md     ADRs, layer diagram, storage layout, performance, security
│   ├── contracts.md        project state, step log, run metadata, data hash, staleness
│   ├── registry.md         capability flags, model catalog, YAML schema
│   ├── rules.md            rules catalog — every rule with an ID and test name
│   └── tabs.md             per-tab inputs/outputs/acceptance criteria + gating matrix
├── registry/
│   └── models.yaml         model registry data: capability flags + hyperparameter schemas
├── data/samples/           deterministic sample CSVs for tests and demos
├── scripts/
│   ├── make_sample_data.py regenerate sample datasets
│   └── scaffold_rule_tests.py  generate/sync rule stub tests from docs/rules.md (--check in CI)
├── src/ml_workbench/
│   ├── tabs.py             tab metadata (id, order, design section, phase)
│   ├── state.py            ProjectState and related dataclasses (docs/contracts.md)
│   ├── registry.py         registry loader + validation + task/phase filtering
│   ├── rules/              pure rules: gating (GATE-*), staleness (STALE-*), task (TASK-*), cleaning (CLEAN-*),
│   │                       split (SPLIT-*), preprocessing (ENC/SCALE/FEAT/DR/TT/PIPE), eda (EDA-*, HINT-*),
│   │                       modelling (MODEL-*, HINT-01), metrics (METRIC-*), training (TRAIN-*), prediction
│   │                       (PRED-*), warnings (WARN-*), performance (PERF-*)
│   ├── services/           service layer: workspace (ADR-004 storage), data_service, cleaning_service,
│   │                       preprocessing_service, eda_service, training_service, prediction_service,
│   │                       model_cache (bounded LRU)
│   ├── ui/                 renderers: data_tab (insert + task), cleaning_tab (options + step log),
│   │                       preprocessing_tab (split/encode/scale/build), eda_tab (views),
│   │                       modelling_tab (model forms + queue), training_tab (leaderboard + runs),
│   │                       prediction_tab (test eval + single/batch prediction)
│   └── app.py              Streamlit shell: navigation, gating, stale banners, tab renderers
└── tests/
    ├── rules/              one file per rules.md section (124 rule tests)
    ├── test_app_shell.py   Streamlit AppTest: boot, locking, staleness end-to-end
    ├── test_rule_coverage.py  docs ↔ tests ↔ engine consistency
    ├── test_registry.py    registry schema and filtering
    ├── test_tabs.py        tab metadata ↔ docs/contracts.md
    └── test_samples.py     sample dataset integrity
```

## Open questions for review

Items marked ⚠ in the docs are inferred (the design doc is silent) and need a decision at review:

1. **Outcome tab gating** — assumed to require ≥1 trained model (doc lists only Prediction/Error/Explain).
2. **EDA gating** — assumed to require a cleaned dataset only (chain shows it after Preprocessing).
3. **Split defaults** — assumed 80% holdout / 5-fold CV; doc specifies strategies but not ratios.
4. **MVP model list** — CatBoost included or deferred? Multilabel classification in MVP?
5. ~~**Data hash method**~~ — resolved in Phase 2: sha256 of Parquet bytes (pinned writer settings)
   via `hash_file` in `services/workspace.py`; the schema sidecar carries the report.
6. **Dataframe default** — pandas at the service boundary for MVP; Polars/DuckDB when >1M rows.

## Development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

streamlit run src/ml_workbench/app.py   # run the app shell
pytest                                   # all tests (rule stubs show as pending)
ruff check . && ruff format --check .    # lint + format
mypy                                     # strict type check (src/)
python scripts/scaffold_rule_tests.py --check   # verify docs/rules.md ↔ tests sync
python scripts/make_sample_data.py       # regenerate sample datasets
```
