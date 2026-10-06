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
| Phases 4–9 — Implementation | not started |

Current state: `docs/` and `registry/` hold the Phase 0 spec (124 rules, tab matrix, contracts);
`src/` holds the foundation (project state, rules engine for gating/staleness, registry loader,
Streamlit shell). Phases 2–3 are live: Data Insertion, Data Cleaning, Data Preprocessing, and EDA
tabs with versioned Parquet storage, schema reports, the step log with undo, task inference
(TASK-01..03), cleaning rules (CLEAN-01..06), and leakage-safe split/encode/scale preprocessing
(SPLIT-01..09, ENC-*, SCALE-*, FEAT-*, DR-*, TT-01, PIPE-01) plus task-adaptive EDA (EDA-01..03,
EDA-09, HINT-02) — all checks green (ruff, mypy strict, 110 passing tests, 70 rule tests still
pending for later phases).

## MVP scope (delivered in Phases 2–5)

- Tasks: binary and multiclass classification, regression on tabular data
- Models: linear baselines and gradient boosting (scikit-learn, XGBoost, LightGBM)
- All 10 workflow tabs present end-to-end: state, gating, staleness, step log, SHAP, report and script export
- Deferred: unsupervised depth (Phase 6), neural networks (Phase 7), time series / autoencoders /
  association rules / auto-compare (Phase 8), production hardening (Phase 9)

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

### Implementation

| # | Phase | Status | Content | Exit criteria |
|---|---|---|---|---|
| 2 | Walking skeleton | ✅ | Project state manager, Data Insertion (schema report, hashing), Cleaning decision tree, step log + undo | Data → versioned cleaned Parquet; stale flags propagate |
| 3 | Preprocessing + EDA | ✅ | Split/encoding/scaling rules from registry, leakage-safe pipeline build, task-adaptive EDA views | Leakage-safety tests pass; EDA adapts to task |
| 4 | Modelling, Training, Prediction | next | Registry filtering, hyperparameter forms, CV/tuning, leaderboard, artifacts + metadata sidecars, prediction tab | Train → leaderboard → predict on tabular cls/reg; LRU lazy-load works |
| 5 | Error Analysis, Explainability, Outcome | — | Task-specific error views, SHAP as background jobs, report + script export + model card + ONNX | **MVP complete:** full 10-tab flow on sample data with reproducible script |
| 6 | Unsupervised | — | Clustering, PCA/UMAP, anomaly detection, surrogate explanations | Design doc Phase 2 delivered |
| 7 | Neural networks | — | MLP/transformers, GPU routing, live loss curves, gradient-based explanations | Design doc Phase 3 delivered |
| 8 | Advanced | — | Time series, autoencoders, association rules, auto-compare | Design doc Phase 4 delivered |
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
│   │                       split (SPLIT-*), preprocessing (ENC/SCALE/FEAT/DR/TT/PIPE), eda (EDA-*, HINT-*)
│   ├── services/           service layer: workspace (ADR-004 storage), data_service, cleaning_service,
│   │                       preprocessing_service, eda_service
│   ├── ui/                 renderers: data_tab (insert + task), cleaning_tab (options + step log),
│   │                       preprocessing_tab (split/encode/scale/build), eda_tab (views)
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
