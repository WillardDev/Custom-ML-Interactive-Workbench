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
| Phase 0 — Design & contracts (no code) | **complete — awaiting review** |
| Phase 1 — Foundation (scaffolding) | not started |
| Phases 2–9 — Implementation | not started |

No application code exists yet. Everything currently in the repo (`README.md`, `docs/`, `registry/`)
is the Phase 0 deliverable: the spec the implementation will be built and tested against.

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

**Exit criteria:** reviewed spec; every rule has an ID and a named test case. → *awaiting your review*

#### Phase 1 — Foundation (scaffolding only)

- [ ] Repo layout mirroring the service layer; pinned dependencies (`pyproject.toml`)
- [ ] ruff + mypy + pytest configured; CI pipeline
- [ ] Sample datasets checked in for tests and demos
- [ ] App shell: 10 tab stubs, navigation, working gating driven by `docs/rules.md`
- [ ] Rules-engine skeleton with the Phase 0 test names wired up

**Exit criteria:** `streamlit run` boots, gated tabs behave correctly, CI green.

### Implementation

| # | Phase | Content | Exit criteria |
|---|---|---|---|
| 2 | Walking skeleton | Project state manager, Data Insertion (schema report, hashing), Cleaning decision tree, step log + undo | Data → versioned cleaned Parquet; stale flags propagate |
| 3 | Preprocessing + EDA | Pipeline rules (encoding/scaling/split from registry), EDA views per task | Leakage-safety tests pass; EDA adapts to task |
| 4 | Modelling, Training, Prediction | Registry filtering, hyperparameter forms, CV/tuning, leaderboard, artifacts + metadata sidecars, prediction tab | Train → leaderboard → predict on tabular cls/reg; LRU lazy-load works |
| 5 | Error Analysis, Explainability, Outcome | Task-specific error views, SHAP as background jobs, report + script export + model card + ONNX | **MVP complete:** full 10-tab flow on sample data with reproducible script |
| 6 | Unsupervised | Clustering, PCA/UMAP, anomaly detection, surrogate explanations | Design doc Phase 2 delivered |
| 7 | Neural networks | MLP/transformers, GPU routing, live loss curves, gradient-based explanations | Design doc Phase 3 delivered |
| 8 | Advanced | Time series, autoencoders, association rules, auto-compare | Design doc Phase 4 delivered |
| 9 | Production (optional) | Background job queue (Redis/Celery), multi-user projects, ONNX serving, monitoring | Design doc Phase 5 delivered |

## Repository layout

```
ml_workbench/
├── README.md               this file: overview + phase plan
├── docs/
│   ├── architecture.md     ADRs, layer diagram, storage layout, performance, security
│   ├── contracts.md        project state, step log, run metadata, data hash, staleness
│   ├── registry.md         capability flags, model catalog, YAML schema
│   ├── rules.md            rules catalog — every rule with an ID and test name
│   └── tabs.md             per-tab inputs/outputs/acceptance criteria + gating matrix
├── registry/
│   └── models.yaml         model registry data: capability flags + hyperparameter schemas
├── src/                    application code (created in Phase 1)
└── tests/                  pytest suites (created in Phase 1)
```

## Open questions for review

Items marked ⚠ in the docs are inferred (the design doc is silent) and need a decision at review:

1. **Outcome tab gating** — assumed to require ≥1 trained model (doc lists only Prediction/Error/Explain).
2. **EDA gating** — assumed to require a cleaned dataset only (chain shows it after Preprocessing).
3. **Split defaults** — assumed 80% holdout / 5-fold CV; doc specifies strategies but not ratios.
4. **MVP model list** — CatBoost included or deferred? Multilabel classification in MVP?
5. **Data hash method** — sha256 of Parquet bytes (pinned writer settings) vs content digest.
6. **Dataframe default** — pandas at the service boundary for MVP; Polars/DuckDB when >1M rows.

## Development

Commands (install, lint, typecheck, test) are added in Phase 1.
