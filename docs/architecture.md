# Architecture

Source: design document §2 (system architecture), §7–§11 (NN, performance, state, storage, technology options).
Status: accepted (Phase 0). Decisions below are recorded as ADRs; revisit triggers noted.

## ADR-001 — UI: Streamlit (Option A)

- **Context:** §11 offers Streamlit (fast start) vs Vue.js + TypeScript (production).
- **Decision:** Streamlit for Phases 0–8. Single Python codebase; charts via Plotly (`st.plotly_chart`).
- **Consequences:** Streamlit re-runs the script on every interaction, so
  - the project state manager must live in `st.session_state` (ADR-006),
  - tabs must never call each other — only read state,
  - heavy work must run outside the request (ADR-003).
- **Revisit:** Phase 9 if a richer UI or multi-user concurrency is required (then FastAPI + Vue, §11 Option B).

## ADR-002 — API: in-process service layer

- **Decision:** No HTTP API for MVP. Tabs call Python services directly (Data, Preprocessing, Training,
  Evaluation, Explainability, Report services — §2).
- **Consequences:** services stay pure and UI-free so the rules engine and pipelines are unit-testable
  without the UI. FastAPI is introduced only in Phase 9.

## ADR-003 — Jobs: process pool behind a JobQueue interface

- **Decision:** Training, tuning, and SHAP run in a `concurrent.futures.ProcessPoolExecutor` behind a
  small `JobQueue` interface (submit / progress / result), reporting progress into `st.session_state`.
- **Consequences:** the UI never freezes (§8); the interface is swappable for Celery/RQ/Arq + Redis in Phase 9
  without touching services. Neural network training (Phase 7) runs in its own worker with GPU detection.

## ADR-004 — Storage: local disk, Parquet, §10 layout

- **Decision:** Local workspace with the exact layout from §10 (see below). Datasets as Parquet,
  artifacts as joblib / framework checkpoints, metadata as JSON, cache as Parquet/JSON.
- **Consequences:** reproducible and inspectable; object storage + PostgreSQL deferred to Phase 9.

```
workspace/
  projects/<project_id>/
    datasets/        versioned Parquet files and schema reports
    steps.json       ordered cleaning and preprocessing log
    models/<run_id>/
      pipeline        fitted pipeline or checkpoint
      metrics.json    scores, curves, fold results
      meta.json       task, params, seed, data hash, versions
    cache/            SHAP arrays, embeddings, profiling reports
    reports/          generated HTML and PDF
```

## ADR-005 — Charts: Plotly

- **Decision:** Plotly Python on the server (§11 Option A). Plotly.js/ECharts only if Phase 9 moves the UI.

## ADR-006 — State: single ProjectState in `st.session_state`

- **Decision:** One `ProjectState` object is the single source of truth: dataset version, task, split,
  fitted pipeline, trained models, results, stale flags (§2). Tabs read from it and never talk to each other.
- **Consequences:** serialization contract defined in [`contracts.md`](contracts.md); staleness is a pure
  function of tab order (rules `STALE-*`).

## ADR-007 — Dataframes: pandas at the service boundary for MVP

- **Context:** §8 recommends Parquet, Polars, DuckDB for fast I/O and larger-than-memory data;
  scikit-learn and Plotly consume pandas.
- **Decision:** pandas is the dataframe type passed between services and models during MVP.
- **Trigger to switch:** datasets >1M rows or larger-than-memory → adopt Polars/DuckDB for I/O and
  aggregation steps, converting to pandas only at the model/chart boundary.

## Layered architecture (§2)

```
Presentation layer     10 workflow tabs · navigation and gating
        │
Application layer      project state manager · rules engine (pure functions)
                       model registry with capability flags · step log + script exporter
        │
Service layer          data · preprocessing · training · evaluation & error ·
                       explainability · report
        │
Background workers     job queue · CPU and GPU workers (progress reported to UI)
        │
Storage                datasets: Parquet · artifacts: models, metrics, metadata
                       cache: SHAP, embeddings, profiles
```

## Component responsibilities (§2)

| Component | Responsibility |
|---|---|
| Project state manager | Single source of truth: dataset version, task, split, fitted pipeline, trained models, results. Tabs read from it and never talk to each other. |
| Rules engine | Pure functions answering "which scaler?", "which metrics?", "is this plot allowed?" from task + model capabilities. Fully unit-testable without the UI ([`rules.md`](rules.md)). |
| Model registry | One entry per model: task types, family, hyperparameter schema, capability flags ([`registry.md`](registry.md)). |
| Step log | Ordered record of cleaning and preprocessing actions. Powers undo, script export, and reports. |
| Job queue + workers | Run training, tuning, and SHAP outside the UI process, with progress reporting. |

## Performance and loading strategy (§8)

- Load model once, cache in memory; lazy loading with an LRU cap (2–3 models).
- Metadata sidecar (`meta.json`) per model — most tabs never load the heavy artifact.
- Render only the active tab; hidden tabs do no work.
- Background workers for training, tuning, SHAP; disk cache keyed by data hash + model ID + params.
- Sampling above row thresholds keeps EDA, SHAP, t-SNE, silhouette fast.
- ONNX Runtime for faster, lighter inference (Phase 5); memory mapping and shared workers to avoid
  duplicate model copies; lazy imports for fast startup.

## Reproducibility, leakage, safety (§9)

- Project state carries dataset version, task, split, step log, fitted pipeline, model runs, result
  caches, with stale flags propagated downstream.
- Imputers, encoders, scalers, selectors, resamplers fitted on training folds only; target encoding and
  SMOTE inside CV folds; group and time structure respected in every split.
- Unsupported-combination warnings enforced as rules `WARN-01`…`WARN-05`.
- **Security:** only load model artifacts the app produced (pickle formats can execute code); validate
  and size-limit uploads; isolate user projects; never log raw data in reports without consent.

## Technology stack (MVP, §11 Option A)

| Layer | Choice |
|---|---|
| UI | Streamlit |
| API | In-process services |
| Jobs | Process pool behind `JobQueue` interface |
| Storage | Local disk (§10 layout) |
| Charts | Plotly |
| ML | scikit-learn, XGBoost, LightGBM, CatBoost, SHAP, Optuna, UMAP, statsmodels (+ neural = scikit-learn MLP from Phase 7, no torch dependency) |
