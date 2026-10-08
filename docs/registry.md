# Model capability registry

Source: design document §4 (taxonomy/catalog), §5 (capability flags), §6.5 (modelling tab).
Status: accepted (Phase 0). Data lives in [`../registry/models.yaml`](../registry/models.yaml).

Tabs query **flags**, never model names (§5). The registry is the only place model-specific
behaviour is defined.

## Capability flags (§5)

| Flag | Type | Meaning | Effect |
|---|---|---|---|
| `family` | enum | `linear, tree, kernel, distance, centroid, density, hierarchical, neural` | Drives most defaults (encoding, scaling, explain method) |
| `needs_scaling` | bool | Sensitive to feature scale | Show scaler options, otherwise hide |
| `handles_nan` | bool | Native missing value support | Allow "leave as NaN" |
| `needs_k` | bool | Requires cluster or component count | Show k-sweep tools |
| `has_predict` | bool | Can score new rows | Enable/disable Prediction features |
| `has_transform` | bool | Can project new rows into embedding space | Enable dim-reduction projection |
| `has_proba` | bool | Outputs probabilities | Enable threshold and calibration tools |
| `early_stopping` | bool | Supports validation-based stopping | Show patience settings |
| `uses_gpu` | bool | Can run on GPU | Route job to GPU worker |
| `explain_method` | enum | `linear, tree_shap, kernel_shap, gradient, surrogate, loadings` | Selects the Explainability panel |
| `viz_only` | bool | Not reusable on new data (t-SNE) | Warn and block downstream use |

## Model catalog (§4.1)

| Task | Classical | Neural network | Enabled in |
|---|---|---|---|
| Classification | Logistic Regression, Naive Bayes, KNN, SVM, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost | MLP, TabNet, FT-Transformer, ResNet-style tabular net, 1D-CNN | Phase 4 (linear + boosted first) / Phase 7 |
| Regression | Linear, Ridge, Lasso, ElasticNet, SVR, KNN, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost | MLP, TabNet, FT-Transformer, ResNet-style tabular net | Phase 4 / Phase 7 |
| Time series | Naive, Seasonal Naive, lag-feature boosting (shipped); ARIMA/SARIMA, ETS, Prophet (catalog) | LSTM, GRU, Temporal CNN, N-BEATS, TFT | Phase 8 |
| Clustering | K-Means, MiniBatch K-Means, GMM, Agglomerative, DBSCAN, HDBSCAN, Spectral, BIRCH, K-Prototypes | SOM, Deep Embedded Clustering | Phase 6 |
| Dim. reduction | PCA, Kernel PCA, TruncatedSVD, NMF, UMAP, t-SNE | Autoencoder, VAE | Phase 6 / Phase 7 |
| Anomaly detection | Isolation Forest, LOF, One-Class SVM, Elliptic Envelope, GMM density | Autoencoder reconstruction error, Deep SVDD | Phase 6 / Phase 7 |
| Association rules | Apriori (shipped); FP-Growth (catalog) | n/a | Phase 8 |

Phase 6 implementation (`registry/models.yaml`, `enabled_phase: 6`):
- `kmeans` — clustering, `family: centroid`, `has_predict`, `explain_method: surrogate`;
  params `n_clusters` (default 8) and `n_init`.
- `pca` — dimensionality reduction, `family: linear`, `has_transform`, `explain_method: loadings`;
  params `n_components` (nullable) or `variance_target` (0.95) — the training step resolves one
  component count from the other.
- `isolation_forest` — anomaly detection, `family: tree`, `has_predict`, `explain_method: tree_shap`;
  params `n_estimators` (200), `contamination` (0.001–0.5), `max_features` (1.0).

Phase 8 implementation (`registry/models.yaml`, `enabled_phase: 8`, `library: ml_workbench`):
- `naive` — forecasting, `family: forecast`, `has_predict`, `explain_method: forecast`;
  no hyperparameters (last-value baseline).
- `seasonal_naive` — forecasting, `family: forecast`, `has_predict`, `explain_method: forecast`;
  param `period` (nullable int, resolved from the EDA seasonal estimate when null).
- `lag_boosting` — forecasting, `family: forecast`, `has_predict`, `explain_method: forecast`;
  params `lags` (12), `learning_rate` (0.1), `max_iter` (200) — HistGradientBoosting on lag
  windows of the target.
- `apriori` — association, `family: association`, `has_predict`, `explain_method: rules`;
  params `min_support` (0.1), `min_confidence` (0.3), `min_lift` (1.0), `max_len` (3) — pure
  Python rule mining (`rules/association.py`); `predict` recommends items per basket.

## YAML entry schema (`registry/models.yaml`)

```yaml
- id: lightgbm                    # stable identifier used in run meta.json
  name: LightGBM
  library: lightgbm               # import path hint
  tasks: [classification, regression]
  family: tree
  enabled_phase: 4                # earliest phase the Modelling tab shows it
  flags:
    needs_scaling: false
    handles_nan: true
    has_predict: true
    has_transform: false
    has_proba: true
    early_stopping: true
    uses_gpu: false
    viz_only: false
    explain_method: tree_shap
  hyperparameters:                # generated form (§6.5 item 3)
    - name: n_estimators
      type: int
      default: 500
      min: 50
      max: 5000
    - name: early_stopping_rounds
      type: int
      default: 50
      visible_when: {early_stopping: true}   # conditional prompt (§6.5 item 4)
```

Field notes:

- `family` is stored at the entry's top level (as shown) and exposed to the rules engine as if it
  were a flag — rule functions receive it alongside `flags`.
- `hyperparameters[].type`: `int | float | bool | str | enum | nullable_int`
- `visible_when`: conditions evaluated against flags or task, e.g.
  `{needs_k: true}`, `{task: classification}`, `{early_stopping: true}`
- Required conditional prompts (§6.5 item 4): cluster count/range for `needs_k`,
  `eps` + `min_samples` for DBSCAN, `n_components`/variance target for PCA,
  `contamination` for anomaly models, class weights for imbalanced classification.
- `explain_method` defaults by family: `linear` → linear, `tree` → tree_shap,
  `kernel`/`distance` → kernel_shap, `neural` → gradient, clustering → surrogate,
  PCA-family → loadings.

## Neural network builder (§6.5 item 5, Phase 7)

- Shipped (MLP, `family: neural`): presets Small / Medium / Large (layer widths), activation,
  optimizer, learning rate, batch size, weight decay, epochs, patience (shown with early stopping).
- Catalog-only until their phase: dropout/scheduler/batch-norm variants, autoencoder family
  (bottleneck size + reconstruction loss), sequence models (window length + horizon).

## Forecast/association builder (§6.5 item 5, Phase 8)

- Shipped: `seasonal_naive` shows `period`; `lag_boosting` shows `lags` / `learning_rate` /
  `max_iter`; `apriori` shows `min_support` / `min_confidence` / `min_lift` / `max_len`.
- All four use `library: ml_workbench` — no external forecasting/association packages.

## Guidance rule (§4.1, §13)

On small tabular datasets (≈ under 10–50k rows), gradient boosting usually matches or beats neural
networks with far less tuning. The Modelling tab shows this as a hint (`HINT-01`) and recommends
including one boosted baseline in every comparison. Same warning fires for neural networks chosen
on small data (`WARN-06`).

## Enablement policy

- An entry appears in the Modelling tab only when `enabled_phase <= current phase` **and**
  `tasks` contains the project's task.
- A model's `hyperparameters` schema must be complete before it is enabled — otherwise it stays
  catalog-only (documented in the catalog table above).
- MVP entries with full schemas are in `registry/models.yaml`; schemas for the remaining models
  are written in the phase that enables them (9).
