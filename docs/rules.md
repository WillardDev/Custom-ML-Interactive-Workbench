# Rules catalog

Source: design document §3–§9. Every rule is a **pure function** of `(ProjectState, registry)` —
unit-testable without the UI (§2). Each rule has a stable ID and a reserved pytest name; tests are
written starting Phase 1 and must pass before the feature phase that depends on them is "done".

Legend: ⚠ = inferred (design doc silent, confirm at review). Sources reference design-doc sections.

## 1. Gating and navigation (§3)

| ID | Given / When / Then | Test |
|---|---|---|
| GATE-01 | Given no split and no fitted pipeline, when the user opens Training, then it is locked with an explanation | `test_gate_training_requires_split_or_pipeline` |
| GATE-02 | Given zero trained models, when the user opens Prediction, Error Analysis, or Explainability, then all three are locked with an explanation | `test_gate_tabs_require_trained_model` |
| GATE-03 | Given ≥1 trained model and a fitted split/pipeline, when the user opens any of the 10 tabs, then each is editable or read-only per the matrix in `tabs.md` | `test_gate_matrix_conformance` |
| GATE-04 ⚠ | Given zero trained models, when the user opens Outcome, then it is locked (doc lists only Prediction/Error/Explain — open question 1) | `test_gate_outcome_requires_model` |

## 2. Staleness and state (§3)

| ID | Given / When / Then | Test |
|---|---|---|
| STALE-01 | Given tabs 1…10 in fixed order, when tab *k* is edited, then every tab with index > *k* is marked stale | `test_edit_upstream_marks_downstream_stale` |
| STALE-02 | Given a stale tab, when the user views it, then a banner shows which upstream change caused staleness and offers re-run | `test_stale_banner_and_rerun_prompt` |
| STALE-03 | Given downstream tabs stale, when tab *k* re-runs successfully, then only `stale[k]` clears; downstream flags persist | `test_rerun_clears_only_rerun_tab` |
| STALE-04 | Given a locked (gated) tab, when an upstream edit occurs, then staleness is still recorded but editing remains blocked | `test_stale_and_gated_are_orthogonal` |

## 3. Task and split selection (§6.1, §6.3, §6.6)

| ID | Given / When / Then | Test |
|---|---|---|
| TASK-01 | Given a supervised project, when a target is chosen, then task type is suggested: 2 unique → binary; integer with few uniques (≤20 ⚠) → multiclass; numeric many uniques → regression | `test_task_suggestion_from_target` |
| TASK-02 | Given a datetime column, when the user confirms forecasting, then task becomes time series; otherwise datetime is an ordinary feature | `test_task_forecasting_requires_confirmation` |
| TASK-03 | Given unsupervised, when a target is set, then it is rejected; an optional eval-labels column is held aside and never seen by fitting | `test_unsupervised_eval_labels_never_fit` |
| SPLIT-01 | Given time-ordered data, when splitting, then chronological split, never shuffled | `test_split_chronological_no_shuffle` |
| SPLIT-02 | Given a group column, when splitting/CV, then no group crosses folds | `test_split_group_never_crosses_folds` |
| SPLIT-03 | Given classification, when splitting, then stratified | `test_split_stratified_classification` |
| SPLIT-04 | Given regression, when splitting, then random | `test_split_random_regression` |
| SPLIT-05 | Given unsupervised with eval labels, when splitting, then stratified holdout used for scoring only | `test_split_eval_labels_holdout_only` |
| SPLIT-06 | Given unsupervised without labels or predict support, when splitting, then optional stability holdout or no split | `test_split_unsupervised_optional` |
| SPLIT-07 | Given a group column exists, when any split/CV is created, then group-aware strategy is enforced (overrides stratified/random) | `test_split_group_overrides_default` |
| SPLIT-08 | Given CV selection: time series → `TimeSeriesSplit`; grouped → `GroupKFold`; classification → `StratifiedKFold`; regression → `KFold` | `test_cv_strategy_by_task` |
| SPLIT-09 ⚠ | Given no other rule applies, defaults: 80% holdout, 5-fold CV (doc specifies strategies, not ratios — open question 3) | `test_default_split_ratio_and_folds` |

## 4. Cleaning (§6.2)

| ID | Given / When / Then | Test |
|---|---|---|
| CLEAN-01 | Always, when cleaning runs: remove duplicates and fix dtypes; report before/after counts | `test_clean_dedup_and_dtype_fix` |
| CLEAN-02 | Given supervised with missing target values, when cleaning runs: drop those rows; never impute the target | `test_clean_never_imputes_target` |
| CLEAN-03 | Given a feature column with missing values: if the selected model's `handles_nan` is true → offer "leave as NaN"; else → require imputation choice (median/mean/KNN/iterative/mode/constant) | `test_clean_nan_policy_by_capability` |
| CLEAN-04a | Given task = anomaly detection, when outlier policy is chosen: removal is forbidden — outliers are the signal; default keep + flag | `test_clean_no_outlier_removal_anomaly` |
| CLEAN-04b | Given a tree-family model, when outlier policy is chosen: default keep and flag ("the model labels noise itself") | `test_clean_outliers_keep_and_flag_trees` |
| CLEAN-04c | Given a non-tree, non-anomaly model, when outliers exist: recommend winsorize/clip/robust scaling; treatment optional | `test_clean_outlier_treatment_recommended` |
| CLEAN-05 | Given classification with minority class < 20%, when cleaning completes: class balance shown and imbalance flagged | `test_clean_flag_minority_under_20_pct` |
| CLEAN-06 | Given any cleaning step, when it is applied: it is appended to the step log with before/after dataset hashes; undo restores the previous dataset and removes the entry | `test_clean_undo_restores_counts` |

## 5. Encoding, scaling, features, target transforms (§6.3)

| ID | Given / When / Then | Test |
|---|---|---|
| ENC-01 | Given tree family: encode ordinal or use native categorical handling; never one-hot by default | `test_encode_tree_ordinal_or_native` |
| ENC-02 | Given linear/distance/kernel family: one-hot; if supervised + high cardinality (≈ >20 ⚠ unique values): target encoding **inside CV folds** | `test_encode_high_cardinality_target_in_folds` |
| ENC-03 | Given neural family: one-hot, or learned embeddings for high cardinality | `test_encode_neural_embeddings` |
| ENC-04 | Given clustering with categorical features: use K-Prototypes or Gower distance | `test_encode_clustering_with_categoricals` |
| ENC-05 | Always: encoders are fitted on training folds only, never on validation/test | `test_encoder_fit_within_fold` |
| SCALE-01 | Given `needs_scaling: true`: offer Standard, Robust, MinMax (default Standard) | `test_scale_options_when_needed` |
| SCALE-02 | Given `needs_scaling: false`: skip scaling and hide scaler options | `test_scale_hidden_for_trees` |
| SCALE-03 | Given neural models: scaling is mandatory — pipeline cannot be fitted without a scaler | `test_scale_mandatory_for_neural` |
| SCALE-04 | Always: scalers fitted on training folds only | `test_scaler_fit_within_fold` |
| FEAT-01 | Given neural models with missing values: impute **and** add missing-indicator columns | `test_neural_missing_indicators` |
| FEAT-02a | Given supervised feature selection: offer ANOVA, chi², mutual info, RFE, L1, model-based | `test_feature_selection_supervised_menu` |
| FEAT-02b | Given unsupervised feature selection: only variance threshold and correlation filter | `test_feature_selection_unsupervised_menu` |
| FEAT-03 | Always: selectors fitted inside CV folds | `test_selector_fit_within_fold` |
| FEAT-04a | Given imbalanced classification: offer class weights, SMOTE, or threshold tuning | `test_imbalance_options_offered` |
| FEAT-04b | Given SMOTE: resampling happens only inside CV folds, never before the split | `test_smote_inside_folds_only` |
| FEAT-04c | Given regression: SMOTE is rejected with a warning (WARN-03) | `test_smote_rejected_for_regression` |
| DR-01 | Given >≈30 features and a centroid/density clustering model: suggest PCA targeting 90–95% variance | `test_pca_suggested_for_highdim_clustering` |
| DR-02 | Given tree models: never suggest PCA | `test_pca_never_suggested_for_trees` |
| DR-03 | Given t-SNE or UMAP: they are visualization-only and never used as clustering input | `test_umap_tsne_not_clustering_input` |
| TT-01 | Given regression + kernel/neural model + \|skew(target)\| > 1: offer log, Box-Cox, or Yeo-Johnson target transform | `test_target_transform_offer_when_skewed` |
| PIPE-01 | Always: every fitted step (imputer, encoder, scaler, selector, resampler) lives in one pipeline fitted on training data only (§9) | `test_pipeline_fit_leakage_safe` |

## 6. Model selection and hyperparameters (§6.5)

| ID | Given / When / Then | Test |
|---|---|---|
| MODEL-01 | Given the project task, when the Modelling tab opens: the registry is filtered to compatible models only | `test_registry_filtered_by_task` |
| MODEL-02 | Given a chosen mode (Auto-compare / Manual / Auto with tuning), when a model is picked: the hyperparameter form is generated from its registry schema | `test_hyperparameter_form_from_schema` |
| MODEL-03 | Given conditional-prompt rules: `needs_k` → cluster count/search range; DBSCAN → `eps` + `min_samples`; PCA → `n_components`/variance target; anomaly → `contamination`; imbalanced classification → class weights | `test_conditional_prompts_by_capability` |
| MODEL-04 | Given a hidden/incompatible option (e.g. scaler for `needs_scaling: false`), when submitting: it is never silently applied — the form cannot express it | `test_incompatible_options_not_submittable` |

## 7. Training and validation (§6.6)

| ID | Given / When / Then | Test |
|---|---|---|
| TRAIN-01 | Always, when training starts: set seed, log config and data hash to `meta.json` | `test_train_logs_seed_config_hash` |
| TRAIN-02 | Given boosting models: fit with CV and use early stopping on validation | `test_boosting_early_stopping` |
| TRAIN-03 | Given neural models: mini-batch training on GPU if available, early stopping on validation loss, live loss curves, checkpoint best epoch | `test_neural_training_loop` |
| TRAIN-04 | Given tuning enabled: Optuna or randomized search with pruning; results still logged per TRAIN-01 | `test_tuning_uses_pruning` |
| TRAIN-05 | Always: leaderboard shows mean and std across folds plus train-vs-validation gap | `test_leaderboard_fold_stats` |
| TRAIN-06 | Always, when training completes: save artifact + `metrics.json` + `meta.json`; model appears in `ProjectState.models` with `status: done` | `test_train_persists_artifacts` |
| TRAIN-07 | Given `uses_gpu: true` and a GPU present: route to GPU worker; else CPU with a time estimate and warning (§7) | `test_job_routing_gpu_vs_cpu` |

## 8. Metrics (§6.6)

| ID | Given / When / Then | Test |
|---|---|---|
| METRIC-01 | Balanced classification → Accuracy, F1 | `test_metrics_balanced_classification` |
| METRIC-02 | Imbalanced classification (minority <20%) → PR-AUC, macro-F1, MCC, balanced accuracy | `test_metrics_imbalanced_classification` |
| METRIC-03 | Regression → RMSE, MAE, R²; if target skewed/outliers: primary metric becomes MAE or RMSLE | `test_metrics_regression_skew_prefers_mae` |
| METRIC-04 | Time series → MAE, RMSE, MAPE/sMAPE, MASE | `test_metrics_time_series` |
| METRIC-05 | Clustering → silhouette, Davies-Bouldin, Calinski-Harabasz, stability ARI; + ARI/NMI if labels exist | `test_metrics_clustering` |
| METRIC-06 | Dim. reduction → explained variance, reconstruction error, trustworthiness | `test_metrics_dim_reduction` |
| METRIC-07 | Anomaly → ROC-AUC/PR-AUC if labeled, otherwise score distribution | `test_metrics_anomaly_labeled_vs_not` |
| METRIC-08 | Association rules → filter rules by support, confidence, lift | `test_metrics_association_filters` |

## 9. Prediction (§6.7)

| ID | Given / When / Then | Test |
|---|---|---|
| PRED-01 | Binary classification: test-set evaluation, single-row form, batch upload, probabilities, adjustable threshold, calibration curve | `test_pred_binary_threshold_calibration` |
| PRED-02 | Regression: test-set evaluation, predicted-vs-actual, prediction intervals (quantile models or conformal) | `test_pred_regression_intervals` |
| PRED-03 | Time series: horizon selector, confidence bands, backtest view | `test_pred_forecast_horizon_backtest` |
| PRED-04 | Clustering: if `has_predict` → assign cluster with distance/membership probability; else warn and offer surrogate nearest-centroid/kNN trained on cluster labels | `test_pred_clustering_surrogate_fallback` |
| PRED-05 | Dim. reduction: if `has_transform` → project new rows; t-SNE → disabled with explanation | `test_pred_tsne_projection_blocked` |
| PRED-06 | Anomaly: score new rows, threshold slider, flagged-rows table; LOF only in novelty mode | `test_pred_anomaly_lof_novelty_only` |
| PRED-07 | Association rules: recommend items for a given basket | `test_pred_basket_recommendations` |

## 10. Error analysis (§6.8)

| ID | Given / When / Then | Test |
|---|---|---|
| ERR-01 | Binary → confusion matrix, ROC, PR curve, threshold analysis, calibration | `test_err_view_binary` |
| ERR-02 | Multiclass → normalized confusion matrix, per-class precision/recall, most-confused pairs | `test_err_view_multiclass` |
| ERR-03 | Regression/forecasting → residuals-vs-predicted, residual histogram, Q-Q, heteroscedasticity check, error by target quantile | `test_err_view_regression` |
| ERR-04 | Clustering → per-sample silhouette, low-silhouette points, size imbalance, stability warning | `test_err_view_clustering` |
| ERR-05 | Dim. reduction → reconstruction error per row, poorly embedded points | `test_err_view_dim_reduction` |
| ERR-06 | Anomaly → score distribution, top flagged rows, FP/FN if labeled | `test_err_view_anomaly` |
| ERR-07 | Any task: worst-N rows and segment slicing by feature bins/categories available | `test_err_common_tools_worst_n_segments` |
| ERR-08 | Neural model in any task: add learning curves, overfitting diagnostics, per-epoch metrics | `test_err_neural_diagnostics_added` |

## 11. Explainability (§6.9)

| ID | Given / When / Then | Test |
|---|---|---|
| EXPL-01 | `explain_method: linear` → standardized coefficients, odds ratios | `test_expl_linear_coefficients` |
| EXPL-02 | `explain_method: tree_shap` → TreeExplainer SHAP + built-in importances shown with bias caveat | `test_expl_tree_shap_with_bias_note` |
| EXPL-03 | `explain_method: kernel_shap` → permutation importance + KernelExplainer on sampled background | `test_expl_kernel_on_sampled_background` |
| EXPL-04 | `explain_method: gradient` (neural) → Deep/GradientExplainer SHAP, Integrated Gradients, attention maps for transformers | `test_expl_neural_gradient_methods` |
| EXPL-05 | Any supervised model → PDP, ICE, ALE + local explanation (SHAP waterfall or LIME) | `test_expl_global_and_local_supervised` |
| EXPL-06 | Clustering → centroid heatmap, ANOVA per feature, surrogate decision tree with SHAP, auto-generated personas | `test_expl_clustering_surrogate` |
| EXPL-07 | PCA → loadings and biplot; UMAP/t-SNE/autoencoder → color embedding by features; autoencoder → per-feature reconstruction error | `test_expl_embedding_panels` |
| EXPL-08 | Anomaly → SHAP for Isolation Forest, per-feature deviation, autoencoder per-feature error | `test_expl_anomaly_panels` |
| EXPL-09 | Association rules → rule network, lift-vs-confidence | `test_expl_association_panels` |
| EXPL-10 | Always: auto-warnings — importance split across correlated features; permutation/kernel run on a sample; surrogate shows fidelity score | `test_expl_warnings_auto_shown` |
| EXPL-11 | Always: SHAP/embeddings computed as background jobs and disk-cached by data hash + model + params (§8) | `test_shap_background_job_and_cache` |

## 12. EDA (§6.4)

| ID | Given / When / Then | Test |
|---|---|---|
| EDA-01 | All tasks → shape, dtypes, missingness heatmap, distributions, correlation matrix (Pearson/Spearman numeric, Cramér's V categorical) | `test_eda_base_views_all_tasks` |
| EDA-02 | Classification → class balance, feature-by-class distributions, chi-square tests | `test_eda_classification_views` |
| EDA-03 | Regression → target histogram/skew, Q-Q plot, feature-vs-target scatter; \|skew\| > 1 → suggest log transform | `test_eda_regression_views_skew_hint` |
| EDA-04 | Time series → decomposition, ACF/PACF, rolling statistics, stationarity test | `test_eda_timeseries_views` |
| EDA-05 | Clustering → Hopkins statistic, PCA/UMAP preview, scree plot | `test_eda_clustering_views` |
| EDA-06 | Anomaly → univariate z-score/IQR flags, Mahalanobis distance | `test_eda_anomaly_views` |
| EDA-07 | Dim. reduction → correlation groups, VIF | `test_eda_dimred_views` |
| EDA-08 | Association → item frequency, basket size distribution | `test_eda_association_views` |
| EDA-09 | Always: sampling above row thresholds so EDA stays fast (§8) | `test_eda_samples_large_data` |

## 13. Outcome and export (§6.10)

| ID | Given / When / Then | Test |
|---|---|---|
| OUT-01 | Supervised outcome → optional refit on all data, exported pipeline, predictions file, chosen threshold/interval method, model card | `test_out_supervised_deliverables` |
| OUT-02 | Clustering outcome → dataset with cluster IDs, profiles, personas | `test_out_clustering_deliverables` |
| OUT-03 | Dim. reduction outcome → transformed dataset, variance/reconstruction summary, reusable transformer | `test_out_dimred_deliverables` |
| OUT-04 | Anomaly outcome → flagged rows with scores and threshold used | `test_out_anomaly_deliverables` |
| OUT-05 | Association outcome → rules table | `test_out_association_deliverables` |
| EXPORT-01 | Always → HTML/PDF report, reproducible script generated from the step log, data hash, library versions, random seed | `test_export_bundle_complete` |
| EXPORT-02 | Always → native artifact format (joblib/checkpoint) + ONNX for portable inference | `test_export_native_and_onnx` |

## 14. Warnings and unsupported combinations (§7, §9, §13)

| ID | Given / When / Then | Test |
|---|---|---|
| WARN-01 | t-SNE selected + Prediction requested → block with explanation | `test_warn_tsne_with_prediction` |
| WARN-02 | DBSCAN + new-row assignment → block; offer surrogate (PRED-04) | `test_warn_dbscan_new_rows` |
| WARN-03 | SMOTE + regression → block | `test_warn_smote_regression` |
| WARN-04 | Target-based feature selection + unsupervised task → block | `test_warn_target_selection_unsupervised` |
| WARN-05 | Outlier removal + anomaly detection → block (CLEAN-04a) | `test_warn_outlier_removal_anomaly` |
| WARN-06 | Neural model + small data → warn and recommend a boosting baseline | `test_warn_neural_small_data` |
| WARN-07 | `viz_only: true` model (t-SNE) used in a pipeline meant for new rows → block downstream use | `test_warn_viz_only_blocks_downstream` |

## 15. Hints (§4.1, §6.4)

| ID | Given / When / Then | Test |
|---|---|---|
| HINT-01 | Tabular dataset < ≈10–50k rows → Modelling tab shows hint: boosting usually matches/beats NN with less tuning; recommend one boosted baseline per comparison | `test_hint_boosting_baseline_small_data` |
| HINT-02 | Regression \|skew(target)\| > 1 (EDA) → suggest log transform (overlaps EDA-03/TT-01) | `test_hint_log_transform_skewed_target` |

## 16. Performance (§8)

| ID | Given / When / Then | Test |
|---|---|---|
| PERF-01 | Given a tab that needs only metadata → read `meta.json`; heavy artifact is not loaded | `test_perf_metadata_sidecar_only` |
| PERF-02 | Given the in-memory model cache exceeds 2–3 models → evict least recently used | `test_perf_lru_eviction` |
| PERF-03 | Given a result cached under (data hash, model ID, params) → return it without recompute | `test_perf_disk_cache_key_reuse` |
| PERF-04 | Given row counts above thresholds → EDA/SHAP/t-SNE/silhouette run on a sample | `test_perf_sampling_thresholds` |
| PERF-05 | Given a heavy job (training, tuning, SHAP) → it runs in the job queue, UI stays responsive, progress reported | `test_perf_heavy_jobs_off_ui_thread` |

---

**Coverage check:** GATE 4 · STALE 4 · TASK/SPLIT 12 · CLEAN 8 · ENCODE/SCALE/FEAT/DR/TT/PIPE 21 ·
MODEL 4 · TRAIN 7 · METRIC 8 · PRED 7 · ERR 8 · EXPL 11 · EDA 9 · OUT/EXPORT 7 · WARN 7 · HINT 2 · PERF 5 = **124 rules**.
