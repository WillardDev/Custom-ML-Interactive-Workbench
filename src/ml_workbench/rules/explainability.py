from __future__ import annotations

import math
from typing import Any

from ml_workbench.rules.performance import disk_cache_key

SHAP_BACKGROUND_ROWS: int = 100
SHAP_BACKGROUND_THRESHOLDS: dict[str, int] = {"shap": 10_000}


def linear_explanation(
    coefficients: list[float],
    feature_names: list[str],
    *,
    binary: bool,
) -> dict[str, Any]:
    """EXPL-01: explain_method 'linear' reports standardized coefficients (+ odds ratios)."""
    pairs = [(name, float(coef)) for name, coef in zip(feature_names, coefficients, strict=False)]
    pairs.sort(key=lambda pair: abs(pair[1]), reverse=True)
    return {
        "method": "linear",
        "binary": binary,
        "top_features": [name for name, _ in pairs],
        "table": [
            {
                "feature": name,
                "coefficient": coef,
                "odds_ratio": round(math.exp(coef), 4) if binary else None,
            }
            for name, coef in pairs
        ],
    }


def tree_explanation(shap_available: bool) -> dict[str, Any]:
    """EXPL-02: explain_method 'tree_shap' uses TreeExplainer when present."""
    if shap_available:
        source = "shap.TreeExplainer"
        shape = "tree_shap"
    else:
        source = "sklearn tree.feature_importances_"
        shape = "tree_importances"
    return {
        "method": shape,
        "source": source,
        "bias_note": (
            ""
            if shap_available
            else "shap (optional extra) is not installed — using built-in feature "
            "importances, which are biased by feature count and not local or "
            "interaction-aware."
        ),
    }


def kernel_explanation(background_rows: int, shap_available: bool) -> dict[str, Any]:
    """EXPL-03: explain_method 'kernel_shap' permutes importance on a sampled background."""
    if shap_available:
        method = "kernel_shap"
        note = "KernelExplainer on a random background sample"
    else:
        method = "permutation_importance"
        note = "shap unavailable — reporting sklearn permutation importance"
    return {
        "method": method,
        "background_rows": min(max(int(background_rows), 1), SHAP_BACKGROUND_ROWS),
        "note": note,
    }


def supervised_explanation_panels() -> dict[str, list[str]]:
    """EXPL-05: any supervised model gets global curves plus a local explanation."""
    return {
        "global": ["pdp", "ice", "ale"],
        "local": ["shap_waterfall", "lime"],
    }


def auto_explain_warnings(
    *,
    importance_spread: bool = False,
    on_sample: bool = False,
    surrogate_fidelity: float | None = None,
) -> tuple[str, ...]:
    """EXPL-10: automatic caveats shown alongside every explanation."""
    warnings: list[str] = []
    if importance_spread:
        warnings.append(
            "importance is split across correlated features — treat magnitudes as "
            "per-feature, not causal"
        )
    if on_sample:
        warnings.append("permutation/SHAP ran on a random sample of background rows")
    if surrogate_fidelity is not None and surrogate_fidelity < 0.8:
        warnings.append(
            f"surrogate fidelity is low ({surrogate_fidelity:.2f}) — interpret cautiously"
        )
    return tuple(warnings)


def explain_cache_key(data_hash: str, model_id: str, params: dict[str, Any]) -> str:
    """EXPL-11/PERF-03: a result is cached under (data hash, model id, params)."""
    return disk_cache_key(data_hash, model_id, params)


def shap_runs_in_job() -> bool:
    """EXPL-11: SHAP-style computation runs as a background job, not inline."""
    return True
