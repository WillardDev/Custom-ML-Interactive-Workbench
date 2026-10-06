from __future__ import annotations

import pytest

from ml_workbench.rules.performance import (
    lru_eviction,
    metadata_sidecar_only,
    sampling_thresholds,
)
from ml_workbench.services.model_cache import ModelCache


def test_perf_metadata_sidecar_only() -> None:
    assert metadata_sidecar_only() == "metrics.json"


def test_perf_lru_eviction() -> None:
    cache = ModelCache(capacity=2)
    cache.put("run_a", object())
    cache.put("run_b", object())
    assert cache.size == 2
    cache.put("run_c", object())
    assert "run_a" in cache.evicted()
    assert cache.get("run_c") is not None
    assert cache.get("run_a") is None

    cache.put("run_b", object())
    cache.put("run_d", object())
    assert "run_c" in cache.evicted()
    assert lru_eviction()["max_cached"] == 2


@pytest.mark.skip(
    reason="disk-cache-by-(hash,model,params) reuse arrives in Phase 5 (docs/rules.md PERF-03)"
)
def test_perf_disk_cache_key_reuse() -> None:
    raise NotImplementedError("docs/rules.md PERF-03")


def test_perf_sampling_thresholds() -> None:
    thresholds = sampling_thresholds()
    assert thresholds["eda"] == 100_000
    assert thresholds["shap"] == 10_000
    assert set(thresholds) >= {"eda", "shap", "neighbors"}


@pytest.mark.skip(
    reason="job queue / progress reporting arrives in Phase 5 (docs/rules.md PERF-05)"
)
def test_perf_heavy_jobs_off_ui_thread() -> None:
    raise NotImplementedError("docs/rules.md PERF-05")
