from __future__ import annotations

from ml_workbench.rules.performance import (
    disk_cache_key,
    heavy_job_kinds,
    heavy_jobs_off_ui,
    lru_eviction,
    metadata_sidecar_only,
    sampling_thresholds,
)
from ml_workbench.services.jobs import JobQueue
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


def test_perf_disk_cache_key_reuse() -> None:
    assert disk_cache_key("h", "m", {}) == disk_cache_key("h", "m", {})
    assert disk_cache_key("h", "m", {"a": 1}) != disk_cache_key("h", "m", {})
    assert disk_cache_key("h1", "m", {}) != disk_cache_key("h2", "m", {})
    assert len(disk_cache_key("h", "m", {"a": 1})) == 32


def test_perf_sampling_thresholds() -> None:
    thresholds = sampling_thresholds()
    assert thresholds["eda"] == 100_000
    assert thresholds["shap"] == 10_000
    assert set(thresholds) >= {"eda", "shap", "neighbors"}


def test_perf_heavy_jobs_off_ui_thread() -> None:
    assert heavy_jobs_off_ui() is True
    assert set(heavy_job_kinds()) == {"training", "tuning", "shap"}

    queue = JobQueue(max_workers=1)
    try:
        handle = queue.submit(
            "training",
            lambda update: (update(0.5, "halfway"), 7)[1],
        )
        assert handle.state in {"queued", "running", "done"}
        done = queue.wait(handle.job_id, timeout=10)
        assert done.state == "done"
        assert done.progress == 1.0
        assert done.result == 7
    finally:
        queue.shutdown()
