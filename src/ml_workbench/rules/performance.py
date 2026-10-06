from __future__ import annotations

MAX_CACHED_MODELS: int = 2


def sampling_thresholds() -> dict[str, int]:
    """PERF-04: heavy compute steps subsample above these row counts."""
    return {
        "eda": 100_000,
        "shap": 10_000,
        "neighbors": 50_000,
    }


def metadata_sidecar_only() -> str:
    """PERF-01: leaderboard/metrics read a JSON sidecar — never the fitted pipeline."""
    return "metrics.json"


def lru_eviction() -> dict[str, int]:
    """PERF-02: a bounded LRU keeps fitted pipelines in memory."""
    return {"max_cached": MAX_CACHED_MODELS}


def heavy_jobs_off_ui() -> bool:
    """PERF-03: long-running jobs are dispatched off the UI thread."""
    return True


def background_hybrid_cache() -> bool:
    """PERF-05: sampling uses an in-memory + disk hybrid cache."""
    return True
