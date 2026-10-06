from __future__ import annotations

from collections import OrderedDict
from typing import Any

from ml_workbench.rules.performance import MAX_CACHED_MODELS

SENTINEL = object()


class ModelCache:
    """PERF-02: bounded LRU cache for fitted pipelines (loaded lazily)."""

    def __init__(self, capacity: int = MAX_CACHED_MODELS) -> None:
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._capacity = max(1, capacity)
        self.hits = 0
        self.misses = 0
        self._evicted: list[str] = []

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return len(self._store)

    def has(self, key: str) -> bool:
        return key in self._store

    def get(self, key: str) -> Any:
        value = self._store.pop(key, SENTINEL)
        if value is SENTINEL:
            self.misses += 1
            return None
        self._store[key] = value
        self.hits += 1
        return value

    def evicted(self) -> list[str]:
        return list(self._evicted)

    def put(self, key: str, value: Any) -> Any:
        self._store.pop(key, None)
        self._store[key] = value
        while len(self._store) > self._capacity:
            self._evicted.append(self._store.popitem(last=False)[0])
        return value

    def clear(self) -> None:
        self._store.clear()
        self._evicted.clear()
