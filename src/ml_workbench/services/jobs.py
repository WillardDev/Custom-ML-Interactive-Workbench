from __future__ import annotations

import secrets
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

ProgressUpdate = Callable[[float, str], None]
JobFn = Callable[[ProgressUpdate], Any]

JOB_ID_RE = "job_[A-Za-z0-9]{8}"

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"


def new_job_id() -> str:
    return f"job_{secrets.token_hex(4)}"


@dataclass
class JobHandle:
    """PERF-05: a submitted job reports its own progress off the UI thread."""

    job_id: str
    kind: str
    state: str = QUEUED
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: str = ""
    updated_at: float = field(default_factory=lambda: time.monotonic())


class JobQueue:
    """ADR-003/PERF-05: heavyweight jobs (training, tuning, SHAP) run in a queue.

    An in-process thread pool backs the queue for the MVP; the interface
    (submit / poll / wait) is what the UI layer depends on, so the executor can
    be swapped for a process pool or a Celery broker later without UI changes.
    """

    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="workbench")
        self._jobs: dict[str, JobHandle] = {}
        self._futures: dict[str, Future[Any]] = {}

    def submit(self, kind: str, fn: JobFn) -> JobHandle:
        handle = JobHandle(job_id=new_job_id(), kind=kind)
        self._jobs[handle.job_id] = handle

        def run() -> Any:
            handle.state = RUNNING
            handle.updated_at = time.monotonic()
            try:
                result = fn(_progress_update(handle))
            except Exception as exc:  # noqa: BLE001 - surfaced via poll()/wait()
                handle.state = FAILED
                handle.error = str(exc)
                handle.updated_at = time.monotonic()
                return handle
            handle.state = DONE
            handle.progress = 1.0
            handle.result = result
            handle.updated_at = time.monotonic()
            return handle

        self._futures[handle.job_id] = self._executor.submit(run)
        return handle

    def poll(self, job_id: str) -> JobHandle:
        return self._jobs[job_id]

    def is_running(self, job_id: str) -> bool:
        handle = self._jobs[job_id]
        return handle.state in {QUEUED, RUNNING}

    def wait(self, job_id: str, timeout: float = 120.0) -> JobHandle:
        future = self._futures[job_id]
        future.result(timeout=timeout)
        return self._jobs[job_id]

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)


def _progress_update(handle: JobHandle) -> ProgressUpdate:
    def update(progress: float, message: str) -> None:
        handle.progress = max(0.0, min(float(progress), 1.0))
        handle.message = message
        handle.updated_at = time.monotonic()

    return update
