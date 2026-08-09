"""Performance timing utilities for OrthoRehab AI Step 5.

Provides lightweight timing helpers to measure MediaPipe processing time,
DTW time, comparison time, LLM latency and total feedback latency. Timings are
recorded in milliseconds for reporting.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator, Optional


class Stopwatch:
    """A simple stopwatch that records elapsed seconds."""

    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.elapsed: float = 0.0

    def start(self) -> None:
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        if self.start_time is None:
            return 0.0
        self.elapsed = time.perf_counter() - self.start_time
        self.start_time = None
        return self.elapsed


class Timer:
    """Collects named timings (in milliseconds) across code paths."""

    def __init__(self) -> None:
        self._timings: Dict[str, float] = {}

    @contextmanager
    def time(self, name: str) -> Iterator["Timer"]:
        """Context manager that records elapsed time for ``name`` (ms)."""
        start = time.perf_counter()
        try:
            yield self
        finally:
            self._timings[name] = (time.perf_counter() - start) * 1000.0

    def record(self, name: str, seconds: float) -> None:
        """Manually record an elapsed time (seconds -> ms)."""
        self._timings[name] = seconds * 1000.0

    def get(self, name: str, default: float = 0.0) -> float:
        return self._timings.get(name, default)

    def to_dict(self) -> Dict[str, float]:
        return {k: round(v, 3) for k, v in sorted(self._timings.items())}

    def summary(self) -> str:
        parts = [f"{k}={v:.2f}ms" for k, v in self.to_dict().items()]
        return ", ".join(parts) if parts else "(no timings)"
