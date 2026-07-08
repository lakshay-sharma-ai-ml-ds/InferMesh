"""Benchmark metrics collector stub."""
from __future__ import annotations

class BenchmarkMetricsCollector:
    """Collects and aggregates metrics during benchmark runs."""

    def __init__(self):
        self._metrics: dict = {}

    def record(self, key: str, value: float) -> None:
        if key not in self._metrics:
            self._metrics[key] = []
        self._metrics[key].append(value)

    def summary(self) -> dict:
        import statistics
        return {
            k: {
                "mean": statistics.mean(v),
                "min": min(v),
                "max": max(v),
                "count": len(v),
            }
            for k, v in self._metrics.items()
            if v
        }
