"""
Prometheus Metrics — Request latency, agent calls, tool execution, cache hits.

Exposes /metrics endpoint for Prometheus scraping.
"""

from __future__ import annotations

import time
from typing import Any


class MetricsCollector:
    """Simple in-process metrics collector (Prometheus-compatible)."""

    def __init__(self):
        self._counters: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = {}
        self._gauges: dict[str, float] = {}

    def increment(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Increment a counter."""
        key = self._key(name, labels)
        self._counters[key] = self._counters.get(key, 0) + value

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Record a histogram observation."""
        key = self._key(name, labels)
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Set a gauge value."""
        key = self._key(name, labels)
        self._gauges[key] = value

    def _key(self, name: str, labels: dict[str, str] | None = None) -> str:
        if labels:
            label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return name

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []

        for key, value in sorted(self._counters.items()):
            lines.append(f"# TYPE {key.split('{')[0]} counter")
            lines.append(f"{key} {value}")

        for key, values in sorted(self._histograms.items()):
            base_name = key.split("{")[0]
            lines.append(f"# TYPE {base_name} histogram")
            lines.append(f'{key.replace(base_name, base_name + "_count")} {len(values)}')
            lines.append(f'{key.replace(base_name, base_name + "_sum")} {sum(values):.4f}')
            if values:
                lines.append(f'{key.replace(base_name, base_name + "_avg")} {sum(values)/len(values):.4f}')

        for key, value in sorted(self._gauges.items()):
            lines.append(f"# TYPE {key.split('{')[0]} gauge")
            lines.append(f"{key} {value}")

        return "\n".join(lines) + "\n"

    def get_summary(self) -> dict[str, Any]:
        """Get a JSON summary of all metrics."""
        summary: dict[str, Any] = {}

        for key, value in self._counters.items():
            summary[key] = {"type": "counter", "value": value}

        for key, values in self._histograms.items():
            summary[key] = {
                "type": "histogram",
                "count": len(values),
                "sum": round(sum(values), 4),
                "avg": round(sum(values) / len(values), 4) if values else 0,
                "min": round(min(values), 4) if values else 0,
                "max": round(max(values), 4) if values else 0,
            }

        for key, value in self._gauges.items():
            summary[key] = {"type": "gauge", "value": value}

        return summary


# Well-known metric names
REQUEST_COUNT = "nba_api_requests_total"
REQUEST_LATENCY = "nba_api_request_duration_seconds"
AGENT_CALL_COUNT = "nba_agent_calls_total"
AGENT_LATENCY = "nba_agent_duration_seconds"
TOOL_CALL_COUNT = "nba_tool_calls_total"
TOOL_LATENCY = "nba_tool_duration_seconds"
CACHE_HIT_COUNT = "nba_cache_hits_total"
CACHE_MISS_COUNT = "nba_cache_misses_total"


# Global metrics instance
_metrics: MetricsCollector | None = None


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics
