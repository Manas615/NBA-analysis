"""
Structured Logging — JSON-formatted logs with context.

Uses structlog for structured, machine-readable log output
with automatic context binding (request_id, agent, tool, etc.).
"""

from __future__ import annotations

import logging
import os
import sys
import json
from datetime import datetime, timezone
from typing import Any


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


class StructuredFormatter(logging.Formatter):
    """JSON-formatted log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields
        for key in ("event", "agent", "tool", "request_id", "elapsed_ms",
                     "error", "success", "task_id", "query", "iteration",
                     "iterations", "tools_called", "persist_dir", "collection",
                     "file", "chunks", "total_chunks", "count", "teams",
                     "path", "task"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])

        return json.dumps(log_entry, default=str)


class StructuredLogger:
    """
    Structured logger wrapper.

    Provides structlog-like API with key-value logging:
        logger.info("event_name", key1="value1", key2=42)
    """

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def _log(self, level: int, event: str, **kwargs: Any) -> None:
        """Log with structured key-value pairs."""
        extra = {"event": event}
        extra.update(kwargs)
        self._logger.log(level, event, extra=extra)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        self._log(logging.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, event, **kwargs)

    def critical(self, event: str, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, event, **kwargs)


def setup_logging() -> None:
    """Configure structured logging for the application."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # Remove existing handlers
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())
    root.addHandler(handler)

    # Quiet noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name)
