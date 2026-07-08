"""
Request Tracing — Middleware for request-level tracing.

Assigns a unique X-Request-ID to every request and propagates
it through agent calls, tool executions, and log entries.
"""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from observability.logging import get_logger
from observability.metrics import (
    get_metrics,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)

logger = get_logger(__name__)


class TracingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware that:
    1. Assigns a unique X-Request-ID to every request
    2. Measures request latency
    3. Logs structured request/response info
    4. Records Prometheus metrics
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Generate or use existing request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:12])
        start_time = time.time()

        # Store in request state for downstream access
        request.state.request_id = request_id

        logger.info(
            "request_start",
            request_id=request_id,
            method=request.method,
            path=str(request.url.path),
        )

        try:
            response = await call_next(request)
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                "request_error",
                request_id=request_id,
                error=str(e),
                elapsed_ms=round(elapsed * 1000, 1),
            )
            raise

        elapsed = time.time() - start_time

        # Add tracing headers to response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{elapsed * 1000:.1f}ms"

        # Record metrics
        metrics = get_metrics()
        labels = {
            "method": request.method,
            "path": str(request.url.path),
            "status": str(response.status_code),
        }
        metrics.increment(REQUEST_COUNT, labels=labels)
        metrics.observe(REQUEST_LATENCY, elapsed, labels=labels)

        logger.info(
            "request_complete",
            request_id=request_id,
            method=request.method,
            path=str(request.url.path),
            status=response.status_code,
            elapsed_ms=round(elapsed * 1000, 1),
        )

        return response
