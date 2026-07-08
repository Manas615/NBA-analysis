"""GET /health — Health check and metrics endpoints."""

from fastapi import APIRouter

from api.schemas import HealthResponse
from cache.redis_cache import get_cache
from observability.metrics import get_metrics
from workers.tasks import get_task_result

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check with dependency status."""
    services = {}

    # Redis
    cache = get_cache()
    services["redis"] = "healthy" if cache.available else "unavailable"

    # PostgreSQL
    try:
        from db.engine import get_engine
        engine = get_engine()
        services["postgres"] = "configured"
    except Exception:
        services["postgres"] = "unavailable"

    # ML Models
    try:
        from nba_analysis.models import get_player_model, get_matchup_model
        get_player_model()
        get_matchup_model()
        services["ml_models"] = "healthy"
    except Exception:
        services["ml_models"] = "unavailable"

    # Overall status
    status = "healthy" if services.get("ml_models") == "healthy" else "degraded"

    return HealthResponse(
        status=status,
        version="3.0.0",
        services=services,
    )


@router.get("/metrics")
async def prometheus_metrics() -> str:
    """Prometheus metrics endpoint."""
    metrics = get_metrics()
    return metrics.to_prometheus()


@router.get("/metrics/json")
async def metrics_json() -> dict:
    """JSON metrics summary."""
    metrics = get_metrics()
    cache = get_cache()

    return {
        "metrics": metrics.get_summary(),
        "cache": cache.get_stats(),
    }


@router.get("/task/{task_id}")
async def get_task_status(task_id: str) -> dict:
    """Check the status of a background task."""
    result = get_task_result(task_id)
    if result is None:
        return {"task_id": task_id, "status": "pending"}
    return {"task_id": task_id, **result}
