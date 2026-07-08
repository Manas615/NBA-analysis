"""API endpoint tests using httpx AsyncClient."""

import pytest
from httpx import AsyncClient, ASGITransport

from api.main import app


@pytest.mark.asyncio
async def test_root_endpoint():
    """Test GET / returns service info."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "NBA Agentic AI Trade Simulator"
    assert data["version"] == "3.0.0"
    assert "agents" in data
    assert len(data["agents"]) == 10


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test GET /health returns status."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["version"] == "3.0.0"


@pytest.mark.asyncio
async def test_metrics_endpoint():
    """Test GET /metrics/json returns metrics."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/metrics/json")

    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data


@pytest.mark.asyncio
async def test_ask_requires_query():
    """Test POST /ask validates required fields."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/ask", json={})

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_trade_requires_fields():
    """Test POST /trade validates required fields."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/trade", json={"team_a": "Lakers"})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_task_status_pending():
    """Test GET /task/{id} returns pending for unknown tasks."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/task/nonexistent-id")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "pending"
