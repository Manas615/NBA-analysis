"""
Pydantic Schemas — All request/response models for the API.

Typed, validated models for every endpoint.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Request Models ────────────────────────────────────────────

class AskRequest(BaseModel):
    """Natural language query to the General Manager Agent."""
    query: str = Field(..., description="Natural language query", min_length=3, max_length=2000)
    session_id: str | None = Field(None, description="Session ID for follow-up queries")
    simulations: int = Field(200, ge=50, le=2000, description="Number of Monte Carlo simulations")


class TradeRequest(BaseModel):
    """Structured trade analysis request."""
    team_a: str = Field(..., description="First team name")
    team_b: str = Field(..., description="Second team name")
    player_a: str = Field(..., description="Player from Team A being traded")
    player_b: str = Field(..., description="Player from Team B being traded")
    session_id: str | None = None
    simulations: int = Field(200, ge=50, le=2000)


class OptimizeRequest(BaseModel):
    """Roster optimization request."""
    team: str = Field(..., description="Team to optimize")
    session_id: str | None = None


class ChampionshipRequest(BaseModel):
    """Championship odds request."""
    team: str = Field(..., description="Team to simulate")
    simulations: int = Field(200, ge=50, le=2000)
    session_id: str | None = None


class RosterRequest(BaseModel):
    """Roster analysis request."""
    team: str = Field(..., description="Team to analyze")
    session_id: str | None = None


class DraftRequest(BaseModel):
    """Draft recommendation request."""
    team: str = Field(..., description="Team drafting")
    draft_position: int = Field(1, ge=1, le=60, description="Draft pick number")
    session_id: str | None = None


class FreeAgencyRequest(BaseModel):
    """Free agency recommendation request."""
    team: str = Field(..., description="Team looking for free agents")
    budget: float = Field(10_000_000, ge=0, description="Salary budget in dollars")
    session_id: str | None = None


class CompareRequest(BaseModel):
    """Player comparison request."""
    player_a: str = Field(..., description="First player name")
    player_b: str = Field(..., description="Second player name")
    session_id: str | None = None


# ── Response Models ───────────────────────────────────────────

class AgentResponse(BaseModel):
    """Standard response from any agent endpoint."""
    success: bool = True
    agent: str = ""
    response: str = ""
    trade_grade: str | None = None
    confidence_score: float | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = None
    request_id: str | None = None
    elapsed_ms: float | None = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "3.0.0"
    services: dict[str, str] = Field(default_factory=dict)


class TaskResponse(BaseModel):
    """Background task response."""
    task_id: str
    status: str = "submitted"
    message: str = ""
