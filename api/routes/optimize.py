"""POST /optimize — Roster optimization."""

from fastapi import APIRouter, Depends
from api.schemas import OptimizeRequest, AgentResponse
from api.dependencies import get_roster_agent
from agents.roster_optimizer import RosterOptimizerAgent

router = APIRouter(tags=["Optimize"])


@router.post("/optimize", response_model=AgentResponse)
async def optimize_roster(
    request: OptimizeRequest,
    agent: RosterOptimizerAgent = Depends(get_roster_agent),
) -> AgentResponse:
    """Optimize a team's starting lineup, bench, and rotation."""
    result = agent.execute(
        user_query=f"Optimize the roster for {request.team}",
        context={"team": request.team},
    )
    return AgentResponse(
        success=True,
        agent=result.get("agent", ""),
        response=result.get("response", ""),
        tool_calls=result.get("tool_calls", []),
        tool_results=result.get("tool_results", {}),
        session_id=request.session_id,
        request_id=result.get("request_id"),
        elapsed_ms=result.get("elapsed_ms"),
    )
