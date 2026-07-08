"""POST /roster — Roster analysis with chemistry and injury assessment."""

from fastapi import APIRouter, Depends
from api.schemas import RosterRequest, AgentResponse
from api.dependencies import get_chemistry_agent
from agents.chemistry import ChemistryAgent

router = APIRouter(tags=["Roster"])


@router.post("/roster", response_model=AgentResponse)
async def roster_analysis(
    request: RosterRequest,
    agent: ChemistryAgent = Depends(get_chemistry_agent),
) -> AgentResponse:
    """Get roster chemistry analysis for a team."""
    result = agent.execute(
        user_query=f"Analyze the roster chemistry for {request.team}",
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
