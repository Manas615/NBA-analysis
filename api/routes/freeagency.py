"""POST /freeagency — Free agency recommendations."""

from fastapi import APIRouter, Depends
from api.schemas import FreeAgencyRequest, AgentResponse
from api.dependencies import get_free_agency_agent
from agents.free_agency import FreeAgencyAgent

router = APIRouter(tags=["Free Agency"])


@router.post("/freeagency", response_model=AgentResponse)
async def free_agency(
    request: FreeAgencyRequest,
    agent: FreeAgencyAgent = Depends(get_free_agency_agent),
) -> AgentResponse:
    """Get free agency recommendations for a team."""
    result = agent.execute(
        user_query=f"Recommend free agents for {request.team} with budget {request.budget}",
        context={
            "team": request.team,
            "budget": request.budget,
        },
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
