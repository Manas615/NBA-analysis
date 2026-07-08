"""POST /compare — Player comparison."""

from fastapi import APIRouter, Depends
from api.schemas import CompareRequest, AgentResponse
from api.dependencies import get_gm_agent
from agents.general_manager import GeneralManagerAgent

router = APIRouter(tags=["Compare"])


@router.post("/compare", response_model=AgentResponse)
async def compare_players(
    request: CompareRequest,
    gm: GeneralManagerAgent = Depends(get_gm_agent),
) -> AgentResponse:
    """Compare two NBA players side-by-side."""
    result = gm.execute_with_memory(
        user_query=f"Compare {request.player_a} vs {request.player_b}",
        session_id=request.session_id,
        context={
            "player_a": request.player_a,
            "player_b": request.player_b,
        },
    )
    return AgentResponse(
        success=True,
        agent=result.get("agent", ""),
        response=result.get("response", ""),
        tool_calls=result.get("tool_calls", []),
        tool_results=result.get("tool_results", {}),
        session_id=result.get("session_id"),
        request_id=result.get("request_id"),
        elapsed_ms=result.get("elapsed_ms"),
    )
