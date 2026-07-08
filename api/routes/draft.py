"""POST /draft — Draft recommendation."""

from fastapi import APIRouter, Depends
from api.schemas import DraftRequest, AgentResponse
from api.dependencies import get_draft_agent
from agents.draft import DraftAgent

router = APIRouter(tags=["Draft"])


@router.post("/draft", response_model=AgentResponse)
async def draft_recommendation(
    request: DraftRequest,
    agent: DraftAgent = Depends(get_draft_agent),
) -> AgentResponse:
    """Get draft pick recommendation for a team at a given position."""
    result = agent.execute(
        user_query=f"Recommend a draft pick for {request.team} at pick #{request.draft_position}",
        context={
            "team": request.team,
            "draft_position": request.draft_position,
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
