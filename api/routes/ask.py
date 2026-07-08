"""POST /ask — Natural language query to the General Manager Agent."""

from fastapi import APIRouter, Depends

from api.schemas import AskRequest, AgentResponse
from api.dependencies import get_gm_agent
from agents.general_manager import GeneralManagerAgent

router = APIRouter(tags=["Ask"])


@router.post("/ask", response_model=AgentResponse)
async def ask(
    request: AskRequest,
    gm: GeneralManagerAgent = Depends(get_gm_agent),
) -> AgentResponse:
    """
    Send a natural language query to the General Manager Agent.

    The GM Agent classifies your intent, creates an execution plan,
    delegates to specialized agents, and returns a comprehensive response.

    Supports follow-up queries via session_id.
    """
    result = gm.execute_with_memory(
        user_query=request.query,
        session_id=request.session_id,
    )

    return AgentResponse(
        success=True,
        agent=result.get("agent", "GeneralManagerAgent"),
        response=result.get("response", ""),
        trade_grade=result.get("trade_grade"),
        confidence_score=result.get("confidence_score"),
        tool_calls=result.get("tool_calls", []),
        tool_results=result.get("tool_results", {}),
        session_id=result.get("session_id"),
        request_id=result.get("request_id"),
        elapsed_ms=result.get("elapsed_ms"),
    )
