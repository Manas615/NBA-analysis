"""POST /trade — Structured trade analysis."""

from fastapi import APIRouter, Depends

from api.schemas import TradeRequest, AgentResponse
from api.dependencies import get_trade_agent
from agents.trade_analysis import TradeAnalysisAgent

router = APIRouter(tags=["Trade"])


@router.post("/trade", response_model=AgentResponse)
async def analyze_trade(
    request: TradeRequest,
    agent: TradeAnalysisAgent = Depends(get_trade_agent),
) -> AgentResponse:
    """
    Analyze a structured trade between two teams.

    Runs salary validation, player evaluation, chemistry analysis,
    matchup prediction, and injury risk assessment.
    """
    context = {
        "team_a": request.team_a,
        "team_b": request.team_b,
        "player_a": request.player_a,
        "player_b": request.player_b,
        "team_a_players": [request.player_a],
        "team_b_players": [request.player_b],
        "simulations": request.simulations,
    }

    result = agent.execute(
        user_query=(
            f"Analyze this trade: {request.player_a} ({request.team_a}) "
            f"for {request.player_b} ({request.team_b})"
        ),
        context=context,
    )

    return AgentResponse(
        success=True,
        agent=result.get("agent", "TradeAnalysisAgent"),
        response=result.get("response", ""),
        tool_calls=result.get("tool_calls", []),
        tool_results=result.get("tool_results", {}),
        session_id=request.session_id,
        request_id=result.get("request_id"),
        elapsed_ms=result.get("elapsed_ms"),
    )
