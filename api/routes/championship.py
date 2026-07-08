"""POST /championship — Championship odds simulation."""

from fastapi import APIRouter, Depends, BackgroundTasks
from api.schemas import ChampionshipRequest, AgentResponse, TaskResponse
from api.dependencies import get_championship_agent
from agents.championship_sim import ChampionshipSimAgent
from workers.tasks import run_championship_simulation_task, create_task_id

router = APIRouter(tags=["Championship"])


@router.post("/championship", response_model=AgentResponse)
async def championship_odds(
    request: ChampionshipRequest,
    agent: ChampionshipSimAgent = Depends(get_championship_agent),
) -> AgentResponse:
    """Run Monte Carlo championship simulation for a team."""
    result = agent.execute(
        user_query=f"Predict championship odds for {request.team} using {request.simulations} simulations",
        context={
            "team": request.team,
            "simulations": request.simulations,
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


@router.post("/championship/async", response_model=TaskResponse)
async def championship_odds_async(
    request: ChampionshipRequest,
    background_tasks: BackgroundTasks,
) -> TaskResponse:
    """Run championship simulation as a background task."""
    task_id = create_task_id()
    background_tasks.add_task(
        run_championship_simulation_task,
        task_id=task_id,
        team=request.team,
        simulations=request.simulations,
    )
    return TaskResponse(
        task_id=task_id,
        status="submitted",
        message=f"Championship simulation for {request.team} started",
    )
