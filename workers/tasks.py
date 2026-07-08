"""
Background Tasks — Async task execution for heavy computations.

Uses FastAPI BackgroundTasks for:
- Monte Carlo simulations
- Injury prediction batch jobs
- Large report generation
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from observability.logging import get_logger

logger = get_logger(__name__)


# In-memory task store
_task_results: dict[str, dict[str, Any]] = {}


def get_task_result(task_id: str) -> dict[str, Any] | None:
    """Get the result of a background task."""
    return _task_results.get(task_id)


def set_task_result(task_id: str, result: dict[str, Any]) -> None:
    """Store the result of a background task."""
    _task_results[task_id] = result


async def run_championship_simulation_task(
    task_id: str,
    team: str,
    simulations: int = 500,
    trade_config: dict[str, str] | None = None,
) -> None:
    """Background task: Run Monte Carlo championship simulation."""
    logger.info("bg_task_start", task_id=task_id, task="championship_simulation", team=team)

    try:
        from tools.simulate_championship import simulate_championship

        result = simulate_championship(
            team=team,
            simulations=simulations,
            trade_player_a=trade_config.get("player_a", "") if trade_config else "",
            trade_team_a=trade_config.get("team_a", "") if trade_config else "",
            trade_player_b=trade_config.get("player_b", "") if trade_config else "",
            trade_team_b=trade_config.get("team_b", "") if trade_config else "",
        )

        set_task_result(task_id, {
            "status": "completed",
            "result": result,
        })

        logger.info("bg_task_complete", task_id=task_id, task="championship_simulation")

    except Exception as e:
        logger.error("bg_task_error", task_id=task_id, error=str(e))
        set_task_result(task_id, {
            "status": "failed",
            "error": str(e),
        })


async def run_full_trade_analysis_task(
    task_id: str,
    team_a: str,
    player_a: str,
    team_b: str,
    player_b: str,
    simulations: int = 200,
) -> None:
    """Background task: Run complete trade analysis with all agents."""
    logger.info("bg_task_start", task_id=task_id, task="full_trade_analysis")

    try:
        from tools.salary_validation import salary_validation
        from tools.predict_player import predict_player
        from tools.calculate_chemistry import calculate_chemistry
        from tools.predict_injury import predict_injury
        from tools.simulate_championship import simulate_championship

        results = {}

        # Step 1: Salary validation
        results["salary"] = salary_validation(
            team_a_players=[player_a],
            team_b_players=[player_b],
            team_a_name=team_a,
            team_b_name=team_b,
        )

        # Step 2: Player predictions
        results["player_a"] = predict_player(player_name=player_a)
        results["player_b"] = predict_player(player_name=player_b)

        # Step 3: Chemistry
        results["chemistry_a"] = calculate_chemistry(
            team=team_a,
            trade_player_out=player_a,
            trade_player_in=player_b,
            trade_team_in=team_b,
        )

        # Step 4: Injury risk
        results["injury_a"] = predict_injury(team=team_a)

        # Step 5: Championship simulation
        results["championship"] = simulate_championship(
            team=team_a,
            simulations=simulations,
            trade_player_a=player_a,
            trade_team_a=team_a,
            trade_player_b=player_b,
            trade_team_b=team_b,
        )

        set_task_result(task_id, {
            "status": "completed",
            "result": results,
        })

        logger.info("bg_task_complete", task_id=task_id, task="full_trade_analysis")

    except Exception as e:
        logger.error("bg_task_error", task_id=task_id, error=str(e))
        set_task_result(task_id, {"status": "failed", "error": str(e)})


async def run_injury_batch_task(
    task_id: str,
    teams: list[str],
) -> None:
    """Background task: Run injury prediction for multiple teams."""
    logger.info("bg_task_start", task_id=task_id, task="injury_batch")

    try:
        from tools.predict_injury import predict_injury

        results = {}
        for team in teams:
            results[team] = predict_injury(team=team)

        set_task_result(task_id, {"status": "completed", "result": results})
        logger.info("bg_task_complete", task_id=task_id, task="injury_batch", teams=len(teams))

    except Exception as e:
        logger.error("bg_task_error", task_id=task_id, error=str(e))
        set_task_result(task_id, {"status": "failed", "error": str(e)})


def create_task_id() -> str:
    """Generate a unique task ID."""
    return str(uuid.uuid4())[:12]
