"""
Tool: salary_validation — Validate trade against NBA salary cap rules.

Wraps the SalaryCapValidator to check CBA compliance, compute
salary differentials, and suggest salary filler if needed.
"""

from __future__ import annotations

from tools.registry import tool
from nba_analysis.config import resolve_team_name


@tool(
    name="salary_validation",
    description=(
        "Validate a trade against NBA salary cap rules (CBA 125% rule). "
        "Checks if incoming salary is within allowed limits for both teams. "
        "Returns detailed salary breakdown and compliance status. "
        "Suggests salary filler players if the trade is invalid."
    ),
)
def salary_validation(
    team_a_players: list,
    team_b_players: list,
    team_a_name: str = "",
    team_b_name: str = "",
) -> dict:
    """Validate a trade against NBA salary cap rules."""
    from salary_cap import SalaryCapValidator

    validator = SalaryCapValidator()

    validation = validator.validate_trade(team_a_players, team_b_players)

    # Get individual salaries
    player_salaries = {}
    for player in team_a_players + team_b_players:
        salary = validator.get_player_salary(player)
        player_salaries[player] = {
            "salary": salary,
            "formatted": validator.format_salary(salary),
        }

    result = {
        "success": True,
        "is_valid": validation.is_valid,
        "reason": validation.reason,
        "details": validation.details,
        "team_a_outgoing_salary": validation.team_a_outgoing_salary,
        "team_a_outgoing_formatted": validator.format_salary(validation.team_a_outgoing_salary),
        "team_b_outgoing_salary": validation.team_b_outgoing_salary,
        "team_b_outgoing_formatted": validator.format_salary(validation.team_b_outgoing_salary),
        "team_a_max_incoming": validation.team_a_max_incoming,
        "team_a_max_incoming_formatted": validator.format_salary(validation.team_a_max_incoming),
        "team_b_max_incoming": validation.team_b_max_incoming,
        "team_b_max_incoming_formatted": validator.format_salary(validation.team_b_max_incoming),
        "player_salaries": player_salaries,
    }

    # Suggest filler if invalid
    if not validation.is_valid and team_a_name:
        team_a_resolved = resolve_team_name(team_a_name)
        gap = abs(validation.team_a_incoming_salary - validation.team_a_max_incoming)
        fillers = validator.suggest_salary_filler(
            team_a_resolved, gap, exclude_players=team_a_players
        )
        result["suggested_filler_team_a"] = fillers

    if not validation.is_valid and team_b_name:
        team_b_resolved = resolve_team_name(team_b_name)
        gap = abs(validation.team_b_incoming_salary - validation.team_b_max_incoming)
        fillers = validator.suggest_salary_filler(
            team_b_resolved, gap, exclude_players=team_b_players
        )
        result["suggested_filler_team_b"] = fillers

    result["explanation"] = (
        f"Trade {'PASSES' if validation.is_valid else 'FAILS'} salary cap validation. "
        f"{validation.reason}. "
        f"Team A sends {validator.format_salary(validation.team_a_outgoing_salary)}, "
        f"Team B sends {validator.format_salary(validation.team_b_outgoing_salary)}."
    )
    result["confidence"] = 0.95

    return result
