"""
Tool: recommend_free_agents — Recommend free agent signings.

Analyzes available players to find best fits, cheapest upgrades,
and sleeper picks for a given team.
"""

from __future__ import annotations

from tools.registry import tool
from nba_analysis.data_loader import get_player_stats_with_predictions
from nba_analysis.team_utils import get_team_roster, calculate_team_strength
from nba_analysis.config import resolve_team_name


@tool(
    name="recommend_free_agents",
    description=(
        "Recommend free agent signings for a team. "
        "Returns best available players, cheapest upgrades, and sleeper picks "
        "based on team needs, salary constraints, and player impact projections."
    ),
)
def recommend_free_agents(team: str, budget: float = 10_000_000.0) -> dict:
    """Recommend free agents for a team."""
    from salary_cap import SalaryCapValidator

    team_resolved = resolve_team_name(team)
    stats = get_player_stats_with_predictions()
    validator = SalaryCapValidator()

    try:
        roster = get_team_roster(team_resolved, stats)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    # Current team players
    team_players = set(roster["full_name"].str.lower().dropna().tolist())
    strength = calculate_team_strength(roster)

    # Find non-team players sorted by impact
    all_players = stats[
        ~stats["full_name"].str.lower().isin(team_players)
    ].copy()
    all_players = all_players.sort_values("future_impact", ascending=False)

    # Get salaries
    def get_salary(name: str) -> float:
        return validator.get_player_salary(name)

    # Best available (top impact)
    best_available = []
    for _, player in all_players.head(30).iterrows():
        name = str(player.get("full_name", ""))
        salary = get_salary(name)
        if salary > 0:
            best_available.append({
                "player": name,
                "team": str(player.get("playerteamName", "")),
                "impact": round(float(player.get("future_impact", 0)), 4),
                "salary": validator.format_salary(salary),
                "salary_raw": salary,
                "points": round(float(player.get("points", 0) or 0), 1),
                "assists": round(float(player.get("assists", 0) or 0), 1),
                "rebounds": round(float(player.get("reboundsTotal", 0) or 0), 1),
            })
        if len(best_available) >= 5:
            break

    # Cheapest upgrades (good impact, low salary)
    cheap_upgrades = []
    for _, player in all_players.iterrows():
        name = str(player.get("full_name", ""))
        impact = float(player.get("future_impact", 0))
        salary = get_salary(name)

        if 0 < salary <= budget and impact > 0.03:
            value = impact / max(salary / 1_000_000, 0.1)
            cheap_upgrades.append({
                "player": name,
                "team": str(player.get("playerteamName", "")),
                "impact": round(impact, 4),
                "salary": validator.format_salary(salary),
                "salary_raw": salary,
                "value_score": round(value, 4),
            })

        if len(cheap_upgrades) >= 20:
            break

    cheap_upgrades.sort(key=lambda x: x["value_score"], reverse=True)
    cheap_upgrades = cheap_upgrades[:5]

    # Sleeper picks (moderate impact, very low salary, high upside)
    sleepers = []
    low_salary_players = all_players[
        all_players["future_impact"] > 0.02
    ].tail(50)

    for _, player in low_salary_players.iterrows():
        name = str(player.get("full_name", ""))
        impact = float(player.get("future_impact", 0))
        salary = get_salary(name)

        if 0 < salary <= budget * 0.3:
            sleepers.append({
                "player": name,
                "team": str(player.get("playerteamName", "")),
                "impact": round(impact, 4),
                "salary": validator.format_salary(salary),
                "upside_note": (
                    "High efficiency per dollar"
                    if impact / max(salary / 1_000_000, 0.1) > 0.02
                    else "Developmental upside"
                ),
            })
        if len(sleepers) >= 5:
            break

    return {
        "success": True,
        "team": team_resolved,
        "budget": validator.format_salary(budget),
        "best_available": best_available,
        "cheapest_upgrades": cheap_upgrades,
        "sleeper_picks": sleepers,
        "explanation": (
            f"Free agent recommendations for {team_resolved} "
            f"(budget: {validator.format_salary(budget)}). "
            f"Found {len(best_available)} top targets, "
            f"{len(cheap_upgrades)} value picks, and "
            f"{len(sleepers)} sleepers."
        ),
        "confidence": 0.65,
    }
