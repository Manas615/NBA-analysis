"""
Tool: predict_matchup — Predict win probability between two teams.

Wraps the Logistic Regression matchup model to compute
head-to-head win probability with team strength differentials.
"""

from __future__ import annotations

from tools.registry import tool
from nba_analysis.data_loader import get_player_stats_with_predictions
from nba_analysis.team_utils import (
    get_team_roster,
    calculate_team_strength,
    predict_win_probability,
)
from nba_analysis.config import resolve_team_name


@tool(
    name="predict_matchup",
    description=(
        "Predict win probability for a head-to-head matchup between two NBA teams. "
        "Returns Team A's win probability, strength metrics for both teams, "
        "and contributing factors."
    ),
)
def predict_matchup(team_a: str, team_b: str) -> dict:
    """Predict head-to-head win probability between two teams."""
    team_a_resolved = resolve_team_name(team_a)
    team_b_resolved = resolve_team_name(team_b)

    stats = get_player_stats_with_predictions()

    try:
        roster_a = get_team_roster(team_a_resolved, stats)
        roster_b = get_team_roster(team_b_resolved, stats)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    strength_a = calculate_team_strength(roster_a)
    strength_b = calculate_team_strength(roster_b)

    win_prob = predict_win_probability(strength_a, strength_b)

    # Determine advantage
    net_rating_diff = strength_a.net_rating - strength_b.net_rating
    if win_prob > 60:
        advantage = f"{team_a_resolved} has a strong advantage"
    elif win_prob > 52:
        advantage = f"{team_a_resolved} has a slight advantage"
    elif win_prob > 48:
        advantage = "This is essentially a coin flip"
    elif win_prob > 40:
        advantage = f"{team_b_resolved} has a slight advantage"
    else:
        advantage = f"{team_b_resolved} has a strong advantage"

    return {
        "success": True,
        "team_a": team_a_resolved,
        "team_b": team_b_resolved,
        "team_a_win_probability": round(win_prob, 2),
        "team_b_win_probability": round(100 - win_prob, 2),
        "team_a_strength": {
            "net_rating": round(strength_a.net_rating, 2),
            "offensive_rating": round(strength_a.offensive_rating, 2),
            "defensive_rating": round(strength_a.defensive_rating, 2),
        },
        "team_b_strength": {
            "net_rating": round(strength_b.net_rating, 2),
            "offensive_rating": round(strength_b.offensive_rating, 2),
            "defensive_rating": round(strength_b.defensive_rating, 2),
        },
        "net_rating_differential": round(net_rating_diff, 2),
        "explanation": (
            f"{team_a_resolved} vs {team_b_resolved}: "
            f"{team_a_resolved} wins {win_prob:.1f}% of the time. "
            f"{advantage}. Net rating differential: {net_rating_diff:+.2f}."
        ),
        "confidence": 0.78,
    }
