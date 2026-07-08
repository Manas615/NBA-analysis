"""
Tool: predict_injury — Predict injury risk and expected games missed.

Wraps the InjuryPredictor to assess injury probability
for individual players or entire team rosters.
"""

from __future__ import annotations

from tools.registry import tool
from nba_analysis.config import resolve_team_name


@tool(
    name="predict_injury",
    description=(
        "Predict injury risk for a player or team roster. "
        "Returns injury probability, expected games missed, risk level, "
        "and adjusted future impact accounting for injury risk. "
        "Can predict for a single player or an entire team."
    ),
)
def predict_injury(
    team: str = "",
    player_name: str = "",
) -> dict:
    """Predict injury risk for a player or team."""
    from injury_predictor import InjuryPredictor

    predictor = InjuryPredictor()

    # Team report
    if team:
        team_resolved = resolve_team_name(team)
        try:
            report = predictor.get_team_injury_report(team_resolved)
        except ValueError as e:
            return {"success": False, "error": str(e)}

        players = []
        high_risk_count = 0
        total_expected_missed = 0.0

        for p in report:
            entry = {
                "player_name": p.player_name,
                "team": p.team,
                "injury_risk": round(p.injury_risk, 3),
                "risk_level": p.risk_level,
                "predicted_games_missed": round(p.predicted_games_missed, 1),
                "original_impact": round(p.original_impact, 4),
                "adjusted_impact": round(p.adjusted_impact, 4),
                "impact_reduction": round(p.original_impact - p.adjusted_impact, 4),
            }
            players.append(entry)
            total_expected_missed += p.predicted_games_missed
            if p.risk_level == "High":
                high_risk_count += 1

        return {
            "success": True,
            "mode": "team_report",
            "team": team_resolved,
            "total_players": len(players),
            "high_risk_count": high_risk_count,
            "total_expected_games_missed": round(total_expected_missed, 1),
            "players": players,
            "explanation": (
                f"{team_resolved} injury report: {high_risk_count} high-risk players, "
                f"{total_expected_missed:.0f} total expected player-games missed. "
                f"Highest risk: {report[0].player_name} ({report[0].injury_risk:.1%})."
                if report else f"No injury data for {team_resolved}."
            ),
            "confidence": 0.70,
        }

    # Single player
    if player_name:
        from nba_analysis.data_loader import get_player_stats_with_predictions

        stats = get_player_stats_with_predictions()
        mask = stats["full_name"].str.lower().str.contains(
            player_name.strip().lower(), na=False
        )
        matches = stats[mask]

        if len(matches) == 0:
            return {"success": False, "error": f"Player '{player_name}' not found"}

        player = matches.iloc[0]

        risk = predictor.predict_injury_risk({
            "seasons_in_league": 5,
            "minutes_load": float(player.get("numMinutes", 0) or 0),
            "usage_burden": float(player.get("usagePercentage", 0) or 0),
            "scoring_load": float(player.get("points", 0) or 0),
            "availability_ratio": 0.80,
            "hist_availability": 0.80,
        })

        risk_level = predictor.get_risk_level(risk)
        games_missed = risk * 82

        return {
            "success": True,
            "mode": "single_player",
            "player_name": str(player.get("full_name", "")),
            "team": str(player.get("playerteamName", "")),
            "injury_risk": round(risk, 3),
            "risk_level": risk_level,
            "predicted_games_missed": round(games_missed, 1),
            "explanation": (
                f"{player.get('full_name', '')} has {risk_level} injury risk ({risk:.1%}). "
                f"Expected to miss ~{games_missed:.0f} games next season. "
                f"Based on {player.get('numMinutes', 0):.0f} MPG workload and "
                f"{player.get('usagePercentage', 0):.0f}% usage rate."
            ),
            "confidence": 0.68,
        }

    return {"success": False, "error": "Provide either 'team' or 'player_name'"}
