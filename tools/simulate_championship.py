"""
Tool: simulate_championship — Run Monte Carlo championship simulation.

Wraps the ChampionshipSimulator to estimate playoff
probabilities, conference finals odds, and championship probability.
"""

from __future__ import annotations

from tools.registry import tool
from nba_analysis.config import resolve_team_name, FAST_SIMULATIONS


@tool(
    name="simulate_championship",
    description=(
        "Run a Monte Carlo simulation to predict championship odds for a team. "
        "Simulates thousands of NBA seasons to estimate expected wins, "
        "playoff probability, conference finals, NBA Finals, and championship probability. "
        "Can also compare odds before and after a trade."
    ),
)
def simulate_championship(
    team: str,
    simulations: int = 200,
    trade_player_a: str = "",
    trade_team_a: str = "",
    trade_player_b: str = "",
    trade_team_b: str = "",
) -> dict:
    """Run championship simulation for a team, optionally with a trade scenario."""
    from championship_simulator import ChampionshipSimulator

    team_resolved = resolve_team_name(team)
    sim_count = min(max(simulations, 50), 2000)

    # Trade scenario
    if trade_player_a and trade_team_a and trade_player_b and trade_team_b:
        team_a = resolve_team_name(trade_team_a)
        team_b = resolve_team_name(trade_team_b)

        try:
            sim = ChampionshipSimulator(n_simulations=sim_count)
            results = sim.simulate_with_trade(
                team_a, trade_player_a,
                team_b, trade_player_b,
                verbose=False,
            )

            focus = results.get("focus_teams", {})
            comparison = {}
            for t_name, data in focus.items():
                comparison[t_name] = {
                    "wins_before": round(data["wins_before"], 1),
                    "wins_after": round(data["wins_after"], 1),
                    "wins_change": round(data["wins_change"], 1),
                    "championship_before": f"{data['champ_before']:.1%}",
                    "championship_after": f"{data['champ_after']:.1%}",
                    "championship_change": f"{data['champ_change']:+.1%}",
                }

            return {
                "success": True,
                "scenario": "trade_comparison",
                "simulations": sim_count,
                "trade": f"{trade_player_a} ({team_a}) ↔ {trade_player_b} ({team_b})",
                "comparison": comparison,
                "explanation": (
                    f"After simulating {sim_count} seasons: "
                    + "; ".join(
                        f"{t}: {d['championship_change']} championship odds"
                        for t, d in comparison.items()
                    )
                ),
                "confidence": min(0.60 + sim_count / 5000, 0.92),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # Single team scenario
    try:
        sim = ChampionshipSimulator(n_simulations=sim_count)
        results = sim.run(verbose=False)
        odds = results.get_team(team_resolved)

        return {
            "success": True,
            "scenario": "baseline",
            "simulations": sim_count,
            "team": odds.team,
            "conference": odds.conference,
            "expected_wins": round(odds.expected_wins, 1),
            "expected_losses": round(odds.expected_losses, 1),
            "win_std": round(odds.win_std, 1),
            "playoff_probability": f"{odds.playoff_probability:.1%}",
            "conference_finals_probability": f"{odds.conf_finals_probability:.1%}",
            "finals_probability": f"{odds.finals_probability:.1%}",
            "championship_probability": f"{odds.championship_probability:.1%}",
            "explanation": (
                f"{odds.team} ({odds.conference}ern Conference): "
                f"Expected record {odds.expected_wins:.0f}-{odds.expected_losses:.0f}. "
                f"Playoff: {odds.playoff_probability:.0%}, "
                f"Championship: {odds.championship_probability:.1%}."
            ),
            "confidence": min(0.60 + sim_count / 5000, 0.92),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
