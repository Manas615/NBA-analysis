"""
Championship Simulation Agent — Monte Carlo championship prediction.

Responsibilities:
- Run Monte Carlo simulations
- Predict expected wins, playoff/conference/finals/championship probabilities
- Compare before/after trade scenarios
"""

from __future__ import annotations

from agents.base import BaseAgent


class ChampionshipSimAgent(BaseAgent):
    """Agent specialized in championship probability simulation."""

    agent_name = "ChampionshipSimAgent"

    system_prompt = """You are the Championship Simulation Agent for the NBA Trade Simulator.

Your job is to run Monte Carlo simulations to predict championship odds.

WORKFLOW:
1. Call simulate_championship with the team name
2. If evaluating a trade, include trade parameters for before/after comparison
3. Interpret the results in basketball context

WHAT YOU PREDICT:
- Expected wins (out of 82)
- Playoff probability
- Conference Finals probability
- NBA Finals probability
- Championship probability

RULES:
- Always cite the number of simulations run
- More simulations = higher confidence
- Explain what drives the odds (team strength, conference, etc.)
- Compare to other contenders when relevant
- Note that simulations have inherent variance

OUTPUT FORMAT:
## Championship Odds: [Team]
### Expected Record: [W]-[L]
### Probabilities
| Stage | Probability |
|-------|-------------|
| Playoffs | X% |
| Conference Finals | X% |
| NBA Finals | X% |
| Championship | X% |

### Confidence: [based on simulation count]
"""

    available_tools = ["simulate_championship", "predict_matchup"]

    def _derive_tool_args(self, tool_name, user_query, context):
        if not context:
            return None
        if tool_name == "simulate_championship":
            return {
                "team": context.get("team", ""),
                "simulations": context.get("simulations", 200),
                "trade_player_a": context.get("player_a", ""),
                "trade_team_a": context.get("team_a", ""),
                "trade_player_b": context.get("player_b", ""),
                "trade_team_b": context.get("team_b", ""),
            }
        elif tool_name == "predict_matchup":
            if context.get("team_a") and context.get("team_b"):
                return {"team_a": context["team_a"], "team_b": context["team_b"]}
        return None
