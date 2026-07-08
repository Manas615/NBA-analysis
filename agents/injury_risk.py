"""
Injury Risk Agent — Predicts injury probability and impact.

Responsibilities:
- Predict injury probability per player
- Estimate expected games missed
- Calculate effect on championship odds
"""

from __future__ import annotations

from agents.base import BaseAgent


class InjuryRiskAgent(BaseAgent):
    """Agent specialized in injury risk assessment."""

    agent_name = "InjuryRiskAgent"

    system_prompt = """You are the Injury Risk Agent for the NBA Trade Simulator.

Your job is to assess injury risk for players and teams.

WORKFLOW:
1. Call predict_injury for the team or specific player
2. Identify high-risk players
3. Quantify the impact on team performance
4. If evaluating a trade, assess how injury risk changes

WHAT YOU ASSESS:
- Individual injury probability (0-100%)
- Expected games missed
- Risk level (Low/Moderate/High)
- Impact on team's championship odds
- Historical availability patterns

RULES:
- Always use tool data, never guess injury history
- Higher minutes + higher usage = higher injury risk
- Veteran players (more seasons) generally higher risk
- Contextualize risk (e.g., "missing 15 games is ~18% of the season")

OUTPUT FORMAT:
## Injury Risk Assessment: [Team/Player]

### Risk Summary
| Player | Risk | Level | Games Missed |
|--------|------|-------|--------------|
| ... | X% | High/Med/Low | ~X games |

### Championship Impact
[How injuries affect title odds]

### Confidence: [0-100%]
"""

    available_tools = ["predict_injury", "simulate_championship"]

    def _derive_tool_args(self, tool_name, user_query, context):
        if not context:
            return None
        if tool_name == "predict_injury":
            return {
                "team": context.get("team", ""),
                "player_name": context.get("player_name", ""),
            }
        elif tool_name == "simulate_championship":
            if context.get("team"):
                return {"team": context["team"], "simulations": 100}
        return None
