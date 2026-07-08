"""
Free Agency Agent — Recommends free agent signings.

Responsibilities:
- Identify best available players
- Find cheapest upgrades (value picks)
- Discover sleeper players
- Match recommendations to team needs
"""

from __future__ import annotations

from agents.base import BaseAgent


class FreeAgencyAgent(BaseAgent):
    """Agent specialized in free agency recommendations."""

    agent_name = "FreeAgencyAgent"

    system_prompt = """You are the Free Agency Agent for the NBA Trade Simulator.

Your job is to recommend free agent signings for a team.

WORKFLOW:
1. Call recommend_free_agents with the team name and budget
2. Analyze each recommendation's fit with the team
3. Prioritize based on team needs

THREE CATEGORIES OF RECOMMENDATIONS:
1. Best Available: Highest impact players regardless of cost
2. Cheapest Upgrades: Best value per dollar (impact/salary ratio)
3. Sleeper Picks: Under-the-radar players with upside

RULES:
- Always consider salary constraints
- Match recommendations to team needs (position, playstyle)
- Sleepers should have clear upside arguments
- Include value score (impact per $1M) for each player
"""

    available_tools = ["recommend_free_agents", "predict_player", "calculate_chemistry"]

    def _derive_tool_args(self, tool_name, user_query, context):
        if not context:
            return None
        if tool_name == "recommend_free_agents":
            return {
                "team": context.get("team", ""),
                "budget": context.get("budget", 10_000_000),
            }
        elif tool_name == "predict_player":
            if context.get("player_name"):
                return {"player_name": context["player_name"]}
        elif tool_name == "calculate_chemistry":
            return {"team": context.get("team", "")}
        return None
