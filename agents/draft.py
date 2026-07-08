"""
Draft Agent — Recommends draft picks based on team needs.

Responsibilities:
- Analyze team positional needs
- Recommend best prospect at given draft position
- Provide ceiling projection and player comparison
"""

from __future__ import annotations

from agents.base import BaseAgent


class DraftAgent(BaseAgent):
    """Agent specialized in NBA draft recommendations."""

    agent_name = "DraftAgent"

    system_prompt = """You are the Draft Agent for the NBA Trade Simulator.

Your job is to recommend draft picks based on team needs and prospect analysis.

WORKFLOW:
1. Call recommend_draft_pick with team and draft position
2. Analyze the team's positional needs
3. Match prospects to needs
4. Provide detailed scouting report

WHAT YOU EVALUATE:
- Team positional gaps (where is the roster weakest?)
- Prospect ceiling (superstar, all-star, starter, rotation)
- Player comparison (historical NBA comp)
- Fit score (how well does the prospect fill the need?)
- Strengths and weaknesses

RULES:
- Best Player Available (BPA) vs. need is a real tension — address it
- Higher ceiling prospects are worth more risk
- Always provide the player comparison
- Include both strengths AND weaknesses
"""

    available_tools = ["recommend_draft_pick", "optimize_lineup"]

    def _derive_tool_args(self, tool_name, user_query, context):
        if not context:
            return None
        if tool_name == "recommend_draft_pick":
            return {
                "team": context.get("team", ""),
                "draft_position": context.get("draft_position", 1),
            }
        elif tool_name == "optimize_lineup":
            return {"team": context.get("team", "")}
        return None
