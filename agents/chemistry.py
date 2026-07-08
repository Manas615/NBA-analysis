"""
Chemistry Agent — Evaluates lineup chemistry and fit.

Responsibilities:
- Predict lineup chemistry (Roster Synergy Index)
- Evaluate spacing, offensive/defensive fit
- Identify usage conflicts and playmaking compatibility
- Return a chemistry score
"""

from __future__ import annotations

from agents.base import BaseAgent


class ChemistryAgent(BaseAgent):
    """Agent specialized in lineup chemistry evaluation."""

    agent_name = "ChemistryAgent"

    system_prompt = """You are the Chemistry Agent for the NBA Trade Simulator.

Your job is to evaluate how well players fit together on a roster.

WHAT YOU EVALUATE:
1. Roster Synergy Index (RSI): Overall chemistry score 0-1
   - < 0.3 = Poor chemistry (usage conflicts)
   - 0.3-0.6 = Average chemistry
   - 0.6-0.8 = Good chemistry (complementary styles)
   - > 0.8 = Elite chemistry (rare)

2. Pairwise chemistry between key player pairs:
   - Usage conflict: Do players fight for the ball?
   - Assist synergy: Do playmakers complement scorers?
   - Shooting spacing: Do shooters spread the floor?
   - Minutes overlap: Do rotation patterns work?

WORKFLOW:
1. Call calculate_chemistry for the team
2. If trade scenario, include trade parameters for before/after
3. Identify the best and worst player pairs
4. Explain chemistry in basketball terms

RULES:
- Two ball-dominant guards = usage conflict = bad chemistry
- One playmaker + shooters = good spacing = good chemistry
- Always explain WHY chemistry is good or bad with specific pairs
- Chemistry affects team strength by the multiplier (0.70 to 1.00)
"""

    available_tools = ["calculate_chemistry", "predict_player"]

    def _derive_tool_args(self, tool_name, user_query, context):
        if not context:
            return None
        if tool_name == "calculate_chemistry":
            return {
                "team": context.get("team", ""),
                "trade_player_out": context.get("trade_player_out", ""),
                "trade_player_in": context.get("trade_player_in", ""),
                "trade_team_in": context.get("trade_team_in", ""),
            }
        elif tool_name == "predict_player":
            if context.get("player_name"):
                return {"player_name": context["player_name"]}
        return None
