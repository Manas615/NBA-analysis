"""
Roster Optimizer Agent — Optimal lineup and rotation advice.

Responsibilities:
- Suggest best starting five
- Recommend bench rotation
- Allocate minutes optimally
- Identify positional gaps
"""

from __future__ import annotations

from agents.base import BaseAgent


class RosterOptimizerAgent(BaseAgent):
    """Agent specialized in roster optimization."""

    agent_name = "RosterOptimizerAgent"

    system_prompt = """You are the Roster Optimizer Agent for the NBA Trade Simulator.

Your job is to optimize a team's lineup and rotation.

WORKFLOW:
1. Call optimize_lineup to get the optimal starting five, bench, and minutes
2. Call calculate_chemistry to assess how the lineup fits together
3. Synthesize into actionable rotation advice

WHAT YOU RECOMMEND:
- Best starting five (with positions)
- Best bench rotation (6th-10th man)
- Minute allocation for each player
- Optimal rotation patterns

RULES:
- Balance impact with positional fit
- Consider minutes load and injury risk
- Bench depth matters for playoff success
- Depth ratio (bench impact / starter impact) should be > 0.4 for contenders
- Always explain your lineup choices
"""

    available_tools = ["optimize_lineup", "calculate_chemistry", "predict_injury"]

    def _derive_tool_args(self, tool_name, user_query, context):
        if not context:
            return None
        if tool_name == "optimize_lineup":
            return {"team": context.get("team", "")}
        elif tool_name == "calculate_chemistry":
            return {"team": context.get("team", "")}
        elif tool_name == "predict_injury":
            return {"team": context.get("team", "")}
        return None
