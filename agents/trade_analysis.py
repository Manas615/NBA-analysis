"""
Trade Analysis Agent — Validates and evaluates NBA trades.

Responsibilities:
- Validate the trade is executable
- Evaluate player value on both sides
- Calculate team improvement/decline
- Compare before vs. after roster
- Generate a detailed trade report
"""

from __future__ import annotations

from agents.base import BaseAgent


class TradeAnalysisAgent(BaseAgent):
    """Agent specialized in trade evaluation and analysis."""

    agent_name = "TradeAnalysisAgent"

    system_prompt = """You are the Trade Analysis Agent for the NBA Trade Simulator.

Your job is to thoroughly evaluate a proposed NBA trade using the available tools.

WORKFLOW:
1. First, call salary_validation to check CBA compliance
2. Call predict_player for each player involved to get their projected value
3. Call calculate_chemistry for affected teams to assess fit changes
4. Call predict_matchup to see how team strength changes
5. Synthesize all data into a trade evaluation

RULES:
- NEVER speculate about player stats. Always call predict_player first.
- NEVER guess salary numbers. Always call salary_validation first.
- Grade the trade from A+ to F based on combined factors
- Clearly state which team wins the trade and why
- Include quantitative evidence for every claim
- Note any red flags (injury risk, chemistry issues, salary problems)

OUTPUT FORMAT:
## Trade Evaluation: [Player A] ↔ [Player B]

### Salary Cap Compliance
[Results from salary_validation]

### Player Value Assessment
[Results from predict_player for each player]

### Chemistry Impact
[Results from calculate_chemistry]

### Team Strength Change
[Results from predict_matchup]

### Trade Grade: [A+ to F]
### Winner: [Team]
### Confidence: [0-100%]
"""

    available_tools = [
        "predict_player",
        "predict_matchup",
        "salary_validation",
        "calculate_chemistry",
        "predict_injury",
    ]

    def _derive_tool_args(self, tool_name, user_query, context):
        if not context:
            return None

        if tool_name == "salary_validation":
            players_a = context.get("team_a_players", [])
            players_b = context.get("team_b_players", [])
            if players_a and players_b:
                return {
                    "team_a_players": players_a,
                    "team_b_players": players_b,
                    "team_a_name": context.get("team_a", ""),
                    "team_b_name": context.get("team_b", ""),
                }
        elif tool_name == "predict_player":
            player = context.get("player_a") or context.get("player_name")
            if player:
                return {"player_name": player}
        elif tool_name == "calculate_chemistry":
            team = context.get("team_a") or context.get("team")
            if team:
                return {"team": team}
        elif tool_name == "predict_matchup":
            if context.get("team_a") and context.get("team_b"):
                return {"team_a": context["team_a"], "team_b": context["team_b"]}
        elif tool_name == "predict_injury":
            team = context.get("team_a") or context.get("team")
            if team:
                return {"team": team}

        return None
