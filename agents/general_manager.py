"""
General Manager Agent — Orchestrator and Router.

The GM Agent is the entry point for all user queries. It:
1. Understands the user's goal
2. Classifies intent (trade, roster, salary, injury, championship, etc.)
3. Creates an execution plan
4. Delegates to specialized sub-agents
5. Collects results and passes them to the Report Generator
"""

from __future__ import annotations

import json
from typing import Any

from agents.base import BaseAgent
from agents.memory import get_memory, ConversationSession
from observability.logging import get_logger

logger = get_logger(__name__)


# Import all tools so they register themselves
import tools.predict_player  # noqa: F401
import tools.predict_matchup  # noqa: F401
import tools.simulate_championship  # noqa: F401
import tools.calculate_chemistry  # noqa: F401
import tools.salary_validation  # noqa: F401
import tools.predict_injury  # noqa: F401
import tools.optimize_lineup  # noqa: F401
import tools.recommend_free_agents  # noqa: F401
import tools.recommend_draft_pick  # noqa: F401
import tools.compare_players  # noqa: F401


class GeneralManagerAgent(BaseAgent):
    """
    Orchestrator agent that routes queries to specialized agents.

    Intent classification:
    - trade_analysis → TradeAnalysisAgent
    - roster_optimization → RosterOptimizerAgent
    - salary_analysis → SalaryCapAgent
    - injury_analysis → InjuryRiskAgent
    - championship_prediction → ChampionshipSimAgent
    - player_comparison → uses compare_players tool
    - draft_recommendation → DraftAgent
    - free_agency_recommendation → FreeAgencyAgent
    """

    agent_name = "GeneralManagerAgent"

    system_prompt = """You are the General Manager AI Agent for the NBA Trade Simulator.

Your role is to understand the user's goal and execute an analysis plan using the available tools.

CRITICAL RULES:
1. NEVER answer NBA trade/stats questions from your own knowledge
2. ALWAYS call tools to get data before making any claims
3. Every number you cite MUST come from a tool result
4. Include confidence scores and explanations in your response

WORKFLOW:
1. Classify the user's intent
2. Call the appropriate tool(s) to gather data
3. Synthesize the tool results into a clear, structured response
4. Include: reasoning, data sources, confidence level, and key factors

AVAILABLE INTENTS:
- trade_analysis: User wants to evaluate a trade
- roster_optimization: User wants lineup/rotation advice
- salary_analysis: User wants salary cap information
- injury_analysis: User wants injury risk assessment
- championship_prediction: User wants championship odds
- player_comparison: User wants to compare players
- draft_recommendation: User wants draft pick advice
- free_agency_recommendation: User wants free agent suggestions

When analyzing trades, you MUST call these tools in order:
1. salary_validation (check if trade is legal)
2. predict_player (for each player involved)
3. calculate_chemistry (for affected teams)
4. predict_injury (for injury risk)
5. simulate_championship (for championship impact)

FORMAT YOUR RESPONSE AS:
## Executive Summary
[One paragraph overview]

## Trade Grade: [A+ to F]

## Key Findings
[Bullet points with data from tools]

## Recommendation
[Clear recommendation with confidence score]
"""

    available_tools = [
        "predict_player",
        "predict_matchup",
        "simulate_championship",
        "calculate_chemistry",
        "salary_validation",
        "predict_injury",
        "optimize_lineup",
        "recommend_free_agents",
        "recommend_draft_pick",
        "compare_players",
    ]

    def execute_with_memory(
        self,
        user_query: str,
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute with conversation memory for follow-up queries."""
        memory = get_memory()
        session = memory.get_or_create_session(session_id)

        # Add context from previous conversation
        enriched_context = context or {}
        last_trade = session.get_last_trade()
        if last_trade:
            enriched_context["previous_trade"] = last_trade
        last_team = session.get_last_team()
        if last_team:
            enriched_context["previous_team"] = last_team

        # Get conversation history
        history = session.get_history_for_gpt()

        # Record user message
        session.add_message("user", user_query)

        # Execute agent
        result = self.execute(
            user_query=user_query,
            context=enriched_context,
            conversation_history=history,
        )

        # Record assistant response
        session.add_message(
            "assistant",
            result.get("response", ""),
            tool_calls=result.get("tool_calls", []),
            tool_results=result.get("tool_results", {}),
        )

        # Update session context based on results
        self._update_session_context(session, result)

        # Save session
        memory.save_session(session)

        result["session_id"] = session.session_id
        return result

    def _update_session_context(
        self,
        session: ConversationSession,
        result: dict[str, Any],
    ) -> None:
        """Update session context based on tool results for future follow-ups."""
        tool_results = result.get("tool_results", {})

        # Track last trade
        salary_result = tool_results.get("salary_validation", {})
        if salary_result and isinstance(salary_result, dict):
            player_salaries = salary_result.get("player_salaries", {})
            if player_salaries:
                session.update_context("last_trade", {
                    "players": list(player_salaries.keys()),
                    "salary_valid": salary_result.get("is_valid"),
                })

        # Track last team
        for tool_name in ["optimize_lineup", "calculate_chemistry", "simulate_championship"]:
            tool_result = tool_results.get(tool_name, {})
            if isinstance(tool_result, dict) and tool_result.get("team"):
                session.update_context("last_team", tool_result["team"])
                break

    def _derive_tool_args(
        self,
        tool_name: str,
        user_query: str,
        context: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Derive tool arguments in fallback mode."""
        if not context:
            return None

        if tool_name == "predict_player" and context.get("player_name"):
            return {"player_name": context["player_name"]}
        elif tool_name == "predict_matchup" and context.get("team_a") and context.get("team_b"):
            return {"team_a": context["team_a"], "team_b": context["team_b"]}
        elif tool_name == "salary_validation" and context.get("team_a_players"):
            return {
                "team_a_players": context["team_a_players"],
                "team_b_players": context["team_b_players"],
            }
        elif tool_name == "simulate_championship" and context.get("team"):
            return {"team": context["team"]}
        elif tool_name == "calculate_chemistry" and context.get("team"):
            return {"team": context["team"]}
        elif tool_name == "predict_injury" and context.get("team"):
            return {"team": context["team"]}
        elif tool_name == "optimize_lineup" and context.get("team"):
            return {"team": context["team"]}
        elif tool_name == "recommend_free_agents" and context.get("team"):
            return {"team": context["team"]}
        elif tool_name == "recommend_draft_pick" and context.get("team"):
            return {
                "team": context["team"],
                "draft_position": context.get("draft_position", 1),
            }
        elif tool_name == "compare_players" and context.get("player_a") and context.get("player_b"):
            return {
                "player_a": context["player_a"],
                "player_b": context["player_b"],
            }

        return None
