"""
Report Generator Agent — Synthesizes outputs from all agents.

Combines results from specialized agents into a unified executive report
with grade, impact analysis, and actionable recommendations.
"""

from __future__ import annotations

import json
from typing import Any

from agents.base import BaseAgent
from observability.logging import get_logger

logger = get_logger(__name__)


class ReportGeneratorAgent(BaseAgent):
    """Agent that synthesizes all sub-agent outputs into a final report."""

    agent_name = "ReportGeneratorAgent"

    system_prompt = """You are the Report Generator Agent for the NBA Trade Simulator.

Your job is to combine outputs from multiple analysis agents into a single
executive report. You will receive pre-computed results from other agents.

DO NOT call any tools yourself. Your role is SYNTHESIS ONLY.

REPORT FORMAT (follow this exactly):

## Executive Summary
[2-3 sentence overview of the analysis]

## Trade Grade: [A+ / A / A- / B+ / B / B- / C+ / C / C- / D+ / D / D- / F]

## Player Impact
[For each player involved, summarize their projected value and contribution]

## Salary Analysis
[Salary cap compliance status and breakdown]

## Championship Odds
[How the trade/decision affects championship probability]

## Chemistry Impact
[How lineup chemistry changes]

## Risk Assessment
[Injury risks, salary risks, chemistry risks]

## Recommendation
[Clear, actionable recommendation with reasoning]

## Confidence Score: [0-100]%
[Based on data quality, simulation count, and analysis depth]

RULES:
- Every number MUST come from the provided agent results
- Be specific and quantitative, not vague
- The trade grade should reflect the combined analysis
- Confidence score reflects how much data supports the conclusion
"""

    available_tools = []  # Report generator doesn't call tools

    def generate_report(
        self,
        query: str,
        agent_results: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Generate a unified report from multiple agent outputs.

        Parameters
        ----------
        query : str
            Original user query.
        agent_results : dict
            Results from each sub-agent, keyed by agent name.
        """
        # Build context with all agent results
        context = {
            "original_query": query,
            "agent_results": agent_results,
        }

        # In GPT mode, send all results for synthesis
        if self.client:
            enriched_query = (
                f"Generate a comprehensive report for this query: '{query}'\n\n"
                f"Here are the results from each analysis agent:\n\n"
                f"{json.dumps(agent_results, default=str, indent=2)}"
            )
            return self.execute(enriched_query, context=context)

        # Fallback: construct report from raw data
        return self._build_fallback_report(query, agent_results)

    def _build_fallback_report(
        self,
        query: str,
        agent_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a structured report without GPT synthesis."""
        report_sections = []

        # Executive Summary
        report_sections.append("## Executive Summary")
        report_sections.append(f"Analysis for: {query}\n")

        # Trade Grade
        grade = self._compute_grade(agent_results)
        report_sections.append(f"## Trade Grade: {grade}")

        # Player Impact
        player_results = agent_results.get("predict_player", {})
        if player_results and isinstance(player_results, dict):
            report_sections.append("\n## Player Impact")
            if player_results.get("explanation"):
                report_sections.append(player_results["explanation"])

        # Salary Analysis
        salary_results = agent_results.get("salary_validation", {})
        if salary_results and isinstance(salary_results, dict):
            report_sections.append("\n## Salary Analysis")
            status = "✅ VALID" if salary_results.get("is_valid") else "❌ INVALID"
            report_sections.append(f"Status: {status}")
            if salary_results.get("explanation"):
                report_sections.append(salary_results["explanation"])

        # Championship Odds
        champ_results = agent_results.get("simulate_championship", {})
        if champ_results and isinstance(champ_results, dict):
            report_sections.append("\n## Championship Odds")
            if champ_results.get("explanation"):
                report_sections.append(champ_results["explanation"])

        # Chemistry
        chem_results = agent_results.get("calculate_chemistry", {})
        if chem_results and isinstance(chem_results, dict):
            report_sections.append("\n## Chemistry Impact")
            if chem_results.get("explanation"):
                report_sections.append(chem_results["explanation"])

        # Risk Assessment
        injury_results = agent_results.get("predict_injury", {})
        if injury_results and isinstance(injury_results, dict):
            report_sections.append("\n## Risk Assessment")
            if injury_results.get("explanation"):
                report_sections.append(injury_results["explanation"])

        # Recommendation
        report_sections.append("\n## Recommendation")
        report_sections.append(
            "Based on the combined analysis above, review all sections "
            "to make an informed decision."
        )

        # Confidence
        confidence = self._compute_confidence(agent_results)
        report_sections.append(f"\n## Confidence Score: {confidence}%")

        report_text = "\n".join(report_sections)

        return {
            "agent": self.agent_name,
            "response": report_text,
            "trade_grade": grade,
            "confidence_score": confidence,
            "sections": {
                "salary": salary_results,
                "championship": champ_results,
                "chemistry": chem_results,
                "injury": injury_results,
                "player": player_results,
            },
            "tool_calls": [],
            "tool_results": agent_results,
            "request_id": self.request_id,
        }

    def _compute_grade(self, results: dict[str, Any]) -> str:
        """Compute an overall trade grade from agent results."""
        score = 50  # Start at C

        # Salary compliance
        salary = results.get("salary_validation", {})
        if isinstance(salary, dict):
            if salary.get("is_valid"):
                score += 10
            else:
                score -= 20

        # Chemistry
        chem = results.get("calculate_chemistry", {})
        if isinstance(chem, dict):
            rsi = chem.get("rsi", 0.5)
            if rsi > 0.6:
                score += 15
            elif rsi < 0.3:
                score -= 15

            trade_comp = chem.get("trade_comparison", {})
            if trade_comp:
                change = trade_comp.get("rsi_change", 0)
                score += int(change * 50)

        # Championship
        champ = results.get("simulate_championship", {})
        if isinstance(champ, dict):
            comparison = champ.get("comparison", {})
            for team_data in comparison.values():
                if isinstance(team_data, dict):
                    champ_change = team_data.get("championship_change", "0%")
                    if isinstance(champ_change, str):
                        try:
                            pct = float(champ_change.strip("%+")) / 100
                            score += int(pct * 200)
                        except ValueError:
                            pass

        # Map score to grade
        if score >= 90:
            return "A+"
        elif score >= 85:
            return "A"
        elif score >= 80:
            return "A-"
        elif score >= 75:
            return "B+"
        elif score >= 70:
            return "B"
        elif score >= 65:
            return "B-"
        elif score >= 60:
            return "C+"
        elif score >= 50:
            return "C"
        elif score >= 40:
            return "C-"
        elif score >= 30:
            return "D+"
        elif score >= 20:
            return "D"
        elif score >= 10:
            return "D-"
        return "F"

    def _compute_confidence(self, results: dict[str, Any]) -> int:
        """Compute confidence score based on data completeness."""
        confidence = 40  # Base

        # More data sources = higher confidence
        if results.get("salary_validation"):
            confidence += 10
        if results.get("calculate_chemistry"):
            confidence += 10
        if results.get("simulate_championship"):
            confidence += 15
        if results.get("predict_injury"):
            confidence += 10
        if results.get("predict_player"):
            confidence += 10

        return min(confidence, 95)
