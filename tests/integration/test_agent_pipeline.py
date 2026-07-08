"""Integration test for the agent pipeline."""

import pytest


class TestAgentPipeline:
    """Integration tests for the multi-agent pipeline."""

    def test_general_manager_init(self):
        """GM Agent initializes without errors."""
        from agents.general_manager import GeneralManagerAgent
        gm = GeneralManagerAgent()
        assert gm.agent_name == "GeneralManagerAgent"
        assert len(gm.available_tools) == 10

    def test_all_agents_instantiate(self):
        """All agents can be instantiated."""
        from agents.trade_analysis import TradeAnalysisAgent
        from agents.salary_cap import SalaryCapAgent
        from agents.championship_sim import ChampionshipSimAgent
        from agents.injury_risk import InjuryRiskAgent
        from agents.chemistry import ChemistryAgent
        from agents.roster_optimizer import RosterOptimizerAgent
        from agents.free_agency import FreeAgencyAgent
        from agents.draft import DraftAgent
        from agents.report_generator import ReportGeneratorAgent

        agents = [
            TradeAnalysisAgent(),
            SalaryCapAgent(),
            ChampionshipSimAgent(),
            InjuryRiskAgent(),
            ChemistryAgent(),
            RosterOptimizerAgent(),
            FreeAgencyAgent(),
            DraftAgent(),
            ReportGeneratorAgent(),
        ]

        for agent in agents:
            assert agent.agent_name is not None
            assert len(agent.system_prompt) > 50

    def test_all_tools_registered(self):
        """All 10 tools are registered after importing agents."""
        from tools.registry import ToolRegistry

        # Import to trigger registration
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

        names = ToolRegistry.get_tool_names()
        expected = [
            "predict_player", "predict_matchup", "simulate_championship",
            "calculate_chemistry", "salary_validation", "predict_injury",
            "optimize_lineup", "recommend_free_agents", "recommend_draft_pick",
            "compare_players",
        ]

        for tool_name in expected:
            assert tool_name in names, f"Tool '{tool_name}' not registered"

    def test_report_generator_fallback(self):
        """Report generator produces output from raw results."""
        from agents.report_generator import ReportGeneratorAgent

        agent = ReportGeneratorAgent()
        result = agent._build_fallback_report(
            query="Test trade query",
            agent_results={
                "salary_validation": {"is_valid": True, "explanation": "Trade passes"},
                "calculate_chemistry": {"rsi": 0.65, "explanation": "Good chemistry"},
            },
        )

        assert result["agent"] == "ReportGeneratorAgent"
        assert "Trade Grade" in result["response"]
        assert result["confidence_score"] > 0

    def test_report_grade_computation(self):
        """Trade grade computation works correctly."""
        from agents.report_generator import ReportGeneratorAgent

        agent = ReportGeneratorAgent()

        # Valid salary + good chemistry should give decent grade
        grade = agent._compute_grade({
            "salary_validation": {"is_valid": True},
            "calculate_chemistry": {"rsi": 0.7},
        })
        assert grade in ["A+", "A", "A-", "B+", "B", "B-", "C+", "C"]
