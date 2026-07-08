"""
Dependency Injection — FastAPI dependencies for agents, DB, cache.

Provides injectable instances of agents, database sessions,
cache clients, and memory managers.
"""

from __future__ import annotations

from functools import lru_cache

from agents.general_manager import GeneralManagerAgent
from agents.trade_analysis import TradeAnalysisAgent
from agents.salary_cap import SalaryCapAgent
from agents.championship_sim import ChampionshipSimAgent
from agents.injury_risk import InjuryRiskAgent
from agents.chemistry import ChemistryAgent
from agents.roster_optimizer import RosterOptimizerAgent
from agents.free_agency import FreeAgencyAgent
from agents.draft import DraftAgent
from agents.report_generator import ReportGeneratorAgent
from agents.memory import get_memory, AgentMemory
from cache.redis_cache import get_cache, RedisCache


@lru_cache(maxsize=1)
def get_gm_agent() -> GeneralManagerAgent:
    """Get the General Manager Agent (singleton)."""
    return GeneralManagerAgent()


@lru_cache(maxsize=1)
def get_trade_agent() -> TradeAnalysisAgent:
    return TradeAnalysisAgent()


@lru_cache(maxsize=1)
def get_salary_agent() -> SalaryCapAgent:
    return SalaryCapAgent()


@lru_cache(maxsize=1)
def get_championship_agent() -> ChampionshipSimAgent:
    return ChampionshipSimAgent()


@lru_cache(maxsize=1)
def get_injury_agent() -> InjuryRiskAgent:
    return InjuryRiskAgent()


@lru_cache(maxsize=1)
def get_chemistry_agent() -> ChemistryAgent:
    return ChemistryAgent()


@lru_cache(maxsize=1)
def get_roster_agent() -> RosterOptimizerAgent:
    return RosterOptimizerAgent()


@lru_cache(maxsize=1)
def get_free_agency_agent() -> FreeAgencyAgent:
    return FreeAgencyAgent()


@lru_cache(maxsize=1)
def get_draft_agent() -> DraftAgent:
    return DraftAgent()


@lru_cache(maxsize=1)
def get_report_agent() -> ReportGeneratorAgent:
    return ReportGeneratorAgent()


def get_agent_memory() -> AgentMemory:
    return get_memory()


def get_redis_cache() -> RedisCache:
    return get_cache()
