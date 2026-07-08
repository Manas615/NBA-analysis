"""
Database Repository — CRUD operations for all tables.

Provides typed, async methods for persisting and querying
trade history, simulations, sessions, and favorites.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from db.engine import get_db_session
from db.models import (
    TradeHistory,
    Simulation,
    UserSession,
    SavedTeam,
    FavoriteTrade,
    ConversationMessage,
)


class TradeRepository:
    """Repository for trade-related database operations."""

    # ── Trade History ─────────────────────────────────────────

    async def save_trade(
        self,
        team_a: str,
        team_b: str,
        players_a: list[str],
        players_b: list[str],
        is_valid: bool = False,
        salary_valid: bool | None = None,
        trade_grade: str | None = None,
        confidence_score: float | None = None,
        win_change: float | None = None,
        championship_change_a: float | None = None,
        championship_change_b: float | None = None,
        chemistry_change: float | None = None,
        full_report: dict | None = None,
        session_id: str | None = None,
    ) -> str:
        """Save a trade to the database. Returns the trade ID."""
        async with get_db_session() as session:
            trade = TradeHistory(
                team_a=team_a,
                team_b=team_b,
                players_a=players_a,
                players_b=players_b,
                is_valid=is_valid,
                salary_valid=salary_valid,
                trade_grade=trade_grade,
                confidence_score=confidence_score,
                win_change=win_change,
                championship_change_a=championship_change_a,
                championship_change_b=championship_change_b,
                chemistry_change=chemistry_change,
                full_report=full_report,
                session_id=uuid.UUID(session_id) if session_id else None,
            )
            session.add(trade)
            await session.flush()
            return str(trade.id)

    async def get_trade(self, trade_id: str) -> dict | None:
        """Get a trade by ID."""
        async with get_db_session() as session:
            result = await session.execute(
                select(TradeHistory).where(TradeHistory.id == uuid.UUID(trade_id))
            )
            trade = result.scalar_one_or_none()
            if trade:
                return {
                    "id": str(trade.id),
                    "team_a": trade.team_a,
                    "team_b": trade.team_b,
                    "players_a": trade.players_a,
                    "players_b": trade.players_b,
                    "is_valid": trade.is_valid,
                    "trade_grade": trade.trade_grade,
                    "confidence_score": trade.confidence_score,
                    "full_report": trade.full_report,
                    "created_at": str(trade.created_at),
                }
            return None

    async def get_recent_trades(self, limit: int = 20) -> list[dict]:
        """Get recent trades."""
        async with get_db_session() as session:
            result = await session.execute(
                select(TradeHistory)
                .order_by(desc(TradeHistory.created_at))
                .limit(limit)
            )
            trades = result.scalars().all()
            return [
                {
                    "id": str(t.id),
                    "team_a": t.team_a,
                    "team_b": t.team_b,
                    "players_a": t.players_a,
                    "players_b": t.players_b,
                    "trade_grade": t.trade_grade,
                    "created_at": str(t.created_at),
                }
                for t in trades
            ]

    # ── Simulations ───────────────────────────────────────────

    async def save_simulation(
        self,
        team: str,
        simulation_count: int,
        results: dict[str, Any],
        trade_id: str | None = None,
    ) -> str:
        """Save a simulation result."""
        async with get_db_session() as session:
            sim = Simulation(
                team=team,
                simulation_count=simulation_count,
                expected_wins=results.get("expected_wins"),
                playoff_probability=results.get("playoff_probability"),
                conference_finals_probability=results.get("conference_finals_probability"),
                finals_probability=results.get("finals_probability"),
                championship_probability=results.get("championship_probability"),
                scenario="trade" if trade_id else "baseline",
                trade_id=uuid.UUID(trade_id) if trade_id else None,
                full_results=results,
            )
            session.add(sim)
            await session.flush()
            return str(sim.id)

    # ── Sessions ──────────────────────────────────────────────

    async def save_session(self, session_id: str, data: str) -> None:
        """Save or update a user session."""
        async with get_db_session() as session:
            result = await session.execute(
                select(UserSession).where(UserSession.id == uuid.UUID(session_id))
            )
            existing = result.scalar_one_or_none()

            if existing:
                existing.session_data = json.loads(data)
            else:
                user_session = UserSession(
                    id=uuid.UUID(session_id),
                    session_data=json.loads(data),
                )
                session.add(user_session)

    async def get_session(self, session_id: str) -> str | None:
        """Get session data by ID."""
        async with get_db_session() as session:
            result = await session.execute(
                select(UserSession).where(UserSession.id == uuid.UUID(session_id))
            )
            user_session = result.scalar_one_or_none()
            if user_session and user_session.session_data:
                return json.dumps(user_session.session_data)
            return None

    # ── Saved Teams ───────────────────────────────────────────

    async def save_team(
        self,
        team_name: str,
        roster: list[str],
        custom_name: str | None = None,
        notes: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Save a team configuration."""
        async with get_db_session() as session:
            saved = SavedTeam(
                team_name=team_name,
                roster=roster,
                custom_name=custom_name,
                notes=notes,
                session_id=uuid.UUID(session_id) if session_id else None,
            )
            session.add(saved)
            await session.flush()
            return str(saved.id)

    # ── Favorites ─────────────────────────────────────────────

    async def save_favorite(
        self,
        trade_id: str,
        label: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Bookmark a trade as a favorite."""
        async with get_db_session() as session:
            fav = FavoriteTrade(
                trade_id=uuid.UUID(trade_id),
                label=label,
                session_id=uuid.UUID(session_id) if session_id else None,
            )
            session.add(fav)
            await session.flush()
            return str(fav.id)

    async def get_favorites(self, session_id: str | None = None, limit: int = 50) -> list[dict]:
        """Get favorite trades."""
        async with get_db_session() as session:
            query = select(FavoriteTrade).order_by(desc(FavoriteTrade.created_at)).limit(limit)
            if session_id:
                query = query.where(FavoriteTrade.session_id == uuid.UUID(session_id))
            result = await session.execute(query)
            favs = result.scalars().all()
            return [
                {
                    "id": str(f.id),
                    "trade_id": str(f.trade_id),
                    "label": f.label,
                    "created_at": str(f.created_at),
                }
                for f in favs
            ]
