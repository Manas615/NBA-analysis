"""Integration tests for the database layer."""

import pytest


class TestDatabaseModels:
    """Tests for database ORM models."""

    def test_import_models(self):
        """All ORM models can be imported."""
        from db.models import (
            TradeHistory,
            Simulation,
            UserSession,
            SavedTeam,
            FavoriteTrade,
            ConversationMessage,
        )

        assert TradeHistory.__tablename__ == "trade_history"
        assert Simulation.__tablename__ == "simulations"
        assert UserSession.__tablename__ == "user_sessions"
        assert SavedTeam.__tablename__ == "saved_teams"
        assert FavoriteTrade.__tablename__ == "favorite_trades"
        assert ConversationMessage.__tablename__ == "conversation_messages"

    def test_repository_import(self):
        """Repository can be imported."""
        from db.repository import TradeRepository

        repo = TradeRepository()
        assert repo is not None
