"""
Database ORM Models — PostgreSQL tables for the NBA Agentic AI system.

Tables:
- trade_history: All simulated trades with results
- simulations: Championship simulation runs
- user_sessions: Session tracking for agent memory
- saved_teams: User-saved team configurations
- favorite_trades: Bookmarked trades
- conversation_messages: Agent memory storage
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from db.engine import Base


class TradeHistory(Base):
    """Record of every trade simulation."""
    __tablename__ = "trade_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("user_sessions.id"), nullable=True)
    team_a = Column(String(64), nullable=False, index=True)
    team_b = Column(String(64), nullable=False, index=True)
    players_a = Column(JSONB, nullable=False)  # List of player names from team A
    players_b = Column(JSONB, nullable=False)  # List of player names from team B
    is_valid = Column(Boolean, default=False)
    salary_valid = Column(Boolean, nullable=True)
    trade_grade = Column(String(4), nullable=True)
    confidence_score = Column(Float, nullable=True)
    win_change = Column(Float, nullable=True)
    championship_change_a = Column(Float, nullable=True)
    championship_change_b = Column(Float, nullable=True)
    chemistry_change = Column(Float, nullable=True)
    full_report = Column(JSONB, nullable=True)  # Full agent report
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    session = relationship("UserSession", back_populates="trades")


class Simulation(Base):
    """Record of championship simulation runs."""
    __tablename__ = "simulations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    team = Column(String(64), nullable=False, index=True)
    simulation_count = Column(Integer, nullable=False)
    expected_wins = Column(Float, nullable=True)
    playoff_probability = Column(Float, nullable=True)
    conference_finals_probability = Column(Float, nullable=True)
    finals_probability = Column(Float, nullable=True)
    championship_probability = Column(Float, nullable=True)
    scenario = Column(String(32), default="baseline")  # baseline or trade
    trade_id = Column(UUID(as_uuid=True), ForeignKey("trade_history.id"), nullable=True)
    full_results = Column(JSONB, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)


class UserSession(Base):
    """User session for agent memory."""
    __tablename__ = "user_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_data = Column(JSONB, nullable=True)  # Serialized ConversationSession
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    last_active = Column(DateTime, server_default=func.now(), onupdate=func.now())

    trades = relationship("TradeHistory", back_populates="session")
    messages = relationship("ConversationMessage", back_populates="session")


class SavedTeam(Base):
    """User-saved team configurations."""
    __tablename__ = "saved_teams"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("user_sessions.id"), nullable=True)
    team_name = Column(String(64), nullable=False)
    roster = Column(JSONB, nullable=False)  # List of player names
    custom_name = Column(String(128), nullable=True)  # User label
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)


class FavoriteTrade(Base):
    """Bookmarked trades for quick reference."""
    __tablename__ = "favorite_trades"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("user_sessions.id"), nullable=True)
    trade_id = Column(UUID(as_uuid=True), ForeignKey("trade_history.id"), nullable=False)
    label = Column(String(256), nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)


class ConversationMessage(Base):
    """Individual conversation messages for agent memory."""
    __tablename__ = "conversation_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("user_sessions.id"), nullable=False, index=True)
    role = Column(String(16), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    tool_calls = Column(JSONB, nullable=True)
    tool_results = Column(JSONB, nullable=True)
    metadata_ = Column("metadata", JSONB, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    session = relationship("UserSession", back_populates="messages")
