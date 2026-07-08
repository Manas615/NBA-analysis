"""
Database Engine — SQLAlchemy async engine configuration.

Provides async engine, session factory, and base model for ORM.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://nba:nba_password@localhost:5432/nba_trades",
)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# Engine (lazy — created on first use)
_engine = None
_session_factory = None


def get_engine():
    """Get or create the async engine."""
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            DATABASE_URL,
            echo=False,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory():
    """Get or create the async session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a transactional async database session."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def init_db() -> None:
    """Create all tables (for development; use Alembic in production)."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Dispose the engine connection pool."""
    global _engine, _session_factory
    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None
