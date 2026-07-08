"""
Shared test fixtures for the NBA Agentic AI test suite.
"""

import os
import sys
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set test environment
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake-key")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///test.db")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")


@pytest.fixture
def sample_trade_context():
    """Sample trade context for testing."""
    return {
        "team_a": "Lakers",
        "team_b": "Celtics",
        "player_a": "LeBron James",
        "player_b": "Jayson Tatum",
        "team_a_players": ["LeBron James"],
        "team_b_players": ["Jayson Tatum"],
    }


@pytest.fixture
def sample_team():
    """Sample team name for testing."""
    return "Lakers"


@pytest.fixture
def sample_player():
    """Sample player name for testing."""
    return "LeBron James"
