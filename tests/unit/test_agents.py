"""Unit tests for the Agent Layer."""

import pytest
from agents.memory import AgentMemory, ConversationSession


class TestAgentMemory:
    """Tests for agent conversation memory."""

    def test_create_session(self):
        memory = AgentMemory()
        session = memory.get_or_create_session()
        assert session.session_id is not None
        assert len(session.messages) == 0

    def test_add_message(self):
        session = ConversationSession()
        session.add_message("user", "What if Lakers trade LeBron?")
        session.add_message("assistant", "Let me analyze that trade...")

        assert len(session.messages) == 2
        assert session.messages[0].role == "user"
        assert session.messages[1].role == "assistant"

    def test_update_context(self):
        session = ConversationSession()
        session.update_context("last_team", "Lakers")
        session.update_context("last_trade", {"player_a": "LeBron"})

        assert session.get_last_team() == "Lakers"
        assert session.get_last_trade()["player_a"] == "LeBron"

    def test_get_history_for_gpt(self):
        session = ConversationSession()
        for i in range(15):
            session.add_message("user", f"Message {i}")

        history = session.get_history_for_gpt(max_messages=5)
        assert len(history) == 5

    def test_session_persistence(self):
        memory = AgentMemory()
        session = memory.get_or_create_session("test-session-123")
        session.add_message("user", "Test message")
        memory.save_session(session)

        # Retrieve
        retrieved = memory.get_session("test-session-123")
        assert retrieved is not None
        assert len(retrieved.messages) == 1

    def test_session_serialization(self):
        session = ConversationSession()
        session.add_message("user", "Hello")
        session.update_context("key", "value")

        data = session.to_dict()
        restored = ConversationSession.from_dict(data)

        assert restored.session_id == session.session_id
        assert len(restored.messages) == 1
        assert restored.context["key"] == "value"

    def test_delete_session(self):
        memory = AgentMemory()
        session = memory.get_or_create_session("to-delete")
        memory.save_session(session)

        assert memory.delete_session("to-delete") is True
        assert memory.get_session("to-delete") is None
        assert memory.delete_session("nonexistent") is False
