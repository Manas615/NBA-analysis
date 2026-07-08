"""
Agent Memory — Conversation state management for follow-up queries.

Stores previous interactions so agents can handle:
  "What if we also trade our center?"
  "Show me the salary impact of that last trade"

Uses PostgreSQL for persistence and Redis for hot state.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationMessage:
    """Single message in a conversation."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationSession:
    """Full conversation session with history."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[ConversationMessage] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def add_message(
        self,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_results: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a message to the conversation."""
        self.messages.append(
            ConversationMessage(
                role=role,
                content=content,
                tool_calls=tool_calls or [],
                tool_results=tool_results or {},
                metadata=metadata or {},
            )
        )
        self.last_active = time.time()

    def update_context(self, key: str, value: Any) -> None:
        """Update conversation context (e.g., last trade analyzed)."""
        self.context[key] = value

    def get_history_for_gpt(self, max_messages: int = 10) -> list[dict[str, str]]:
        """Get recent conversation history in GPT message format."""
        history = []
        for msg in self.messages[-max_messages:]:
            history.append({
                "role": msg.role,
                "content": msg.content,
            })
        return history

    def get_last_trade(self) -> dict[str, Any] | None:
        """Get the last trade that was analyzed in this session."""
        return self.context.get("last_trade")

    def get_last_team(self) -> str | None:
        """Get the last team that was referenced."""
        return self.context.get("last_team")

    def to_dict(self) -> dict[str, Any]:
        """Serialize session for storage."""
        return {
            "session_id": self.session_id,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp,
                    "tool_calls": m.tool_calls,
                    "tool_results": m.tool_results,
                    "metadata": m.metadata,
                }
                for m in self.messages
            ],
            "context": self.context,
            "created_at": self.created_at,
            "last_active": self.last_active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationSession:
        """Deserialize session from storage."""
        session = cls(
            session_id=data.get("session_id", str(uuid.uuid4())),
            created_at=data.get("created_at", time.time()),
            last_active=data.get("last_active", time.time()),
            context=data.get("context", {}),
        )

        for msg_data in data.get("messages", []):
            session.messages.append(
                ConversationMessage(
                    role=msg_data["role"],
                    content=msg_data["content"],
                    timestamp=msg_data.get("timestamp", time.time()),
                    tool_calls=msg_data.get("tool_calls", []),
                    tool_results=msg_data.get("tool_results", {}),
                    metadata=msg_data.get("metadata", {}),
                )
            )

        return session


class AgentMemory:
    """
    Agent memory manager.

    Manages conversation sessions with in-memory storage
    and optional PostgreSQL/Redis persistence.
    """

    def __init__(self):
        self._sessions: dict[str, ConversationSession] = {}

    def get_or_create_session(
        self, session_id: str | None = None
    ) -> ConversationSession:
        """Get existing session or create a new one."""
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]

        session = ConversationSession(
            session_id=session_id or str(uuid.uuid4())
        )
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> ConversationSession | None:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def save_session(self, session: ConversationSession) -> None:
        """Save session to memory (and optionally to DB)."""
        self._sessions[session.session_id] = session

    def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        return list(self._sessions.keys())

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    async def save_to_db(self, session: ConversationSession) -> None:
        """Persist session to PostgreSQL (async)."""
        try:
            from db.repository import TradeRepository
            repo = TradeRepository()
            await repo.save_session(
                session_id=session.session_id,
                data=json.dumps(session.to_dict(), default=str),
            )
        except Exception:
            pass  # DB not available, keep in memory

    async def load_from_db(self, session_id: str) -> ConversationSession | None:
        """Load session from PostgreSQL (async)."""
        try:
            from db.repository import TradeRepository
            repo = TradeRepository()
            data = await repo.get_session(session_id)
            if data:
                session = ConversationSession.from_dict(json.loads(data))
                self._sessions[session.session_id] = session
                return session
        except Exception:
            pass
        return None


# Global memory instance
_memory = AgentMemory()


def get_memory() -> AgentMemory:
    """Get the global agent memory instance."""
    return _memory
