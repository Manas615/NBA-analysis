"""
Base Agent — GPT-powered agent with function calling.

Every agent:
1. Has a system prompt defining its role and constraints
2. Uses OpenAI function calling to invoke tools — never answers directly
3. Returns structured outputs with explainability
4. Integrates with agent memory for follow-up queries
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

from openai import OpenAI

from tools.registry import ToolRegistry
from observability.logging import get_logger

logger = get_logger(__name__)


class BaseAgent:
    """
    Base class for all agents in the NBA Agentic AI system.

    Each agent wraps an OpenAI GPT model with a specific system prompt
    and a subset of tools. The agent loop:
      1. Send system prompt + user query + tool schemas to GPT
      2. GPT responds with tool_calls (never direct text)
      3. Execute tools, collect results
      4. Send results back to GPT for synthesis
      5. Return structured response with explainability
    """

    agent_name: str = "BaseAgent"
    system_prompt: str = "You are an AI agent."
    available_tools: list[str] = []
    max_iterations: int = 5

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.client = self._init_client()
        self.request_id = str(uuid.uuid4())[:8]

    def _init_client(self) -> OpenAI | None:
        """Initialize OpenAI client."""
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key.startswith("sk-your"):
            logger.warning(
                "openai_not_configured",
                agent=self.agent_name,
                message="No valid OPENAI_API_KEY found. Agent will use fallback mode.",
            )
            return None

        return OpenAI(api_key=api_key)

    def _get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI function-calling schemas for this agent's tools."""
        all_schemas = ToolRegistry.get_schemas()
        if not self.available_tools:
            return all_schemas

        return [
            s for s in all_schemas
            if s["function"]["name"] in self.available_tools
        ]

    def _execute_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a registered tool and return its result."""
        start = time.time()
        logger.info(
            "tool_call_start",
            agent=self.agent_name,
            tool=name,
            arguments=arguments,
            request_id=self.request_id,
        )

        try:
            result = ToolRegistry.execute(name, arguments)
            elapsed = time.time() - start
            logger.info(
                "tool_call_complete",
                agent=self.agent_name,
                tool=name,
                elapsed_ms=round(elapsed * 1000, 1),
                success=True,
                request_id=self.request_id,
            )
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(
                "tool_call_error",
                agent=self.agent_name,
                tool=name,
                error=str(e),
                elapsed_ms=round(elapsed * 1000, 1),
                request_id=self.request_id,
            )
            return {"error": str(e), "tool": name}

    def execute(
        self,
        user_query: str,
        context: dict[str, Any] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """
        Execute the agent loop.

        1. Build messages with system prompt, history, and user query
        2. Call GPT with tool schemas
        3. Process tool calls iteratively
        4. Return final structured response
        """
        start_time = time.time()
        logger.info(
            "agent_execute_start",
            agent=self.agent_name,
            query=user_query[:200],
            request_id=self.request_id,
        )

        # If no OpenAI client, use fallback mode
        if self.client is None:
            return self._fallback_execute(user_query, context)

        # Build messages
        messages = self._build_messages(user_query, context, conversation_history)
        tool_schemas = self._get_tool_schemas()

        tool_calls_made = []
        tool_results = {}
        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1

            try:
                # Call GPT
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tool_schemas if tool_schemas else None,
                    tool_choice="auto" if tool_schemas else None,
                    temperature=0.1,
                )
            except Exception as e:
                logger.error(
                    "gpt_call_error",
                    agent=self.agent_name,
                    error=str(e),
                    iteration=iterations,
                    request_id=self.request_id,
                )
                return self._fallback_execute(user_query, context)

            choice = response.choices[0]
            message = choice.message

            # If GPT wants to call tools
            if message.tool_calls:
                messages.append(message.model_dump())

                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    try:
                        func_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        func_args = {}

                    result = self._execute_tool(func_name, func_args)
                    tool_calls_made.append({
                        "tool": func_name,
                        "arguments": func_args,
                    })
                    tool_results[func_name] = result

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, default=str),
                    })

                continue

            # GPT returned a final text response
            elapsed = time.time() - start_time
            final_text = message.content or ""

            logger.info(
                "agent_execute_complete",
                agent=self.agent_name,
                iterations=iterations,
                tools_called=len(tool_calls_made),
                elapsed_ms=round(elapsed * 1000, 1),
                request_id=self.request_id,
            )

            return {
                "agent": self.agent_name,
                "response": final_text,
                "tool_calls": tool_calls_made,
                "tool_results": tool_results,
                "iterations": iterations,
                "elapsed_ms": round(elapsed * 1000, 1),
                "request_id": self.request_id,
            }

        # Max iterations reached
        elapsed = time.time() - start_time
        logger.warning(
            "agent_max_iterations",
            agent=self.agent_name,
            iterations=iterations,
            request_id=self.request_id,
        )

        return {
            "agent": self.agent_name,
            "response": "Analysis completed with maximum tool iterations.",
            "tool_calls": tool_calls_made,
            "tool_results": tool_results,
            "iterations": iterations,
            "elapsed_ms": round(elapsed * 1000, 1),
            "request_id": self.request_id,
        }

    def _build_messages(
        self,
        user_query: str,
        context: dict[str, Any] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build the message list for GPT."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        # Add conversation history for follow-up context
        if conversation_history:
            for msg in conversation_history[-10:]:  # Last 10 messages
                messages.append(msg)

        # Add context if provided
        if context:
            messages.append({
                "role": "system",
                "content": f"Additional context:\n{json.dumps(context, default=str)}",
            })

        messages.append({"role": "user", "content": user_query})
        return messages

    def _fallback_execute(
        self,
        user_query: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Fallback execution when GPT is not available.

        Directly calls relevant tools based on agent type and
        returns raw tool results without GPT synthesis.
        """
        logger.info(
            "agent_fallback_mode",
            agent=self.agent_name,
            request_id=self.request_id,
        )

        tool_results = {}
        tool_calls = []

        for tool_name in self.available_tools:
            try:
                # Try to call tool with context-derived arguments
                args = self._derive_tool_args(tool_name, user_query, context)
                if args is not None:
                    result = self._execute_tool(tool_name, args)
                    tool_results[tool_name] = result
                    tool_calls.append({"tool": tool_name, "arguments": args})
            except Exception as e:
                logger.warning(
                    "fallback_tool_skip",
                    tool=tool_name,
                    error=str(e),
                    request_id=self.request_id,
                )

        return {
            "agent": self.agent_name,
            "response": "Analysis completed in fallback mode (no GPT). Raw tool results attached.",
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "iterations": 1,
            "elapsed_ms": 0,
            "request_id": self.request_id,
            "fallback_mode": True,
        }

    def _derive_tool_args(
        self,
        tool_name: str,
        user_query: str,
        context: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """
        Derive tool arguments from query and context.

        Override in subclasses for agent-specific logic.
        """
        if context:
            return context
        return None
