"""
Tool Registry — Auto-generates OpenAI function-calling schemas from Python functions.

The @tool decorator inspects type hints, docstrings, and Pydantic models
to produce the JSON schema that GPT uses for function calling.

Usage:
    from tools.registry import tool, ToolRegistry

    @tool(name="predict_player", description="Predict a player's future impact")
    def predict_player(player_name: str, team: str) -> dict:
        ...

    schemas = ToolRegistry.get_schemas()  # OpenAI-compatible tool definitions
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable, get_type_hints

from pydantic import BaseModel


class ToolRegistry:
    """Global registry of all callable tools."""

    _tools: dict[str, dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        func: Callable,
        description: str,
        parameters_schema: dict[str, Any],
    ) -> None:
        """Register a tool with its metadata and schema."""
        cls._tools[name] = {
            "function": func,
            "name": name,
            "description": description,
            "parameters": parameters_schema,
        }

    @classmethod
    def get_tool(cls, name: str) -> dict[str, Any]:
        """Get a registered tool by name."""
        if name not in cls._tools:
            raise KeyError(f"Tool '{name}' not registered. Available: {list(cls._tools.keys())}")
        return cls._tools[name]

    @classmethod
    def execute(cls, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a registered tool by name with the given arguments."""
        tool_entry = cls.get_tool(name)
        func = tool_entry["function"]
        return func(**arguments)

    @classmethod
    def get_schemas(cls) -> list[dict[str, Any]]:
        """
        Return all tool schemas in OpenAI function-calling format.

        Returns a list of dicts, each with:
            {"type": "function", "function": {"name", "description", "parameters"}}
        """
        schemas = []
        for name, entry in cls._tools.items():
            schemas.append({
                "type": "function",
                "function": {
                    "name": entry["name"],
                    "description": entry["description"],
                    "parameters": entry["parameters"],
                },
            })
        return schemas

    @classmethod
    def get_tool_names(cls) -> list[str]:
        """Return all registered tool names."""
        return list(cls._tools.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered tools (useful for testing)."""
        cls._tools.clear()


def _python_type_to_json_schema(annotation: Any) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema type."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}

    origin = getattr(annotation, "__origin__", None)

    if annotation is str:
        return {"type": "string"}
    elif annotation is int:
        return {"type": "integer"}
    elif annotation is float:
        return {"type": "number"}
    elif annotation is bool:
        return {"type": "boolean"}
    elif origin is list:
        args = getattr(annotation, "__args__", (Any,))
        return {
            "type": "array",
            "items": _python_type_to_json_schema(args[0] if args else Any),
        }
    elif origin is dict:
        return {"type": "object"}
    elif annotation is type(None):
        return {"type": "string"}
    elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation.model_json_schema()
    else:
        return {"type": "string"}


def _build_parameters_schema(func: Callable) -> dict[str, Any]:
    """Build an OpenAI-compatible parameters JSON schema from function signature."""
    sig = inspect.signature(func)
    hints = get_type_hints(func)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        annotation = hints.get(param_name, inspect.Parameter.empty)
        prop = _python_type_to_json_schema(annotation)

        # Extract description from docstring if available
        prop["description"] = f"Parameter: {param_name}"

        properties[param_name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def tool(
    name: str | None = None,
    description: str | None = None,
) -> Callable:
    """
    Decorator to register a function as a callable tool.

    Usage:
        @tool(name="predict_player", description="Predict future impact")
        def predict_player(player_name: str) -> dict:
            '''Predict a player's future impact score.'''
            ...
    """

    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_desc = description or (func.__doc__ or "").strip().split("\n")[0]
        schema = _build_parameters_schema(func)

        ToolRegistry.register(
            name=tool_name,
            func=func,
            description=tool_desc,
            parameters_schema=schema,
        )

        func._tool_name = tool_name
        return func

    return decorator
