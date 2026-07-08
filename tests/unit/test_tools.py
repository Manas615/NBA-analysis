"""Unit tests for the Tool Layer."""

import pytest
from tools.registry import ToolRegistry, tool


class TestToolRegistry:
    """Tests for the tool registry and decorator."""

    def setup_method(self):
        """Clear registry before each test."""
        ToolRegistry.clear()

    def test_register_tool(self):
        @tool(name="test_tool", description="A test tool")
        def test_func(name: str, value: int) -> dict:
            return {"name": name, "value": value}

        assert "test_tool" in ToolRegistry.get_tool_names()

    def test_execute_tool(self):
        @tool(name="add_numbers", description="Add two numbers")
        def add(a: int, b: int) -> dict:
            return {"result": a + b}

        result = ToolRegistry.execute("add_numbers", {"a": 3, "b": 5})
        assert result == {"result": 8}

    def test_get_schemas(self):
        @tool(name="greet", description="Greet someone")
        def greet(name: str) -> dict:
            return {"greeting": f"Hello {name}"}

        schemas = ToolRegistry.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "greet"
        assert "name" in schemas[0]["function"]["parameters"]["properties"]

    def test_tool_not_found(self):
        with pytest.raises(KeyError):
            ToolRegistry.get_tool("nonexistent")

    def test_schema_generation_types(self):
        @tool(name="typed_tool", description="Tool with types")
        def typed(name: str, count: int, ratio: float, active: bool) -> dict:
            return {}

        schemas = ToolRegistry.get_schemas()
        props = schemas[0]["function"]["parameters"]["properties"]
        assert props["name"]["type"] == "string"
        assert props["count"]["type"] == "integer"
        assert props["ratio"]["type"] == "number"
        assert props["active"]["type"] == "boolean"

    def test_required_vs_optional(self):
        @tool(name="optional_tool", description="Tool with optional params")
        def func(required: str, optional: str = "default") -> dict:
            return {}

        schemas = ToolRegistry.get_schemas()
        required = schemas[0]["function"]["parameters"]["required"]
        assert "required" in required
        assert "optional" not in required
