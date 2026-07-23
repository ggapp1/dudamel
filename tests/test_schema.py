from datetime import date
from enum import Enum
from typing import Literal

import pytest

from dudamel.contract.schema import ToolSchema


class Color(Enum):
    RED = "red"
    BLUE = "blue"


async def sample_tool(
    query: str,
    limit: int = 5,
    weight: float | None = None,
    since: date | None = None,
    mode: Literal["fast", "deep"] = "fast",
    color: Color = Color.RED,
) -> str:
    """Search things."""
    return query


def test_json_schema_golden():
    schema = ToolSchema(sample_tool).json_schema
    props = schema["properties"]
    assert props["query"] == {"type": "string"}
    assert props["limit"] == {"type": "integer", "default": 5}
    assert props["weight"]["anyOf"] == [{"type": "number"}, {"type": "null"}]
    assert {"type": "string", "format": "date"} in props["since"]["anyOf"]
    assert props["mode"]["enum"] == ["fast", "deep"]
    assert props["color"]["enum"] == ["red", "blue"]
    assert schema["required"] == ["query"]
    assert schema["additionalProperties"] is False


def test_description_from_docstring():
    assert ToolSchema(sample_tool).description == "Search things."


def test_missing_type_hint_rejected():
    async def bad(x):  # no hint
        """Doc."""
        return x

    with pytest.raises(TypeError, match="needs a type hint"):
        ToolSchema(bad)


def _assert_no_refs(node) -> None:
    """Recursively assert a schema (sub)tree has no $ref/$defs anywhere."""
    if isinstance(node, dict):
        assert "$ref" not in node, f"found dangling $ref in {node!r}"
        assert "$defs" not in node, f"found leftover $defs in {node!r}"
        for value in node.values():
            _assert_no_refs(value)
    elif isinstance(node, list):
        for item in node:
            _assert_no_refs(item)


def test_container_of_enum_has_no_dangling_refs():
    async def list_tool(colors: list[Color]) -> str:
        """List colors."""
        return ""

    schema = ToolSchema(list_tool).json_schema
    _assert_no_refs(schema)
    assert schema["properties"]["colors"]["items"]["enum"] == ["red", "blue"]


def test_enum_default_matches_use_enum_values():
    async def color_tool(color: Color = Color.RED) -> str:
        """Pick a color."""
        return ""

    schema = ToolSchema(color_tool)
    assert schema.validate({}) == {"color": "red"}
    assert schema.validate({"color": "blue"}) == {"color": "blue"}
