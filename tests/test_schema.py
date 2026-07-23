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
