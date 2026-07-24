from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from dudamel.contract.types import Tool
from dudamel.llm.types import Completion, Message


@dataclass
class ToolSpec:
    """What a provider needs to advertise one tool. 1:1 with an MCP tool."""

    name: str
    description: str
    json_schema: dict[str, Any]

    @classmethod
    def from_tool(cls, tool: Tool) -> ToolSpec:
        return cls(
            name=tool.name,
            description=tool.description,
            json_schema=tool.schema.json_schema,
        )


class Provider(Protocol):
    name: str

    async def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion: ...
