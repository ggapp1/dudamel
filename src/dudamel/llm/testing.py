"""Public test harness: script a provider, assert on captured requests.

App developers use this to test their apps' LLM-dependent jobs and the chat
path deterministically, with no model running.
"""

from __future__ import annotations

from typing import Any

from dudamel.exceptions import LLMError
from dudamel.llm.provider import ToolSpec
from dudamel.llm.types import Completion, Message, ToolCall, Usage


def fake_text(text: str, *, tokens_in: int = 10, tokens_out: int = 5) -> Completion:
    return Completion(
        message=Message(role="assistant", text=text),
        usage=Usage(tokens_in=tokens_in, tokens_out=tokens_out),
        stop_reason="end",
    )


def fake_tool_call(
    name: str,
    args: dict[str, Any],
    *,
    id: str = "tc1",
    tokens_in: int = 10,
    tokens_out: int = 5,
) -> Completion:
    return Completion(
        message=Message(
            role="assistant",
            tool_calls=[ToolCall(id=id, name=name, args=args)],
        ),
        usage=Usage(tokens_in=tokens_in, tokens_out=tokens_out),
        stop_reason="tool_calls",
    )


class FakeProvider:
    name = "fake"

    def __init__(self, script: list[Completion | Exception]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        self.calls.append(
            {
                "model": model,
                "messages": list(messages),
                "tools": list(tools) if tools else None,
                "max_tokens": max_tokens,
                "json_schema": json_schema,
            }
        )
        if not self._script:
            raise LLMError(f"FakeProvider script exhausted after {len(self.calls) - 1} calls")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
