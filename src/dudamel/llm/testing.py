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
    """Create a fake completion with text response.

    Args:
        text: The assistant's text response.
        tokens_in: Tokens in the input (default 10).
        tokens_out: Tokens in the output (default 5).

    Returns:
        A Completion with the given text and stop_reason="end".
    """
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
    """Create a fake completion with a tool call.

    Args:
        name: The tool name.
        args: The tool arguments.
        id: The tool call ID (default "tc1").
        tokens_in: Tokens in the input (default 10).
        tokens_out: Tokens in the output (default 5).

    Returns:
        A Completion with a ToolCall and stop_reason="tool_calls".
    """
    return Completion(
        message=Message(
            role="assistant",
            tool_calls=[ToolCall(id=id, name=name, args=args)],
        ),
        usage=Usage(tokens_in=tokens_in, tokens_out=tokens_out),
        stop_reason="tool_calls",
    )


class FakeProvider:
    """Mock LLM provider for deterministic testing.

    Records all calls in .calls, a list of dicts with keys:
    model, messages, tools, max_tokens, json_schema.
    Element objects are aliased (shallow copies); lists are copied.
    Replays a scripted sequence of Completions or raises them.

    Raises:
        LLMError: When the script is exhausted.
    """

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
        """Call the LLM and record the request.

        Records a shallow copy of the call (element objects aliased,
        lists copied). Returns the next item from the script.

        Raises:
            LLMError: If the script is exhausted.
        """
        self.calls.append(
            {
                "model": model,
                "messages": list(messages),
                "tools": list(tools) if tools is not None else None,
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
