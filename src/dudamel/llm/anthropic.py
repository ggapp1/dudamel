"""Provider for Anthropic's native /v1/messages API."""

from __future__ import annotations

import json
from typing import Any

import httpx

from dudamel.exceptions import LLMError
from dudamel.llm.provider import ToolSpec
from dudamel.llm.types import Completion, Message, ToolCall, Usage

_RETRYABLE = {408, 429, 500, 502, 503, 504, 529}
_VERSION = "2023-06-01"


def _render_messages(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Anthropic wants system as a top-level param and tool results as user
    tool_result blocks."""
    system_parts: list[str] = []
    wire: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            system_parts.append(m.text)
        elif m.role == "user":
            wire.append({"role": "user", "content": [{"type": "text", "text": m.text}]})
        elif m.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if m.text:
                blocks.append({"type": "text", "text": m.text})
            for tc in m.tool_calls:
                blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.args})
            if not blocks:
                blocks.append({"type": "text", "text": "(no content)"})
            wire.append({"role": "assistant", "content": blocks})
        else:  # tool result
            wire.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.tool_call_id,
                            "content": m.text,
                            "is_error": m.is_error,
                        }
                    ],
                }
            )
    return "\n".join(system_parts), wire


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=120.0)
        self._headers = {"x-api-key": api_key, "anthropic-version": _VERSION}
        self._api_key = api_key

    def _redact(self, text: str) -> str:
        if self._api_key and len(self._api_key) > 6:
            text = text.replace(self._api_key, "***")
        return text

    async def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        if json_schema is not None:
            raise LLMError(
                "structured output (json_schema) is not supported on the anthropic "
                "provider in v1 — use an openai-compatible tier or model a tool"
            )
        system, wire = _render_messages(messages)
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": wire,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.json_schema}
                for t in tools
            ]
        try:
            resp = await self._client.post(
                f"{self._base}/v1/messages", json=payload, headers=self._headers
            )
        except httpx.HTTPError as e:
            raise LLMError(f"LLM endpoint unreachable: {e}", retryable=True) from e
        if resp.status_code >= 400:
            raise LLMError(
                f"anthropic returned HTTP {resp.status_code}: {self._redact(resp.text[:300])}",
                retryable=resp.status_code in _RETRYABLE,
            )
        try:
            body = resp.json()
        except json.JSONDecodeError as e:
            raise LLMError(
                f"provider returned non-JSON response body: {self._redact(resp.text[:300])}"
            ) from e
        return self._parse(body)

    def _parse(self, data: dict[str, Any]) -> Completion:
        if not isinstance(data, dict):
            raise LLMError(
                f"malformed completion response: expected object, got {type(data).__name__}"
            )
        try:
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_calls.append(
                        ToolCall(id=block["id"], name=block["name"], args=block.get("input", {}))
                    )
            stop = data.get("stop_reason")
            stop_reason = (
                "tool_calls"
                if stop == "tool_use"
                else "max_tokens"
                if stop == "max_tokens"
                else "end"
            )
            usage = data.get("usage") or {}
            return Completion(
                message=Message(role="assistant", text="".join(text_parts), tool_calls=tool_calls),
                usage=Usage(
                    tokens_in=usage.get("input_tokens", 0),
                    tokens_out=usage.get("output_tokens", 0),
                ),
                stop_reason=stop_reason,
            )
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            raise LLMError(f"malformed completion response: {e}") from e
