"""Provider for any OpenAI-compatible /chat/completions endpoint
(Ollama, LM Studio, vLLM, OpenRouter, ...)."""

from __future__ import annotations

import json
from typing import Any

import httpx

from dudamel.exceptions import LLMError
from dudamel.llm.provider import ToolSpec
from dudamel.llm.types import Completion, Message, ToolCall, Usage

_RETRYABLE = {408, 429, 500, 502, 503, 504}


def _render(m: Message) -> dict[str, Any]:
    if m.role == "tool":
        return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.text}
    if m.role == "assistant" and m.tool_calls:
        return {
            "role": "assistant",
            "content": m.text or None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.args)},
                }
                for tc in m.tool_calls
            ],
        }
    return {"role": m.role, "content": m.text}


def _parse_args(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # Feed the garbage through the normal validation-retry branch instead
        # of special-casing it: schema validation will reject the sentinel key.
        return {"__unparseable__": raw}
    return parsed if isinstance(parsed, dict) else {"__unparseable__": raw}


class OpenAICompatProvider:
    name = "openai-compatible"

    def __init__(
        self,
        base_url: str,
        api_key: str = "unused",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=120.0)
        self._headers = {"Authorization": f"Bearer {api_key}"}

    async def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [_render(m) for m in messages],
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.json_schema,
                    },
                }
                for t in tools
            ]
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": json_schema, "strict": True},
            }
        try:
            resp = await self._client.post(
                f"{self._base}/chat/completions", json=payload, headers=self._headers
            )
        except httpx.HTTPError as e:
            raise LLMError(f"LLM endpoint unreachable: {e}", retryable=True) from e
        if resp.status_code >= 400:
            raise LLMError(
                f"LLM endpoint returned HTTP {resp.status_code}: {resp.text[:300]}",
                retryable=resp.status_code in _RETRYABLE,
            )
        return self._parse(resp.json())

    def _parse(self, data: dict[str, Any]) -> Completion:
        try:
            choice = data["choices"][0]
            wire_msg = choice["message"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"malformed completion response: {e}") from e
        tool_calls = [
            ToolCall(
                id=tc.get("id", f"call_{i}"),
                name=tc["function"]["name"],
                args=_parse_args(tc["function"].get("arguments", "{}")),
            )
            for i, tc in enumerate(wire_msg.get("tool_calls") or [])
        ]
        finish = choice.get("finish_reason", "stop")
        stop_reason = "tool_calls" if tool_calls else "max_tokens" if finish == "length" else "end"
        usage = data.get("usage") or {}
        return Completion(
            message=Message(
                role="assistant", text=wire_msg.get("content") or "", tool_calls=tool_calls
            ),
            usage=Usage(
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
            ),
            stop_reason=stop_reason,
        )
