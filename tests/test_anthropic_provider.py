import json

import httpx
import pytest

from dudamel.exceptions import LLMError
from dudamel.llm.anthropic import AnthropicProvider
from dudamel.llm.provider import ToolSpec
from dudamel.llm.types import Message, ToolCall


def make_provider(handler, api_key: str = "k") -> AnthropicProvider:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    return AnthropicProvider(api_key=api_key, client=client)


async def test_request_shape_and_text_parse() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 9, "output_tokens": 3},
            },
        )

    p = make_provider(handler)
    c = await p.complete(
        model="claude-sonnet-5",
        messages=[
            Message(role="system", text="be brief"),
            Message(role="user", text="hi"),
        ],
        tools=[ToolSpec(name="t", description="d", json_schema={"type": "object"})],
        max_tokens=64,
    )
    assert c.message.text == "hello" and c.stop_reason == "end"
    assert c.usage.tokens_in == 9 and c.usage.tokens_out == 3
    body = seen["body"]
    assert body["system"] == "be brief"
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert body["tools"][0]["input_schema"] == {"type": "object"}
    assert seen["headers"]["x-api-key"] == "k"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"


async def test_tool_use_parse_and_tool_result_render() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": "calling"},
                    {"type": "tool_use", "id": "tu1", "name": "log_workout", "input": {"sets": 3}},
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    msgs = [
        Message(role="user", text="go"),
        Message(
            role="assistant",
            text="calling",
            tool_calls=[ToolCall(id="tu0", name="t", args={"a": 1})],
        ),
        Message(role="tool", text="ok!", tool_call_id="tu0", is_error=True),
    ]
    c = await make_provider(handler).complete(model="m", messages=msgs)
    assert c.stop_reason == "tool_calls"
    assert c.message.tool_calls == [ToolCall(id="tu1", name="log_workout", args={"sets": 3})]
    wire = seen["body"]["messages"]
    assert wire[1]["content"][-1] == {
        "type": "tool_use",
        "id": "tu0",
        "name": "t",
        "input": {"a": 1},
    }
    assert wire[2] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "tu0", "content": "ok!", "is_error": True}
        ],
    }


async def test_json_schema_unsupported() -> None:
    p = AnthropicProvider(api_key="k")
    with pytest.raises(LLMError, match="structured output"):
        await p.complete(model="m", messages=[], json_schema={"type": "object"})


async def test_http_error_mapping() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(529, json={"error": {"message": "overloaded"}})

    with pytest.raises(LLMError) as exc:
        await make_provider(handler).complete(model="m", messages=[])
    assert exc.value.retryable is True


async def test_malformed_tool_use_block_raises_llm_error() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content": [{"type": "tool_use", "name": "t", "input": {}}],
                "stop_reason": "tool_use",
            },
        )

    with pytest.raises(LLMError, match="malformed"):
        await make_provider(handler).complete(model="m", messages=[])


async def test_non_json_body_raises_llm_error() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content="<html>oops</html>", headers={"content-type": "text/html"}
        )

    with pytest.raises(LLMError, match="non-JSON"):
        await make_provider(handler).complete(model="m", messages=[])


async def test_api_key_redacted_in_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            content=f"Unauthorized: {request.headers['x-api-key']}",
        )

    with pytest.raises(LLMError) as exc:
        await make_provider(handler, api_key="sk-secret123").complete(model="m", messages=[])
    assert "sk-secret123" not in str(exc.value)
    assert "***" in str(exc.value)


async def test_non_dict_json_body_raises_llm_error() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[1, 2, 3])

    with pytest.raises(LLMError, match="malformed"):
        await make_provider(handler).complete(model="m", messages=[])


async def test_empty_assistant_message_renders_placeholder_block() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "response"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    msgs = [
        Message(role="user", text="hi"),
        Message(role="assistant"),  # empty assistant message
    ]
    await make_provider(handler).complete(model="m", messages=msgs)
    wire = seen["body"]["messages"]
    assistant_msg = wire[-1]
    assert assistant_msg["role"] == "assistant"
    assert len(assistant_msg["content"]) == 1
    assert assistant_msg["content"][0] == {"type": "text", "text": "(no content)"}
